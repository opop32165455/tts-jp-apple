# Architecture

## `speech_server.py`

FastAPI app, default `127.0.0.1:7790`. Config is module-level constants read from env vars (`SPEECH_SERVER_*`, falling back to `YUNA_TTS_*`); `/v1/config` echoes them.

### Lifecycle

- `lifespan`: patches the tokenizer loader, sets `mx.set_cache_limit(2 GB)` and `mx.set_wired_limit(WIRED_LIMIT_MB)` (default 3072; pins weights so macOS can't swap them out — see agent-facts), then applies `DEFAULT_TTS_ENABLED` (default on) / `DEFAULT_STT_ENABLED` (default off).
- Engine switches: `set_tts_enabled` / `set_stt_enabled`, exposed as `POST /v1/services/{tts|stt}/{on|off}`; state `TTS_STATE` / `STT_STATE` is `off | loading | on`, not persisted (restart → env defaults). `SERVICE_LOCK` serializes switching. TTS unload takes `MODEL_LOCK`; STT unload takes `STT_LOCK` (held during each transcription) and clears `mlx_whisper.transcribe.ModelHolder` (the library's own model cache; preload uses the same fp16 dtype). Both then `gc.collect()` + `mx.clear_cache()`.
- TTS follows client heartbeats: `TTS_LAST_ACTIVE` is touched by `POST /v1/heartbeat`, `/v1/tts` (before and after), `tts/on` and each load. Heartbeat with TTS off calls `ensure_tts_loading` (one background `tts-loader` thread, returns at once) or loads inline with `?wait=true`. `/v1/tts` loads on demand via `set_tts_enabled(True)` (blocks on `SERVICE_LOCK` if a load is in progress). `_idle_watcher` (asyncio task, every 30 s) calls `unload_tts_if_idle` after `TTS_IDLE_UNLOAD_SECONDS` (default 120, 0 = never). Unload must call `qwen3_fast.reset()`: compiled rollouts close over the model and would otherwise keep ~2.7 GB alive. A reloaded model is wired again (the wired limit is process-wide); load failures are kept in `TTS_LAST_ERROR` (`tts_error` in status).
- STT is on-demand: `_run_whisper` calls `set_stt_enabled(True)` when not loaded, then re-checks under `STT_LOCK` and retries if an unload slipped in. `_idle_watcher` calls `unload_stt_if_idle` after `STT_IDLE_UNLOAD_SECONDS` (default 120, 0 = never) since `STT_LAST_USED`. Lock order is always `SERVICE_LOCK` → `STT_LOCK`; never load while holding `STT_LOCK` (deadlock).
- `/health` returns 503 only while TTS is `loading`; otherwise 200 with engine states. Neither `/v1/tts` nor `/v1/transcribe` rejects for engine state.
- At import, PATH gets `.venv/bin`, `~/.local/bin`, Homebrew; if no `ffmpeg` is on PATH, imageio-ffmpeg's binary is symlinked to `.venv/bin/ffmpeg` (mlx-whisper shells out to `ffmpeg`).

### TTS — `POST /v1/tts` → `synthesize_wav_bytes`

- Sync endpoint (runs in FastAPI threadpool). `MODEL.generate()` is not thread-safe: all synthesis runs under `MODEL_LOCK`, so requests are serialized.
- Fast path (`FAST_TTS`, default on): when `qwen3_fast.supports()` (base model, x-vector path i.e. no `ref_text`, no `\n` in text) synthesis goes through `qwen3_fast.generate`, a copy of mlx-audio's single-segment loop where the per-token 15-step code-predictor rollout is one `mx.compile` graph (random state threaded through) and the per-step `mx.clear_cache()` is dropped. Bit-identical to `Model.generate` for the same seed; `scripts/check_fast_tts.py` verifies it. Compiled graphs are cached per (temperature, top_k, top_p); `_warm_up_tts` traces the default one after model load. Per token: talker ~16 ms (memory-bandwidth bound at 8-bit), code predictor ~20 ms → ~12 ms compiled.
- Speaker profile caching: `SPEAKER_PROFILE_CACHE` (`collections.OrderedDict`, true LRU bounded to 16 entries) caches speaker embedding vectors and metadata (`model_id`, `dim`, timestamps/sizes of watched files). `_resolve_speaker_profile` resolves: for default audio, explicit `SPEECH_SERVER_SPEAKER_PROFILE` takes top priority; otherwise sidecar `<stem>_speaker_profile.npz` is loaded (old `.npy` sidecars are ignored). Sidecars record `model_id`; every loaded vector is checked against `(1, speaker_encoder_config.enc_dim)` (`_check_embed_shape`). A sidecar with another model_id, wrong shape, or older than the source WAV is ignored and re-extracted. Sidecar saving is restricted to `DEFAULT_REF_AUDIO` and `ref-audio/` files. On fast path, `qwen3_fast.generate` receives `speaker_embed`, bypassing WAV load, resampling, STFT, and speaker encoder (22s WAV: ~260ms cold, ~16ms warm; bit-identical audio). There is no runtime retry: a bad profile is rejected before generation. Direct `.npy`/`.npz` passed via `ref_audio` is supported on the single-segment `/v1/tts` fast path; it is rejected with HTTP 400 on streaming, multi-line text, with `ref_text`, or if it fails to load / has the wrong shape.
- Streaming (`POST /v1/tts/stream`): Generates audio chunks iteratively. To avoid slow network clients holding `MODEL_LOCK` across socket transmissions and starving concurrent STT/TTS requests, chunk iteration `next(gen)` is protected under `MODEL_LOCK`, while `yield` (both WAV header and PCM audio bytes) runs outside the lock. Client disconnections (`GeneratorExit`) cleanly close the generator under lock.
- `max_tokens`: if the caller omits it and text ≤ 20 chars, clamp to `max(32, min(default, len*8))` to stop repetition loops on short text.
- Codec decode correctness: `mlx_fixes.apply()` (called at import in `speech_server.py` and `scripts/run_yuna_demo.py`) replaces `nn.ConvTranspose1d.__call__` with a time-chunked version. MLX 0.30.3's Metal `conv_transpose1d` corrupts the start of the output once batch × output length exceeds ~32768 frames, and the codec decoder's upsampling layers produce 24 kHz audio, so every clip > ~1.36 s lost its opening words. Items are run one by one and chunked so each call's batch × output ≤ 16384; overlaps are summed with `.at[].add`; matches CPU to float error, +~15 ms per decode. `scripts/check_mlx_fixes.py` verifies it. The model naturally starts many clips with ~0.5 s of near-silence, so leading silence is not a failure signal.
- Post-processing: 20 ms linear fade-in (kills vocoder DC pop); output < 0.3 s is rejected as truncation (error, not silent).
- After each request: `mx.clear_cache()` + `gc.collect()`.
- Response headers: `X-TTS-Ref-Audio` = ref audio actually used; `X-TTS-Profile-Cached` = `true` if the fast path used a speaker profile vector (`false` on the original path or streaming).

### STT — `POST /v1/transcribe` and `/v1/stt` (same handler)

- Async handler; writes the body to a temp file (suffix sniffed by `_guess_audio_suffix`), runs `_do_transcribe` in a thread with word timestamps.
- Segment post-processing: tighten start/end to first/last word timestamp, drop segments ≤ 0.2 s, de-duplicate and re-order overlapping segments.

### Logging

`logs/speech_server.log`, rotated at midnight, 30 days kept.

## Batch script — `scripts/run_yuna_demo.py`

- Loads the model in-process (does not use the server). Input CSV columns `id,content`; output `<out-dir>/<id>.wav`, skips existing unless `--overwrite`.
- Dynamic batches by `--batch-size` and `--max-batch-chars`; on failure retries with `--fallback-batch-size`. `--start-row` resumes.
- Post-processing (unless `--no-postprocess`): fade-in `--fade-seconds`, padded tail silence `--tail-silence-seconds` with a tiny noise floor `--tail-floor-amplitude` so downstream importers don't trim exact digital silence.

## `manage_service.sh`

Generates `~/Library/LaunchAgents/com.local.speech-server.plist` (RunAtLoad + KeepAlive, `--port 7790`), no `WorkingDirectory` (server uses absolute paths). launchd stdout/stderr → `~/Library/Logs/speech-server/`.

## `main.py`

Upstream interactive CLI. Resolves models relative to CWD (`./models/<folder>`), so run from repo root; enrolled voices in `voices/`.
