# Architecture

## `speech_server.py`

FastAPI app, default `127.0.0.1:7790`. Config is module-level constants read from env vars (`SPEECH_SERVER_*`, falling back to `YUNA_TTS_*`); `/v1/config` echoes them.

### Lifecycle

- `lifespan`: patches the tokenizer loader, sets `mx.set_cache_limit(2 GB)` and `mx.set_wired_limit(WIRED_LIMIT_MB)` (default 3072; pins weights so macOS can't swap them out — see agent-facts), then applies `DEFAULT_TTS_ENABLED` (default on) / `DEFAULT_STT_ENABLED` (default off).
- Engine switches: `set_tts_enabled` / `set_stt_enabled`, exposed as `POST /v1/services/{tts|stt}/{on|off}`; state `TTS_STATE` / `STT_STATE` is `off | loading | on`, not persisted (restart → env defaults). `SERVICE_LOCK` serializes switching. TTS unload takes `MODEL_LOCK`; STT unload takes `STT_LOCK` (held during each transcription) and clears `mlx_whisper.transcribe.ModelHolder` (the library's own model cache; preload uses the same fp16 dtype). Both then `gc.collect()` + `mx.clear_cache()`.
- STT is on-demand: `_run_whisper` calls `set_stt_enabled(True)` when not loaded, then re-checks under `STT_LOCK` and retries if an unload slipped in. `_stt_idle_watcher` (asyncio task, every 30 s) calls `unload_stt_if_idle` after `STT_IDLE_UNLOAD_SECONDS` (default 600, 0 = never) since `STT_LAST_USED`. Lock order is always `SERVICE_LOCK` → `STT_LOCK`; never load while holding `STT_LOCK` (deadlock).
- `/health` returns 503 only while TTS is `loading`; otherwise 200 with engine states. `/v1/tts` returns 503 when TTS is not `on`; `/v1/transcribe` never rejects for STT state.
- At import, PATH gets `.venv/bin`, `~/.local/bin`, Homebrew; if no `ffmpeg` is on PATH, imageio-ffmpeg's binary is symlinked to `.venv/bin/ffmpeg` (mlx-whisper shells out to `ffmpeg`).

### TTS — `POST /v1/tts` → `synthesize_wav_bytes`

- Sync endpoint (runs in FastAPI threadpool). `MODEL.generate()` is not thread-safe: all synthesis runs under `MODEL_LOCK`, so requests are serialized.
- `max_tokens`: if the caller omits it and text ≤ 20 chars, clamp to `max(32, min(default, len*8))` to stop repetition loops on short text.
- Post-processing: 20 ms linear fade-in (kills vocoder DC pop); output < 0.3 s is rejected as truncation (error, not silent).
- After each request: `mx.clear_cache()` + `gc.collect()`.
- Response header `X-TTS-Ref-Audio` = ref audio actually used.

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
