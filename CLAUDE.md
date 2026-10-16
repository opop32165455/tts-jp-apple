# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Local, offline TTS (Qwen3-TTS via `mlx-audio`) and STT (`mlx-whisper`) for Apple Silicon. Runs only on macOS with Metal; there is no CPU path. The README is in Chinese and is the canonical user-facing doc (API spec, env vars, FAQ).

There is no test suite, linter config, or build step. Verification means running the server or a script against a real model in `models/` and listening to / transcribing the output.

## Commands

```bash
# Setup: Python 3.13 venv at .venv/ (requirements pin 3.13-only backports like audioop-lts; manage_service.sh hard-codes .venv)
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# Run the HTTP server in the foreground (127.0.0.1:7790)
.venv/bin/python speech_server.py --port 7790

# Run as a launchd LaunchAgent (label com.local.speech-server)
./manage_service.sh {install|start|stop|restart|status|logs|uninstall}

# Smoke-test a running server
curl -s http://127.0.0.1:7790/health
curl -X POST http://127.0.0.1:7790/v1/tts -H "Content-Type: application/json" \
  -d '{"text": "こんにちは"}' --output outputs/test.wav
curl -X POST "http://127.0.0.1:7790/v1/transcribe?language=ja" \
  --data-binary @outputs/test.wav -H "Content-Type: application/octet-stream"

# Interactive terminal app (CustomVoice / VoiceDesign / Clone, 1.7B or 0.6B); must run from repo root
.venv/bin/python main.py

# Offline batch cloning from CSV (loads the model in-process, does NOT use the server)
.venv/bin/python scripts/run_yuna_demo.py --limit 10 --batch-size 5 --out-dir outputs/yuna-demo-10
```

If the launchd service is installed, it holds port 7790 and the model in memory; stop it (`./manage_service.sh stop`) before running the server in the foreground. After editing `speech_server.py`, `./manage_service.sh restart` is required for the daemon to pick it up.

## Upstream tracking

This repo is published as `opop32165455/tts-jp-apple` (remote `origin`) and is a customized fork of `kapi2800/qwen3-tts-apple-silicon` (remote `upstream`, fetch-only; push URL disabled). The pristine upstream code is kept untouched on branch `upstream-main` (tracks `upstream/main`), and each pulled upstream version gets a tag `upstream-YYYY-MM-DD` (first snapshot: `upstream-2026-05-02` = `f74e52e`). Never commit local changes to `upstream-main`. Full Chinese guide: `docs/UPSTREAM.md`.

```bash
git fetch upstream                                  # check for new upstream commits
git log --oneline upstream-main..upstream/main      # what's new
git diff upstream-main upstream/main --stat         # which files changed
git branch -f upstream-main upstream/main && git tag upstream-<date> upstream/main   # record the new snapshot
git merge upstream-main                             # (on main) bring updates into the customized code
git diff upstream-2026-05-02 main                   # all local customizations vs the original
```

The repo is public. Voice references (`ref-audio/*`), batch input data (`data/*`), `legacy/`, company-specific scripts and `CLAUDE.local.md` are gitignored on purpose — never force-add them, and keep machine paths, bucket names and credentials out of tracked files.

## Architecture

- **`speech_server.py`** — the production FastAPI service (TTS + STT). Key points that span the file:
  - The TTS model is loaded once in the `lifespan` hook; `/health` returns 503 until it is loaded. `MODEL.generate()` is not thread-safe, so every synthesis runs under `MODEL_LOCK`, and `mx.clear_cache()` + `gc.collect()` run after each request (MLX cache capped at 2 GB).
  - `/v1/tts` is a sync endpoint (FastAPI threadpool); `/v1/stt` and `/v1/transcribe` are the same async handler that offloads to a thread. The Whisper model is loaded lazily by `mlx_whisper` on first call (cold ~20s).
  - Audio post-processing in `synthesize_wav_bytes`: 20 ms linear fade-in to kill vocoder DC pop; outputs < 0.3 s are rejected as truncation; when the caller does not pass `max_tokens` and text is ≤ 20 chars, `max_tokens` is clamped to `max(32, min(default, len*8))` to prevent repetition loops.
  - STT post-processing tightens segment bounds to word timestamps, drops segments shorter than 0.2 s, and de-duplicates/de-orders overlapping segments.
  - Config is entirely env-var driven: `SPEECH_SERVER_*`, with legacy `YUNA_TTS_*` names still honored as fallbacks — keep both when adding new settings.
  - Calls `patch_transformers_tokenizer_loader()` (forces `fix_mistral_regex=True`) before loading; `scripts/run_yuna_demo.py` carries the same patch. Keep them in sync.
- **STT is deliberately MLX-only.** The CPU `openai-whisper` fallback was removed on purpose (worse Japanese accuracy, hallucinations, timestamp drift that hides MLX failures). Do not reintroduce a CPU fallback; failures should surface as errors.
- **`scripts/run_yuna_demo.py`** — standalone batch generator: reads an `id,content` CSV (default `data/yuna-missing-audio.csv`), groups rows into dynamic batches (`--batch-size`, `--max-batch-chars`) with a smaller fallback batch size on failure, and post-processes each WAV (fade-in, padded tail silence with a tiny noise floor so downstream importers don't trim it). `--start-row` supports resuming.
- **`main.py`** — upstream interactive CLI; resolves models relative to the CWD (`./models/<folder>`) and stores enrolled voices in `voices/`.

## Data and paths

- `models/` (gitignored) holds MLX weights downloaded from `mlx-community` on HuggingFace; the server defaults to `models/Qwen3-TTS-12Hz-1.7B-Base-8bit`.
- `ref-audio/` holds the default voice references (`yuna_short_4s.wav` is the server default, `yuna.wav` the batch script default) and `data/` holds batch input CSVs. Both are gitignored (only `.gitkeep` is tracked); users supply their own files or override via `SPEECH_SERVER_REF_AUDIO` / script flags.
- `logs/` (gitignored): `speech_server.log` rotates at midnight (30 days kept). The generated launchd plist omits `WorkingDirectory` and writes launchd stdout/stderr to `~/Library/Logs/speech-server/`.
- `~/.local/bin` is prepended to PATH by `speech_server.py` and the generated plist.
- Machine-specific notes live in the gitignored `CLAUDE.local.md`.
