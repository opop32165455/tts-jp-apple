# AGENTS.md

Shared instructions for AI coding agents. Human-facing usage (install, API, env vars) lives in `README.md`.

Local offline TTS (Qwen3-TTS via `mlx-audio`) + STT (`mlx-whisper`) HTTP server for Apple Silicon. macOS + Metal only. No test suite, linter, or build step.

## Read Path

Read only this file, then the one matching row below, then the code. Do not preload `docs/` or `README.md`.

| Task | Read |
|---|---|
| `speech_server.py`: TTS/STT behavior, post-processing, concurrency, config | `docs/architecture.md` |
| `scripts/run_yuna_demo.py` batch generation | `docs/architecture.md` §Batch script |
| launchd / ports / ffmpeg / env setup problems | `docs/agent-facts.md` |
| Sync upstream, git remotes, publishing | `docs/UPSTREAM.md` |
| API shape, env var list, user docs | `README.md` (the matching section only) |

New verified facts or pitfalls go into `docs/agent-facts.md`, not into private agent memory.

## Hard Rules

- **Public repo.** `ref-audio/*`, `data/*`, `legacy/`, company scripts and `CLAUDE.local.md` are gitignored on purpose: never `git add -f` them. Keep machine paths, bucket names, credentials out of tracked files.
- STT stays MLX-only. Never add a CPU / `openai-whisper` fallback; failures must surface as errors.
- New config: `SPEECH_SERVER_*` env var, keep the legacy `YUNA_TTS_*` fallback. Document it in the README env-var table.
- `patch_transformers_tokenizer_loader()` exists in both `speech_server.py` and `scripts/run_yuna_demo.py`; change both together.
- Never commit to `upstream-main`. `upstream` remote is fetch-only.
- Do not modify or run anything in `legacy/`.

## Verify

```bash
./manage_service.sh restart          # required after editing speech_server.py (daemon holds port 7790)
curl -s http://127.0.0.1:7790/health
curl -X POST http://127.0.0.1:7790/v1/tts -H "Content-Type: application/json" -d '{"text": "こんにちは"}' --output outputs/test.wav
curl -X POST "http://127.0.0.1:7790/v1/transcribe?language=ja" --data-binary @outputs/test.wav -H "Content-Type: application/octet-stream"
```

Verification = run against the real model and check the transcription / listen to the output.

## Docs Sync

Update `README.md` when API shape, env vars or user-visible behavior change; update `docs/architecture.md` when internal mechanics described there change. No change history in docs.
