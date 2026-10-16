# Agent Facts

Verified environment facts and pitfalls. Add new ones here.

- **Python 3.13 only**: `requirements.txt` pins `audioop-lts` (3.13 backport). Venv must be `.venv/` (hard-coded in `manage_service.sh` and the server's PATH setup).
- **Port 7790 busy**: the launchd daemon holds it and the model in memory. `./manage_service.sh stop` before running the server in the foreground.
- **Code changes not taking effect**: the daemon keeps the old process; `./manage_service.sh restart`.
- **launchd EX_CONFIG (78)** happened when the repo lived on an external volume (chdir / log file open failed; Python needed removable-volume access). Fix was moving the repo to the internal disk; plist also omits `WorkingDirectory` and logs to `~/Library/Logs/speech-server/`.
- **STT fails with ffmpeg not found**: mlx-whisper shells out to `ffmpeg`; the server symlinks imageio-ffmpeg's versioned binary to `.venv/bin/ffmpeg` at startup. Check that `imageio-ffmpeg` is installed or `brew install ffmpeg`.
- **Tokenizer load error**: transformers 5.0.0rc3 needs `fix_mistral_regex=True` for the Qwen tokenizer; handled by `patch_transformers_tokenizer_loader()`.
- **Git push "did not receive expected object"**: repo was a shallow clone; `git fetch --unshallow upstream` fixed it.
