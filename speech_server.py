#!/usr/bin/env python3
"""
Local Speech Server (TTS & STT)
Provides local GPU-accelerated Text-to-Speech (Qwen3-TTS on MLX)
and Speech-to-Text (Whisper) on Apple Silicon.
"""

import argparse
import asyncio
import io
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.parse
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Ensure local venv bin and tools are in PATH
venv_bin = os.path.join(BASE_DIR, ".venv", "bin")
os.environ["PATH"] = (
    f"{venv_bin}:{os.path.expanduser('~/.local/bin')}:/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}"
)

# mlx-whisper shells out to `ffmpeg`; imageio-ffmpeg's bundled binary has a versioned
# name (ffmpeg-macos-aarch64-vX), so expose it as .venv/bin/ffmpeg when none is on PATH.
try:
    import shutil

    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg

        ffmpeg_link = os.path.join(venv_bin, "ffmpeg")
        if os.path.islink(ffmpeg_link):
            os.remove(ffmpeg_link)
        os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg_link)
except Exception:
    pass

from mlx_audio.tts.utils import load_model

# Environment Configurations (with backward compatibility)
DEFAULT_HOST = os.environ.get(
    "SPEECH_SERVER_HOST", os.environ.get("YUNA_TTS_HOST", "127.0.0.1")
)
DEFAULT_PORT = int(
    os.environ.get("SPEECH_SERVER_PORT", os.environ.get("YUNA_TTS_PORT", "7790"))
)
DEFAULT_MODEL = os.environ.get(
    "SPEECH_SERVER_MODEL",
    os.environ.get(
        "YUNA_TTS_MODEL",
        os.path.join(BASE_DIR, "models", "Qwen3-TTS-12Hz-1.7B-Base-8bit"),
    ),
)
DEFAULT_REF_AUDIO = os.environ.get(
    "SPEECH_SERVER_REF_AUDIO",
    os.environ.get(
        "YUNA_TTS_REF_AUDIO",
        # 默认参考音频放在项目内 ref-audio/（不纳入 git，需自行放置），也可用环境变量指定。
        os.path.join(BASE_DIR, "ref-audio", "yuna_short_4s.wav"),
    ),
)
DEFAULT_REF_TEXT = (
    os.environ.get(
        "SPEECH_SERVER_REF_TEXT", os.environ.get("YUNA_TTS_REF_TEXT", "")
    ).strip()
    or None
)
DEFAULT_LANG = os.environ.get(
    "SPEECH_SERVER_LANG", os.environ.get("YUNA_TTS_LANG", "japanese")
)
DEFAULT_TEMPERATURE = float(
    os.environ.get(
        "SPEECH_SERVER_TEMPERATURE", os.environ.get("YUNA_TTS_TEMPERATURE", "0.65")
    )
)
DEFAULT_TOP_P = float(
    os.environ.get("SPEECH_SERVER_TOP_P", os.environ.get("YUNA_TTS_TOP_P", "0.95"))
)
DEFAULT_TOP_K = int(
    os.environ.get("SPEECH_SERVER_TOP_K", os.environ.get("YUNA_TTS_TOP_K", "50"))
)
DEFAULT_REPETITION_PENALTY = float(
    os.environ.get(
        "SPEECH_SERVER_REPETITION_PENALTY",
        os.environ.get("YUNA_TTS_REPETITION_PENALTY", "1.1"),
    )
)
DEFAULT_MAX_TOKENS = int(
    os.environ.get(
        "SPEECH_SERVER_MAX_TOKENS", os.environ.get("YUNA_TTS_MAX_TOKENS", "512")
    )
)
# STT 只走 MLX（Apple Silicon 统一内存/GPU）；CPU 版 openai-whisper 已禁用。
# 模型可用 SPEECH_SERVER_MLX_WHISPER_MODEL 覆盖。
DEFAULT_MLX_WHISPER_MODEL = os.environ.get(
    "SPEECH_SERVER_MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
)


# ==========================================
# Logging Configuration (Daily Rotating)
# ==========================================
def setup_logging():
    log_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    daily_log_file = os.path.join(LOG_DIR, "speech_server.log")
    file_handler = TimedRotatingFileHandler(
        filename=daily_log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    for uvicorn_log in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        u_logger = logging.getLogger(uvicorn_log)
        u_logger.handlers = []
        u_logger.addHandler(file_handler)
        u_logger.addHandler(console_handler)
        u_logger.propagate = False

    return logging.getLogger("speech_server")


logger = setup_logging()

MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_LOAD_TIME = 0.0


class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    lang_code: Optional[str] = None


def patch_transformers_tokenizer_loader():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return

    original_from_pretrained = AutoTokenizer.from_pretrained

    def from_pretrained_with_mistral_fix(*args, **kwargs):
        if "fix_mistral_regex" in kwargs:
            return original_from_pretrained(*args, **kwargs)
        try:
            return original_from_pretrained(*args, fix_mistral_regex=True, **kwargs)
        except TypeError as exc:
            # transformers 5.0.0rc3 passes fix_mistral_regex twice internally, so an
            # explicit value always raises; for the Qwen tokenizer the fix does not
            # change tokenization anyway, so fall back to the plain load.
            if "fix_mistral_regex" not in str(exc):
                raise
            return original_from_pretrained(*args, **kwargs)

    AutoTokenizer.from_pretrained = from_pretrained_with_mistral_fix


def synthesize_wav_bytes(req: TTSRequest) -> tuple[bytes, str, Optional[str]]:
    normalized = req.text.strip()
    if not normalized:
        raise ValueError("text is empty")

    temperature = (
        req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    )
    top_p = req.top_p if req.top_p is not None else DEFAULT_TOP_P
    top_k = req.top_k if req.top_k is not None else DEFAULT_TOP_K
    repetition_penalty = (
        req.repetition_penalty
        if req.repetition_penalty is not None
        else DEFAULT_REPETITION_PENALTY
    )
    max_tokens = (
        req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS
    )
    ref_audio = req.ref_audio if req.ref_audio is not None else DEFAULT_REF_AUDIO
    ref_text = req.ref_text if req.ref_text is not None else DEFAULT_REF_TEXT
    if ref_text == "":
        ref_text = None
    lang_code = req.lang_code if req.lang_code is not None else DEFAULT_LANG

    # Prevent autoregressive repetition on short texts/words
    if req.max_tokens is None and len(normalized) <= 20:
        max_tokens = max(32, min(max_tokens, len(normalized) * 8))

    start_time = time.time()
    logger.info(
        f"[TTS START] text={normalized!r} | ref_audio={ref_audio!r} | "
        f"ref_text={ref_text!r} | lang={lang_code!r} | temp={temperature} | top_p={top_p} | "
        f"top_k={top_k} | rep_pen={repetition_penalty} | max_tokens={max_tokens}"
    )

    try:
        with MODEL_LOCK:
            if MODEL is None:
                raise RuntimeError("model not loaded")
            result_count = 0
            audio = None
            sample_rate = None
            for result in MODEL.generate(
                text=normalized,
                ref_audio=ref_audio,
                ref_text=ref_text,
                lang_code=lang_code,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                verbose=False,
            ):
                audio = result.audio
                sample_rate = result.sample_rate
                result_count += 1
                break

        if result_count == 0 or audio is None or sample_rate is None:
            raise RuntimeError("no audio generated")

        import numpy as np

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        audio_dur = len(audio) / sample_rate
        if audio_dur < 0.3:
            raise RuntimeError(
                f"generated audio too short ({audio_dur:.2f}s), possible early truncation"
            )

        # 消除神经声码器起始隐状态瞬态直流冲击（DC Pop / 咚的一声）
        fade_len = min(len(audio), int(sample_rate * 0.02))
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            audio[:fade_len] = audio[:fade_len] * fade_in

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        wav_bytes = buffer.getvalue()
        elapsed = time.time() - start_time
        rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

        logger.info(
            f"[TTS SUCCESS] text={normalized!r} | audio_duration={audio_dur:.2f}s | "
            f"synthesis_time={elapsed:.2f}s | rtf={rtf:.2f}x | wav_bytes={len(wav_bytes)}"
        )
        return wav_bytes, ref_audio, ref_text
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(
            f"[TTS ERROR] text={normalized!r} | elapsed={elapsed:.2f}s | error={exc!r}"
        )
        raise
    finally:
        try:
            import gc
            import mlx.core as mx

            mx.clear_cache()
            gc.collect()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_LOAD_TIME
    started = time.time()
    logger.info(f"Loading TTS model from: {DEFAULT_MODEL}")
    patch_transformers_tokenizer_loader()
    try:
        import mlx.core as mx

        mx.set_cache_limit(2 * 1024 * 1024 * 1024)
    except Exception:
        pass
    MODEL = load_model(DEFAULT_MODEL)
    MODEL_LOAD_TIME = time.time() - started
    logger.info(f"TTS Model loaded successfully in {MODEL_LOAD_TIME:.2f}s")
    yield


app = FastAPI(
    title="Local Speech Server (TTS & STT)",
    version="1.0.0",
    description="GPU-accelerated local Text-to-Speech and Speech-to-Text service for Apple Silicon",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    if MODEL is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {
        "status": "ok",
        "service": "speech_server",
        "model_load_seconds": round(MODEL_LOAD_TIME, 3),
        "lang_code": DEFAULT_LANG,
    }


@app.get("/v1/config")
async def get_config():
    return {
        "model": DEFAULT_MODEL,
        "ref_audio": DEFAULT_REF_AUDIO,
        "ref_text": DEFAULT_REF_TEXT,
        "lang_code": DEFAULT_LANG,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "max_tokens": DEFAULT_MAX_TOKENS,
        # STT 固定走 MLX（CPU 版 openai-whisper 已禁用）
        "whisper_model": DEFAULT_MLX_WHISPER_MODEL,
        "whisper_backend": "mlx",
    }


@app.post("/v1/tts")
def tts(request: TTSRequest):
    try:
        wav, ref_audio, ref_text = synthesize_wav_bytes(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"tts failed: {exc}") from exc

    headers = {
        "X-TTS-Ref-Audio": ref_audio,
        "X-TTS-Ref-Text": urllib.parse.quote(ref_text or ""),
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


# 注意：原先的 CPU 版 openai-whisper（`get_whisper_model` + `import whisper`）已整体移除，
# STT 只走 MLX。不要再引入 CPU 兜底——它精度更低，且会掩盖 MLX 的真实故障。


def _guess_audio_suffix(body: bytes) -> str:
    if len(body) < 12:
        return ".bin"
    head = body[:12]
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if head[:4] == b"RIFF":
        return ".wav"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".bin"


def _run_whisper(tmp_path: str, language: Optional[str]) -> dict:
    """只走 MLX（Apple Silicon 统一内存/GPU）做 STT，返回与 openai-whisper 同构的 result dict。

    CPU 版 openai-whisper 已**整体禁用**。实测同一首 4 分 36 秒的日语歌曲：
      - openai-whisper `small` 走 CPU：46.9–53.9s，日语识别质量差（会出现重复幻觉），
        锚点还会出现 2 处约 8 秒的错配；
      - MLX `large-v3-turbo`：冷启动 ~22s、热态 **10–13s**，锚点无错配。
    保留精度更低的 CPU 兜底，只会在 MLX 偶发失败时**静默产出劣质时间轴**，
    反而更难排查——因此直接让失败显性化（抛错，由调用方记 ERROR 并向后端返回失败）。
    """
    import mlx_whisper

    res = mlx_whisper.transcribe(
        tmp_path,
        path_or_hf_repo=DEFAULT_MLX_WHISPER_MODEL,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=0.0,
        verbose=False,
    )
    if not res.get("segments"):
        raise RuntimeError(
            f"mlx whisper 返回空结果（模型异常或音频不可识别）: {DEFAULT_MLX_WHISPER_MODEL}"
        )
    return res


def _do_transcribe(body: bytes, suffix: str, language: Optional[str] = "ja"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        res = _run_whisper(tmp_path, language)

        raw_segments = []
        for s in res.get("segments", []):
            text_val = str(s.get("text", "")).strip()
            if not text_val:
                continue
            start_val = float(s.get("start", 0.0))
            end_val = float(s.get("end", 0.0))
            words = s.get("words", [])
            if words:
                first_w = next((w for w in words if w.get("start") is not None), None)
                if first_w:
                    start_val = float(first_w["start"])
                last_w = next((w for w in reversed(words) if w.get("end") is not None), None)
                if last_w:
                    end_val = float(last_w["end"])
            if end_val > start_val + 0.2:
                raw_segments.append(
                    (float(round(start_val, 2)), float(round(end_val, 2)), text_val)
                )

        raw_segments.sort(key=lambda x: x[0])
        segments = []
        for start, end, text in raw_segments:
            if segments:
                prev = segments[-1]
                if abs(prev["start"] - start) < 0.5 and prev["text"] == text:
                    continue
                if start < prev["start"]:
                    continue
            segments.append(
                {"start": float(start), "end": float(end), "text": str(text)}
            )

        elapsed = time.time() - t0
        logger.info(
            f"[Transcribe SUCCESS] segments={len(segments)} elapsed={elapsed:.2f}s"
        )
        return {"segments": segments}
    except Exception as e:
        logger.error(f"[Transcribe ERROR] error={e!r}")
        raise HTTPException(
            status_code=500, detail=f"transcription failed: {e}"
        ) from e
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/v1/transcribe")
@app.post("/v1/stt")
async def transcribe(request: Request, language: Optional[str] = "ja"):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="request body is empty")
    return await asyncio.to_thread(
        _do_transcribe, body, _guess_audio_suffix(body), language
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Speech HTTP Server (TTS & STT)"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
