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
import re
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
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

import mlx_fixes
import qwen3_fast

mlx_fixes.apply()  # MLX Metal conv_transpose1d bug truncated the start of every clip > ~1.36 s

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


def _env_flag(name: str, legacy: str, default: bool) -> bool:
    value = os.environ.get(name, os.environ.get(legacy))
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# 启动时各引擎是否加载；运行中可用 POST /v1/services/{tts|stt}/{on|off} 切换（重启后恢复为这里的值）。
# STT 很少用：默认不预加载，收到识别请求时自动加载，空闲 STT_IDLE_UNLOAD_SECONDS 后自动卸载（0 = 不卸载），
# 省下 Whisper 约 1.5~2 GB 内存。
DEFAULT_TTS_ENABLED = _env_flag("SPEECH_SERVER_TTS_ENABLED", "YUNA_TTS_TTS_ENABLED", True)
DEFAULT_STT_ENABLED = _env_flag("SPEECH_SERVER_STT_ENABLED", "YUNA_TTS_STT_ENABLED", False)
# MLX 默认不锁定内存（wired limit = 0），内存紧张时 macOS 会把闲置的模型权重换出到交换区，
# 久未调用后的第一次合成要先读回 ~2.7 GB，会慢到 6~15 s。设置后模型权重被锁定（wired），不会被换出。
# 实测这不是硬上限：STT 加载时也会被锁定（约 +2 GB），STT 空闲自动卸载后释放。0 = 不锁定。
WIRED_LIMIT_MB = int(
    os.environ.get("SPEECH_SERVER_WIRED_LIMIT_MB", os.environ.get("YUNA_TTS_WIRED_LIMIT_MB", "3072"))
)
# 用 qwen3_fast 的编译版生成循环（同种子下与 mlx-audio 原版逐位一致，约快 15%）；0 = 用原版 Model.generate。
FAST_TTS = _env_flag("SPEECH_SERVER_FAST_TTS", "YUNA_TTS_FAST_TTS", True)
# TTS 跟着客户端心跳走：POST /v1/heartbeat（或 TTS 请求）会加载并保持 TTS，
# 超过这么多秒没有心跳/请求就卸载模型、释放内存（0 = 常驻不卸载）。
TTS_IDLE_UNLOAD_SECONDS = int(
    os.environ.get(
        "SPEECH_SERVER_TTS_IDLE_UNLOAD_SECONDS",
        os.environ.get("YUNA_TTS_TTS_IDLE_UNLOAD_SECONDS", "600"),
    )
)
STT_IDLE_UNLOAD_SECONDS = int(
    os.environ.get(
        "SPEECH_SERVER_STT_IDLE_UNLOAD_SECONDS",
        os.environ.get("YUNA_TTS_STT_IDLE_UNLOAD_SECONDS", "600"),
    )
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

# Engine state: "off" | "loading" | "on". SERVICE_LOCK serializes on/off switching;
# STT_LOCK keeps the Whisper model from being unloaded mid-transcription.
# Lock order is always SERVICE_LOCK -> STT_LOCK; never load while holding STT_LOCK.
SERVICE_LOCK = threading.Lock()
STT_LOCK = threading.Lock()
TTS_STATE = "off"
STT_STATE = "off"
STT_LOAD_TIME = 0.0
STT_LAST_USED = 0.0  # time.monotonic() of the last load or transcription
TTS_LAST_ACTIVE = 0.0  # time.monotonic() of the last heartbeat, TTS request or load
TTS_LAST_ERROR: Optional[str] = None  # last background load failure, cleared on success
_TTS_LOADER: Optional[threading.Thread] = None
_TTS_LOADER_LOCK = threading.Lock()


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
    streaming_interval: Optional[float] = 0.5
    format: Optional[str] = "pcm"


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


# 日语振假名（Furigana / 括号注音）正则：匹配 [汉字] 紧跟 [全角/半角括号包围的纯平假名/片假名/长音符]
# 将 "漢字（ふりがな）" 提取为 "ふりがな"，以便 TTS 准确发音，避免大模型把汉字与注音连续朗读两遍。
# 括号内含汉字、英文、数字时（如 "東京（羽田）発"、"午後（14時）"）不匹配，保留原样朗读。
RUBY_PATTERN = re.compile(r"([\u4e00-\u9faf]+)[（\(]([ぁ-んァ-ヶー]+)[）\)]")


def synthesize_wav_bytes(req: TTSRequest) -> tuple[bytes, str, Optional[str]]:
    raw_text = req.text.strip()
    if not raw_text:
        raise ValueError("text is empty")

    normalized = RUBY_PATTERN.sub(r"\2", raw_text)
    if normalized != raw_text:
        logger.info(f"[TTS FURIGANA] {raw_text!r} -> {normalized!r}")

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
            if FAST_TTS and qwen3_fast.supports(MODEL, normalized, ref_audio, ref_text):
                audio, sample_rate = qwen3_fast.generate(
                    MODEL,
                    normalized,
                    ref_audio=ref_audio,
                    lang_code=lang_code,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                )
                result_count = 1
            else:
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


def _make_wav_header(sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = 0x7FFFF000  # Streaming continuous length
    riff_size = data_size + 36
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )


def stream_audio_generator(req: TTSRequest):
    raw_text = req.text.strip()
    if not raw_text:
        raise ValueError("text is empty")

    normalized = RUBY_PATTERN.sub(r"\2", raw_text)
    if normalized != raw_text:
        logger.info(f"[TTS FURIGANA] {raw_text!r} -> {normalized!r}")

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
    streaming_interval = req.streaming_interval if req.streaming_interval is not None else 0.5
    audio_format = (req.format or "pcm").strip().lower()

    if req.max_tokens is None and len(normalized) <= 20:
        max_tokens = max(32, min(max_tokens, len(normalized) * 8))

    start_time = time.time()
    logger.info(
        f"[TTS STREAM START] text={normalized!r} | interval={streaming_interval}s | format={audio_format} | "
        f"ref_audio={ref_audio!r} | temp={temperature} | top_p={top_p} | top_k={top_k} | rep_pen={repetition_penalty}"
    )

    sample_rate = 24000
    is_first_chunk = True
    total_samples = 0
    total_chunks = 0

    try:
        with MODEL_LOCK:
            if MODEL is None:
                raise RuntimeError("model not loaded")

            if audio_format == "wav":
                yield _make_wav_header(sample_rate=sample_rate)

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
                stream=True,
                streaming_interval=streaming_interval,
                verbose=False,
            ):
                chunk = result.audio
                if chunk is None or len(chunk) == 0:
                    continue
                if not isinstance(chunk, np.ndarray):
                    chunk = np.array(chunk, dtype=np.float32)

                sample_rate = result.sample_rate

                if is_first_chunk:
                    fade_len = min(len(chunk), int(sample_rate * 0.02))
                    if fade_len > 0:
                        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                        chunk[:fade_len] = chunk[:fade_len] * fade_in
                    ttfa = time.time() - start_time
                    logger.info(f"[TTS STREAM TTFA] first chunk ready in {ttfa*1000:.1f}ms ({len(chunk)} samples)")
                    is_first_chunk = False

                total_samples += len(chunk)
                total_chunks += 1

                pcm_data = np.clip(chunk, -1.0, 1.0)
                pcm_bytes = (pcm_data * 32767.0).astype(np.int16).tobytes()
                yield pcm_bytes

        elapsed = time.time() - start_time
        total_dur = total_samples / sample_rate if sample_rate > 0 else 0.0
        rtf = elapsed / total_dur if total_dur > 0 else 0.0
        logger.info(
            f"[TTS STREAM SUCCESS] text={normalized!r} | chunks={total_chunks} | audio_duration={total_dur:.2f}s | "
            f"synthesis_time={elapsed:.2f}s | rtf={rtf:.2f}x"
        )
    except GeneratorExit:
        logger.info(f"[TTS STREAM CANCEL] client disconnected after {total_chunks} chunks")
    except Exception as exc:
        logger.error(f"[TTS STREAM ERROR] text={normalized!r} | error={exc!r}")
        raise
    finally:
        touch_tts()
        _release_mlx_memory()


def _release_mlx_memory():
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _warm_up_tts():
    """Trace the compiled fast path for the default sampling params (~4 s) before serving."""
    if not FAST_TTS or not qwen3_fast.supports(MODEL, "こんにちは", DEFAULT_REF_AUDIO, DEFAULT_REF_TEXT or None):
        return
    started = time.time()
    try:
        with MODEL_LOCK:
            qwen3_fast.generate(
                MODEL,
                "こんにちは",
                ref_audio=DEFAULT_REF_AUDIO,
                lang_code=DEFAULT_LANG,
                temperature=DEFAULT_TEMPERATURE,
                top_k=DEFAULT_TOP_K,
                top_p=DEFAULT_TOP_P,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                max_tokens=32,
            )
        logger.info(f"TTS fast path warmed up in {time.time() - started:.2f}s")
    except Exception as exc:
        logger.warning(f"TTS warm-up failed (requests will still work): {exc!r}")
    finally:
        _release_mlx_memory()


def touch_tts():
    """Mark TTS as in use; postpones the idle unload."""
    global TTS_LAST_ACTIVE
    TTS_LAST_ACTIVE = time.monotonic()


def _unload_tts_locked():
    """Caller holds SERVICE_LOCK."""
    global MODEL, TTS_STATE
    with MODEL_LOCK:  # waits for an in-flight synthesis to finish
        MODEL = None
        qwen3_fast.reset()
    TTS_STATE = "off"
    _release_mlx_memory()


def set_tts_enabled(enabled: bool):
    global MODEL, MODEL_LOAD_TIME, TTS_STATE, TTS_LAST_ERROR
    with SERVICE_LOCK:
        if enabled and TTS_STATE != "on":
            TTS_STATE = "loading"
            logger.info(f"Loading TTS model from: {DEFAULT_MODEL}")
            started = time.time()
            try:
                model = load_model(DEFAULT_MODEL)
            except Exception as exc:
                TTS_STATE = "off"
                TTS_LAST_ERROR = repr(exc)
                raise
            with MODEL_LOCK:
                MODEL = model
            MODEL_LOAD_TIME = time.time() - started
            _warm_up_tts()
            TTS_STATE = "on"
            TTS_LAST_ERROR = None
            touch_tts()
            logger.info(f"TTS Model loaded successfully in {MODEL_LOAD_TIME:.2f}s")
        elif not enabled and TTS_STATE != "off":
            _unload_tts_locked()
            logger.info("TTS model unloaded")


def _load_tts_in_background():
    try:
        set_tts_enabled(True)
    except Exception as exc:
        logger.error(f"[TTS LOAD ERROR] {exc!r}")


def ensure_tts_loading():
    """Start loading TTS in a background thread if it is off; returns immediately."""
    global _TTS_LOADER
    with _TTS_LOADER_LOCK:
        if TTS_STATE == "on" or (_TTS_LOADER is not None and _TTS_LOADER.is_alive()):
            return
        _TTS_LOADER = threading.Thread(target=_load_tts_in_background, name="tts-loader", daemon=True)
        _TTS_LOADER.start()


def unload_tts_if_idle():
    with SERVICE_LOCK:
        idle = time.monotonic() - TTS_LAST_ACTIVE
        if TTS_STATE == "on" and idle >= TTS_IDLE_UNLOAD_SECONDS:
            _unload_tts_locked()
            logger.info(f"TTS model unloaded after {idle:.0f}s without heartbeat")


def _unload_stt_locked():
    """Caller holds SERVICE_LOCK and STT_LOCK."""
    global STT_STATE
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = None
    ModelHolder.model_path = None
    STT_STATE = "off"
    _release_mlx_memory()


def set_stt_enabled(enabled: bool):
    global STT_STATE, STT_LOAD_TIME, STT_LAST_USED
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    with SERVICE_LOCK:
        if enabled and STT_STATE != "on":
            STT_STATE = "loading"
            logger.info(f"Loading Whisper model: {DEFAULT_MLX_WHISPER_MODEL}")
            started = time.time()
            try:
                # Same cache and dtype that mlx_whisper.transcribe() uses (fp16 default).
                ModelHolder.get_model(DEFAULT_MLX_WHISPER_MODEL, mx.float16)
            except Exception:
                STT_STATE = "off"
                raise
            STT_LOAD_TIME = time.time() - started
            STT_LAST_USED = time.monotonic()
            STT_STATE = "on"
            logger.info(f"Whisper model loaded in {STT_LOAD_TIME:.2f}s")
        elif not enabled and STT_STATE != "off":
            with STT_LOCK:  # waits for an in-flight transcription to finish
                _unload_stt_locked()
            logger.info("Whisper model unloaded")


def unload_stt_if_idle():
    with SERVICE_LOCK, STT_LOCK:
        idle = time.monotonic() - STT_LAST_USED
        if STT_STATE == "on" and idle >= STT_IDLE_UNLOAD_SECONDS:
            _unload_stt_locked()
            logger.info(f"Whisper model unloaded after {idle:.0f}s idle")


async def _idle_watcher():
    while True:
        await asyncio.sleep(30)
        now = time.monotonic()
        if (
            TTS_IDLE_UNLOAD_SECONDS > 0
            and TTS_STATE == "on"
            and now - TTS_LAST_ACTIVE >= TTS_IDLE_UNLOAD_SECONDS
        ):
            try:
                await asyncio.to_thread(unload_tts_if_idle)
            except Exception as exc:
                logger.error(f"[TTS IDLE UNLOAD ERROR] {exc!r}")
        if (
            STT_IDLE_UNLOAD_SECONDS > 0
            and STT_STATE == "on"
            and now - STT_LAST_USED >= STT_IDLE_UNLOAD_SECONDS
        ):
            try:
                await asyncio.to_thread(unload_stt_if_idle)
            except Exception as exc:
                logger.error(f"[STT IDLE UNLOAD ERROR] {exc!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    patch_transformers_tokenizer_loader()
    try:
        import mlx.core as mx

        mx.set_cache_limit(2 * 1024 * 1024 * 1024)
        if WIRED_LIMIT_MB > 0:
            mx.set_wired_limit(WIRED_LIMIT_MB * 1024 * 1024)
            logger.info(f"MLX wired limit set to {WIRED_LIMIT_MB} MB")
    except Exception as exc:
        logger.warning(f"MLX memory limits not applied: {exc!r}")
    set_tts_enabled(DEFAULT_TTS_ENABLED)
    set_stt_enabled(DEFAULT_STT_ENABLED)
    watcher = asyncio.create_task(_idle_watcher())
    yield
    watcher.cancel()


app = FastAPI(
    title="Local Speech Server (TTS & STT)",
    version="1.0.0",
    description="GPU-accelerated local Text-to-Speech and Speech-to-Text service for Apple Silicon",
    lifespan=lifespan,
)


def _services_status() -> dict:
    return {
        "tts": TTS_STATE,
        "stt": STT_STATE,
        "tts_idle_unload_seconds": TTS_IDLE_UNLOAD_SECONDS,
        "tts_unload_in_seconds": (
            max(0, int(TTS_IDLE_UNLOAD_SECONDS - (time.monotonic() - TTS_LAST_ACTIVE)))
            if TTS_STATE == "on" and TTS_IDLE_UNLOAD_SECONDS > 0
            else None
        ),
        "tts_error": TTS_LAST_ERROR,
        "model_load_seconds": round(MODEL_LOAD_TIME, 3),
        "stt_load_seconds": round(STT_LOAD_TIME, 3),
        "stt_idle_unload_seconds": STT_IDLE_UNLOAD_SECONDS,
        "stt_unload_in_seconds": (
            max(0, int(STT_IDLE_UNLOAD_SECONDS - (time.monotonic() - STT_LAST_USED)))
            if STT_STATE == "on" and STT_IDLE_UNLOAD_SECONDS > 0
            else None
        ),
    }


@app.get("/health")
async def health():
    if TTS_STATE == "loading":
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {
        "status": "ok",
        "service": "speech_server",
        "lang_code": DEFAULT_LANG,
        **_services_status(),
    }


@app.get("/v1/services")
async def get_services():
    return _services_status()


@app.post("/v1/heartbeat")
async def heartbeat(wait: bool = False):
    """Client is active: keep TTS loaded (wired in RAM) for another TTS_IDLE_UNLOAD_SECONDS.

    Loads TTS in the background when it is off and returns at once; wait=true blocks
    until it is ready (or the load fails).
    """
    touch_tts()
    if TTS_STATE != "on":
        if wait:
            try:
                await asyncio.to_thread(set_tts_enabled, True)
            except Exception as exc:
                logger.error(f"[TTS LOAD ERROR] {exc!r}")
                raise HTTPException(status_code=500, detail=f"tts load failed: {exc}") from exc
        else:
            ensure_tts_loading()
    return {"ready": TTS_STATE == "on", **_services_status()}


@app.post("/v1/services/{name}/{action}")
def switch_service(name: str, action: str):
    setters = {"tts": set_tts_enabled, "stt": set_stt_enabled}
    if name not in setters or action not in ("on", "off"):
        raise HTTPException(status_code=404, detail="use /v1/services/{tts|stt}/{on|off}")
    if name == "tts" and action == "on":
        touch_tts()
    try:
        setters[name](action == "on")
    except Exception as exc:
        logger.error(f"[SERVICE ERROR] {name} {action} failed: {exc!r}")
        raise HTTPException(status_code=500, detail=f"{name} {action} failed: {exc}") from exc
    return _services_status()


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
    touch_tts()  # before loading, so the idle watcher can't unload in between
    if TTS_STATE != "on":
        try:
            set_tts_enabled(True)  # load on demand; waits for an in-progress load
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"tts load failed: {exc}") from exc
    try:
        wav, ref_audio, ref_text = synthesize_wav_bytes(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"tts failed: {exc}") from exc
    finally:
        touch_tts()

    headers = {
        "X-TTS-Ref-Audio": ref_audio,
        "X-TTS-Ref-Text": urllib.parse.quote(ref_text or ""),
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


@app.post("/v1/tts/stream")
def tts_stream(request: TTSRequest):
    touch_tts()
    if TTS_STATE != "on":
        try:
            set_tts_enabled(True)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"tts load failed: {exc}") from exc

    fmt = (request.format or "pcm").strip().lower()
    media_type = "audio/wav" if fmt == "wav" else "audio/pcm"
    headers = {
        "X-Sample-Rate": "24000",
        "X-Channels": "1",
        "X-Audio-Format": "pcm_s16le",
        "X-TTS-Ref-Audio": request.ref_audio or DEFAULT_REF_AUDIO,
        "X-TTS-Ref-Text": urllib.parse.quote(request.ref_text or DEFAULT_REF_TEXT or ""),
    }
    try:
        return StreamingResponse(
            stream_audio_generator(request),
            media_type=media_type,
            headers=headers,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"tts stream failed: {exc}") from exc


@app.get("/v1/tts/stream")
def tts_stream_get(
    text: str,
    format: Optional[str] = "pcm",
    streaming_interval: Optional[float] = 0.5,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    ref_audio: Optional[str] = None,
    lang_code: Optional[str] = None,
):
    req = TTSRequest(
        text=text,
        format=format,
        streaming_interval=streaming_interval,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        ref_audio=ref_audio,
        lang_code=lang_code,
    )
    return tts_stream(req)


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
    global STT_LAST_USED
    import mlx_whisper

    while True:
        if STT_STATE != "on":
            set_stt_enabled(True)  # load on demand; outside STT_LOCK to keep lock order
        with STT_LOCK:
            if STT_STATE != "on":
                continue  # unloaded between load and lock (idle/manual off); load again
            try:
                res = mlx_whisper.transcribe(
                    tmp_path,
                    path_or_hf_repo=DEFAULT_MLX_WHISPER_MODEL,
                    language=language,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    verbose=False,
                )
            finally:
                STT_LAST_USED = time.monotonic()
            break
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
