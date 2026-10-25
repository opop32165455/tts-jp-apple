#!/usr/bin/env python3
"""Check that qwen3_fast.generate matches mlx-audio's Model.generate bit-for-bit, and compare speed.

Run after upgrading mlx / mlx-audio (loads the model in-process; ~1 min):
    .venv/bin/python scripts/check_fast_tts.py
Exits non-zero if any output differs; then disable the fast path (SPEECH_SERVER_FAST_TTS=0) until fixed.
"""

import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx

import qwen3_fast
import speech_server as S

TEXTS = [
    "きょうはとてもいいてんきですね",
    "あしたはあさからあめがふるそうです",
    "電車がもうすぐ駅に着きます。",
    "このお店のラーメンはとてもおいしいです。",
]
PARAMS = dict(
    temperature=S.DEFAULT_TEMPERATURE,
    top_k=S.DEFAULT_TOP_K,
    top_p=S.DEFAULT_TOP_P,
    repetition_penalty=S.DEFAULT_REPETITION_PENALTY,
)


def max_tokens(text):  # same short-text clamp as speech_server
    return max(32, min(S.DEFAULT_MAX_TOKENS, len(text) * 8)) if len(text) <= 20 else S.DEFAULT_MAX_TOKENS


def run_original(model, text):
    for result in model.generate(
        text=text, ref_audio=S.DEFAULT_REF_AUDIO, ref_text=None, lang_code=S.DEFAULT_LANG,
        max_tokens=max_tokens(text), verbose=False, **PARAMS,
    ):
        return result.audio


def run_fast(model, text):
    audio, _ = qwen3_fast.generate(
        model, text, ref_audio=S.DEFAULT_REF_AUDIO, lang_code=S.DEFAULT_LANG,
        max_tokens=max_tokens(text), **PARAMS,
    )
    return audio


def run_fast_profile(model, text, speaker_embed):
    audio, _ = qwen3_fast.generate(
        model, text, ref_audio=S.DEFAULT_REF_AUDIO, speaker_embed=speaker_embed,
        lang_code=S.DEFAULT_LANG, max_tokens=max_tokens(text), **PARAMS,
    )
    return audio


def main():
    S.patch_transformers_tokenizer_loader()
    mx.set_cache_limit(2 * 1024**3)
    model = S.load_model(S.DEFAULT_MODEL)
    assert qwen3_fast.supports(model, TEXTS[0], S.DEFAULT_REF_AUDIO, None)

    from mlx_audio.utils import load_audio
    ref_arr = load_audio(S.DEFAULT_REF_AUDIO, sample_rate=model.sample_rate)
    cached_embed = model.extract_speaker_embedding(ref_arr)
    mx.eval(cached_embed)

    run_original(model, TEXTS[0])
    run_fast(model, TEXTS[0])  # compile
    run_fast_profile(model, TEXTS[0], cached_embed)

    ok = True
    timings = {"original": [], "fast": [], "profile": []}
    for i, text in enumerate(TEXTS):
        seed = 1000 + i
        mx.random.seed(seed)
        t0 = time.perf_counter()
        a = run_original(model, text)
        timings["original"].append(time.perf_counter() - t0)
        mx.clear_cache()

        mx.random.seed(seed)
        t0 = time.perf_counter()
        b = run_fast(model, text)
        timings["fast"].append(time.perf_counter() - t0)
        mx.clear_cache()

        mx.random.seed(seed)
        t0 = time.perf_counter()
        c = run_fast_profile(model, text, cached_embed)
        timings["profile"].append(time.perf_counter() - t0)
        mx.clear_cache()

        same_fast = a.shape == b.shape and bool(mx.array_equal(a, b))
        same_profile = a.shape == c.shape and bool(mx.array_equal(a, c))
        same = same_fast and same_profile
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {text}  ({a.shape[0] / model.sample_rate:.2f}s audio)")

    for name, values in timings.items():
        print(f"{name:8s} median {statistics.median(values):.2f}s")
    print("PASS: bit-identical" if ok else "FAIL: outputs differ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
