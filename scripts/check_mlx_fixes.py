#!/usr/bin/env python3
"""Check mlx_fixes against a CPU reference, and whether the MLX GPU bug it works around still exists.

    .venv/bin/python scripts/check_mlx_fixes.py
Exits non-zero if the patched GPU result differs from CPU. "raw GPU bug: gone" after an mlx upgrade
means mlx_fixes.apply() is no longer needed (keeping it is harmless).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_fixes

TOL = 1e-2  # normal float32 GPU-vs-CPU error is ~1e-3; the bug gives errors > 1


def main():
    mx.random.seed(0)
    raw_call = nn.ConvTranspose1d.__call__
    raw_bug = False
    ok = True
    # channel/stride pairs of the Qwen3-TTS codec decoder upsampling blocks
    for c_in, c_out, stride in [(768, 384, 8), (384, 192, 5), (192, 96, 4), (96, 48, 3)]:
        conv = nn.ConvTranspose1d(c_in, c_out, 2 * stride, stride=stride)
        conv.bias = mx.random.normal(conv.bias.shape)
        for out_len in (30000, 61440, 160000):
            x = mx.random.normal((1, out_len // stride, c_in))
            with mx.stream(mx.cpu):
                ref = np.array(raw_call(conv, x))
            raw_err = float(np.abs(np.array(raw_call(conv, x)) - ref).max())
            fixed_err = float(np.abs(np.array(mlx_fixes._chunked_call(conv, x)) - ref).max())
            raw_bug |= raw_err > TOL
            ok &= fixed_err <= TOL
            print(f"stride {stride} out {ref.shape[1]:6d}: raw GPU err {raw_err:8.4f}  fixed GPU err {fixed_err:.5f}")
    print(f"raw GPU bug: {'present' if raw_bug else 'gone'}")
    print("PASS: fixed path matches CPU" if ok else "FAIL: fixed path differs from CPU")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
