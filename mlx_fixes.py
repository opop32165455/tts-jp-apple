"""Workarounds for MLX bugs that affect this service. Call `apply()` once before decoding audio.

MLX 0.30.3 Metal `conv_transpose1d` returns wrong values at the start of the output once
batch * output length exceeds ~32768 frames (CPU is correct, plain conv1d is fine). The Qwen3-TTS codec
decoder upsamples with ConvTranspose1d to 24 kHz, so every clip longer than ~1.36 s had its first
part overwritten: the opening words came out as near-silence or garbage. We compute the op per batch
item in time chunks whose outputs stay well below that size and add the overlaps; the result matches
the CPU reference to normal float error, and is faster with a lower memory peak than the raw op on
long clips. Re-check with scripts/check_mlx_fixes.py after upgrading mlx.
"""

import mlx.core as mx
import mlx.nn as nn

_MAX_OUT = 16384  # batch * chunk output length, well below the ~32768 where the Metal kernel goes wrong
_orig_call = nn.ConvTranspose1d.__call__


def conv_transpose1d_chunked(x: mx.array, weight: mx.array, stride: int) -> mx.array:
    """Same as mx.conv_transpose1d(x, weight, stride) with padding 0, dilation 1, computed in chunks."""
    batch, length, _ = x.shape
    kernel = weight.shape[1]
    out_len = (length - 1) * stride + kernel
    if batch * out_len <= _MAX_OUT:
        return mx.conv_transpose1d(x, weight, stride)
    if batch > 1:  # the limit is on batch * length, so run the items one by one
        return mx.concatenate(
            [conv_transpose1d_chunked(x[i : i + 1], weight, stride) for i in range(batch)], axis=0
        )
    frames = max(1, (_MAX_OUT - kernel) // stride + 1)
    y = mx.zeros((batch, out_len, weight.shape[0]), dtype=x.dtype)
    for start in range(0, length, frames):
        chunk = mx.conv_transpose1d(x[:, start : start + frames], weight, stride)
        y = y.at[:, start * stride : start * stride + chunk.shape[1]].add(chunk)
    return y


def _chunked_call(self, x: mx.array) -> mx.array:
    if self.padding != 0 or self.dilation != 1 or self.output_padding != 0:
        return _orig_call(self, x)
    y = conv_transpose1d_chunked(x, self.weight, self.stride)
    return y + self.bias if "bias" in self else y


def apply():
    """Patch every nn.ConvTranspose1d in the process (idempotent)."""
    nn.ConvTranspose1d.__call__ = _chunked_call
