"""Faster non-streaming Qwen3-TTS generation for the base model's x-vector path.

Replicates `Model.generate` of mlx-audio (pinned commit 9349644, qwen3_tts.py) for one segment,
with two changes that do not alter the math:
  - the 15-step code-predictor rollout of each token runs as a single `mx.compile` graph
    (random state is threaded through, so sampling draws are the same);
  - the per-token `mx.clear_cache()` is dropped.
For the same seed the audio is bit-identical to `Model.generate`, ~15% faster on 1.7B-8bit.
Verify after any mlx / mlx-audio upgrade with `scripts/check_fast_tts.py`.
"""

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_audio.tts.models.qwen3_tts.talker import apply_rotary_pos_emb
from mlx_audio.utils import load_audio
from mlx_lm.sample_utils import apply_top_k, apply_top_p, categorical_sampling

_ROLLOUTS = {}


def reset():
    """Drop compiled rollouts; they hold the model, so call this when unloading it."""
    _ROLLOUTS.clear()


def supports(model, text: str, ref_audio, ref_text: Optional[str]) -> bool:
    """True when Model.generate would take the base x-vector path with a single segment."""
    if getattr(model.config, "tts_model_type", "base") != "base":
        return False
    if not hasattr(getattr(model, "talker", None), "code_predictor") or model.speech_tokenizer is None:
        return False
    if "\n" in text:  # Model.generate splits on "\n"; leave multi-segment input to it
        return False
    use_icl = ref_audio is not None and ref_text is not None and model.speech_tokenizer.has_encoder
    return not use_icl


def _layer(layer, x, cos, sin, mask, kv):
    attn = layer.self_attn
    h = layer.input_layernorm(x)
    b, length, _ = h.shape
    q = attn.q_norm(attn.q_proj(h).reshape(b, length, attn.num_heads, attn.head_dim))
    k = attn.k_norm(attn.k_proj(h).reshape(b, length, attn.num_kv_heads, attn.head_dim))
    v = attn.v_proj(h).reshape(b, length, attn.num_kv_heads, attn.head_dim)
    q, k, v = (mx.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if kv is not None:
        k = mx.concatenate([kv[0], k], axis=2)
        v = mx.concatenate([kv[1], v], axis=2)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
    o = mx.transpose(o, (0, 2, 1, 3)).reshape(b, length, -1)
    x = x + attn.o_proj(o)
    return x + layer.mlp(layer.post_attention_layernorm(x)), (k, v)


def _make_rollout(model, temperature: float, top_k: int, top_p: float):
    talker = model.talker
    cp = talker.code_predictor
    inner = cp.model
    embed = talker.get_input_embeddings()
    num_groups = model.config.talker_config.num_code_groups

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def rollout(hidden_last, first_token):
        tokens = [first_token]
        kvs = [None] * len(inner.layers)
        pos = 0
        for ci in range(num_groups - 1):
            if ci == 0:
                x = mx.concatenate([hidden_last, embed(first_token)], axis=1)
            else:
                x = cp.codec_embedding[ci - 1](tokens[-1])
            if cp.small_to_mtp_projection is not None:
                x = cp.small_to_mtp_projection(x)
            length = x.shape[1]
            cos, sin = inner.rotary_emb(x, mx.arange(pos, pos + length)[None, :])
            mask = (
                nn.MultiHeadAttention.create_additive_causal_mask(length).astype(x.dtype)
                if length > 1
                else None
            )
            for li, layer in enumerate(inner.layers):
                x, kvs[li] = _layer(layer, x, cos, sin, mask, kvs[li])
            pos += length
            logits = cp.lm_head[ci](inner.norm(x))[:, -1, :]
            if 0 < top_k < logits.shape[-1]:
                logits = apply_top_k(logits, top_k)
            if 0.0 < top_p < 1.0:
                logits = apply_top_p(logits, top_p)
            tokens.append(categorical_sampling(logits, temperature)[:, None])
        codec_embed = embed(first_token)
        for i, c in enumerate(tokens[1:]):
            codec_embed = codec_embed + cp.codec_embedding[i](c)
        return mx.concatenate(tokens, axis=1), codec_embed

    return rollout


def generate(
    model,
    text: str,
    ref_audio=None,
    lang_code: str = "auto",
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 4096,
):
    """Returns (audio mx.array [samples], sample_rate). Caller holds the model lock."""
    key = (id(model), temperature, top_k, top_p)
    if key not in _ROLLOUTS:
        _ROLLOUTS[key] = _make_rollout(model, temperature, top_k, top_p)
    rollout = _ROLLOUTS[key]

    if ref_audio is not None:
        ref_audio = load_audio(ref_audio, sample_rate=model.sample_rate)
    input_embeds, trailing, pad_embed = model._prepare_generation_inputs(
        text, language=lang_code, speaker=None, ref_audio=ref_audio, ref_text=None
    )
    talker = model.talker
    cfg = model.config.talker_config
    eos = cfg.codec_eos_token_id
    suppress = [i for i in range(cfg.vocab_size - 1024, cfg.vocab_size) if i != eos]
    cache = talker.make_cache()
    codes, generated = [], []
    trailing_idx = 0
    for _ in range(max_tokens):
        logits, hidden = talker(input_embeds, cache=cache)
        next_token = model._sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated or None,
            suppress_tokens=suppress,
            eos_token_id=eos,
        )
        token = int(next_token[0, 0])
        if token == eos:
            break
        generated.append(token)
        all_codes, codec_embed = rollout(hidden[:, -1:, :], next_token)
        codes.append(all_codes)
        if trailing_idx < trailing.shape[1]:
            text_embed = trailing[:, trailing_idx : trailing_idx + 1, :]
            trailing_idx += 1
        else:
            text_embed = pad_embed
        input_embeds = text_embed + codec_embed
        mx.eval(input_embeds)

    if not codes:
        raise RuntimeError("no audio generated")
    audio, lengths = model.speech_tokenizer.decode(mx.stack(codes, axis=1))
    audio = audio[0]
    valid = int(lengths[0])
    if 0 < valid < audio.shape[0]:
        audio = audio[:valid]
    mx.eval(audio)
    return audio, model.sample_rate
