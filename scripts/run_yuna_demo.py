import argparse
import csv
import gc
import os
import re
import time
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

import mlx_fixes

mlx_fixes.apply()  # MLX Metal conv_transpose1d bug truncated the start of every clip > ~1.36 s

# 输入数据集与参考音频默认放在项目内 data/ 与 ref-audio/（不纳入 git，需自行放置），也可用参数指定。
DEFAULT_CSV = str(PROJECT_ROOT / "data" / "yuna-missing-audio.csv")
DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base-8bit")
DEFAULT_REF_AUDIO = str(PROJECT_ROOT / "ref-audio" / "yuna.wav")
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "outputs" / "yuna-demo-10")

DEFAULT_FADE_SECONDS = 0.15
DEFAULT_TAIL_SILENCE_SECONDS = 0.5
DEFAULT_TAIL_FLOOR_AMPLITUDE = 1.0 / 32767.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a small Yuna audio demo with Qwen3-TTS MLX."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--fallback-batch-size", type=int, default=5)
    parser.add_argument("--max-batch-chars", type=int, default=300)
    parser.add_argument("--lang-code", default="japanese")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--fade-seconds",
        type=float,
        default=float(os.environ.get("FADE_SECONDS", DEFAULT_FADE_SECONDS)),
    )
    parser.add_argument(
        "--tail-silence-seconds",
        type=float,
        default=float(
            os.environ.get("TAIL_SILENCE_SECONDS", DEFAULT_TAIL_SILENCE_SECONDS)
        ),
    )
    parser.add_argument(
        "--tail-floor-amplitude",
        type=float,
        default=float(
            os.environ.get("TAIL_FLOOR_AMPLITUDE", DEFAULT_TAIL_FLOOR_AMPLITUDE)
        ),
        help=(
            "Use tiny non-zero tail samples so importers do not trim exact digital "
            "silence. Set 0 for pure zero silence."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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


def safe_filename(name):
    value = re.sub(r"[^\w.-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return value or "untitled"


def read_rows(csv_path, out_dir, start_row, limit, overwrite):
    rows = []
    path = Path(csv_path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        missing = {"id", "content"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing columns: {sorted(missing)}. Found: {reader.fieldnames}"
            )

        for source_row, row in enumerate(reader, start=1):
            if source_row < start_row:
                continue

            item_id = str(row.get("id", "")).strip()
            content = str(row.get("content", "")).strip()
            if not item_id or not content:
                continue

            safe_id = safe_filename(item_id)
            out_path = out_dir / f"{safe_id}.wav"
            if out_path.exists() and not overwrite:
                continue

            rows.append(
                {
                    "id": item_id,
                    "safe_id": safe_id,
                    "content": content,
                    "source_row": source_row,
                    "out_path": out_path,
                }
            )

            if len(rows) >= limit:
                break

    return rows


def make_dynamic_batches(rows, max_items, max_chars):
    batch = []
    char_count = 0

    for row in rows:
        row_chars = len(row["content"])
        if batch and (len(batch) >= max_items or char_count + row_chars > max_chars):
            yield batch
            batch = []
            char_count = 0

        batch.append(row)
        char_count += row_chars

    if batch:
        yield batch


def postprocess_wav(wav_path, args):
    audio, sr = audio_read(wav_path, always_2d=True, dtype="float32")

    silence_samples = int(sr * args.tail_silence_seconds)
    fade_samples = int(sr * args.fade_seconds)

    if silence_samples > 0:
        if args.tail_floor_amplitude > 0:
            tail = np.empty((silence_samples, audio.shape[1]), dtype=audio.dtype)
            tail[0::2] = args.tail_floor_amplitude
            tail[1::2] = -args.tail_floor_amplitude
            silence = tail
        else:
            silence = np.zeros((silence_samples, audio.shape[1]), dtype=audio.dtype)
        audio = np.concatenate([audio, silence], axis=0)

    if fade_samples > 0 and audio.shape[0] > silence_samples + fade_samples:
        start = audio.shape[0] - silence_samples - fade_samples
        end = audio.shape[0] - silence_samples
        fade_curve = np.linspace(1.0, 0.0, end - start, dtype=audio.dtype).reshape(
            -1, 1
        )
        audio[start:end] *= fade_curve

    if audio.shape[1] == 1:
        audio = audio[:, 0]

    audio_write(wav_path, audio, sr, format="wav")


def generate_one(model, row, ref_audio, args):
    started = time.time()
    result_count = 0

    for result in model.generate(
        text=row["content"],
        ref_audio=ref_audio,
        ref_text=None,
        lang_code=args.lang_code,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    ):
        if result_count > 0:
            raise RuntimeError(f"Expected one audio segment, got more for {row['id']}")

        audio_write(row["out_path"], result.audio, result.sample_rate, format="wav")
        result_count += 1

    if result_count == 0:
        raise RuntimeError(f"No audio generated for {row['id']}")

    if not args.no_postprocess:
        postprocess_wav(row["out_path"], args)

    elapsed = time.time() - started
    print(
        f"saved id={row['id']} source_row={row['source_row']} "
        f"chars={len(row['content'])} time={elapsed:.2f}s path={row['out_path']}"
    )


def generate_rows(model, rows, ref_audio, args):
    generated = 0
    completed_max_source_row = 0

    for row in rows:
        generate_one(model, row, ref_audio, args)
        generated += 1
        completed_max_source_row = max(completed_max_source_row, row["source_row"])
        gc.collect()

    return generated, completed_max_source_row


def run_with_fallback(model, batch_rows, ref_audio, args):
    generated = 0
    completed_max_source_row = 0

    for size in [args.fallback_batch_size, 1]:
        print(f"Falling back to batch size: {size}")
        for small_batch in make_dynamic_batches(
            batch_rows, size, max(150, min(args.max_batch_chars, 400))
        ):
            try:
                batch_generated, batch_max = generate_rows(
                    model, small_batch, ref_audio, args
                )
            except RuntimeError as exc:
                print(f"fallback batch failed: {exc!r}")
                if size == 1:
                    for row in small_batch:
                        print(
                            f"FAILED id={row['id']} source_row={row['source_row']} "
                            f"chars={len(row['content'])}"
                        )
                    return generated, completed_max_source_row, False
                continue

            generated += batch_generated
            completed_max_source_row = max(completed_max_source_row, batch_max)

    return generated, completed_max_source_row, True


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    model_path = Path(args.model)
    ref_audio = Path(args.ref_audio)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    rows = read_rows(csv_path, out_dir, args.start_row, args.limit, args.overwrite)

    print("=== Demo Setup ===")
    print(f"csv: {csv_path}")
    print(f"model: {model_path}")
    print(f"ref_audio: {ref_audio}")
    print(f"out_dir: {out_dir}")
    print(f"selected rows: {len(rows)}")
    print(
        "params: "
        f"batch_size={args.batch_size}, "
        f"fallback_batch_size={args.fallback_batch_size}, "
        f"max_batch_chars={args.max_batch_chars}, "
        f"lang_code={args.lang_code}, "
        f"fade={args.fade_seconds}s, "
        f"tail_silence={args.tail_silence_seconds}s, "
        f"tail_floor_amplitude={args.tail_floor_amplitude}"
    )

    if not rows:
        print("No pending rows found.")
        return

    print("\n=== Loading Model ===")
    load_started = time.time()
    patch_transformers_tokenizer_loader()
    model = load_model(str(model_path))
    print(f"model loaded in {time.time() - load_started:.2f}s")

    generated_total = 0
    completed_max_source_row = 0
    total_started = time.time()

    for batch_index, batch_rows in enumerate(
        make_dynamic_batches(rows, args.batch_size, args.max_batch_chars),
        start=1,
    ):
        batch_chars = sum(len(row["content"]) for row in batch_rows)
        batch_max = max(row["source_row"] for row in batch_rows)
        print(
            f"\n=== Batch {batch_index} | size={len(batch_rows)} "
            f"| chars={batch_chars} | max_source_row={batch_max} ==="
        )

        try:
            generated, batch_completed_max = generate_rows(
                model, batch_rows, str(ref_audio), args
            )
            generated_total += generated
            completed_max_source_row = max(
                completed_max_source_row, batch_completed_max
            )
        except RuntimeError as exc:
            print(f"batch failed: {exc!r}")
            generated, batch_completed_max, ok = run_with_fallback(
                model, batch_rows, str(ref_audio), args
            )
            generated_total += generated
            completed_max_source_row = max(
                completed_max_source_row, batch_completed_max
            )
            if not ok:
                print("Stop because current batch failed.")
                break

    print("\n=== Finished ===")
    print(f"generated files: {generated_total}")
    print(f"completed_max_source_row: {completed_max_source_row}")
    print(f"total generation time: {time.time() - total_started:.2f}s")


if __name__ == "__main__":
    main()
