# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthesize a tiny LibriSpeech-PC-shaped corpus and emit both JSONL flavors.

Lets us compare on-disk size of the inline-data form (``metadata.audio_data``)
vs the path form (``metadata.audio_path``) without downloading the real
dataset. Bytes are random — this is only an accounting test.

Usage::

    python benchmarks/librispeech_pc/demo_size_compare.py \\
        --num-clips 200 --clip-bytes 56000

Defaults approximate test-clean (~2.4k 7s WAVs, ~56 KB each).
"""

import argparse
import base64
import json
import random
import tempfile
from pathlib import Path


def _human(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(n_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} {units[-1]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clips", type=int, default=2417, help="default ≈ test-clean count")
    parser.add_argument("--clip-bytes", type=int, default=56000, help="default ≈ 7s 16kHz 16-bit WAV size")
    parser.add_argument("--num-seeds", type=int, default=8, help="seeds simulated for materialized output blow-up")
    args = parser.parse_args()

    rng = random.Random(0)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        audio_dir = td_path / "audio"
        audio_dir.mkdir()

        # 1. Synthesize fake audio files.
        for i in range(args.num_clips):
            (audio_dir / f"clip_{i:06d}.wav").write_bytes(rng.randbytes(args.clip_bytes))

        # 2. Inline-data input JSONL (metadata.audio_data form).
        inline_path = td_path / "input_inline.jsonl"
        with inline_path.open("w") as f:
            for i in range(args.num_clips):
                wav = audio_dir / f"clip_{i:06d}.wav"
                b64 = base64.b64encode(wav.read_bytes()).decode("ascii")
                f.write(
                    json.dumps(
                        {
                            "responses_create_params": {
                                "metadata": {"audio_data": f"data:audio/wav;base64,{b64}"},
                            },
                            "expected_answer": "lorem ipsum",
                            "sample_id": f"clip_{i:06d}",
                        }
                    )
                    + "\n"
                )

        # 3. Path-mode input JSONL (this POC).
        path_jsonl = td_path / "input_path.jsonl"
        with path_jsonl.open("w") as f:
            for i in range(args.num_clips):
                f.write(
                    json.dumps(
                        {
                            "responses_create_params": {
                                "metadata": {"audio_path": f"clip_{i:06d}.wav"},
                            },
                            "expected_answer": "lorem ipsum",
                            "sample_id": f"clip_{i:06d}",
                        }
                    )
                    + "\n"
                )

        inline_size = inline_path.stat().st_size
        path_size = path_jsonl.stat().st_size
        audio_size = sum(p.stat().st_size for p in audio_dir.iterdir())

        print(f"clips={args.num_clips}, raw audio total = {_human(audio_size)}")
        print(f"  input JSONL (audio_data): {_human(inline_size)}")
        print(f"  input JSONL (audio_path): {_human(path_size)}")
        print(f"  audio_data / audio_path size ratio: {inline_size / max(path_size, 1):.1f}x")
        print(f"  materialized outputs across {args.num_seeds} seeds (each duplicates input row):")
        print(f"    audio_data: ~{_human(inline_size * args.num_seeds)} of redundant audio")
        print(f"    audio_path: ~{_human(path_size * args.num_seeds)} (paths only)")


if __name__ == "__main__":
    main()
