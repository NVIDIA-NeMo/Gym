# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Path-mode prepare for LibriSpeech-PC (POC for size-friendly audio JSONL).

Mirrors :mod:`prepare` but writes ``responses_create_params.metadata.audio_path``
instead of inlining the WAV as a base64 data-URI on
``metadata.audio_data``. Effect:

* The on-disk JSONL drops from ~210 MB (test-clean, base64-inline) to a few
  MB — only paths + transcripts. For benchmarks with bigger audio (musan,
  audiobench) and multi-seed materialized rollouts the saving compounds.
* The ``vllm_model`` server resolves each path against its ``audio_root``
  config and base64-encodes at request time, mirroring how NeMo Skills'
  ``VLLMMultimodalModel`` already handles ``{"audio": {"path": ...}}`` rows.

This is a POC; merging path-mode into the canonical prepare is a follow-up.
"""

import argparse
import json
from pathlib import Path
from typing import Iterator

from prepare import (
    DATA_DIR,
    DEFAULT_SPLITS,
    _download_audio,
    _download_manifests,
    _split_filename,
)
from tqdm import tqdm


def _path_split_filename(split: str) -> str:
    """Distinct filename so path- and inline-mode JSONLs don't collide."""
    return f"librispeech_pc_path_{split.replace('-', '_')}.jsonl"


def _iter_split_rows_path(
    split: str,
    work_dir: Path,
    audio_dir: Path,
    audio_root: Path,
) -> Iterator[dict]:
    """Yield rows carrying ``metadata.audio_path`` instead of an audio data-URI.

    Paths are emitted relative to ``audio_root`` so the JSONL is portable
    across mounts as long as the consumer (the vLLM model server) is
    configured with a matching ``audio_root``.
    """
    manifest_file = work_dir / f"{split}.json"
    with open(manifest_file, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for entry in tqdm(entries, desc=f"Indexing {split}"):
        audio_filepath = entry.get("audio_filepath", "")
        text = entry.get("text", "")
        if not audio_filepath or not text:
            continue

        rel_audio_path = audio_filepath.lstrip("/")
        if rel_audio_path.startswith("LibriSpeech/"):
            rel_audio_path = rel_audio_path[len("LibriSpeech/") :]
        local_audio_path = audio_dir / "LibriSpeech" / rel_audio_path
        if not local_audio_path.exists():
            continue

        rel_to_root = local_audio_path.relative_to(audio_root)
        sample_id = local_audio_path.stem

        yield {
            "responses_create_params": {
                "metadata": {"audio_path": str(rel_to_root)},
            },
            "expected_answer": text,
            "sample_id": sample_id,
            "split": split,
        }


def prepare_path(
    work_dir: Path | None = None,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> Path:
    """Path-mode counterpart to :func:`prepare.prepare`.

    The audio download/extraction step is shared with :mod:`prepare` (same
    on-disk layout). Only the JSONL emission differs.
    """
    work_dir = work_dir or DATA_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _download_manifests(work_dir)
    for split in splits:
        _download_audio(split, work_dir)

    audio_root = work_dir
    primary_path: Path | None = None
    for split in splits:
        out_path = DATA_DIR / _path_split_filename(split)
        count = 0
        with open(out_path, "w") as f:
            for row in _iter_split_rows_path(split, work_dir, work_dir, audio_root):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Wrote {count} rows ({split}) to {out_path} [{size_mb:.2f} MB]")

        # Print the size delta vs the inline-base64 sibling if it exists.
        inline_path = DATA_DIR / _split_filename(split)
        if inline_path.exists():
            inline_mb = inline_path.stat().st_size / (1024 * 1024)
            ratio = inline_mb / max(size_mb, 1e-9)
            print(f"  inline-base64 sibling: {inline_path.name} = {inline_mb:.2f} MB ({ratio:.1f}x larger)")

        if split == "test-clean":
            primary_path = out_path

    if primary_path is None:
        raise RuntimeError("prepare_path() ran without test-clean.")
    print(f"\naudio_root for the matching vllm_model config: {audio_root}")
    return primary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech-PC (path-mode) for Gym")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=["test-clean", "test-other"],
        default=list(DEFAULT_SPLITS),
    )
    args = parser.parse_args()
    prepare_path(
        work_dir=Path(args.work_dir) if args.work_dir else None,
        splits=tuple(args.splits),
    )


if __name__ == "__main__":
    main()
