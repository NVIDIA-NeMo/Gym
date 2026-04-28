# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare LibriSpeech-PC benchmark data for NeMo Gym.

Downloads OpenSLR-145 manifests (with punctuation+capitalization) and
OpenSLR-12 audio for the requested splits (default test-clean), then
writes one JSONL per split. Each row carries the audio data-URI on
``responses_create_params.metadata.audio_url`` (the sidechannel ``vllm_model``
consumes) and the reference transcript on ``expected_answer``.
``responses_create_params.input`` is left empty — Gym's ``prompt_config``
materializes the system+user messages from
``benchmarks/librispeech_pc/prompts/default.yaml`` at rollout time.

The on-disk JSONLs are large (~210 MB for test-clean alone — ~50 KB
base64 per ~7 s WAV × 2,417 utterances) and are gitignored. Smoke-test
data with silence-WAV placeholders lives at
``resources_servers/asr_with_pc/data/example.jsonl``.
"""

import argparse
import base64
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

MANIFESTS_URL = "https://www.openslr.org/resources/145/manifests.tar.gz"
AUDIO_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}

# test-clean is the standard LibriSpeech-PC eval split; test-other (~2.9k
# harder utterances) can be enabled either by adding a sibling benchmark
# dir (the idiomatic shape, since `ng_prepare_benchmark` enforces one
# dataset per agent) or by passing `--splits test-other` and pointing a
# separate config at the resulting JSONL. Default below emits only
# test-clean.
DEFAULT_SPLITS = ("test-clean",)
ALL_SPLITS = ("test-clean", "test-other")


def _split_filename(split: str) -> str:
    """Map ``test-clean`` → ``librispeech_pc_test_clean.jsonl`` etc.

    The on-disk JSONL names mirror the dataset entries in config.yaml
    (one JSONL per dataset entry, hyphen→underscore for filename hygiene).
    """
    return f"librispeech_pc_{split.replace('-', '_')}.jsonl"


def _download_with_progress(url: str, output_path: Path, desc: str) -> None:
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:

        def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
            if pbar.total != totalsize:
                pbar.total = totalsize
            downloaded = blocknum * blocksize
            pbar.update(max(0, downloaded - pbar.n))

        urllib.request.urlretrieve(url, output_path, reporthook)


def _download_manifests(work_dir: Path) -> None:
    """Extract just the test-clean.json and test-other.json manifests."""
    if (work_dir / "test-clean.json").exists() and (work_dir / "test-other.json").exists():
        return

    tar_path = work_dir / "manifests.tar.gz"
    _download_with_progress(MANIFESTS_URL, tar_path, "Downloading manifests")

    wanted = {"test-clean.json", "test-other.json"}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in wanted:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            with open(work_dir / name, "wb") as fout:
                shutil.copyfileobj(fobj, fout)
    os.remove(tar_path)


def _download_audio(split: str, audio_dir: Path) -> None:
    """Extract OpenSLR-12 LibriSpeech audio for the given split."""
    split_dir = audio_dir / "LibriSpeech" / split.replace("-", "_")
    if split_dir.exists():
        return

    tar_path = audio_dir / f"{split}.tar.gz"
    _download_with_progress(AUDIO_URLS[split], tar_path, f"Downloading {split}")

    with tarfile.open(tar_path, "r:gz") as tar:
        if sys.version_info >= (3, 11, 4):
            tar.extractall(audio_dir, filter="data")
        else:
            tar.extractall(audio_dir)
    os.remove(tar_path)


def _audio_file_to_base64(audio_path: Path) -> str:
    return base64.b64encode(audio_path.read_bytes()).decode("ascii")


def _iter_split_rows(split: str, work_dir: Path, audio_dir: Path) -> Iterator[dict]:
    """Yield one raw Gym JSONL row per utterance in ``split``.

    Rows carry the audio data-URI on ``responses_create_params.metadata.audio_url``
    (consumed by ``vllm_model``) and the reference transcript on
    ``expected_answer``. ``responses_create_params.input`` is intentionally
    NOT pre-populated — the benchmark's ``prompt_config`` materializes the
    system+user messages at rollout time, which is the canonical Gym pattern
    and lets the prompt template change without re-preparing the JSONL.
    """
    manifest_file = work_dir / f"{split}.json"
    with open(manifest_file, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for entry in tqdm(entries, desc=f"Encoding {split}"):
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

        audio_b64 = _audio_file_to_base64(local_audio_path)
        sample_id = local_audio_path.stem

        yield {
            "responses_create_params": {
                "metadata": {"audio_url": f"data:audio/wav;base64,{audio_b64}"},
            },
            "expected_answer": text,
            "sample_id": sample_id,
            "split": split,
        }


def prepare(work_dir: Path | None = None, splits: tuple[str, ...] = DEFAULT_SPLITS) -> Path:
    """Download LibriSpeech-PC and write per-split benchmark JSONLs.

    Args:
        work_dir: Directory to use for manifest + audio downloads. Defaults to
            ``benchmarks/librispeech_pc/data``. Reusing the same path across
            runs makes the prepare step idempotent — extracted audio + manifests
            persist between invocations.
        splits: Which splits to download + emit. Defaults to ``("test-clean",)``.
            When extra splits are requested they each land in their own
            JSONL alongside test_clean (e.g. for users wiring up a sibling
            benchmark dir for test-other).

    Returns:
        Path to ``librispeech_pc_test_clean.jsonl`` — the file the benchmark
        config's ``datasets[0].jsonl_fpath`` references. Returning a single
        Path matches the contract ``ng_prepare_benchmark`` enforces.
    """
    work_dir = work_dir or DATA_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _download_manifests(work_dir)
    for split in splits:
        _download_audio(split, work_dir)

    primary_path: Path | None = None
    for split in splits:
        out_path = DATA_DIR / _split_filename(split)
        count = 0
        with open(out_path, "w") as f:
            for row in _iter_split_rows(split, work_dir, work_dir):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        print(f"Wrote {count} rows ({split}) to {out_path}")
        if split == "test-clean":
            primary_path = out_path

    # Always return the test-clean path (the dataset entry in config.yaml).
    # If the caller asked only for test-other (sibling benchmark dir use case),
    # they shouldn't have called this prepare — they'd have their own.
    if primary_path is None:
        raise RuntimeError(
            "prepare() ran without test-clean. Add 'test-clean' to splits or "
            "use a benchmark dir whose dataset.jsonl_fpath matches the split "
            "you actually want."
        )
    return primary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech-PC benchmark for Gym")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for manifest + audio downloads (default: benchmarks/librispeech_pc/data).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=list(AUDIO_URLS.keys()),
        default=list(DEFAULT_SPLITS),
        help=(
            "Which LibriSpeech splits to download + emit. Default is "
            "test-clean only. Each emitted split lands in its own JSONL "
            "(librispeech_pc_test_clean.jsonl, librispeech_pc_test_other.jsonl)."
        ),
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else None
    prepare(work_dir=work_dir, splits=tuple(args.splits))


if __name__ == "__main__":
    main()
