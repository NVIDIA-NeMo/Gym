# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare LibriSpeech-PC benchmark data for NeMo Gym.

Downloads OpenSLR-145 manifests (with punctuation+capitalization) and OpenSLR-12
audio (test-clean and test-other), then writes a single JSONL where each row's
audio WAV is base64-inlined into ``responses_create_params.input`` as an
``input_audio`` content block. This mirrors the ``circle_click`` pattern of
baking multimodal content into the JSONL at prep time, so the Gym vllm_model
server can forward it without any audio-loading logic.

The on-disk JSONL is large (~270 MB across both splits — ~50 KB base64 per
~30 s WAV × 5,500 utterances). It is gitignored; the sister
``resources_servers/librispeech_pc/data/example.jsonl`` provides 5 silence-WAV
placeholders for smoke testing without the real audio.
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
OUTPUT_FPATH = DATA_DIR / "librispeech_pc_benchmark.jsonl"

MANIFESTS_URL = "https://www.openslr.org/resources/145/manifests.tar.gz"
AUDIO_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}

# Skills' nemo_skills/dataset/librispeech-pc/__init__.py defines
# `EVAL_SPLIT = "test-clean"`. Match that here so the Gym↔Skills parity
# comparison is apples-to-apples by default. Test-other (~2.9k harder
# utterances, higher WER) can be evaluated by passing --splits explicitly.
DEFAULT_SPLITS = ("test-clean",)

SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
USER_PROMPT = "Transcribe the audio with proper punctuation and capitalization."


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


def _make_input_messages() -> list:
    # Plain text-only Responses input. The audio data-URI rides on
    # `responses_create_params.metadata.audio_url`; `vllm_audio_model` reads
    # it there and splices an `audio_url` content block into the user
    # message after Responses→Chat-Completions translation. This sidechannel
    # is required because openai's `ResponseInputContentParam` (the message
    # content union) has no audio variant — putting audio in
    # `input.user.content` directly would be rejected by simple_agent's
    # Pydantic validator at the agent layer.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]


def _iter_split_rows(split: str, work_dir: Path, audio_dir: Path) -> Iterator[dict]:
    """Yield one Gym JSONL row per utterance in ``split``."""
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
                "input": _make_input_messages(),
                # Audio sidechannel: vllm_audio_model reads metadata.audio_url
                # and splices an `audio_url` block into the user message before
                # forwarding to vLLM Chat Completions. See
                # responses_api_models/vllm_audio_model/README.md.
                "metadata": {"audio_url": f"data:audio/wav;base64,{audio_b64}"},
            },
            "expected_answer": text,
            "sample_id": sample_id,
            "split": split,
        }


def prepare(work_dir: Path | None = None, splits: tuple[str, ...] = DEFAULT_SPLITS) -> Path:
    """Download LibriSpeech-PC and write the Gym benchmark JSONL.

    Args:
        work_dir: Directory to use for manifest + audio downloads. Defaults to
            ``benchmarks/librispeech_pc/data``. Reusing the same path across
            runs makes the prepare step idempotent — extracted audio + manifests
            persist between invocations.
        splits: Which splits to include in the output JSONL. Defaults to
            ``("test-clean",)`` to match Skills' ``EVAL_SPLIT``. Pass
            ``("test-clean", "test-other")`` to evaluate both.

    Returns:
        Path to the written benchmark JSONL.
    """
    work_dir = work_dir or DATA_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _download_manifests(work_dir)
    for split in splits:
        _download_audio(split, work_dir)

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for split in splits:
            for row in _iter_split_rows(split, work_dir, work_dir):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} rows ({', '.join(splits)}) to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


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
        help="Which LibriSpeech splits to include in the JSONL. Default matches Skills' EVAL_SPLIT.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else None
    prepare(work_dir=work_dir, splits=tuple(args.splits))


if __name__ == "__main__":
    main()
