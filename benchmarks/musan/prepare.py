# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare MUSAN (Music, Speech, Noise) benchmark data for NeMo Gym.

MUSAN is used as a hallucination detector: the model is given non-speech
audio (or, in the speech category, non-target speech) and asked to
transcribe only speech. The ``asr_with_pc`` server in
``task_type=Hallucination`` mode then scores ``char_rate = (len(hyp) /
audio_duration) * 60``. A char-rate above 1500 chars/min indicates
repetition / hallucination.

Default source is OpenSLR (~11 GB, no API key) which mirrors Skills'
``--source openslr``. Each row carries ``audio_duration`` and
``responses_create_params.metadata.audio_path`` (consumed by
``vllm_model``'s file-path audio sidechannel). ``expected_answer`` is
always ``""`` — the hallucination scorer cares only about hypothesis
length normalized by audio duration.

The audio files are referenced by absolute path (``/data/musan/<cat>/audio/<file>.wav``)
which matches the Skills container mount layout, so the same WAV bytes
serve both pipelines for parity comparison.
"""

import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

import soundfile as sf
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "musan_benchmark.jsonl"

OPENSLR_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

# JSONL audio_path layout matches Skills' container mount:
#   <root>/musan/<category>/audio/musan_<category>_NNNNNN.wav
# Skills' SLURM container bind-mounts the data dir at /data, so /data
# is the canonical root and is what `audio_path` in the JSONL should
# resolve to in the cluster container regardless of where this prepare
# step ran.
DEFAULT_AUDIO_ROOT = "/data"

ALL_CATEGORIES = ("noise", "music", "speech")


def _download_with_progress(url: str, output_path: Path, desc: str) -> None:
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:

        def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
            if pbar.total != totalsize:
                pbar.total = totalsize
            downloaded = blocknum * blocksize
            pbar.update(max(0, downloaded - pbar.n))

        urllib.request.urlretrieve(url, output_path, reporthook)


def _download_openslr(work_dir: Path) -> Path:
    """Download and extract MUSAN from OpenSLR. Returns the extracted ``musan/`` dir."""
    extract_root = work_dir / "musan_openslr"
    musan_root = extract_root / "musan"
    if musan_root.exists():
        return musan_root

    work_dir.mkdir(parents=True, exist_ok=True)
    tar_path = work_dir / "musan.tar.gz"
    if not tar_path.exists():
        _download_with_progress(OPENSLR_URL, tar_path, "Downloading MUSAN (~11 GB)")

    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        if sys.version_info >= (3, 11, 4):
            tar.extractall(extract_root, filter="data")
        else:
            tar.extractall(extract_root)
    tar_path.unlink(missing_ok=True)
    return musan_root


def _audio_duration_seconds(audio_path: Path) -> float:
    """Read just the WAV header to compute duration in seconds (no decode)."""
    info = sf.info(str(audio_path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _build_row(
    *,
    category: str,
    sample_id: int,
    audio_filename: str,
    audio_path: str,
    duration: float,
) -> dict:
    """Shape one Gym JSONL row for a single MUSAN utterance.

    Mirrors Skills' manifest entry on the verifier-relevant fields
    (``audio_duration``, ``expected_answer="" ``, ``subset_for_metrics``,
    ``category``, ``task_type``). ``responses_create_params.input`` is
    intentionally empty — Gym's ``prompt_config`` materializes the
    system+user messages at rollout time.
    """
    return {
        "responses_create_params": {
            "metadata": {"audio_path": audio_path},
        },
        "expected_answer": "",
        "audio_duration": duration,
        "subset_for_metrics": f"musan_{category}",
        "category": category,
        "sample_id": sample_id,
        "original_label": Path(audio_filename).stem,
        "task_type": "Hallucination",
    }


def _iter_category_rows(category: str, dataset_root: Path, audio_root: str) -> Iterator[dict]:
    """Yield one row per WAV file in ``dataset_root/<category>/`` (recursive).

    Filename layout matches Skills (``musan_<category>_NNNNNN.wav``); each
    row's ``audio_path`` resolves under ``<audio_root>/musan/<category>/audio/``.
    """
    cat_dir = dataset_root / category
    if not cat_dir.exists():
        print(f"  [skip] {category}: not found in {dataset_root}")
        return

    wav_files = sorted(cat_dir.glob("**/*.wav"))
    if not wav_files:
        print(f"  [skip] {category}: zero WAVs under {cat_dir}")
        return

    for idx, wav_path in enumerate(tqdm(wav_files, desc=f"Processing {category}")):
        try:
            duration = _audio_duration_seconds(wav_path)
        except Exception as exc:
            print(f"  [skip] {wav_path}: {exc}")
            continue

        audio_filename = f"musan_{category}_{idx:06d}.wav"
        audio_path = f"{audio_root}/musan/{category}/audio/{audio_filename}"
        yield _build_row(
            category=category,
            sample_id=idx,
            audio_filename=audio_filename,
            audio_path=audio_path,
            duration=duration,
        )


def _materialize_flat_layout(dataset_root: Path, target_root: Path, categories: tuple[str, ...]) -> None:
    """Copy WAVs into ``<target_root>/musan/<category>/audio/musan_<cat>_NNNNNN.wav``.

    OpenSLR ships MUSAN with nested per-source directories (e.g.
    ``noise/free-sound/noise-free-sound-0000.wav``). Skills' prepare
    flattens this into a contiguous-index layout that the JSONL's
    ``audio_path`` references. We mirror that layout so a standalone
    Gym run (without a Skills-prepared mount at ``/data``) can also
    serve audio. Idempotent: skips files that already exist.
    """
    for category in categories:
        cat_src = dataset_root / category
        if not cat_src.exists():
            continue
        wav_files = sorted(cat_src.glob("**/*.wav"))
        if not wav_files:
            continue
        cat_dst = target_root / "musan" / category / "audio"
        cat_dst.mkdir(parents=True, exist_ok=True)
        for idx, src in enumerate(tqdm(wav_files, desc=f"Materializing {category}")):
            dst = cat_dst / f"musan_{category}_{idx:06d}.wav"
            if dst.exists():
                continue
            shutil.copy2(src, dst)


def prepare(
    work_dir: Path | None = None,
    categories: tuple[str, ...] = ALL_CATEGORIES,
    audio_root: str = DEFAULT_AUDIO_ROOT,
    materialize_flat_root: Path | None = None,
) -> Path:
    """Download MUSAN and write ``musan_benchmark.jsonl`` for the chosen categories.

    Args:
        work_dir: Directory used for the OpenSLR download + extract. Defaults
            to ``benchmarks/musan/data``. Reusing the same path across runs
            makes the prepare step idempotent.
        categories: Subset of ``("noise", "music", "speech")`` to include.
            Default is all three.
        audio_root: Directory under which ``musan/<cat>/audio/<file>.wav``
            paths in the JSONL resolve at rollout time. Defaults to ``/data``
            (Skills' SLURM container mount). The Gym container must bind-mount
            the same lustre dir at this path so audio reads succeed.
        materialize_flat_root: If set, also copy the WAVs into
            ``<root>/musan/<cat>/audio/musan_<cat>_NNNNNN.wav`` so a
            standalone Gym run can serve audio without a Skills-prepared
            mount. Skip when the parity setup already materializes the
            same layout via the Skills prepare job.

    Returns:
        Path to the written ``musan_benchmark.jsonl``.
    """
    work_dir = work_dir or DATA_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source: OpenSLR ({OPENSLR_URL})")
    dataset_root = _download_openslr(work_dir)
    print(f"Categories: {', '.join(categories)}")
    print(f"audio_root (resolved at rollout time): {audio_root}")

    if materialize_flat_root is not None:
        _materialize_flat_layout(dataset_root, materialize_flat_root, categories)

    count = 0
    with OUTPUT_FPATH.open("w") as fout:
        for category in categories:
            for row in _iter_category_rows(category, dataset_root, audio_root):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MUSAN benchmark for Gym")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for the OpenSLR download (default: benchmarks/musan/data).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(ALL_CATEGORIES),
        default=list(ALL_CATEGORIES),
        help="Which MUSAN categories to include (default: all three).",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default=DEFAULT_AUDIO_ROOT,
        help=(
            "Directory under which musan/<cat>/audio/<file>.wav paths in the "
            f"JSONL resolve at rollout time. Default: {DEFAULT_AUDIO_ROOT}."
        ),
    )
    parser.add_argument(
        "--materialize-flat-root",
        type=str,
        default=None,
        help=(
            "If set, also copy WAVs into <root>/musan/<cat>/audio/ with "
            "Skills' contiguous-index naming, so a standalone Gym run can "
            "serve audio without a Skills-prepared mount. Omit when the "
            "parity setup mounts a Skills-prepared dir at --audio-root."
        ),
    )
    args = parser.parse_args()
    prepare(
        work_dir=Path(args.work_dir) if args.work_dir else None,
        categories=tuple(args.categories),
        audio_root=args.audio_root,
        materialize_flat_root=Path(args.materialize_flat_root) if args.materialize_flat_root else None,
    )


if __name__ == "__main__":
    main()
