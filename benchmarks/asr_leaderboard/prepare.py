# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare the HF Open ASR Leaderboard benchmark for NeMo Gym.

Downloads the 8 ESB test subsets from
``hf-audio/esb-datasets-test-only-sorted`` (librispeech-clean,
librispeech-other, voxpopuli, tedlium, gigaspeech, spgispeech,
earnings22, ami), saves audio to FLAC files on disk, and writes one
combined JSONL the ``asr_leaderboard`` benchmark consumes.

Audio is referenced via ``responses_create_params.metadata.audio_path``
(the file-path sidechannel ``vllm_model`` consumes), NOT inlined as
base64 — the FLAC corpus is multi-GB. Audio paths use the cluster mount
``/dataset/asr-leaderboard/data/<dataset>/<id>.flac`` so other pipelines
running on the same cluster can share the audio cache.

``responses_create_params.input`` is left empty — Gym's ``prompt_config``
materializes the system+user messages from
``benchmarks/asr_leaderboard/prompts/default.yaml`` at rollout time.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

# Audio shorter than this trips the upstream mel-spectrogram extractor.
MIN_AUDIO_DURATION = 0.1

# Cluster-mounted audio root referenced by every JSONL row. Decoupled
# from ``work_dir`` (the on-disk FLAC output) so prepare runs can stage
# audio anywhere while every row points at the canonical mount.
AUDIO_ROOT = "/dataset/asr-leaderboard/data"

# (hf_repo, hf_config, split, text_field, id_field) for each ESB subset.
DATASET_CONFIGS: Dict[str, tuple] = {
    "librispeech_clean": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.clean", "text", "id"),
    "librispeech_other": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.other", "text", "id"),
    "voxpopuli": ("hf-audio/esb-datasets-test-only-sorted", "voxpopuli", "test", "text", "id"),
    "tedlium": ("hf-audio/esb-datasets-test-only-sorted", "tedlium", "test", "text", "id"),
    "gigaspeech": ("hf-audio/esb-datasets-test-only-sorted", "gigaspeech", "test", "text", "id"),
    "spgispeech": ("hf-audio/esb-datasets-test-only-sorted", "spgispeech", "test", "text", "id"),
    "earnings22": ("hf-audio/esb-datasets-test-only-sorted", "earnings22", "test", "text", "id"),
    "ami": ("hf-audio/esb-datasets-test-only-sorted", "ami", "test", "text", "id"),
}


def _audio_filename(raw_id: str) -> str:
    """Filename for a single sample's FLAC: slash-replace then take stem.

    HF dataset ids may contain ``/`` (e.g. AMI's ``EN2002a/EN2002a.Mix-Headset_0``);
    we replace with ``_`` so the FLAC lands as a flat file under the subset dir.
    """
    return f"{Path(str(raw_id).replace('/', '_')).stem}.flac"


def _format_row(
    entry: Dict[str, Any],
    dataset_name: str,
    text_field: str = "text",
    id_field: str = "id",
) -> Optional[Dict[str, Any]]:
    """Format one HF dataset row as a Gym JSONL entry, or return None to skip.

    Skipped when the audio is shorter than ``MIN_AUDIO_DURATION`` or when
    the reference text is empty after strip (HF Open ASR Leaderboard
    convention drops blank references).
    """
    text = entry[text_field].strip()
    if not text:
        return None

    audio_info = entry.get("audio") or {}
    if "array" in audio_info and "sampling_rate" in audio_info:
        duration = len(audio_info["array"]) / audio_info["sampling_rate"]
        if duration < MIN_AUDIO_DURATION:
            return None

    raw_id = entry[id_field]
    return {
        "responses_create_params": {
            "metadata": {"audio_path": f"{AUDIO_ROOT}/{dataset_name}/{_audio_filename(raw_id)}"},
        },
        "expected_answer": text,
        "subset_for_metrics": dataset_name,
        "sample_id": str(raw_id),
    }


def _iter_dataset_rows(dataset_name: str, audio_dir: Path) -> Iterator[Dict[str, Any]]:
    """Load one HF subset, save its FLACs to ``audio_dir``, yield formatted rows."""
    import soundfile as sf
    from datasets import load_dataset
    from tqdm import tqdm

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS)}")
    hf_repo, hf_config, hf_split, text_field, id_field = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {hf_repo} (config={hf_config}, split={hf_split})...")
    ds = load_dataset(hf_repo, hf_config, split=hf_split, trust_remote_code=True)

    audio_dir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(ds, desc=dataset_name):
        row = _format_row(entry, dataset_name, text_field=text_field, id_field=id_field)
        if row is None:
            continue
        audio_info = entry.get("audio") or {}
        if "array" in audio_info and "sampling_rate" in audio_info:
            sf.write(
                str(audio_dir / _audio_filename(entry[id_field])),
                audio_info["array"],
                audio_info["sampling_rate"],
            )
        yield row


def prepare(
    work_dir: Optional[Path] = None,
    datasets: Optional[tuple] = None,
) -> Path:
    """Download all configured ESB subsets and write the combined Gym JSONL.

    Args:
        work_dir: Directory for FLAC files. Defaults to
            ``benchmarks/asr_leaderboard/data``. Per-subset FLACs land
            in ``<work_dir>/<dataset>/<id>.flac``.
        datasets: Subset names to prepare (default: all 8).

    Returns:
        Path to ``asr_leaderboard_benchmark.jsonl`` (the file the
        benchmark config's ``datasets[0].jsonl_fpath`` references).
    """
    work_dir = work_dir or DATA_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = datasets or tuple(DATASET_CONFIGS)
    out_path = DATA_DIR / "asr_leaderboard_benchmark.jsonl"

    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for dataset_name in datasets:
            count = 0
            for row in _iter_dataset_rows(dataset_name, work_dir / dataset_name):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            print(f"Wrote {count} rows from {dataset_name}")
            total += count

    print(f"Combined {total} samples from {len(datasets)} datasets into {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASR-Leaderboard benchmark for Gym")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for FLAC downloads (default: benchmarks/asr_leaderboard/data).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS),
        default=list(DATASET_CONFIGS),
        help="Which ESB subsets to prepare (default: all 8).",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else None
    prepare(work_dir=work_dir, datasets=tuple(args.datasets))


if __name__ == "__main__":
    main()
