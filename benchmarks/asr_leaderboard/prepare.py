# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare the HF Open ASR Leaderboard benchmark for NeMo Gym.

Downloads the 7 ESB test subsets from ``hf-audio/open-asr-leaderboard``
at a pinned revision (librispeech-clean, librispeech-other, voxpopuli,
gigaspeech, spgispeech, earnings22, ami), saves audio to FLAC files on
disk, and writes one combined JSONL the ``asr_leaderboard`` benchmark
consumes. The JSONL appears only once every subset has been written.

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

# The ESB test sets, under the repo's current name (``hf-audio/esb-datasets-test-only-sorted``
# redirects here). Pinned to a commit: the repo's main branch changes, and on 2026-05-27 it deleted
# its TED-LIUM subset because the dataset's authors withdrew it, so TED-LIUM is not prepared.
HF_REPO = "hf-audio/open-asr-leaderboard"
HF_REVISION = "de79e57adf9983993bcc72586d7313460825f5eb"  # pragma: allowlist secret

# (hf_repo, hf_config, split, text_field, id_field) for each ESB subset.
DATASET_CONFIGS: Dict[str, tuple] = {
    "librispeech_clean": (HF_REPO, "librispeech", "test.clean", "text", "id"),
    "librispeech_other": (HF_REPO, "librispeech", "test.other", "text", "id"),
    "voxpopuli": (HF_REPO, "voxpopuli", "test", "text", "id"),
    "gigaspeech": (HF_REPO, "gigaspeech", "test", "text", "id"),
    "spgispeech": (HF_REPO, "spgispeech", "test", "text", "id"),
    "earnings22": (HF_REPO, "earnings22", "test", "text", "id"),
    "ami": (HF_REPO, "ami", "test", "text", "id"),
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

    print(f"Loading {dataset_name} from {hf_repo}@{HF_REVISION} (config={hf_config}, split={hf_split})...")
    ds = load_dataset(hf_repo, hf_config, split=hf_split, revision=HF_REVISION)

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
        datasets: Subset names to prepare (default: all 7).

    Returns:
        Path to ``asr_leaderboard_benchmark.jsonl`` (the file the
        benchmark config's ``datasets[0].jsonl_fpath`` references). The
        file is replaced only after every subset is written; if a subset
        fails, an existing file stays as it was and no partial file is left.
    """
    work_dir = work_dir or DATA_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = datasets or tuple(DATASET_CONFIGS)
    out_path = DATA_DIR / "asr_leaderboard_benchmark.jsonl"
    # Gym treats an existing jsonl_fpath as prepared, so rows go to a sibling file renamed into place at the end.
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")

    total = 0
    try:
        with open(tmp_path, "w", encoding="utf-8") as fout:
            for dataset_name in datasets:
                count = 0
                for row in _iter_dataset_rows(dataset_name, work_dir / dataset_name):
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                print(f"Wrote {count} rows from {dataset_name}")
                total += count
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

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
        help="Which ESB subsets to prepare (default: all 7).",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else None
    prepare(work_dir=work_dir, datasets=tuple(args.datasets))


if __name__ == "__main__":
    main()
