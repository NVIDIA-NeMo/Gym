# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the per-bucket audiobench prepare scripts.

Each top-level ``benchmarks/audiobench/prepare_<bucket>.py`` is a thin
wrapper that calls ``prepare_audiobench_bucket(bucket=...)`` here.

The bucket scripts are independent so that ``ng_prepare_benchmark`` can
satisfy its per-dataset-entry contract: each dataset entry's ``prepare()``
must return that entry's specific ``jsonl_fpath``. The actual download +
schema mapping is shared.

Schema mapping mirrors Skills'
``nemo_skills/dataset/audiobench/prepare.py`` so a row from any AudioBench
dataset comes out the same shape regardless of upstream-row column
naming variations:

  ``instruction`` ← row.instruction || row.text || cfg.instruction || ""
  ``answer``      ← row.answer || row.reference || ""
  ``audio_col``   ← row.context || row.audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from benchmarks.audiobench.DATASETS import DATASETS, DatasetSpec


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"


def _write_audio(audio_dict: Any, out_path: Path, sample_idx: int) -> Path:
    """Write the HF Audio bytes for one sample to ``out_path``.

    With ``Audio(decode=False)`` HF returns ``{"bytes": <raw>, "path": None}``;
    that's the only path we exercise in production. The array fallback is
    there so future ``datasets`` releases that auto-decode don't silently
    break — but it requires ``soundfile`` which is not on the main env.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_bytes = audio_dict.get("bytes") if isinstance(audio_dict, dict) else None
    if raw_bytes:
        out_path.write_bytes(raw_bytes)
        return out_path

    array = audio_dict.get("array") if isinstance(audio_dict, dict) else None
    if array is not None:
        import numpy as np
        import soundfile as sf

        if isinstance(array, list):
            array = np.array(array)
        sr = int(audio_dict.get("sampling_rate", 16000))
        sf.write(str(out_path), array, sr, format="WAV", subtype="PCM_16")
        return out_path

    raise ValueError(
        f"Sample {sample_idx} has no decodable audio (neither bytes nor array). "
        f"Got keys: {list(audio_dict.keys()) if isinstance(audio_dict, dict) else type(audio_dict).__name__}"
    )


def _iter_dataset_rows(slug: str, cfg: DatasetSpec, max_samples: int | None) -> Iterable[dict]:
    """Download one HF sub-dataset and yield Gym-shaped rows.

    Per-row schema:
      * ``responses_create_params.metadata.audio_path`` — absolute path to the
        WAV written under ``data/audio/``. Absolute because ``vllm_model``
        resolves ``audio_root`` against its own (per-server) cwd, which
        differs from the ``ng_run`` launch dir.
      * ``agent_ref.name`` — set by the per-bucket caller; the rollout
        collector uses this to dispatch each row to the right server
        (judge / asr / bleu / exact-match).
      * ``dataset_name`` — the AudioBench sub-dataset slug; carried through
        to the verify response and used by ``compute_subset_metrics`` for
        per-sub-dataset breakdowns inside each agent's metric block.
      * ``instruction`` / ``expected_answer`` — the user message (resolved
        via ``prompt_config: prompts/default.yaml``) and the gold reference
        text the verifier scores against.
    """
    from datasets import Audio, load_dataset

    if cfg.get("hf_data_dir"):
        ds = load_dataset(cfg["hf_repo"], data_dir=cfg["hf_data_dir"], split=cfg["hf_split"])
    else:
        ds = load_dataset(cfg["hf_repo"], split=cfg["hf_split"])

    audio_col = "context" if "context" in ds.column_names else "audio"
    ds = ds.cast_column(audio_col, Audio(decode=False))

    total = len(ds)
    if max_samples is not None and max_samples > 0:
        total = min(total, max_samples)

    for idx in range(total):
        sample = ds[idx]
        audio = sample.get(audio_col)
        if audio is None:
            continue
        audio_filename = f"{slug}_{idx:06d}.wav"
        audio_full_path = AUDIO_DIR / audio_filename
        _write_audio(audio, audio_full_path, idx)

        instruction = cfg.get("instruction") or sample.get("instruction") or sample.get("text") or "Process the audio."
        expected_answer = sample.get("answer", sample.get("reference", ""))

        yield {
            "responses_create_params": {
                "metadata": {"audio_path": str(audio_full_path.resolve())},
            },
            "instruction": instruction,
            "expected_answer": expected_answer,
            "dataset_name": slug,
            "sample_id": f"{slug}_{idx:06d}",
            "split": cfg["hf_split"],
        }


def prepare_audiobench_bucket(
    bucket: str,
    *,
    out_dir: Path | None = None,
    audio_dir: Path | None = None,
    datasets: list[str] | None = None,
    max_samples_per_dataset: int | None = None,
) -> Path:
    """Download every sub-dataset in ``bucket`` and emit a unified JSONL.

    The unified JSONL is the file the matching agent's
    ``datasets[0].jsonl_fpath`` references in ``config.yaml``. Each row
    inside carries its sub-dataset slug in ``dataset_name`` so the
    resource server's ``compute_metrics`` can break results down per
    sub-dataset via ``compute_subset_metrics``.

    Args:
      bucket: one of ``judge``, ``asr``, ``bleu``, ``exact_match`` —
        determines which agent the rollout collector will route this
        bucket's rows to.
      datasets: optional explicit list of sub-dataset slugs to include
        (must all be in DATASETS and match the bucket). Default: all
        sub-datasets in ``bucket``.
      max_samples_per_dataset: cap rows per sub-dataset (smoke testing).

    Returns the path of the bucket's JSONL.
    """
    out_dir = out_dir or DATA_DIR
    audio_dir = audio_dir or AUDIO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    selected = datasets or [s for s, c in DATASETS.items() if c["bucket"] == bucket]
    for slug in selected:
        if slug not in DATASETS:
            raise ValueError(f"unknown audiobench sub-dataset: {slug!r}")
        if DATASETS[slug]["bucket"] != bucket:
            raise ValueError(f"sub-dataset {slug!r} is bucket={DATASETS[slug]['bucket']!r}, not {bucket!r}")

    out_path = out_dir / f"audiobench_{bucket}.jsonl"
    # The judge bucket's agent is renamed to ``..._benchmarks_simple_agent`` to
    # avoid a self-reference clash with the parent ``audiobench_judge_simple_agent``
    # that lives on the server-level config (used by the server's example.jsonl
    # smoke). asr/bleu/exact_match buckets inherit from ``asr_with_pc_simple_agent``
    # whose name doesn't collide, so they keep the simpler form.
    agent_name = (
        "audiobench_judge_benchmarks_simple_agent" if bucket == "judge" else f"audiobench_{bucket}_simple_agent"
    )
    agent_ref = {"name": agent_name}

    count = 0
    skipped = []
    with open(out_path, "w", encoding="utf-8") as f:
        for slug in tqdm(selected, desc=f"audiobench/{bucket}"):
            cfg = DATASETS[slug]
            try:
                for row in _iter_dataset_rows(slug, cfg, max_samples_per_dataset):
                    row["agent_ref"] = agent_ref
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:  # noqa: BLE001
                # License-gated datasets fail with GatedRepoError until the
                # user accepts the license on HF. Skip with a printed
                # warning rather than crashing the whole bucket prepare.
                skipped.append((slug, type(e).__name__, str(e).splitlines()[0]))
                continue

    if skipped:
        print(f"\n  skipped {len(skipped)} sub-datasets (likely license-gated):")
        for slug, kind, msg in skipped:
            print(f"    {slug}: {kind} — {msg[:100]}")

    print(f"\nWrote {count} rows from {len(selected) - len(skipped)} sub-datasets to {out_path}")
    return out_path


def main_for_bucket(bucket: str) -> None:
    """CLI entry point shared by every bucket script."""
    parser = argparse.ArgumentParser(description=f"Prepare audiobench/{bucket} for Gym")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"comma-separated sub-dataset slugs (must be bucket={bucket}); default ALL",
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    args = parser.parse_args()

    selected = args.datasets.split(",") if args.datasets else None
    prepare_audiobench_bucket(
        bucket,
        datasets=selected,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
