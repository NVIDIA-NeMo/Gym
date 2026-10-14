# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the per-bucket audiobench prepare scripts.

Each ``benchmarks/audiobench/<bucket>/prepare.py`` is a thin wrapper that calls
``prepare_audiobench_bucket(bucket=...)`` here. The buckets are separate
benchmarks because ``gym eval prepare`` requires one benchmark dataset per
config; the download and schema mapping are shared.

Schema mapping mirrors Skills'
``nemo_skills/dataset/audiobench/prepare.py`` so a row from any AudioBench
dataset comes out the same shape regardless of upstream-row column
naming variations:

  ``instruction`` ← cfg.instruction || row.instruction || row.text || "Process the audio."
  ``answer``      ← row.reference || row.answer || ""  (``{"text": [...]}`` answers are unwrapped)
  ``audio_col``   ← row.context || row.audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from benchmarks.audiobench.DATASETS import DATASETS, HF_REVISIONS, DatasetSpec


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


def _answer_text(answer: Any) -> str:
    """Return a reference answer as text; Spoken-MQA stores it as ``{"text": ["731"], "audio": None}``."""
    if isinstance(answer, dict):
        answer = answer.get("text", "")
    if isinstance(answer, list):
        answer = " ".join(str(part) for part in answer)
    return str(answer)


def _iter_dataset_rows(slug: str, cfg: DatasetSpec, *, max_samples: int | None, audio_prefix: str) -> Iterable[dict]:
    """Download one HF sub-dataset at its pinned revision and yield Gym-shaped rows.

    Per-row schema:
      * ``responses_create_params.metadata.audio_path`` — ``<audio_prefix>/<slug>_<idx>.wav``.
        The WAV itself is written under ``data/audio/``.
      * ``subset_for_metrics`` — the AudioBench sub-dataset slug; the resources
        server's ``compute_metrics`` reports per-sub-dataset metrics from it.
      * ``instruction`` / ``question`` / ``expected_answer`` — the user message
        (resolved via ``prompts/default.yaml``), the question the judge sees,
        and the gold reference text the verifier scores against.
    """
    from datasets import Audio, load_dataset

    load_kwargs: dict[str, Any] = {"split": cfg["hf_split"], "revision": HF_REVISIONS[cfg["hf_repo"]]}
    if cfg.get("hf_data_dir"):
        load_kwargs["data_dir"] = cfg["hf_data_dir"]
    ds = load_dataset(cfg["hf_repo"], **load_kwargs)

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
        _write_audio(audio, AUDIO_DIR / audio_filename, idx)

        instruction = cfg.get("instruction") or sample.get("instruction") or sample.get("text") or "Process the audio."
        expected_answer = _answer_text(sample.get("reference", sample.get("answer", "")))
        # `question` is what the audiobench judge sees in its rating prompt
        # (alongside the model's generation and the reference answer). For
        # audiobench rows it's the spoken-content transcription
        # (`speech_instruction`) when the upstream HF row has it, otherwise
        # the user-facing instruction. asr_with_pc verifiers ignore this
        # field, so it's harmless on non-judge rows.
        question = sample.get("speech_instruction") or sample.get("question") or instruction

        yield {
            "responses_create_params": {
                "metadata": {"audio_path": f"{audio_prefix.rstrip('/')}/{audio_filename}"},
            },
            "instruction": instruction,
            "question": question,
            "expected_answer": expected_answer,
            "subset_for_metrics": slug,
            "sample_id": f"{slug}_{idx:06d}",
            "split": cfg["hf_split"],
        }


def prepare_audiobench_bucket(
    bucket: str,
    *,
    datasets: list[str] | None = None,
    max_samples_per_dataset: int | None = None,
    audio_prefix: str | None = None,
) -> Path:
    """Download every sub-dataset in ``bucket`` and write ``data/audiobench_<bucket>.jsonl``.

    Args:
      bucket: one of ``judge``, ``asr``, ``bleu``, ``exact_match``.
      datasets: optional explicit list of sub-dataset slugs to include
        (must all be in DATASETS and match the bucket). Default: all
        sub-datasets in ``bucket``.
      max_samples_per_dataset: cap rows per sub-dataset (smoke testing).
      audio_prefix: directory written into each row's ``audio_path``. Defaults
        to the absolute local ``data/audio`` directory; set it to where that
        directory is mounted when the model server runs on another host or in
        a container.

    License-gated sub-datasets (``license_gated=True``) whose license has not
    been accepted on Hugging Face are skipped with a warning; any other
    download failure is raised.

    Returns the path of the bucket's JSONL.
    """
    from datasets.exceptions import DatasetNotFoundError
    from huggingface_hub.errors import GatedRepoError

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_prefix = audio_prefix or str(AUDIO_DIR.resolve())

    selected = datasets or [s for s, c in DATASETS.items() if c["bucket"] == bucket]
    for slug in selected:
        if slug not in DATASETS:
            raise ValueError(f"unknown audiobench sub-dataset: {slug!r}")
        if DATASETS[slug]["bucket"] != bucket:
            raise ValueError(f"sub-dataset {slug!r} is bucket={DATASETS[slug]['bucket']!r}, not {bucket!r}")

    out_path = DATA_DIR / f"audiobench_{bucket}.jsonl"
    count = 0
    skipped = []
    with open(out_path, "w", encoding="utf-8") as f:
        for slug in tqdm(selected, desc=f"audiobench/{bucket}"):
            cfg = DATASETS[slug]
            try:
                for row in _iter_dataset_rows(
                    slug, cfg, max_samples=max_samples_per_dataset, audio_prefix=audio_prefix
                ):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            except (GatedRepoError, DatasetNotFoundError) as e:
                if not cfg.get("license_gated"):
                    raise
                skipped.append((slug, type(e).__name__, str(e).splitlines()[0]))

    if skipped:
        print(f"\n  skipped {len(skipped)} license-gated sub-datasets (accept the license on Hugging Face):")
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
    parser.add_argument("--audio-prefix", type=str, default=None, help="directory written into audio_path")
    args = parser.parse_args()

    selected = args.datasets.split(",") if args.datasets else None
    prepare_audiobench_bucket(
        bucket,
        datasets=selected,
        max_samples_per_dataset=args.max_samples_per_dataset,
        audio_prefix=args.audio_prefix,
    )
