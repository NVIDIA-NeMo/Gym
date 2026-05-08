# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare the Numb3rs TN/ITN benchmark for NeMo Gym.

Numb3rs (`nvidia/Numb3rs`) is a TN/ITN ASR benchmark whose rows carry
both a written reference (``text_tn``, e.g. ``"$100"``) and a spoken
reference (``text_itn``, e.g. ``"one hundred dollars"``). The
``asr_with_pc`` resources server's ``ASR_LEADERBOARD`` task_type scores
the model output against ``expected_answer`` as the primary WER and
emits per-reference ``wer_<suffix>`` for every entry in
``reference_fields``.

This prepare ships the **neutral** prompt variant only (Skills'
``EVAL_SPLIT = "test_neutral"``); the ``tn`` / ``itn`` prompt variants
are reproducible by re-running prepare with a different prompt and are
not committed to keep JSONLs lean.

Audio is referenced via ``responses_create_params.metadata.audio_path``
— the WAV files live on a shared cluster mount (default
``/data/numb3rs/Numb3rs/<CATEGORY>/<filename>.wav``, controlled by
``--audio-prefix``) and the ``vllm_model`` audio sidechannel splices an
``audio_url`` content block from that path at rollout time. The
prepared audio tree is written to
``benchmarks/numb3rs/data/Numb3rs/<CATEGORY>/<filename>.wav``; copy or
mount it to ``--audio-prefix`` on the cluster.

``responses_create_params.input`` is intentionally NOT pre-populated —
``benchmarks/numb3rs/prompts/default.yaml`` materializes the system +
user messages at rollout time, which is the canonical Gym pattern and
lets the prompt template change without re-preparing the JSONL.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "numb3rs_benchmark.jsonl"

# Skills' PROMPT_NEUTRAL — used purely for documentation; the actual prompt
# is materialized at rollout time from prompts/default.yaml.
PROMPT_NEUTRAL = "Transcribe the audio file into English text."

DEFAULT_AUDIO_PREFIX = "/data/numb3rs"
MIN_AUDIO_DURATION = 0.1  # Skip audio shorter than this (matches Skills).


def _format_row(
    entry: Dict[str, Any],
    *,
    audio_prefix: str,
) -> Optional[Dict[str, Any]]:
    """Convert one Numb3rs HF row into a Gym JSONL row (neutral variant).

    Returns ``None`` for rows that should be skipped (short audio, missing
    fields). The returned row is the verify-request payload Gym sends to
    the ``asr_with_pc`` server in ``ASR_LEADERBOARD`` mode:

      * ``expected_answer`` is the spoken form (Skills' default for the
        neutral and ITN variants).
      * ``text_tn`` / ``text_itn`` ride alongside on the request — the
        server's ``ConfigDict(extra="allow")`` lets them through, and the
        aggregator reads them as per-reference corpus inputs.
      * ``reference_fields`` tells the server which sibling fields to
        score against in addition to ``expected_answer``.
    """
    text_tn = (entry.get("original_text") or "").strip()
    text_itn = (entry.get("text") or "").strip()
    file_name = entry.get("file_name") or ""
    duration = entry.get("duration")
    category = (entry.get("category") or "").upper()

    if not text_tn or not text_itn or not file_name or not category:
        return None

    if duration is None or duration < MIN_AUDIO_DURATION:
        return None

    audio_filename = Path(file_name).name
    sample_id = Path(file_name).stem
    audio_path = f"{audio_prefix.rstrip('/')}/Numb3rs/{category}/{audio_filename}"

    return {
        "responses_create_params": {
            "metadata": {"audio_path": audio_path},
        },
        "expected_answer": text_itn,
        "text_tn": text_tn,
        "text_itn": text_itn,
        "audio_duration": float(duration),
        "subset_for_metrics": f"numb3rs_{category}",
        "category": category,
        "sample_id": sample_id,
        "task_type": "ASR_LEADERBOARD",
        "reference_fields": ["text_tn", "text_itn"],
    }


def _save_audio(entry: Dict[str, Any], category_dir: Path) -> None:
    import soundfile as sf

    audio_info = entry.get("audio") or {}
    audio_array = audio_info.get("array")
    sampling_rate = audio_info.get("sampling_rate")
    if audio_array is None or len(audio_array) == 0 or not sampling_rate:
        return

    file_name = entry.get("file_name") or ""
    audio_filename = Path(file_name).name
    sf.write(str(category_dir / audio_filename), audio_array, sampling_rate)


def _iter_rows(
    dataset,
    *,
    audio_prefix: str,
    with_audio: bool,
    audio_dir: Path,
) -> Iterator[Dict[str, Any]]:
    from tqdm import tqdm

    # Cache mkdir per-category — ~10K rows / 12 categories means ~9988 of
    # these would otherwise be redundant exists checks.
    seen_categories: set = set()
    for entry in tqdm(dataset, desc="Encoding numb3rs"):
        row = _format_row(entry, audio_prefix=audio_prefix)
        if row is None:
            continue
        if with_audio:
            category_dir = audio_dir / row["category"]
            if row["category"] not in seen_categories:
                category_dir.mkdir(parents=True, exist_ok=True)
                seen_categories.add(row["category"])
            _save_audio(entry, category_dir)
        yield row


def prepare(
    *,
    audio_prefix: str = DEFAULT_AUDIO_PREFIX,
    with_audio: bool = True,
    categories: Optional[List[str]] = None,
) -> Path:
    """Download Numb3rs and write the neutral-variant Gym JSONL.

    Args:
        audio_prefix: Prefix for the ``audio_path`` metadata in the JSONL.
            Defaults to ``/data/numb3rs`` (the same path Skills uses on
            the cluster mount).
        with_audio: If ``True``, also materializes the WAV tree under
            ``benchmarks/numb3rs/data/Numb3rs/<CATEGORY>/<filename>.wav``.
        categories: Optional category allowlist (e.g.
            ``["MONEY", "DATE"]``). ``None`` means "all categories".

    Returns:
        Path to ``numb3rs_benchmark.jsonl`` — the file the benchmark
        config's ``datasets[0].jsonl_fpath`` references.
    """
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = DATA_DIR / "Numb3rs"
    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    print("Loading nvidia/Numb3rs (split=test) from HuggingFace...")
    dataset = load_dataset("nvidia/Numb3rs", split="test", trust_remote_code=True)

    if categories:
        wanted = {c.upper() for c in categories}
        dataset = dataset.filter(lambda e: (e.get("category") or "").upper() in wanted)

    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as fout:
        for row in _iter_rows(
            dataset,
            audio_prefix=audio_prefix,
            with_audio=with_audio,
            audio_dir=audio_dir,
        ):
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Numb3rs benchmark for Gym (neutral variant)")
    parser.add_argument(
        "--audio-prefix",
        default=DEFAULT_AUDIO_PREFIX,
        help="Prefix for audio paths in the JSONL (default: /data/numb3rs).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip writing WAV files; the JSONL still references audio paths.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help=(
            "Optional category allowlist (e.g. MONEY DATE). Default is all "
            "12 categories: ADDRESS, CARDINAL, DATE, DECIMAL, DIGIT, FRACTION, "
            "MEASURE, MONEY, ORDINAL, PLAIN, TELEPHONE, TIME."
        ),
    )
    args = parser.parse_args()
    prepare(
        audio_prefix=args.audio_prefix,
        with_audio=not args.no_audio,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
