# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic Arena Hard prompts and optional language-specific baseline answers."""

import argparse
import json
import warnings
from pathlib import Path

import datasets
import pyarrow.parquet as pq


HF_DATASET = "anushakamathofficial/indic_arena_hard"
HF_REVISION = "cc08e98842cd04d659092dc8b13cf00a848b435c"
OUTPUT_FPATH = Path(__file__).parent / "data" / "indic_arena_hard_benchmark.jsonl"


def _format_entry(row: dict, language: str) -> dict:
    if row["language_code"] != language:
        raise ValueError(f"Language mismatch for row {row['row_id']}: expected {language}, got {row['language_code']}")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise ValueError(f"Empty prompt for ({language}, {row['row_id']})")
    entry = dict(row)
    entry["question"] = entry.pop("prompt")
    entry["language_name"] = entry["language"]
    entry["language"] = language
    # Translations share row_id; use a distinct task ID for each language.
    entry["uid"] = f"{language}:{row['row_id']}"
    entry["subset_for_metrics"] = language
    entry["category"] = "hard_prompt"
    entry["baseline_answer"] = ""
    return entry


def _load_baselines(path: Path) -> dict[tuple[str, str], str]:
    baselines: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            row = json.loads(line)
            key = (row["language"], str(row["row_id"]))
            answer = row["generation"]
            if key in baselines:
                raise ValueError(f"Duplicate baseline {key} in {path}:{line_number}")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Empty baseline {key} in {path}:{line_number}")
            baselines[key] = answer
    return baselines


def prepare(
    *, languages: list[str] | None = None, baseline_file: str | None = None, dataset_dir: str | None = None
) -> Path:
    """Load local or pinned HF test splits, optionally join baselines, and write Gym JSONL.

    By default include all language configurations, including English. Without a
    baseline file the output supports pointwise scoring or prompt generation;
    pairwise scoring requires baselines. Baseline rows contain language, row_id, generation.
    dataset_dir points to a downloaded repository with data/<language>/test.parquet;
    when supplied, preparation reads those files without accessing the Hub.
    """
    local_dir = Path(dataset_dir).expanduser() if dataset_dir is not None else None
    if local_dir is not None:
        supported = sorted(path.parent.name for path in local_dir.glob("data/*/test.parquet") if path.is_file())
        if not supported:
            raise FileNotFoundError(f"No data/<language>/test.parquet files found under {local_dir}")
    else:
        supported = datasets.get_dataset_config_names(HF_DATASET, revision=HF_REVISION)
    languages = supported if languages is None else list(dict.fromkeys(languages))
    if not languages:
        raise ValueError("Select at least one language.")
    invalid = set(languages) - set(supported)
    if invalid:
        raise ValueError(f"Unsupported languages: {sorted(invalid)}. Supported: {supported}")

    baselines = _load_baselines(Path(baseline_file)) if baseline_file else None
    if baselines is None:
        warnings.warn(
            "No baseline file supplied: baseline_answer will be empty. "
            "Use pointwise mode for reference-free scoring, or --baseline-file before running pairwise judging.",
            stacklevel=2,
        )

    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for language in languages:
        if local_dir is not None:
            path = local_dir / "data" / language / "test.parquet"
            print(f"Loading {path} ...")
            dataset = pq.read_table(path).to_pylist()
        else:
            print(f"Loading {HF_DATASET} ({language}) ...")
            dataset = datasets.load_dataset(HF_DATASET, name=language, split="test", revision=HF_REVISION)
        if not len(dataset):
            raise ValueError(f"No test rows found for {language}.")
        for row in dataset:
            entry = _format_entry(row, language)
            key = (language, str(row["row_id"]))
            if key in seen:
                raise ValueError(f"Duplicate question {key}")
            seen.add(key)
            if baselines is not None:
                if key not in baselines:
                    raise ValueError(f"Missing baseline for {key} in {baseline_file}")
                entry["baseline_answer"] = baselines[key]
            entries.append(entry)

    # Validate every join before replacing an existing prepared dataset.
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for entry in entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", help="HF language configs to include (default: all, including en).")
    parser.add_argument(
        "--baseline-file", help="JSONL with language (code), row_id, generation; required for pairwise judging."
    )
    parser.add_argument("--dataset-dir", help="Local dataset repository containing data/<language>/test.parquet.")
    args = parser.parse_args()
    prepare(languages=args.languages, baseline_file=args.baseline_file, dataset_dir=args.dataset_dir)
