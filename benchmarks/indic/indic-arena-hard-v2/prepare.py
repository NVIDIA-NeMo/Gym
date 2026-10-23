# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic Arena Hard prompts and optional language-specific baseline answers."""

import argparse
import json
import warnings
from pathlib import Path

import datasets
import pyarrow.parquet as pq


HF_DATASET = "ai4bharat/indic-arena-hard-v2"
HF_REVISION = "aebdc931858cb6df854f16a0513e184fc38a3b1e"
OUTPUT_FPATH = Path(__file__).parent / "data" / "indic_arena_hard_v2_benchmark.jsonl"
LANGUAGE_NAMES = {
    "en": "English",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


def _prompt_column(language: str) -> str:
    return "prompt" if language == "en" else f"prompt_{LANGUAGE_NAMES[language]}_translation"


def _language_row(row: dict, language: str, row_index: int) -> dict:
    """Select one language from the published dataset's translation columns."""
    return {key: value for key, value in row.items() if not key.endswith("_translation")} | {
        "row_id": row["uid"],
        "row_index": row_index,
        "prompt": row[_prompt_column(language)],
        "language": LANGUAGE_NAMES[language],
        "language_code": language,
    }


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
    entry["category"] = row.get("category", "hard_prompt")
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

    By default include all available languages, including English. Without a
    baseline file the output supports pointwise scoring or prompt generation;
    pairwise scoring requires baselines. Baseline rows contain language, row_id, generation.
    dataset_dir points to a downloaded repository with test.parquet, or the older
    data/<language>/test.parquet layout; local preparation does not access the Hub.
    """
    local_dir = Path(dataset_dir).expanduser() if dataset_dir is not None else None
    wide_rows: list[dict] | None = None
    if local_dir is not None:
        if (local_dir / "test.parquet").is_file():
            wide_rows = pq.read_table(local_dir / "test.parquet").to_pylist()
        else:
            supported = sorted(path.parent.name for path in local_dir.glob("data/*/test.parquet") if path.is_file())
            if not supported:
                raise FileNotFoundError(
                    f"No data/<language>/test.parquet files or test.parquet found under {local_dir}"
                )
    else:
        supported = datasets.get_dataset_config_names(HF_DATASET, revision=HF_REVISION)
        if supported == ["default"]:
            wide_rows = list(datasets.load_dataset(HF_DATASET, name="default", split="test", revision=HF_REVISION))
    if wide_rows is not None:
        if not wide_rows:
            raise ValueError("No test rows found in the dataset.")
        supported = [language for language in LANGUAGE_NAMES if _prompt_column(language) in wide_rows[0]]
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
            "The default pointwise mode needs no baseline; supply --baseline-file for optional pairwise judging.",
            stacklevel=2,
        )

    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for language in languages:
        if wide_rows is not None:
            dataset = [_language_row(row, language, index) for index, row in enumerate(wide_rows)]
        elif local_dir is not None:
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
    parser.add_argument("--languages", nargs="+", help="Language codes to include (default: all, including en).")
    parser.add_argument(
        "--baseline-file", help="JSONL with language (code), row_id, generation; required for pairwise judging."
    )
    parser.add_argument(
        "--dataset-dir", help="Local repository containing test.parquet or data/<language>/test.parquet."
    )
    args = parser.parse_args()
    prepare(languages=args.languages, baseline_file=args.baseline_file, dataset_dir=args.dataset_dir)
