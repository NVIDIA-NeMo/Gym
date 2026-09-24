# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic AlpacaEval instructions and optional language-specific references."""

import argparse
import json
from pathlib import Path

import datasets
import pyarrow.parquet as pq


HF_DATASET = "anushakamathofficial/indic_alpaca_eval"
HF_REVISION = "5a173582c9e9f184e6a8fdefb0fb12ae485d56fa"
OUTPUT_FPATH = Path(__file__).parent / "data" / "indic_alpaca_eval_benchmark.jsonl"


def _load_references(path: Path) -> dict[tuple[str, str], str]:
    references = {}
    with path.open(encoding="utf-8") as fin:
        for number, line in enumerate(fin, 1):
            row = json.loads(line)
            key = (row["language"], str(row["row_id"]))
            answer = row["generation"]
            if key in references:
                raise ValueError(f"Duplicate reference {key} in {path}:{number}")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Empty reference {key} in {path}:{number}")
            references[key] = answer
    return references


def prepare(
    *, languages: list[str] | None = None, dataset_dir: str | None = None, reference_file: str | None = None
) -> Path:
    """Convert local or pinned HF eval splits into Gym JSONL.

    Keep every instruction and translation-review field. References, if supplied,
    must cover every selected (language, row_id), using language codes and a
    generation field. The default Gym rubric needs no references; the optional
    references can be exported for evaluation by the upstream AlpacaEval package.
    """
    local_dir = Path(dataset_dir).expanduser() if dataset_dir is not None else None
    if local_dir is not None:
        supported = sorted(p.parent.name for p in local_dir.glob("data/*/eval.parquet") if p.is_file())
        if not supported:
            raise FileNotFoundError(f"No data/<language>/eval.parquet files found under {local_dir}")
    else:
        supported = datasets.get_dataset_config_names(HF_DATASET, revision=HF_REVISION)
    selected = supported if languages is None else list(dict.fromkeys(languages))
    if not selected:
        raise ValueError("Select at least one language.")
    if invalid := set(selected) - set(supported):
        raise ValueError(f"Unsupported languages: {sorted(invalid)}. Supported: {supported}")

    references = _load_references(Path(reference_file)) if reference_file else None
    entries = []
    seen = set()
    for language in selected:
        if local_dir is not None:
            source = local_dir / "data" / language / "eval.parquet"
            print(f"Loading {source} ...")
            rows = pq.read_table(source).to_pylist()
        else:
            print(f"Loading {HF_DATASET} ({language}) ...")
            rows = datasets.load_dataset(HF_DATASET, name=language, split="eval", revision=HF_REVISION)
        if not len(rows):
            raise ValueError(f"No eval rows found for {language}.")
        for row in rows:
            if row["language_code"] != language:
                raise ValueError(f"Language mismatch for {row['row_id']}: expected {language}")
            if not isinstance(row["row_id"], str) or not row["row_id"].strip():
                raise ValueError(f"Empty row_id for {language}")
            if not isinstance(row["instruction"], str) or not row["instruction"].strip():
                raise ValueError(f"Empty instruction for ({language}, {row['row_id']})")
            key = (language, row["row_id"])
            if key in seen:
                raise ValueError(f"Duplicate instruction ID {key}")
            seen.add(key)
            if references is not None and key not in references:
                raise ValueError(f"Missing reference for {key} in {reference_file}")
            entries.append(
                {
                    **row,
                    "question": row["instruction"],
                    "language_name": row["language"],
                    "language": language,
                    "uid": f"{language}:{row['row_id']}",
                    "subset_for_metrics": language,
                    "category": "hard_prompt",
                    "baseline_answer": references[key] if references is not None else "",
                }
            )

    # All validation and reference joins complete before replacing prepared data.
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FPATH.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as fout:
            for entry in entries:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        temporary.replace(OUTPUT_FPATH)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {len(entries)} instructions to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", help="Language codes to include (default: all, including en).")
    parser.add_argument("--dataset-dir", help="Local dataset repository containing data/<language>/eval.parquet.")
    parser.add_argument("--reference-file", help="Optional JSONL with language, row_id, generation.")
    args = parser.parse_args()
    prepare(languages=args.languages, dataset_dir=args.dataset_dir, reference_file=args.reference_file)
