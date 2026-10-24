# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare chemeval directly from pinned upstream sources."""

from collections.abc import Iterator
from pathlib import Path

from benchmarks.chemeval.data_utils import materialize, write_rows
from resources_servers.chemeval.task_data import TaskData


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "test.jsonl"

DATA_REVISION = "61d82e727865e9c6c110fffd3ab920e0ab2edc42"  # pragma: allowlist secret
FAMILIES = ("mcq", "true_false", "rule_based", "regression", "judged")


def build_rows() -> Iterator[dict[str, object]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from benchmarks.chemeval.conversion import build_split, check_muzzle_stripping, check_task_coverage, gold_spans
    from benchmarks.chemeval.utils import DATA_FILE, DATASET_NAME, is_three_shot

    source = hf_hub_download(DATASET_NAME, DATA_FILE, repo_type="dataset", revision=DATA_REVISION)
    records = pq.read_table(source, columns=["query", "target", "filename"]).to_pylist()
    check_task_coverage(records)
    check_muzzle_stripping(records)
    spans = gold_spans(records)
    grouped = build_split([r for r in records if not is_three_shot(r["filename"])], spans)
    if sum(map(len, grouped.values())) != 2210:
        raise ValueError("Unexpected ChemEval zero-shot release size")
    for family in FAMILIES:
        for index, row in enumerate(grouped[family]):
            yield {
                "responses_create_params": {
                    "input": [
                        {"role": "user", "content": row["problem"] + row["answer_format"]},
                    ]
                },
                "verifier_metadata": {
                    "id": f"chemeval.test.{family}.{index:05d}",
                    **{key: value for key, value in row.items() if key not in ("problem", "answer_format")},
                },
            }


def prepare(input_path: str | Path | None = None, output_path: str | Path = OUTPUT_FPATH) -> Path:
    """Download and convert raw sources, or validate an explicitly supplied Gym JSONL."""
    options = dict(task_schema=TaskData, identity_key="id", benchmark="chemeval")
    if input_path is not None:
        return materialize(Path(input_path), Path(output_path), **options)
    return write_rows(build_rows(), Path(output_path), **options)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional already converted Gym JSONL; default downloads raw data")
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    args = parser.parse_args()
    print(prepare(args.input, args.output))
