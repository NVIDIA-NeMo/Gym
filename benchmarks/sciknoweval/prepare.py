# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare sciknoweval directly from pinned upstream sources."""

from collections.abc import Iterator
from pathlib import Path

from benchmarks.sciknoweval.data_utils import materialize, write_rows
from resources_servers.sciknoweval.task_data import TaskData


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "test.jsonl"

DATA_REVISION = "92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2"  # pragma: allowlist secret
FAMILIES = ("mcq", "true_false", "filling", "relation_extraction", "open_ended")


def build_rows() -> Iterator[dict[str, object]]:
    import json

    import yaml
    from huggingface_hub import hf_hub_download

    from benchmarks.sciknoweval.conversion import check_muzzle_stripping, format_entry
    from benchmarks.sciknoweval.rubrics import load_rubrics
    from benchmarks.sciknoweval.utils import DATA_FILE, DATASET_NAME

    source = hf_hub_download(DATASET_NAME, DATA_FILE, repo_type="dataset", revision=DATA_REVISION)
    with open(source, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    check_muzzle_stripping(records)
    rubrics = load_rubrics(BENCHMARK_DIR / "data")
    grouped = {family: [] for family in FAMILIES}
    for record in records:
        family, row = format_entry(record, rubrics)
        grouped[family].append(row)
    if len(records) != 28392:
        raise ValueError(f"Unexpected SciKnowEval release size: {len(records)}")
    for family in FAMILIES:
        prompt = yaml.safe_load((BENCHMARK_DIR / "prompts" / (family.replace("_", "-") + ".yaml")).read_text())
        for index, row in enumerate(grouped[family]):
            yield {
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": prompt["system"].format(**row)},
                        {"role": "user", "content": prompt["user"].format(**row)},
                    ]
                },
                "verifier_metadata": {
                    "id": f"sciknoweval.{family}.{index:05d}",
                    **{key: value for key, value in row.items() if key not in ("problem", "instruction")},
                },
            }


def prepare(input_path: str | Path | None = None, output_path: str | Path = OUTPUT_FPATH) -> Path:
    """Download and convert raw sources, or validate an explicitly supplied Gym JSONL."""
    options = dict(task_schema=TaskData, identity_key="id", benchmark="sciknoweval")
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
