# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare RewardBench 2 best-of-n records as response pairs in both candidate orders."""

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from resources_servers.genrm_eval.prepare import atomic_output, make_task


REVISION = "7ff08853b0d5686e79b13fda8677024f566a104a"  # pragma: allowlist secret
PARQUET_FPATH = "data/test-00000-of-00001.parquet"
COLUMNS = ["id", "prompt", "chosen", "rejected", "subset"]
OUTPUT = Path(__file__).parent / "data/rewardbench2_validation.jsonl"

# A chosen-versus-chosen comparison has no better answer, so a judge should decline to separate them.
FLIPPED_LABEL = {"A>B": "B>A", "A=B": "A=B"}


def example_id(item: dict[str, Any]) -> str:
    """Qualify a record ID with its subset; RewardBench 2 reuses IDs across subsets."""
    return f"{item['subset']}:{item['id']}"


def orientations(
    pair_id: str, turns: list[dict[str, str]], first: str, second: str, label: str, metadata: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Emit one comparison in both candidate orders."""
    for flipped in (False, True):
        task_id = f"{pair_id}-flipped" if flipped else pair_id
        yield make_task(
            task_id,
            turns,
            second if flipped else first,
            first if flipped else second,
            metadata | {"pair_id": task_id, "label": FLIPPED_LABEL[label] if flipped else label, "flipped": flipped},
        )


def convert(item: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand one record into its comparisons, each in both candidate orders."""
    chosen, rejected = item["chosen"], item["rejected"]
    if not chosen or not rejected:
        raise ValueError(f"RewardBench 2 record {example_id(item)} needs chosen and rejected responses")
    if not item["prompt"]:
        raise ValueError(f"RewardBench 2 record {example_id(item)} has no prompt")
    turns = [{"role": "user", "content": item["prompt"]}]
    metadata = {
        "example_id": example_id(item),
        "subset": item["subset"],
        # Scoring needs both expected counts to tell a lost record from a truncated one.
        "comparisons": len(chosen) * len(rejected),
        "correct_pairs": len(chosen) * (len(chosen) - 1) // 2,
    }
    for chosen_index, winner in enumerate(chosen):
        for rejected_index, loser in enumerate(rejected):
            pair_id = f"{example_id(item)}:{chosen_index}:{rejected_index}"
            yield from orientations(pair_id, turns, winner, loser, "A>B", metadata)
    # The Ties score needs the spread among the correct answers, which only these measure.
    for first_index, first in enumerate(chosen):
        for second_index, second in enumerate(chosen[first_index + 1 :], start=first_index + 1):
            pair_id = f"{example_id(item)}:c{first_index}:c{second_index}"
            yield from orientations(pair_id, turns, first, second, "A=B", metadata)


def convert_rows(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Convert records, rejecting repeated comparisons."""
    seen: set[str] = set()
    for item in rows:
        for row in convert(item):
            if row["id"] in seen:
                raise ValueError(f"Duplicate RewardBench 2 comparison {row['id']}")
            seen.add(row["id"])
            yield row


def prepare(output_fpath: Path | str = OUTPUT, *, source_jsonl: Path | str | None = None) -> Path:
    """Prepare RewardBench 2 from Hugging Face or a local JSONL file."""
    output_fpath = Path(output_fpath)
    if source_jsonl and Path(source_jsonl).resolve() == output_fpath.resolve():
        raise ValueError("Source and output must be different files")
    if source_jsonl:
        with Path(source_jsonl).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        source = hf_hub_download(
            "allenai/reward-bench-2", repo_type="dataset", filename=PARQUET_FPATH, revision=REVISION
        )
        rows = pq.read_table(source, columns=COLUMNS).to_pylist()
    with atomic_output(output_fpath) as stream:
        for row in convert_rows(rows):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--output-fpath", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(prepare(args.output_fpath, source_jsonl=args.source_jsonl))


if __name__ == "__main__":
    main()
