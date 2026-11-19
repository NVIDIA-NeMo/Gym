# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare JudgeBench response pairs in both candidate orders."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.genrm_judgebench.metrics import base_pair_id
from resources_servers.genrm_eval.prepare import atomic_output, make_task


REVISION = "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308"  # pragma: allowlist secret
OUTPUT = Path(__file__).parent / "data/judgebench_validation.jsonl"


def convert(item: dict[str, Any]) -> dict[str, Any]:
    if item["label"] not in {"A>B", "B>A"}:
        raise ValueError(f"Invalid JudgeBench label for {item['pair_id']}")
    turns = [{"role": "user", "content": item["question"]}]
    first, second = item["response_A"], item["response_B"]
    metadata = {key: item[key] for key in ("pair_id", "source", "label")}
    task_id = str(item["pair_id"])
    metadata = {
        key: value for key, value in metadata.items() if value is not None and value != "unknown" and value != ""
    }
    return make_task(task_id, turns, first, second, metadata)


def convert_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Convert rows, preserving existing JudgeBench flips or adding missing orientations."""
    converted = (convert(item) for item in rows)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in converted:
        groups.setdefault(base_pair_id(row["id"]), []).append(row)
    for pair_id, group in groups.items():
        if len(group) > 2:
            raise ValueError(f"JudgeBench pair {pair_id} has {len(group)} orientations; expected at most two")
        if len(group) == 2:
            first, second = group
            a = first["responses_create_params"]
            b = second["responses_create_params"]
            if (
                first["id"] == second["id"]
                or a["input"] != b["input"]
                or a["metadata"]["response_1"] != b["metadata"]["response_2"]
                or a["metadata"]["response_2"] != b["metadata"]["response_1"]
                or first["verifier_metadata"]["label"] == second["verifier_metadata"]["label"]
            ):
                raise ValueError(f"JudgeBench pair {pair_id} must contain opposite response orders and labels")
        yield from group
        if len(group) == 1:
            row = group[0]
            flipped = json.loads(json.dumps(row))
            flipped["id"] = pair_id + "-flipped" if row["id"] == pair_id else pair_id
            flipped["verifier_metadata"]["pair_id"] = flipped["id"]
            flipped["verifier_metadata"]["label"] = "B>A" if row["verifier_metadata"]["label"] == "A>B" else "A>B"
            candidates = flipped["responses_create_params"]["metadata"]
            candidates["response_1"], candidates["response_2"] = candidates["response_2"], candidates["response_1"]
            yield flipped


def prepare(
    output_fpath: Path | str = OUTPUT,
    *,
    source_jsonl: Path | str | None = None,
    split: str = "gpt",
) -> Path:
    """Prepare JudgeBench from Hugging Face or a local JSONL file."""
    output_fpath = Path(output_fpath)
    if source_jsonl and Path(source_jsonl).resolve() == output_fpath.resolve():
        raise ValueError("Source and output must be different files")
    if source_jsonl:
        with Path(source_jsonl).open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                "ScalerLab/JudgeBench",
                repo_type="dataset",
                filename=f"data/{split}-00000-of-00001.jsonl",
                revision=REVISION,
            )
        )
        with source.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    with atomic_output(output_fpath) as stream:
        for row in convert_rows(rows):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--output-fpath", type=Path, default=OUTPUT)
    parser.add_argument("--split", choices=["gpt", "claude"], default="gpt")
    args = parser.parse_args()
    print(prepare(args.output_fpath, source_jsonl=args.source_jsonl, split=args.split))


if __name__ == "__main__":
    main()
