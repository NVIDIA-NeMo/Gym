# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare MMRB2 reasoning inputs for the selected judge interface."""

import json
from pathlib import Path
from shutil import copyfile


DATASET_REVISION = "78392e6ab989bfc75d3ee19e3e8ca5c1cbe4e323"  # pragma: allowlist secret


def prepare(
    source_jsonl: str | Path | None = None,
    *,
    judge_mode: str = "genrm",
    task: str = "reasoning",
    response_modality: str = "text_only",
) -> Path:
    if task != "reasoning" or response_modality != "text_only":
        raise ValueError("Only task=reasoning with response_modality=text_only is supported")
    if judge_mode not in {"genrm", "llm_as_a_judge"}:
        raise ValueError("judge_mode must be genrm or llm_as_a_judge")
    output = Path(__file__).parent / "data" / f"genrm_mmrb2_reasoning_text_only_{judge_mode}_validation.jsonl"
    source = Path(source_jsonl).expanduser() if source_jsonl else output
    if source_jsonl is None and not output.is_file():
        from huggingface_hub import snapshot_download

        from benchmarks.genrm_mmrb2.build_validation import prepare as build_validation

        # Cache only the reasoning split; the converter retains text-only candidate pairs.
        snapshot = snapshot_download(
            repo_id="rl-research/multimodal-rewardbench-2",
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns="reasoning/test-*.parquet",
        )
        build_validation(Path(snapshot) / "reasoning", output, genrm=judge_mode == "genrm")
    pairs = {}
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            metadata = row["verifier_metadata"]
            if (
                metadata.get("task") != task
                or metadata.get("response_modality") != response_modality
                or metadata.get("pair_type") != "chosen_text__rejected_text"
                or metadata.get("judge_mode") != judge_mode
            ):
                raise ValueError("Input must contain only reasoning/text_only pairs for the selected judge_mode")
            orders = pairs.setdefault(metadata["pair_id"], set())
            if metadata["order"] in orders:
                raise ValueError("Duplicate candidate ordering")
            orders.add(metadata["order"])
    if len(pairs) != 350 or any(orders != {"forward", "reverse"} for orders in pairs.values()):
        raise ValueError("Expected 350 pairs with both candidate orders (700 judgments)")
    if source.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, output)
    return output
