# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared SPEED-Bench prepare helpers.

The unified standalone CLI and the split-specific Gym entrypoints delegate here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


LOG = logging.getLogger(__name__)
DEFAULT_DATASET_REPO = "nvidia/SPEED-Bench"
DATA_DIR = Path(__file__).resolve().parent / "data"
TURNS_PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"

ALL_CONFIGS = (
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)


def _row_to_gym(example: dict, speed_config: str) -> dict:
    """Convert a Skills-style resolved row to Gym JSONL shape."""
    turns = example["turns"]
    if not isinstance(turns, list) or not turns or not all(isinstance(turn, str) and turn.strip() for turn in turns):
        raise ValueError(f"Expected nonempty text turns for {speed_config}, src_id={example.get('src_id')!r}")
    if any(TURNS_PLACEHOLDER in turn for turn in turns):
        raise ValueError(f"Unresolved SPEED-Bench placeholder for {speed_config}, src_id={example.get('src_id')!r}")
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": turn} for turn in turns],
        },
        "verifier_metadata": {
            "src_id": example.get("src_id"),
            "source": example.get("source"),
            "speed_config": speed_config,
            "num_turns": len(turns),
            "sub_category": example.get("sub_category"),
        },
    }


def prepare_one_config(
    speed_config: str,
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_revision: str | None = None,
) -> Path:
    """Download and resolve a single speed-bench config, write Gym JSONL.

    Returns the output JSONL path that the agent's
    `dataset.jsonl_fpath` is expected to match.
    """
    if speed_config not in ALL_CONFIGS:
        raise ValueError(f"Unknown speed-bench config: {speed_config!r}. Allowed: {ALL_CONFIGS}")

    from datasets import load_dataset  # imported lazily

    output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"speed_bench_{speed_config}_benchmark.jsonl"

    LOG.info("Loading %s config %s at revision %s", dataset_repo, speed_config, dataset_revision or "default")
    dataset = load_dataset(dataset_repo, speed_config, split="test", revision=dataset_revision)

    needs_resolution = any(TURNS_PLACEHOLDER in turn for turns in dataset["turns"] for turn in turns)
    if needs_resolution:
        LOG.info("Resolving external dataset placeholders for config %s (%d rows)", speed_config, len(dataset))
        from _resolve_external_data import _resolve_external_data

        resolved = _resolve_external_data(dataset, speed_config)
    else:
        resolved = dataset

    n = 0
    with output_path.open("wt", encoding="utf-8") as f:
        for example in resolved:
            row = _row_to_gym(example, speed_config)
            f.write(json.dumps(row) + "\n")
            n += 1
    LOG.info("Wrote %d rows to %s", n, output_path)
    return output_path


def prepare_cli() -> None:
    """Prepare one configuration or all six from the standalone command line."""
    parser = argparse.ArgumentParser(description="Prepare SPEED-Bench data for Gym.")
    parser.add_argument("--config", choices=(*ALL_CONFIGS, "all"), default="qualitative")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-revision", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configs = ALL_CONFIGS if args.config == "all" else (args.config,)
    for speed_config in configs:
        prepare_one_config(speed_config, dataset_repo=args.dataset_repo, dataset_revision=args.dataset_revision)
