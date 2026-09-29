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

Both `prepare.py` (qualitative) and `prepare_throughput_2k.py` (throughput_2k)
delegate to `prepare_one_config` here so the multi-source resolution logic
isn't duplicated.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import List


LOG = logging.getLogger(__name__)

ALL_CONFIGS = (
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)


def _load_skills_resolver():
    """Import Skills' `_resolve_external_data` by file path.

    Skills' module dir is `nemo_skills/dataset/speed-bench/` — Python's import
    system can't dot-import a hyphenated package name, so we load the file
    directly. Tries the in-container mounted path first, then the local
    recipe checkout for off-cluster usage.
    """
    candidates = [
        Path("/nemo_run/code/nemo_skills/dataset/speed-bench/prepare.py"),
        Path(__file__).resolve().parents[3] / "nemo-skills" / "nemo_skills" / "dataset" / "speed-bench" / "prepare.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("ns_speed_bench_prepare", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod._resolve_external_data
    raise RuntimeError(
        f"Could not find Skills' speed-bench prepare.py to import _resolve_external_data. Tried: {candidates}"
    )


def _row_to_gym(example: dict, speed_config: str) -> dict:
    """Convert a Skills-style resolved row to Gym JSONL shape."""
    turns: List[str] = list(example["turns"])
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


def prepare_one_config(speed_config: str) -> Path:
    """Download and resolve a single speed-bench config, write Gym JSONL.

    Returns the output JSONL path that the agent's
    `dataset.jsonl_fpath` is expected to match.
    """
    if speed_config not in ALL_CONFIGS:
        raise ValueError(f"Unknown speed-bench config: {speed_config!r}. Allowed: {ALL_CONFIGS}")

    from datasets import load_dataset  # imported lazily

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"speed_bench_{speed_config}_benchmark.jsonl"

    LOG.info("Loading nvidia/SPEED-Bench config %s", speed_config)
    dataset = load_dataset("nvidia/SPEED-Bench", speed_config, split="test")

    LOG.info("Resolving external dataset placeholders for config %s (%d rows)", speed_config, len(dataset))
    resolve = _load_skills_resolver()
    resolved = resolve(dataset, speed_config)

    n = 0
    with output_path.open("wt", encoding="utf-8") as f:
        for example in resolved:
            row = _row_to_gym(example, speed_config)
            f.write(json.dumps(row) + "\n")
            n += 1
    LOG.info("Wrote %d rows to %s", n, output_path)
    return output_path
