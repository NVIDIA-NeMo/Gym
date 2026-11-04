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

"""Prepare ToolSandbox benchmark data for NeMo Gym.

Generates one JSONL row per scenario index from the vendored apple/ToolSandbox
scenario registry. Each row is ``{"task_idx": N, "scenario": "<name>"}``; the
resources server resolves ``task_idx`` to the actual scenario at runtime.

Usage::

    python benchmarks/toolsandbox/prepare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_RESOURCES_SERVER_DIR = Path(__file__).resolve().parents[2] / "resources_servers" / "toolsandbox"
if str(_RESOURCES_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_SERVER_DIR))

from tool_sandbox.common.tool_discovery import ToolBackend  # noqa: E402
from tool_sandbox.scenarios import named_scenarios  # noqa: E402


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "toolsandbox_benchmark.jsonl"


def prepare(backend: str = "default") -> Path:
    """Generate the full ToolSandbox scenario set and write it as JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = named_scenarios(preferred_tool_backend=ToolBackend[backend.upper()])
    names = sorted(scenarios.keys())

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for idx, name in enumerate(names):
            f.write(json.dumps({"task_idx": idx, "scenario": name}) + "\n")

    print(f"Wrote {len(names)} scenarios to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
