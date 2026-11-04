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
"""Generate the ToolSandbox dataset (one JSONL row per scenario index).

The resources server maps each dataset row's ``task_idx`` into the sorted list
of scenario names, so the dataset is simply ``{"task_idx": N}`` for N in
``range(num_scenarios)``. Each row also carries the resolved scenario name for
human readability; the server ignores everything but ``task_idx``.

Usage:

    python resources_servers/toolsandbox/prepare_toolsandbox.py \
        --output resources_servers/toolsandbox/data/test_tolsandbox.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tool_sandbox.common.tool_discovery import ToolBackend  # noqa: E402
from tool_sandbox.scenarios import named_scenarios  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(_HERE) / "data" / "test_toolsandbox.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--backend",
        default="default",
        help="Preferred tool backend (default / rapid_api).",
    )
    args = parser.parse_args()

    scenarios = named_scenarios(preferred_tool_backend=ToolBackend[args.backend.upper()])
    names = sorted(scenarios.keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for idx, name in enumerate(names):
            f.write(json.dumps({"task_idx": idx, "scenario": name}) + "\n")
    print(f"Wrote {len(names)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
