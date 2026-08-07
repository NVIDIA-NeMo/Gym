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
"""Build the AgentIF Gym dataset from the upstream ``THU-KEG/AgentIF`` eval.json.

The upstream ``eval.json`` is a JSON array; each row carries an ``input`` list
already in Responses-API ``{"role", "content"}`` form plus a list of scored
``constraints``. Each row becomes a Gym task: ``input`` goes into
``responses_create_params.input`` verbatim, and the routing/gold data travels in
``verifier_metadata`` so the resources server can re-score with the LLM judge and
code checkers. Any pre-existing ``score`` key on a constraint is stripped so the
verifier re-derives it from the model's own generation.

Usage::

    python resources_servers/agentif/prepare_agentif.py
    python resources_servers/agentif/prepare_agentif.py --input eval.json --output train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INPUT = _REPO_ROOT / "benchmarks" / "agentif" / "AgentIF" / "data" / "eval.json"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "train.jsonl"


def _strip_scores(constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop any pre-existing ``score`` so the verifier re-derives it."""
    out: List[Dict[str, Any]] = []
    for constraint in constraints:
        out.append({k: v for k, v in constraint.items() if k != "score"})
    return out


def build_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one upstream AgentIF row into a Gym task row."""
    return {
        "responses_create_params": {"input": row.get("input", [])},
        "verifier_metadata": {
            "query_id": row.get("query_id"),
            "turn_id": row.get("turn_id"),
            "domain": row.get("domain"),
            "agent_name": row.get("agent_name"),
            "prompt_type": row.get("prompt_type"),
            "constraints": _strip_scores(row.get("constraints", [])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the AgentIF Gym dataset.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Upstream eval.json path.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Output JSONL path.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as reader:
        data = json.load(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as writer:
        for row in data:
            writer.write(json.dumps(build_row(row), ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
