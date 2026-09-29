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
"""Prepare the IHEval **gym-native** benchmark dataset.

The IHEval resources server ships ``resources_servers/iheval/prepare_iheval.py``,
which builds ``data/test.jsonl`` (the whole dataset, all tasks/settings) in
**Chat-Completions** shape (forwarded verbatim to ``/chat/completions``).

The gym-native agent path is different: ``simple_agent`` POSTs to ``/v1/responses``
and speaks the **Responses API** (``function_call`` / ``function_call_output``
items, top-level function tools). So this benchmark re-shapes the tool-use rows
(``get-webpage``, ``slack-user``) from Chat to Responses; every other row is
plain ``{role, content}`` messages that are valid in both shapes and pass
through untouched.

The gold/routing fields (``task``, ``domain``, ``setting``, ``instruction``,
``answer``) and the JSON-encoded ``answer`` are left exactly as the resources
server's ``verify()`` expects — only ``responses_create_params`` is converted.

Running the whole gym-native eval against this dataset scores each row with the
same rule-based checkers and then aggregates via the resources server's
``compute_metrics`` → the upstream ``average_final_score.py`` **hierarchy**
``result_score`` (exact task-macro conflict score), surfaced through
``get_key_metrics``. That is the number a per-row-mean driver cannot produce.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


BENCHMARK_DIR = Path(__file__).resolve().parent
GYM_ROOT = BENCHMARK_DIR.parents[1]
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "iheval_benchmark.jsonl"

# Whole-dataset source built by the resources server's own prepare script.
_SRC_PREPARE = GYM_ROOT / "resources_servers" / "iheval" / "prepare_iheval.py"
_SRC_TEST = GYM_ROOT / "resources_servers" / "iheval" / "data" / "test.jsonl"

# Agent that runs this benchmark (see config.yaml). Rows are tagged with it so
# they align with the agent selected at eval time.
_BENCHMARK_AGENT = "iheval_benchmark_simple_agent"


def _chat_tools_to_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chat ``[{type,function:{name,description,parameters}}]`` -> Responses
    ``[{type:function, name, description, parameters}]`` (fields lifted up)."""
    out: List[Dict[str, Any]] = []
    for tool in tools:
        fn = tool["function"]
        out.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                # Responses ``FunctionToolParam`` requires ``strict`` as a key
                # (Optional[bool] — may be null). Chat tools don't carry it.
                "strict": fn.get("strict"),
            }
        )
    return out


def _chat_input_to_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a Chat ``messages`` list to Responses ``input`` items.

    * ``assistant`` with ``tool_calls`` -> one ``function_call`` item per call.
    * ``role: "tool"`` -> a ``function_call_output`` item.
    * everything else (system / user / plain assistant) passes through as a
      ``{role, content}`` message.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call["function"]
                out.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": fn["name"],
                        "arguments": fn["arguments"],
                    }
                )
        elif role == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": msg["tool_call_id"],
                    "output": msg["content"],
                }
            )
        else:
            out.append({"role": role, "content": msg["content"]})
    return out


def convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return *row* with ``responses_create_params`` re-shaped Chat -> Responses."""
    rcp = row["responses_create_params"]
    new_rcp: Dict[str, Any] = {"input": _chat_input_to_responses(rcp["input"])}
    if "tools" in rcp:
        new_rcp["tools"] = _chat_tools_to_responses(rcp["tools"])

    converted = dict(row)
    converted["responses_create_params"] = new_rcp
    if "agent_ref" in converted:
        converted["agent_ref"] = {"type": "responses_api_agents", "name": _BENCHMARK_AGENT}
    # Upstream IHEval uses int ``id`` for some tasks and str for others. Gym's
    # dataset metric aggregator (train_data_utils.aggregate_other_metrics) keys a
    # single accumulator per top-level field and cannot mix numeric (AvgMinMax)
    # and string (Counter) values, so normalize ``id`` to str across all rows.
    if "id" in converted:
        converted["id"] = str(converted["id"])
    return converted


def _ensure_source() -> None:
    """Build the whole Chat-shaped ``test.jsonl`` if it is not already present.

    ``prepare_iheval.main`` downloads ``github.com/ytyz1307zzh/IHEval`` (or reads
    ``IHEVAL_REPO_DIR``) and writes the full dataset.
    """
    if _SRC_TEST.exists():
        return
    spec = importlib.util.spec_from_file_location("iheval_prepare_iheval", _SRC_PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main([])


def prepare() -> Path:
    """Build the gym-native IHEval benchmark JSONL (whole dataset, Responses shape)."""
    _ensure_source()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    with _SRC_TEST.open(encoding="utf-8") as fin, OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(convert_row(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1

    print(f"IHEval: wrote {n} Responses-shaped rows -> {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
