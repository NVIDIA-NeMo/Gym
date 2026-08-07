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
"""Prepare the finance_agent_v2 (FABv2) benchmark data.

Tools-only reuse: the tool JSON schemas wrapped into each sample's
``responses_create_params`` are built **directly from the upstream Vals
``finance_agent`` Tool classes** (name/description/parameters/required), and the
system / question prompts are imported from ``finance_agent.prompt`` — so the
benchmark tracks upstream automatically instead of hand-copying schemas.

Input precedence (first that exists wins):
  1. ``data/labeled.jsonl``  — labeled rows ``{question, expected_answer?, rubric?}``
     (``expected_answer`` enables path-A scoring via our [[N]] judge; ``rubric`` is
     copied through for reference only and is not used for reward).
  2. ``data/public.jsonl``   — rows with at least ``{question}`` (unlabeled dry-run ok).
  3. ``data/public.txt``     — one question per line (FABv2 public format, unlabeled).
  4. fallback: the resource server's example_questions.jsonl (smoke test).

Without labels, samples carry only the question; the resource server's
``/verify`` returns reward 0 (dry-run) so the agent + tools path can be
validated before ground truth is available.
"""

import csv
import json
from pathlib import Path

# Upstream Vals finance-agent-v2 (installed via the resource server requirements).
from finance_agent.prompt import QUESTION_PROMPT, SYSTEM_PROMPT
from finance_agent.tools import (
    VALID_TOOLS,
    Calculator,
    EDGARSearch,
    ParseHtmlPage,
    PriceHistory,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
)


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "finance_agent_v2_benchmark.jsonl"

EXAMPLE_QUESTIONS_FPATH = (
    BENCHMARK_DIR.parent.parent
    / "resources_servers"
    / "finance_agent_v2"
    / "data"
    / "example_questions.jsonl"
)

# Maps upstream tool name -> upstream Tool class (mirrors get_agent.available_tools).
_TOOL_CLASSES = {
    "web_search": TavilyWebSearch,
    "retrieve_information": RetrieveInformation,
    "parse_html_page": ParseHtmlPage,
    "edgar_search": EDGARSearch,
    "calculator": Calculator,
    "price_history": PriceHistory,
}


def _tool_schema(tool_cls) -> dict:
    """Build a responses-API function tool schema from an upstream Tool class.

    Reads class-level ``name`` / ``description`` / ``parameters`` / ``required``
    attributes, so the schema stays in lockstep with the upstream package.
    """
    return {
        "type": "function",
        "name": tool_cls.name,
        "description": tool_cls.description,
        "parameters": {
            "type": "object",
            "properties": dict(tool_cls.parameters),
            "required": list(tool_cls.required),
        },
        "strict": False,
    }


def build_tools(tool_names: list[str] | None = None) -> list[dict]:
    """Build the full v2 tool set (selected tools + submit_final_result)."""
    names = list(tool_names) if tool_names is not None else list(VALID_TOOLS)
    tools = [_tool_schema(_TOOL_CLASSES[name]) for name in names]
    tools.append(_tool_schema(SubmitFinalResult))
    return tools


def _convert_row(row: dict, tools: list[dict]) -> dict:
    """Wrap one input row into Gym benchmark format with v2 prompts + tools."""
    question = row.get("question") or row.get("problem", "")
    out = {**row, "question": question}
    out["responses_create_params"] = {
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT, "type": "message"},
            {"role": "user", "content": QUESTION_PROMPT.format(question=question), "type": "message"},
        ],
        "tools": tools,
    }
    return out


def _load_rows() -> list[dict]:
    """Load input rows using the documented precedence."""
    labeled = DATA_DIR / "labeled.jsonl"
    public_jsonl = DATA_DIR / "public.jsonl"
    public_txt = DATA_DIR / "public.txt"
    public_csv = DATA_DIR / "public.csv"

    if labeled.exists():
        print(f"Using labeled dataset: {labeled}")
        return [json.loads(line) for line in labeled.read_text().splitlines() if line.strip()]
    if public_jsonl.exists():
        print(f"Using public jsonl dataset: {public_jsonl}")
        return [json.loads(line) for line in public_jsonl.read_text().splitlines() if line.strip()]
    if public_txt.exists():
        print(f"Using public txt dataset (one question per line): {public_txt}")
        return [{"question": line.strip()} for line in public_txt.read_text().splitlines() if line.strip()]
    if public_csv.exists():
        print(f"Using public csv dataset: {public_csv}")
        with open(public_csv, newline="") as f:
            reader = csv.DictReader(f)
            return [{"question": (r.get("question") or r.get("prompt") or "").strip()} for r in reader]

    print(
        f"WARNING: no input dataset found in {DATA_DIR} (looked for labeled.jsonl, "
        f"public.jsonl, public.txt, public.csv). Falling back to example questions for a smoke run."
    )
    return [json.loads(line) for line in EXAMPLE_QUESTIONS_FPATH.read_text().splitlines() if line.strip()]


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tools = build_tools()
    rows = _load_rows()

    count = 0
    labeled_count = 0
    with open(OUTPUT_FPATH, "w") as f_out:
        for row in rows:
            if not row.get("question"):
                continue
            if row.get("expected_answer") is not None:
                labeled_count += 1
            f_out.write(json.dumps(_convert_row(row, tools)) + "\n")
            count += 1

    print(f"Wrote {count} benchmark samples ({labeled_count} labeled) to {OUTPUT_FPATH}")
    if labeled_count == 0:
        print(
            "NOTE: all samples are unlabeled — /verify will return reward 0 (dry-run). "
            "Provide data/labeled.jsonl (or set reward_mode + a labeled set) for real scores."
        )
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
