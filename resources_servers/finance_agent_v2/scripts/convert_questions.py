#!/usr/bin/env python3
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
"""Convert the Vals finance-agent-v2 public CSV into NeMo Gym benchmark JSONL.

This mirrors ``resources_servers/finance_sec_search/scripts/convert_questions.py``
(v1) but targets FABv2, with two key differences:

1. Prompts + tools come straight from the **upstream** ``finance_agent`` package
   (tools-only reuse): the system / question prompts are imported from
   ``finance_agent.prompt`` and the tool JSON schemas are built directly from the
   ``finance_agent.tools`` ``Tool`` classes. This keeps the dataset in lockstep
   with upstream (same source of truth as ``benchmarks/finance_agent_v2/prepare.py``)
   instead of hand-copying schemas. Run it in the same environment the
   ``finance_agent_v2`` resource server uses (where ``finance-agent`` is installed).

2. Grading uses **our own** judge (same as ``resources_servers/finance_sec_search``).
   The public FABv2 release ships no official grader, so for the PUBLIC set we
   score each answer with the legacy ``[[0]]/[[1]]/[[2]]`` judge
   (``app.py::verify`` in ``reward_mode: binary``). That judge needs a GOLD
   ``expected_answer``, but the public CSV ships only rubric *criteria* (no single
   gold answer). Every public criterion is a positive factual assertion the answer
   must contain, so we synthesize ``expected_answer`` by joining the criteria into
   a bulleted GOLD reference — the judge then awards ``[[2]]`` (reward 1.0) only
   when the answer covers all required facts.

   The public CSV's ``rubric`` is also copied through **verbatim** for
   reference/completeness only. We deliberately do NOT encode how rubric checks
   map to a reward (Vals's private per-criterion grader is licensed); reward comes
   solely from the ``expected_answer`` + our ``[[N]]`` judge above.

Input CSV columns (Vals public release, e.g. ``data/vals_v2_public_27q.csv``):
    Question, Question Type, Expert time (mins), Rubric
where ``Rubric`` is a JSON string: a list of ``{"operator", "criteria"}`` checks.

Output JSONL (one object per line), consumed by the finance_agent agent loop and
the finance_agent_v2 ``/verify`` endpoint (``reward_mode: binary``)::

    {
      "question": "...",
      "question_type": "...",
      "expert_time_mins": 60,
      "expected_answer": "Required facts:\n- ...\n- ...",   # GOLD for our judge
      "rubric": "[{\"operator\": \"...\", \"criteria\": \"...\"}, ...]",  # reference only
      "responses_create_params": {"input": [system, user], "tools": [...]}
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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

# Maps upstream tool name -> upstream Tool class (mirrors prepare.py / get_agent).
_TOOL_CLASSES = {
    "web_search": TavilyWebSearch,
    "retrieve_information": RetrieveInformation,
    "parse_html_page": ParseHtmlPage,
    "edgar_search": EDGARSearch,
    "calculator": Calculator,
    "price_history": PriceHistory,
}

# CSV-only large field values (rubric JSON can be long).
csv.field_size_limit(10_000_000)


def _tool_schema(tool_cls) -> dict:
    """Build a responses-API function tool schema from an upstream Tool class."""
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


def parse_rubric(raw_rubric: str) -> list[dict]:
    """Parse the public CSV's rubric JSON into a list of checks, verbatim.

    The checks are propagated to the output for reference/completeness only. We
    deliberately do NOT translate operators into any grading vocabulary or encode
    how a rubric maps to a reward — the public release has no official grader and
    Vals's private per-criterion grader is licensed. Reward is computed by our own
    ``[[N]]`` judge against the synthesized ``expected_answer`` (see app.py::verify).
    """
    if not raw_rubric or not raw_rubric.strip():
        return []
    checks = json.loads(raw_rubric)
    if isinstance(checks, dict):
        checks = [checks]

    parsed: list[dict] = []
    for check in checks:
        parsed.append(
            {
                "operator": (check.get("operator") or "").strip(),
                "criteria": check.get("criteria", ""),
            }
        )
    return parsed


def build_expected_answer(rubric: list[dict]) -> str:
    """Synthesize a GOLD reference for our judge from the rubric criteria.

    The public CSV has no single gold answer; each criterion is a required fact.
    We join them into a bulleted reference so the legacy ``[[N]]`` judge can grade
    an answer for completeness against all required facts.
    """
    criteria = [str(c.get("criteria", "")).strip() for c in rubric]
    criteria = [c for c in criteria if c]
    if not criteria:
        return ""
    return "A complete answer must establish all of the following:\n" + "\n".join(
        f"- {c}" for c in criteria
    )


def _maybe_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def convert_row(row: dict, tools: list[dict]) -> dict | None:
    """Convert one CSV row into a Gym benchmark sample with v2 prompts + tools."""
    question = (row.get("Question") or row.get("question") or "").strip()
    if not question:
        return None

    rubric = parse_rubric(row.get("Rubric") or row.get("rubric") or "")

    sample = {
        "question": question,
        "question_type": (row.get("Question Type") or row.get("question_type") or "").strip() or None,
        "expert_time_mins": _maybe_int(row.get("Expert time (mins)") or row.get("expert_time_mins")),
        "expected_answer": build_expected_answer(rubric),
        "rubric": json.dumps(rubric),
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT, "type": "message"},
                {"role": "user", "content": QUESTION_PROMPT.format(question=question), "type": "message"},
            ],
            "tools": tools,
        },
    }
    return sample


def convert_file(input_file: Path, output_file: Path) -> tuple[int, int]:
    """Convert a Vals public CSV to benchmark JSONL. Returns (rows, labeled rows)."""
    tools = build_tools()
    count = 0
    labeled = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, newline="", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for raw in reader:
            sample = convert_row(raw, tools)
            if sample is None:
                continue
            if json.loads(sample["rubric"]):
                labeled += 1
            f_out.write(json.dumps(sample) + "\n")
            count += 1

    return count, labeled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert the Vals FABv2 public CSV to NeMo Gym benchmark JSONL.")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input Vals public CSV (columns: Question, Question Type, Expert time (mins), Rubric).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSONL path. Default: alongside the input with a .jsonl suffix.",
    )
    args = parser.parse_args(argv)

    input_file = Path(args.input)
    if not input_file.exists():
        parser.error(f"Input CSV not found: {input_file}")
    output_file = Path(args.output) if args.output else input_file.with_suffix(".jsonl")

    count, labeled = convert_file(input_file, output_file)
    print(f"Converted {input_file} -> {output_file}")
    print(f"  rows: {count} ({labeled} with a rubric)")

    sample_tools = [t["name"] for t in build_tools()]
    print(f"  tools: {', '.join(sample_tools)}")
    print(f"  expected_answer: synthesized from criteria (GOLD for reward_mode=binary)")
    print(f"  rubric: copied through verbatim (reference only; not used for reward)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
