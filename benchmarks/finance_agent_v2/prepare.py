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

This is the single ``gym eval prepare`` entry point for the environment. It both
(a) converts a raw Vals public export into NeMo Gym benchmark JSONL, and
(b) can convert an explicit input file via ``--input/--output`` (the workflow
formerly in ``scripts/convert_questions.py``, now folded in here).

Tools-only reuse: the tool JSON schemas wrapped into each sample's
``responses_create_params`` are built **directly from the upstream Vals
``finance_agent`` Tool classes** (name/description/parameters/required), and the
system / question prompts are imported from ``finance_agent.prompt`` — so the
benchmark tracks upstream automatically instead of hand-copying schemas. Run it
in the environment where ``finance-agent`` is installed (the resource server's).

Grading uses **our own** judge (same as ``resources_servers/finance_sec_search``).
The public FABv2 release ships no official grader, so for the PUBLIC set we score
each answer with the legacy ``[[0]]/[[1]]/[[2]]`` judge (``app.py::verify`` in
``reward_mode: binary``). That judge needs a GOLD ``expected_answer``, but the
public CSV ships only rubric *criteria* (no single gold answer). Every public
criterion is a positive factual assertion the answer must contain, so we
synthesize ``expected_answer`` by joining the criteria into a bulleted GOLD
reference — the judge then awards ``[[2]]`` (reward 1.0) only when the answer
covers all required facts. The raw ``rubric`` is copied through **verbatim** for
reference/completeness only; we deliberately do NOT encode how rubric checks map
to a reward (Vals's private per-criterion grader is licensed).

Input precedence for the no-args ``prepare()`` path (first that exists wins),
all under ``data/``:
  1. ``labeled.jsonl``  — labeled rows ``{question, expected_answer?, rubric?}``.
  2. ``public.jsonl``   — rows with at least ``{question}`` (unlabeled dry-run ok).
  3. ``public.txt``     — one question per line (FABv2 public format, unlabeled).
  4. ``public.csv``     — raw Vals public CSV (columns: ``Question``,
     ``Question Type``, ``Expert time (mins)``, ``Rubric``).
  5. (fallback) if none of the above exist, the public CSV is downloaded from the
     upstream Vals repo (``CSV_URL``) into ``data/public.csv`` — so ``gym eval
     prepare --benchmark finance_agent_v2`` reproduces the set from scratch.

Without labels/rubric, samples carry only the question; the resource server's
``/verify`` returns reward 0 (dry-run) so the agent + tools path can be validated
before ground truth is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
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

ENV_DIR = Path(__file__).parent
DATA_DIR = ENV_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "vals_v2_public_27q.jsonl"

# Repo-relative output path (must match the benchmark config's `jsonl_fpath`).
# prepare() returns this anchored to the CWD so gym eval prepare's
# `output_fpath.absolute() == jsonl_fpath.absolute()` check holds. Path.absolute()
# does not normalize symlinks, and on this Lustre mount the package imports via its
# /lustre/fs12 realpath while the config resolves against the /lustre/fsw CWD — so
# an __file__-anchored return would spuriously mismatch.
_BENCHMARK_REL_FPATH = "benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl"

# Upstream public question set (used as the reproducible fallback source when no
# local export is present under data/). Pinned to the same repo the tools import from.
CSV_URL = "https://raw.githubusercontent.com/vals-ai/finance-agent-v2/main/data/public.csv"

# Maps upstream tool name -> upstream Tool class (mirrors get_agent.available_tools).
_TOOL_CLASSES = {
    "web_search": TavilyWebSearch,
    "retrieve_information": RetrieveInformation,
    "parse_html_page": ParseHtmlPage,
    "edgar_search": EDGARSearch,
    "calculator": Calculator,
    "price_history": PriceHistory,
}

# Rubric JSON strings can be long; raise the CSV field cap.
csv.field_size_limit(10_000_000)


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


def parse_rubric(raw_rubric) -> list[dict]:
    """Normalize a rubric (raw JSON string, list, or dict) into a list of checks.

    Propagated to the output for reference/completeness only. We deliberately do
    NOT translate operators into any grading vocabulary or encode how a rubric
    maps to a reward — the public release has no official grader and Vals's
    private per-criterion grader is licensed. Reward is computed by our own
    ``[[N]]`` judge against the synthesized ``expected_answer`` (see app.py::verify).
    """
    if not raw_rubric:
        return []
    if isinstance(raw_rubric, str):
        if not raw_rubric.strip():
            return []
        checks = json.loads(raw_rubric)
    else:
        checks = raw_rubric
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
    return "A complete answer must establish all of the following:\n" + "\n".join(f"- {c}" for c in criteria)


def _maybe_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _convert_row(row: dict, tools: list[dict]) -> dict | None:
    """Wrap one input row into Gym benchmark format with v2 prompts + tools.

    Accepts both raw Vals CSV column names (``Question``, ``Question Type``,
    ``Expert time (mins)``, ``Rubric``) and lowercase JSONL keys. When the row
    carries a rubric but no explicit ``expected_answer``, a GOLD reference is
    synthesized from the rubric criteria for ``reward_mode: binary``.
    """
    question = (
        row.get("question") or row.get("Question") or row.get("problem") or row.get("prompt") or ""
    ).strip()
    if not question:
        return None

    raw_rubric = row.get("rubric") if row.get("rubric") is not None else row.get("Rubric")
    rubric = parse_rubric(raw_rubric)

    expected = row.get("expected_answer")
    if not expected:
        expected = build_expected_answer(rubric) or None

    question_type = (row.get("question_type") or row.get("Question Type") or "").strip() or None
    expert_raw = row.get("expert_time_mins")
    if expert_raw is None:
        expert_raw = row.get("Expert time (mins)")

    return {
        "question": question,
        "question_type": question_type,
        "expert_time_mins": _maybe_int(expert_raw),
        "expected_answer": expected,
        "rubric": json.dumps(rubric),
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT, "type": "message"},
                {"role": "user", "content": QUESTION_PROMPT.format(question=question), "type": "message"},
            ],
            "tools": tools,
        },
    }


def _read_source(path: Path) -> list[dict]:
    """Read raw rows from a ``.csv`` / ``.jsonl`` / ``.txt`` source file."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suffix in (".jsonl", ".json"):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".txt":
        return [{"question": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"Unsupported input extension {suffix!r} (expected .csv, .jsonl, or .txt): {path}")


def convert_file(input_file: Path, output_file: Path) -> tuple[int, int]:
    """Convert a raw Vals export to benchmark JSONL. Returns (rows, labeled rows)."""
    tools = build_tools()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    labeled = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for raw in _read_source(input_file):
            sample = _convert_row(raw, tools)
            if sample is None:
                continue
            if sample.get("expected_answer"):
                labeled += 1
            f_out.write(json.dumps(sample) + "\n")
            count += 1
    return count, labeled


def _default_source() -> Path:
    """Find the input source under ``data/`` (precedence), else download the public CSV."""
    for name in ("labeled.jsonl", "public.jsonl", "public.txt", "public.csv"):
        candidate = DATA_DIR / name
        if candidate.exists():
            print(f"Using dataset: {candidate}")
            return candidate

    # Reproducible fallback: fetch the upstream public question set.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "public.csv"
    print(f"No local dataset found; downloading {CSV_URL} -> {dest}")
    urllib.request.urlretrieve(CSV_URL, dest)
    return dest


def prepare() -> Path:
    """``gym eval prepare`` entry point: convert the default source, return output path.

    Output is anchored to the CWD (repo root) via the same repo-relative string the
    benchmark config uses for ``jsonl_fpath`` so the CLI's ``output == jsonl_fpath``
    check holds regardless of the fsw<->fs12 import symlink (see _BENCHMARK_REL_FPATH).
    """
    out = Path(_BENCHMARK_REL_FPATH)
    count, labeled = convert_file(_default_source(), out)
    print(f"Wrote {count} benchmark samples ({labeled} labeled) to {out}")
    if labeled == 0:
        print(
            "NOTE: all samples are unlabeled — /verify will return reward 0 (dry-run). "
            "Provide data/labeled.jsonl (or a rubric column) for real scores."
        )
    return out.absolute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare / convert the Vals FABv2 benchmark data.")
    parser.add_argument(
        "--input", "-i", default=None,
        help="Explicit input file (.csv/.jsonl/.txt). If omitted, uses data/ precedence.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSONL path (only with --input). Default: input with a .jsonl suffix.",
    )
    args = parser.parse_args(argv)

    if args.input is None:
        prepare()
        return 0

    input_file = Path(args.input)
    if not input_file.exists():
        parser.error(f"Input file not found: {input_file}")
    output_file = Path(args.output) if args.output else input_file.with_suffix(".jsonl")

    count, labeled = convert_file(input_file, output_file)
    print(f"Converted {input_file} -> {output_file}")
    print(f"  rows: {count} ({labeled} with a synthesized/explicit expected_answer)")
    print(f"  tools: {', '.join(t['name'] for t in build_tools())}")
    print("  expected_answer: explicit label, else synthesized from rubric criteria (GOLD for reward_mode=binary)")
    print("  rubric: copied through verbatim (reference only; not used for reward)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
