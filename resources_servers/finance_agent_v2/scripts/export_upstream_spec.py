#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot the upstream Vals prompts and tool schemas into the benchmark directory.

`benchmarks/finance_agent_v2/prepare.py` bakes the upstream system/question prompts
and tool JSON schemas into every sample, but cannot import `finance_agent` to get
them: that package is installed only in this server's venv, while `gym eval prepare`
imports the prepare script into the root `gym` process.

So the schemas are still derived from the upstream `Tool` classes, just here at export
time instead of there at prepare time. `tests/test_upstream_spec.py` re-runs this
builder against the installed package and fails if the committed file has drifted.

The upstream commit is read from the `finance-agent` pin in requirements.txt, so
bumping that pin requires re-running this; prepare.py refuses to run when its
`_UPSTREAM_SHA` and the snapshot disagree.

Usage (from the resource server venv):
    python scripts/export_upstream_spec.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

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


SERVER_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_FPATH = SERVER_DIR / "requirements.txt"
SPEC_FPATH = SERVER_DIR.parent.parent / "benchmarks" / "finance_agent_v2" / "upstream_spec.json"

# Maps upstream tool name -> upstream Tool class (mirrors get_agent.available_tools).
_TOOL_CLASSES = {
    "web_search": TavilyWebSearch,
    "retrieve_information": RetrieveInformation,
    "parse_html_page": ParseHtmlPage,
    "edgar_search": EDGARSearch,
    "calculator": Calculator,
    "price_history": PriceHistory,
    "submit_final_result": SubmitFinalResult,
}

# The tool the agent must call to finish; appended after the selected tools so the
# emitted tool order matches what upstream's agent is given.
SUBMIT_TOOL = "submit_final_result"


def _upstream_sha() -> str:
    """Read the pinned upstream commit from the ``finance-agent`` requirement."""
    text = REQUIREMENTS_FPATH.read_text(encoding="utf-8")
    shas = re.findall(r"finance-agent-v2\.git@([0-9a-f]{40})", text)
    if len(shas) != 1:
        raise ValueError(
            f"expected exactly one pinned finance-agent-v2 requirement in {REQUIREMENTS_FPATH}, got {shas}"
        )
    return shas[0]


def _tool_schema(tool_cls) -> Dict[str, Any]:
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


def build_spec() -> Dict[str, Any]:
    """Build the snapshot from the installed ``finance_agent`` package."""
    missing = set(VALID_TOOLS) - set(_TOOL_CLASSES)
    if missing:
        # Exporting without it would produce a dataset whose agent cannot call the
        # tool, which reads as a weak model.
        raise ValueError(
            f"upstream VALID_TOOLS contains {sorted(missing)}, which has no class in _TOOL_CLASSES. "
            "Add it here (and check whether the resource server implements it) before exporting."
        )
    return {
        "upstream_sha": _upstream_sha(),
        "system_prompt": SYSTEM_PROMPT,
        "question_prompt": QUESTION_PROMPT,
        "valid_tools": list(VALID_TOOLS),
        "submit_tool": SUBMIT_TOOL,
        "tools": {name: _tool_schema(cls) for name, cls in _TOOL_CLASSES.items()},
    }


def serialize(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit non-zero if the committed snapshot differs from the installed package.",
    )
    args = parser.parse_args(argv)

    payload = serialize(build_spec())

    if args.check:
        if not SPEC_FPATH.exists():
            print(f"MISSING: {SPEC_FPATH}", file=sys.stderr)
            return 1
        if SPEC_FPATH.read_text(encoding="utf-8") != payload:
            print(
                f"STALE: {SPEC_FPATH} differs from the installed finance_agent package.\n"
                "Re-run `python scripts/export_upstream_spec.py` from the resource server venv.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {SPEC_FPATH} matches the installed finance_agent package.")
        return 0

    SPEC_FPATH.write_text(payload, encoding="utf-8")
    spec = json.loads(payload)
    print(f"Wrote {SPEC_FPATH}")
    print(f"  upstream_sha: {spec['upstream_sha']}")
    print(f"  tools: {', '.join(sorted(spec['tools']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
