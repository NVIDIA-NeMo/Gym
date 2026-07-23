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
"""Convert WebVoyager-style task JSON into this env's ``example.jsonl`` rows.

WebVoyager (https://github.com/MinorJerry/WebVoyager, MIT license) ships one JSON
object per line with these fields::

    {"web_name": "Allrecipes", "id": "Allrecipes--0",
     "ques": "Provide a recipe for vegetarian lasagna ...",
     "web": "https://www.allrecipes.com/"}

This script maps each such row to a NeMo-Gym rollout input for the
``lexmount_browser`` environment: ``responses_create_params`` (the system + user
messages and the browser tool schemas the policy may call) plus the two extra
fields this env reads — ``initial_url`` (consumed in ``seed_session``, set to the
task's ``web``) and ``verifier_metadata`` (consumed in ``verify``).

Cleaning conventions (ported from the validated 0721 WebVoyager pipeline,
`nemorl-webagent@3220bc5` `training/scripts/prepare_webvoyager_data.py`):
  * the source is validated by row count and (optionally) SHA-256 before use;
  * duplicate task ``id``s are rejected;
  * the source ``id`` (e.g. "Allrecipes--3") is preserved as ``task_id`` so a
    smoke subset never silently re-indexes tasks;
  * a task-agnostic browser-agent system prompt is used — no demonstrations,
    answers, or synthetic data are injected.

IMPORTANT — verifier gap. WebVoyager has no rule-checkable ground truth: success
is judged by a trajectory-level LLM judge (see README "Reward"). This env's
in-PR ``verify()`` is rule-based (final_url / url_contains / dom_contains /
answer_equals). We therefore emit a *conservative* ``url_contains`` spec derived
from the task's start-URL host, which checks the agent reached/stayed on the
right site but does NOT check task success. Treat converted rows as scaffolding
for the LLM-judge extension, not as a rule-scored benchmark. Rows whose success
genuinely needs the judge are marked ``"verifier_metadata": {"needs_llm_judge": true}``
when ``--emit-judge-todo`` is passed, so downstream tooling can route them.

Usage::

    # Convert a WebVoyager JSONL export (not bundled — fetch from upstream):
    python scripts/convert_webvoyager.py \
        --source WebVoyager_data.jsonl \
        --output data/webvoyager_example.jsonl --limit 3

    # Self-check against the 3 bundled sample tasks (no external data needed):
    python scripts/convert_webvoyager.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

# Task-agnostic browser-agent contract. Byte-compatible in spirit with the 0721
# validated pipeline's BROWSER_AGENT_SYSTEM_PROMPT: it only tells the model the
# browser tools exist and must be used — no answers, no demonstrations.
SYSTEM = (
    "You are a web agent operating a live browser through tools. Call browser_observe to see the "
    "page (its URL, title, and a numbered list of interactive elements as `[id] role: name`). "
    "Use browser_navigate / browser_click / browser_type to act — element_id values come from the "
    "most recent observation. Call browser_finish when the task is complete."
)

# Same tool schema this env's generate_data.py emits (Responses-API strict tools).
TOOLS = [
    {"type": "function", "name": "browser_observe",
     "description": "Return the current page: URL, title, and a numbered list of interactive elements ([id] role: name).",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "browser_navigate",
     "description": "Navigate to a URL.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"type": "function", "name": "browser_click",
     "description": "Click the interactive element with the given id from the latest observation.",
     "parameters": {"type": "object", "properties": {"element_id": {"type": "integer"}}, "required": ["element_id"]}},
    {"type": "function", "name": "browser_type",
     "description": "Type text into the element with the given id.",
     "parameters": {"type": "object",
                    "properties": {"element_id": {"type": "integer"}, "text": {"type": "string"}},
                    "required": ["element_id", "text"]}},
    {"type": "function", "name": "browser_finish",
     "description": "End the episode, reporting an answer ('' if none).",
     "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": []}},
]

for _t in TOOLS:
    _p = _t["parameters"]
    _p["additionalProperties"] = False
    _p["required"] = list(_p.get("properties", {}).keys())
    _t["strict"] = True

REQUIRED_FIELDS = ("web_name", "id", "ques", "web")

# Three real WebVoyager tasks (verbatim from the upstream MIT dataset) kept in the
# repo so the converter is runnable and testable WITHOUT redistributing the full
# 600-task set. These are the canonical Allrecipes--0 / Amazon--0 / GitHub--0 rows.
SAMPLE_TASKS = [
    {"web_name": "Allrecipes", "id": "Allrecipes--0",
     "ques": "Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people.",
     "web": "https://www.allrecipes.com/"},
    {"web_name": "Amazon", "id": "Amazon--0",
     "ques": "Search an Xbox Wireless controller with green color and rate 4 stars and up.",
     "web": "https://www.amazon.com/"},
    {"web_name": "GitHub", "id": "GitHub--0",
     "ques": "Search for an open-source project related to 'climate change' on GitHub and report the project with the most stars.",
     "web": "https://github.com/"},
]


def host_of(url: str) -> str:
    """Return a bare host (no leading www.) for a conservative url_contains spec."""
    netloc = urlparse(url).netloc or url
    return netloc[4:] if netloc.startswith("www.") else netloc


def to_row(task: dict, emit_judge_todo: bool = False) -> dict:
    for field in REQUIRED_FIELDS:
        if field not in task:
            raise ValueError(f"task {task.get('id', '<no id>')!r} missing required field {field!r}")
    # Conservative, rule-checkable spec: the agent must reach/stay on the task
    # host. Full task success needs the LLM judge (see module docstring).
    verifier_metadata: dict = {"url_contains": host_of(task["web"])}
    if emit_judge_todo:
        verifier_metadata["needs_llm_judge"] = True
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": task["ques"]},
            ],
            "tools": TOOLS,
        },
        "initial_url": task["web"],
        "verifier_metadata": verifier_metadata,
        # Provenance passthrough (ignored by seed_session/verify; handy for audit).
        "task_id": task["id"],
        "website": task["web_name"],
    }


def read_source(path: Path, expected_sha256: str | None, expected_rows: int | None) -> list[dict]:
    raw = path.read_bytes()
    if expected_sha256:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise SystemExit(f"source SHA-256 mismatch: got {actual}, expected {expected_sha256}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if expected_rows is not None and len(rows) != expected_rows:
        raise SystemExit(f"source row mismatch: got {len(rows)}, expected {expected_rows}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("source contains duplicate task IDs")
    return rows


def selftest() -> int:
    rows = [to_row(t) for t in SAMPLE_TASKS]
    assert rows[0]["initial_url"] == "https://www.allrecipes.com/"
    assert rows[0]["verifier_metadata"]["url_contains"] == "allrecipes.com"
    assert rows[1]["verifier_metadata"]["url_contains"] == "amazon.com"
    assert rows[2]["verifier_metadata"]["url_contains"] == "github.com"
    # tool schema is Responses-API-strict
    assert all(t["strict"] is True for t in rows[0]["responses_create_params"]["tools"])
    # each row round-trips through JSON
    for r in rows:
        json.loads(json.dumps(r))
    # judge-todo flag is opt-in
    assert "needs_llm_judge" not in rows[0]["verifier_metadata"]
    assert to_row(SAMPLE_TASKS[0], emit_judge_todo=True)["verifier_metadata"]["needs_llm_judge"] is True
    print("selftest ok: 3 sample WebVoyager tasks convert and validate")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="WebVoyager JSONL export (one task object per line).")
    parser.add_argument("--output", type=Path, help="Destination JSONL (this env's example format).")
    parser.add_argument("--limit", type=int, default=-1, help="Convert at most N tasks (default: all).")
    parser.add_argument("--offset", type=int, default=0, help="Start offset into the source (default: 0).")
    parser.add_argument("--expected-sha256", default=None, help="Fail unless the source hashes to this value.")
    parser.add_argument("--expected-rows", type=int, default=None, help="Fail unless the source has this many rows.")
    parser.add_argument("--emit-judge-todo", action="store_true",
                        help="Mark rows with needs_llm_judge:true (full success requires the LLM judge).")
    parser.add_argument("--selftest", action="store_true", help="Convert the 3 bundled sample tasks and validate.")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.source or not args.output:
        parser.error("--source and --output are required unless --selftest is given")

    rows = read_source(args.source, args.expected_sha256, args.expected_rows)
    if args.offset < 0 or args.offset >= len(rows):
        raise SystemExit(f"offset must be in [0, {len(rows) - 1}], got {args.offset}")
    selected = rows[args.offset:] if args.limit < 0 else rows[args.offset: args.offset + args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task in selected:
            handle.write(json.dumps(to_row(task, emit_judge_todo=args.emit_judge_todo), ensure_ascii=False) + "\n")
    print(json.dumps({
        "source": str(args.source),
        "output": str(args.output),
        "converted_rows": len(selected),
        "source_rows": len(rows),
        "offset": args.offset,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
