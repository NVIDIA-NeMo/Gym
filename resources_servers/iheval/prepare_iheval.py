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
"""Build the IHEval Gym dataset in **Chat-Completions** shape, straight from the
upstream ``ytyz1307zzh/IHEval`` raw ``input_data.json`` files.

Writes:

* ``data/test.jsonl``    — all rows across all tasks.
* ``data/example.jsonl`` — a small mixed sample for smoke testing (committed).

Why Chat-Completions shape
--------------------------
IHEval is driven by the nemo-evaluator ``simple`` solver via the
``gym://...protocol=native`` scheme. The solver forwards
``responses_create_params.input`` and ``responses_create_params.tools``
**verbatim** to the vLLM ``/chat/completions`` endpoint — there is no
Responses→Chat conversion on that path. So the dataset must already be
Chat-Completions-shaped, or the native tool-use rows 400 ("tools.0.function:
Field required").

This reproduces the upstream benchmark's own request builder exactly, so the
prompt the model sees is byte-for-byte what upstream IHEval sends:

* message assembly  → ``src/model/run_model.py::main`` (vLLM backend branch):
  optional ``conversation_history`` (alternating user/assistant), then the
  ``user`` instruction, with ``system`` inserted at position 0.
* tool turn         → ``src/utils/call_api.py::tool_call_openai``: the OpenAI
  chat tool ``definition`` plus an ``assistant`` message carrying ``tool_calls``
  and a ``role: "tool"`` result (``arguments`` JSON-encoded, ``name`` kept).

The assembled chat ``messages`` go into ``responses_create_params.input`` and
the tool ``definition`` into ``responses_create_params.tools`` (both already the
shape vLLM chat accepts). Routing/gold fields ride at the ROW TOP LEVEL
(``task``, ``domain``, ``setting``, ``instruction``, ``answer``) so they survive
the native driver, which forwards top-level scalars but drops nested objects;
``answer`` is JSON-encoded and ``verify()`` JSON-decodes it (see app.py
``_decode_answer``).

Source: set ``IHEVAL_REPO_DIR`` to a local checkout (defaults to the in-repo
``benchmarks/iheval/IHEval`` if present); otherwise the upstream repo is
downloaded as a zip into ``$XDG_CACHE_HOME/byob_iheval`` (or ``~/.cache``).

Usage::

    python resources_servers/iheval/prepare_iheval.py
    python resources_servers/iheval/prepare_iheval.py --example-only
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOG = logging.getLogger(__name__)

_GITHUB_ZIP = "https://github.com/ytyz1307zzh/IHEval/archive/refs/heads/main.zip"
_ZIP_TOP_DIR = "IHEval-main"

_DATA_DIR = Path(__file__).resolve().parent / "data"
_AGENT = "iheval_simple_agent"

# In-repo checkout of the upstream benchmark, used when IHEVAL_REPO_DIR is unset.
_LOCAL_REPO = Path(__file__).resolve().parents[3] / "benchmarks" / "iheval" / "IHEval"

# (domain, task) pairs. ``task`` doubles as the verifier's scorer key.
# ``multi-turn`` rule-following is included: its ``conversation_history`` is
# pre-canned in the data (the assistant turns are fixed, not model-generated),
# and scoring grades only the final response with the same IFEval checker as
# ``single-turn`` — so it is a single generation over a pre-filled context.
_TASKS: Tuple[Tuple[str, str], ...] = (
    ("task-execution", "verb-extract"),
    ("task-execution", "translation"),
    ("task-execution", "lang-detect"),
    ("safety", "system-prompt-extract"),
    ("safety", "user-prompt-hijack"),
    ("tool-use", "slack-user"),
    ("tool-use", "get-webpage"),
    ("rule-following", "single-turn"),
    ("rule-following", "multi-turn"),
)

# Rows sampled (in order) for the committed example.jsonl smoke-test dataset.
_EXAMPLE_PER_TASK = {
    "verb-extract": 1,
    "lang-detect": 1,
    "system-prompt-extract": 1,
    "get-webpage": 1,
    "single-turn": 1,
    "multi-turn": 1,
}


# ── Upstream source ──────────────────────────────────────────────────────


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    cache = Path(base) / "byob_iheval"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _repo_root() -> Path:
    override = os.environ.get("IHEVAL_REPO_DIR")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "benchmark").is_dir():
            raise FileNotFoundError(f"IHEVAL_REPO_DIR missing 'benchmark/': {root}")
        return root

    if (_LOCAL_REPO / "benchmark").is_dir():
        return _LOCAL_REPO.resolve()

    cache = _cache_dir()
    target = cache / _ZIP_TOP_DIR
    sentinel = target / "benchmark"
    if sentinel.is_dir():
        return target

    LOG.info("Downloading IHEval source from %s", _GITHUB_ZIP)
    with urllib.request.urlopen(_GITHUB_ZIP, timeout=120) as resp:
        archive = resp.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        zf.extractall(cache)
    if not sentinel.is_dir():
        raise FileNotFoundError(f"IHEval extraction failed; missing {sentinel}")
    return target


def _iter_rows(root: Path, domain: str, task: str) -> List[Dict[str, Any]]:
    """Load every ``input_data.json`` under a task, tagging its setting."""
    task_root = root / "benchmark" / domain / task
    if not task_root.is_dir():
        raise FileNotFoundError(f"IHEval task directory missing: {task_root}")
    rows: List[Dict[str, Any]] = []
    for path in sorted(task_root.rglob("input_data.json")):
        setting = str(path.parent.relative_to(task_root))
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for row in data:
            row = dict(row)
            row["_setting"] = setting
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No input_data.json under {task_root}")
    return rows


# ── Upstream request building (verbatim from run_model.py + call_api.py) ──


def _tool_call_openai(tool: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Port of ``src/utils/call_api.py::tool_call_openai``.

    Returns the OpenAI-chat ``definition`` list plus the pre-canned ``assistant``
    tool-call message and the ``role: "tool"`` result message.
    """
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    definition = [
        {
            "type": "function",
            "function": {
                "name": raw_definition["name"],
                "description": raw_definition["description"],
                "parameters": {
                    "type": "object",
                    "properties": raw_definition["parameters"],
                    "required": list(raw_definition["parameters"].keys()),
                },
            },
        }
    ]

    tool_call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": raw_tool_call["id"],
                "type": "function",
                "function": {
                    "name": raw_tool_call["name"],
                    "arguments": json.dumps(raw_tool_call["arguments"]),
                },
            }
        ],
    }

    tool_return = {
        "role": "tool",
        "tool_call_id": raw_tool_return["id"],
        "name": raw_tool_return["name"],
        "content": raw_tool_return["content"],
    }

    return definition, tool_call, tool_return


def _build_messages(example: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Assemble chat ``messages`` (+ optional ``tools``) exactly like upstream.

    Mirrors ``run_model.py::main`` (message assembly) followed by
    ``call_api.py::call_openai`` (tool turn extension + system insertion). Net
    order: ``[system?, <history?>, user(instruction), assistant(tool_call)?,
    tool(result)?]``.
    """
    messages: List[Dict[str, Any]] = []

    history = example.get("conversation_history")
    if history:
        messages.extend(
            [
                {"role": "user", "content": msg} if i % 2 == 0 else {"role": "assistant", "content": msg}
                for i, msg in enumerate(history)
            ]
        )

    messages.append({"role": "user", "content": example["instruction"]})

    tools: Optional[List[Dict[str, Any]]] = None
    if "tool" in example and example["tool"] is not None:
        definition, tool_call, tool_return = _tool_call_openai(example["tool"])
        messages.extend([tool_call, tool_return])
        tools = definition

    # System prompt goes first (call_api.py inserts at position 0 after the tool
    # turn is appended, so it precedes everything regardless).
    system = example.get("system")
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})

    return messages, tools


# ── Row → Gym task ───────────────────────────────────────────────────────


def _to_task(row: Dict[str, Any], domain: str, task: str) -> Dict[str, Any]:
    messages, tools = _build_messages(row)

    params: Dict[str, Any] = {"input": messages}
    if tools is not None:
        params["tools"] = tools

    # Routing/gold fields ride at the ROW TOP LEVEL so they survive the
    # nemo-evaluator ``gym://...protocol=native`` driver, which forwards
    # top-level SCALARS but drops nested objects. ``answer`` is a dict/list for
    # safety, rule-following and get-webpage, so JSON-encode it; verify()
    # JSON-decodes via ``_decode_answer``.
    return {
        "responses_create_params": params,
        "id": row.get("id"),
        "task": task,
        "domain": domain,
        "setting": row.get("_setting", ""),
        "instruction": str(row.get("instruction", "")),
        "answer": json.dumps(row.get("answer"), ensure_ascii=False),
        "agent_ref": {"type": "responses_api_agents", "name": _AGENT},
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"IHEval: wrote {len(rows)} rows -> {path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-only", action="store_true", help="Only (re)write data/example.jsonl.")
    args = parser.parse_args(argv)

    root = _repo_root()
    LOG.info("Using IHEval source at %s", root)
    all_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    n_tool_rows = 0
    for domain, task in _TASKS:
        rows = _iter_rows(root, domain, task)
        tasks = [_to_task(r, domain, task) for r in rows]
        n_tool_rows += sum(1 for t in tasks if "tools" in t["responses_create_params"])
        all_rows.extend(tasks)
        for t in tasks[: _EXAMPLE_PER_TASK.get(task, 0)]:
            example_rows.append(t)
        print(f"IHEval: {domain}/{task}: {len(tasks)} rows")

    _write_jsonl(_DATA_DIR / "example.jsonl", example_rows)
    if not args.example_only:
        _write_jsonl(_DATA_DIR / "test.jsonl", all_rows)
    print(f"IHEval: {n_tool_rows} rows carry a tool turn")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
