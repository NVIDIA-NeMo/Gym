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
"""Build the IHEval Gym dataset in **Responses API** shape, straight from the
upstream ``ytyz1307zzh/IHEval`` raw ``input_data.json`` files.

Writes:

* ``data/test.jsonl``          — all rows across all tasks.
* ``data/test_conflict.jsonl`` — only the ``conflict/*`` setting rows.
* ``data/example.jsonl``       — a small mixed sample for smoke testing (committed).

With ``--chat-completions``, each of those gets a ``*_chat.jsonl`` twin holding
the same tasks with the request in Chat Completions shape (see below).

Why a conflict-only dataset
---------------------------
The gym driver "owns the loop": its headline metric is a plain per-row
``mean_reward`` over the whole dataset — it never calls the Gym server's
``compute_metrics``/``get_key_metrics``, so the aggregate conflict
``result_score`` that ``app.py`` computes does not surface in a plain eval run.
IHEval's headline is the **conflict** setting (instruction hierarchy is what the
conflict setting stresses), so ``test_conflict.jsonl`` restricts the dataset to
the ``conflict/*`` rows; the per-row ``mean_reward`` over that file IS the
average conflict score. (Row counts differ across tasks, so this is a per-row
mean, not the task-macro average of upstream ``average_final_score.py`` — that
exact number still comes from the gym-native ``compute_metrics`` path over the
full ``test.jsonl``.)

Responses API shape
-------------------
Rows are Responses-API-native, like every other Gym resources server.
``simple_agent`` POSTs them to ``/v1/responses`` on the model server, which
runs ``ResponsesConverter`` to reach ``/chat/completions``, so the chat shape
is reconstructed downstream rather than stored.

* ``input`` — pre-canned tool turns are ``function_call`` /
  ``function_call_output`` items paired by ``call_id`` (not ``assistant`` with
  ``tool_calls`` + ``role: "tool"``).
* ``tools`` — the flat ``FunctionToolParam`` form: ``name``, ``description``,
  ``parameters`` and ``strict`` at the top level.
  ``NeMoGymResponseCreateParamsNonStreaming.tools`` is typed ``List[ToolParam]``
  and rejects the nested ``{"type": "function", "function": {...}}`` form with a
  422 ("tools.0.FunctionToolParam.name: Field required") before the request ever
  reaches the model server.

The ``--chat-completions`` variant
----------------------------------
Not every harness that runs this dataset speaks the Responses API. Some read a
row and forward its ``input`` and ``tools`` straight to ``/chat/completions``,
which rejects the Responses shapes above.

``--chat-completions`` emits ``*_chat.jsonl`` twins for those callers: the same
tasks with ``input`` and ``tools`` pre-translated to the Chat Completions shape.
Only the request shape differs — the scoring fields are untouched, so both files
score the same rows the same way, and both put the same prompt in front of the
model.

This is a shape change only — the translation back to chat is lossless, so the
prompt the model ultimately sees is still byte-for-byte what upstream IHEval
sends, and the request builder below stays a faithful port:

* message assembly  → ``src/model/run_model.py::main`` (vLLM backend branch):
  optional ``conversation_history`` (alternating user/assistant), then the
  ``user`` instruction, with ``system`` inserted at position 0.
* tool turn         → ``src/utils/call_api.py::tool_call_openai``: the tool
  ``definition`` plus the pre-canned call and its result (``arguments``
  JSON-encoded), re-expressed as Responses items.

The assembled ``messages`` go into ``responses_create_params.input`` and the
tool ``definition`` into ``responses_create_params.tools``. Routing/gold fields
ride at the ROW TOP LEVEL
(``task``, ``domain``, ``setting``, ``instruction``, ``answer``);
``answer`` is JSON-encoded and ``verify()`` JSON-decodes it (see app.py
``_decode_answer``).

Usage::

    python resources_servers/iheval/prepare_iheval.py
    python resources_servers/iheval/prepare_iheval.py --example-only
    python resources_servers/iheval/prepare_iheval.py --chat-completions
"""

from __future__ import annotations

import argparse
import copy
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
    cache = Path(base) / "iheval"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _repo_root() -> Path:
    override = os.environ.get("IHEVAL_REPO_DIR")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "benchmark").is_dir():
            raise FileNotFoundError(f"IHEVAL_REPO_DIR missing 'benchmark/': {root}")
        return root

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

    Returns the ``tools`` ``definition`` list plus the pre-canned tool call and
    tool result, both as Responses API typed input items.
    """
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    # Responses API flat shape (``ToolParam``/``FunctionToolParam``): ``name``, ``description``
    # and ``parameters`` sit at the top level and ``strict`` must be present, because
    # ``NeMoGymResponseCreateParamsNonStreaming.tools`` is typed ``List[ToolParam]``. The nested
    # Chat Completions shape (``{"type": "function", "function": {...}}``) is rejected with a 422
    # before the request reaches the model server.
    definition = [
        {
            "type": "function",
            "name": raw_definition["name"],
            "description": raw_definition["description"],
            "parameters": {
                "type": "object",
                "properties": raw_definition["parameters"],
                "required": list(raw_definition["parameters"].keys()),
            },
            "strict": None,
        }
    ]

    # Responses API typed input items, matching the rest of the Gym benchmarks.
    # Upstream's chat shape (``assistant`` with ``tool_calls`` + ``role: "tool"``)
    # is reconstructed downstream by ``ResponsesConverter``.
    #
    # The result is keyed off the *call's* id rather than ``raw_tool_return["id"]``
    # so the pair can never drift apart. Upstream ships them identical for all
    # 2520 tool rows, so this changes no emitted value.
    call_id = raw_tool_call["id"]

    tool_call = {
        "type": "function_call",
        "call_id": call_id,
        "name": raw_tool_call["name"],
        "arguments": json.dumps(raw_tool_call["arguments"]),
    }

    tool_return = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": raw_tool_return["content"],
    }

    return definition, tool_call, tool_return


def _build_messages(example: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Assemble the ``input`` items (+ optional ``tools``) exactly like upstream.

    Mirrors ``run_model.py::main`` (message assembly) followed by
    ``call_api.py::call_openai`` (tool turn extension + system insertion). Net
    order: ``[system?, <history?>, user(instruction), function_call?,
    function_call_output?]``.
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

    # Routing/gold fields ride at the ROW TOP LEVEL. ``answer`` is a dict/list
    # for safety, rule-following and get-webpage, so JSON-encode it; verify()
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


def _nest_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Flat ``FunctionToolParam`` -> the nested Chat Completions tool form.

    ``strict`` is dropped rather than carried into ``function``: it is always
    ``None`` here (upstream IHEval defines no strict-mode tools) and a null
    ``strict`` is rejected on the chat endpoint.
    """
    assert tool["type"] == "function", f"unexpected tool type {tool['type']!r}"
    function = {"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]}
    return {"type": "function", "function": copy.deepcopy(function)}


def _chat_input(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Responses typed items -> Chat Completions messages (upstream's shape).

    The tool result keeps the ``name`` of the tool it answers. A
    ``function_call_output`` item has no such field, so it is recovered from the
    matching call rather than left off — upstream IHEval sends it.
    """
    messages: List[Dict[str, Any]] = []
    names: Dict[str, str] = {}
    for item in items:
        item_type = item.get("type")
        if item_type == "function_call":
            names[item["call_id"]] = item["name"]
            # No ``content`` key at all, matching upstream: an assistant turn that
            # only calls a tool carries no text.
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item["call_id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": item["arguments"]},
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "content": item["output"],
                    "tool_call_id": item["call_id"],
                    "name": names[item["call_id"]],
                }
            )
        else:
            messages.append(copy.deepcopy(item))
    return messages


def _to_chat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``row`` with the request re-expressed in the Chat Completions shape.

    For callers that forward ``input`` and ``tools`` to ``/chat/completions``
    unchanged. Scoring fields are copied through untouched, so the row still
    verifies identically.
    """
    chat_row = copy.deepcopy(row)
    params = chat_row["responses_create_params"]
    params["input"] = _chat_input(params["input"])
    if params.get("tools") is not None:
        params["tools"] = [_nest_tool(t) for t in params["tools"]]
    return chat_row


def _setting_category(setting: str) -> str:
    """``conflict/foo`` -> ``conflict`` (aligned / conflict / reference)."""
    return setting.split("/", 1)[0] if setting else "unknown"


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"IHEval: wrote {len(rows)} rows -> {path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-only", action="store_true", help="Only (re)write data/example.jsonl.")
    parser.add_argument(
        "--chat-completions",
        action="store_true",
        help="Also write *_chat.jsonl twins whose request is in Chat Completions shape.",
    )
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

    # Conflict-only subset: per-row mean_reward over this file is the
    # average conflict score (see module docstring).
    conflict_rows = [t for t in all_rows if _setting_category(t["setting"]) == "conflict"]

    outputs = [("example", example_rows)]
    if not args.example_only:
        outputs += [("test", all_rows), ("test_conflict", conflict_rows)]

    for stem, rows in outputs:
        _write_jsonl(_DATA_DIR / f"{stem}.jsonl", rows)
        if args.chat_completions:
            _write_jsonl(_DATA_DIR / f"{stem}_chat.jsonl", [_to_chat_row(r) for r in rows])
    print(f"IHEval: {n_tool_rows} rows carry a tool turn")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
