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
"""A recorded attempt: every model request the harness sent while it played one game.

The recording is a ``.snap.jsonl`` capture (schema ``arc3-python-snapshot/v1``). Its ``call`` lines hold each request:
the messages (as a delta of the previous request: keep the first ``messages_keep``, append ``messages_append``), the
sampling parameters and tools when they changed, and the state the request was sent from (``step``: actions so far,
``level``, ``frame_sha256``: the board, ``request_sha256``: the exact POST body).

A recorded call's reply is not stored as such: it is the assistant message that the next request appends after the
call's messages. This module is standard library only; the worker (another interpreter) imports it too.
"""

import hashlib
import json
import re
from typing import Any


SCHEMA = "arc3-python-snapshot/v1"

# The run clock as a tool result prints it: Python repr, JSON, JSON inside a JSON string, or line by line.
CLOCK = re.compile(
    r"""\\?["']?run_elapsed_seconds\\?["']? ?[:=] ?([0-9.eE+-]+)(?:, |\\n|\n| )\\?["']?"""
    r"""time_remaining_seconds\\?["']? ?[:=] ?([0-9.eE+-]+|None|null)"""
)
# The harness's Python retry guard, with the cell counter it had for this call.
GUARD_CALL = re.compile(r'"retry_guard": \{\s*"status": "[a-z_]+",\s*"code_id": "[0-9a-f]+",\s*"request": (\d+)')
# The sandbox memory limit as the harness reports it: the MemoryError cap (GiB, 6 significant digits) and the notice.
MEMORY_GIB = re.compile(r"configured virtual-memory cap: ([0-9.eE+-]+) GiB")
MEMORY_MIB = re.compile(r'"memory_limit":\s*"Python virtual-memory limit: ([0-9.]+) MiB\."')
# The sandbox's working directory name (tempfile, random), which can reach the model in a path.
SANDBOX_DIR = re.compile(r"/rgb_python_tool_([A-Za-z0-9_]{8})\b")
SANDBOX_DIR_PREFIX = "rgb_python_tool_"
TIMEOUT_TEXT = "Tool timed out after"
# Texts of the harness's message that resumes a turn it ended on its yield budget (any resume mode and tone).
YIELD_MARKERS = (
    "You yielded control on the turn time budget",
    "Nothing has been executed since the turn opener above",  # the state_only resume's caption
    "You spent the entire turn budget reasoning without calling a tool",
    "Your last python snippet executed no actions (it only computed",
    "Your previous exchange produced no tool call and executed no actions",
)
# Object addresses and id() values (16-byte aligned) that a tool result may print: no replay reproduces them.
HEX_ADDRESS = re.compile(r"0x[0-9a-f]{8,16}\b")
ID_INTEGER = re.compile(r"(?<![\w.])\d{14,16}(?![\w.])")
# The harness's helper requests (continuations of an assistant message it wrote), and where their replies show.
THINK_CLOSE = "\n</think>\n\n"
FORCE_ACT_CLOSE = (
    "I have thought about this long enough. I must act now: I will write the code for my best plan so far and call "
    "the tool."
)
LEVEL_SUMMARY_PROMPT = re.compile(r"Level (\d+) was completed at action \d+\. Summarize what would have helped")
LEVEL_SUMMARY_SHOWN = "Level summary \u2014 lessons through completed level {level}:\n"
LEVEL_SUMMARY_END = "\nLater levels may change the rules or goal; verify these lessons against the current board."
LEVEL_SUMMARY_FAILED = "Summary update after level {level} failed."
COMPACTION_PROMPT = "HISTORY SUMMARY. To make room"
COMPACTION_SHOWN = "[Summary of earlier history"
# Request fields of a specific engine (the Kaggle kit's SGLang): session and state-ownership metadata, not model input.
ENGINE_FIELDS = frozenset({"gdn_ownership", "kv_evict_conversation", "session_id"})
# Chat-template kwargs that set the thinking effort (a model's template reads one of them).
EFFORT_KWARGS = ("reasoning_effort", "thinking_effort")
# A recorded reply's generated length when only its text is known (the virtual clock's per-token term).
CHARS_PER_TOKEN = 3.6


def parse(raw: bytes) -> dict[str, Any]:
    """``{"header", "actions", "calls"}``; each call carries its rebuilt request as ``payload``. A truncated last line
    (a killed run) is ignored."""
    header, actions, calls = None, [], []
    messages: list[dict[str, Any]] = []
    params: dict[str, Any] | None = None
    tools_by_sha: dict[str, Any] = {}
    lines = raw.decode("utf-8").splitlines()
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except ValueError:
            if i == len(lines) - 1:
                break
            raise
        kind = record.get("t")
        if kind == "header":
            if record.get("schema") != SCHEMA:
                raise ValueError(f"recording schema {record.get('schema')!r}, want {SCHEMA}")
            header = record
        elif kind == "actions":
            if record["from"] != len(actions):
                raise ValueError(f"recording actions from {record['from']}, have {len(actions)}")
            actions.extend(record["rows"])
        elif kind == "call":
            messages = messages[: record["messages_keep"]] + list(record["messages_append"])
            params = record["params"] if record.get("params") is not None else params
            if record.get("tools") is not None:
                tools_by_sha[record["tools_sha256"]] = record["tools"]
            payload = dict(params or {})  # messages / tools are placeholders there: the key order as sent
            payload["messages"] = list(messages)
            if record.get("tools_sha256"):
                payload["tools"] = tools_by_sha[record["tools_sha256"]]
            fields = {k: v for k, v in record.items() if k not in ("messages_append", "params", "tools")}
            calls.append({**fields, "payload": payload})
    if header is None:
        raise ValueError("recording has no header line")
    return {"header": header, "actions": actions, "calls": calls}


def request_sha256(payload: dict[str, Any]) -> str:
    """sha256 of the POST body the harness sends for ``payload`` (``requests.post(url, json=payload)``)."""
    import requests

    body = requests.Request("POST", "http://model.invalid/v1/chat/completions", json=payload).prepare().body
    return hashlib.sha256(body if isinstance(body, bytes) else str(body).encode("utf-8")).hexdigest()


def model_visible(payload: dict[str, Any]) -> dict[str, Any]:
    """The request without engine-only fields: what the model is given (messages, tools, sampling)."""
    return {k: v for k, v in payload.items() if k not in ENGINE_FIELDS}


def effort_of(payload: dict[str, Any]) -> dict[str, Any]:
    """The thinking effort a request carries, as ``{field: value}``: its ``chat_template_kwargs`` keys of
    ``EFFORT_KWARGS``, and ``"@reasoning_effort"`` for a top-level ``reasoning_effort``. Empty: the chat template's
    default."""
    kwargs = payload.get("chat_template_kwargs") or {}
    found = {k: kwargs[k] for k in EFFORT_KWARGS if k in kwargs}
    if "reasoning_effort" in payload:
        found["@reasoning_effort"] = payload["reasoning_effort"]
    return found


def with_effort(payload: dict[str, Any], effort: dict[str, Any]) -> dict[str, Any]:
    """``payload`` with ``effort`` (as :func:`effort_of` returns it) added. A request that carries an effort of its
    own (the harness's effort ladder) is returned unchanged."""
    if not effort or effort_of(payload):
        return payload
    out = dict(payload)
    kwargs = dict(out.get("chat_template_kwargs") or {})
    for field, value in effort.items():
        if field.startswith("@"):
            out[field[1:]] = value
        else:
            kwargs[field] = value
    if kwargs:
        out["chat_template_kwargs"] = kwargs
    return out


def thinks(payload: dict[str, Any]) -> bool:
    """Whether the request asks for reasoning (``chat_template_kwargs.enable_thinking`` is not false)."""
    return (payload.get("chat_template_kwargs") or {}).get("enable_thinking") is not False


def frame_sha256(grid: Any) -> str:
    """The recording's ``frame_sha256``: the board as compact JSON of ints."""
    plain = [[int(c) for c in row] for row in grid]
    return hashlib.sha256(json.dumps(plain, separators=(",", ":")).encode()).hexdigest()


def reply_of(calls: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    """The recorded reply to request ``index``: the assistant message the next request appends after its messages.
    None when the next request does not extend this one (a helper request or a context rewrite in between)."""
    if index + 1 >= len(calls):
        return None
    sent, following = calls[index]["payload"]["messages"], calls[index + 1]["payload"]["messages"]
    if len(following) <= len(sent) or following[: len(sent)] != sent:
        return None
    reply = following[len(sent)]
    return reply if reply.get("role") == "assistant" else None


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def yielded_after(calls: list[dict[str, Any]], index: int) -> bool:
    """Whether the harness ended the turn on its yield budget right after the reply to request ``index``: the next
    request goes on with the harness's resume message. (A reply the harness continued in a force-act request is
    judged after that continuation.)"""
    msgs = calls[index]["payload"]["messages"]
    if index + 1 >= len(calls) or _force_act_of(calls[index + 1], msgs):
        return False
    base = msgs[:-1] if _is_force_act(calls[index]) else msgs
    following = calls[index + 1]["payload"]["messages"]
    if len(following) <= len(base) or following[: len(base)] != base:
        return False
    added = following[len(base) + 1 :]
    return any(m.get("role") == "user" and any(k in _text(m) for k in YIELD_MARKERS) for m in added)


def _continues(call: dict[str, Any]) -> bool:
    messages = call["payload"]["messages"]
    return (
        bool(call["payload"].get("continue_final_message"))
        and bool(messages)
        and (messages[-1].get("role") == "assistant")
    )


def _is_force_act(call: dict[str, Any]) -> bool:
    """A force-act request: the harness closed a reply cut off inside its reasoning and asks for its continuation."""
    if not _continues(call):
        return False
    prefix = _text(call["payload"]["messages"][-1])
    return prefix.endswith(THINK_CLOSE) and FORCE_ACT_CLOSE in prefix


def _force_act_of(call: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    """Whether ``call`` is the force-act continuation of the reply to a request with ``messages``."""
    return _is_force_act(call) and call["payload"]["messages"][:-1] == messages


def recorded_reply(calls: list[dict[str, Any]], index: int) -> tuple | None:
    """``(message, finish_reason)``: the reply to request ``index`` as the model server returned it, recovered from
    where the recording shows it. None when the recording does not show it.

    - A gameplay reply: the assistant message the next request appends (``reply_of``).
    - A reply the harness then force-acted on: its reasoning, cut off at the output limit (``finish_reason`` length);
      the force-act request's prefix holds it, closed by the harness's sentence.
    - The force-act continuation: the tool calls of the closed reply in a later request.
    - A level summary: the text later openers show under the level's summary heading (empty when they show that the
      summary failed).
    - A history compaction: the text the next request shows under its "[Summary of earlier history ...]" header.
    """
    call = calls[index]
    messages = call["payload"]["messages"]
    if _continues(call):
        prompt = _text(messages[-2]) if len(messages) >= 2 else ""
        if _is_force_act(call):
            base = messages[:-1]
            for later in calls[index + 1 :]:
                following = later["payload"]["messages"]
                if len(following) > len(base) and following[: len(base)] == base:
                    closed = following[len(base)]
                    if closed.get("role") == "assistant":
                        reply = {"role": "assistant", "content": closed.get("content")}
                        if closed.get("tool_calls"):
                            reply["tool_calls"] = closed["tool_calls"]
                        return reply, "tool_calls" if closed.get("tool_calls") else "stop"
            return None
        summary = LEVEL_SUMMARY_PROMPT.match(prompt)
        if summary:
            shown = LEVEL_SUMMARY_SHOWN.format(level=summary.group(1))
            failed = LEVEL_SUMMARY_FAILED.format(level=summary.group(1))
            for later in calls[index + 1 :]:
                for message in later["payload"]["messages"]:
                    text = _text(message)
                    start = text.find(shown)
                    if start >= 0:
                        end = text.find(LEVEL_SUMMARY_END, start)
                        if end >= 0:
                            return {"role": "assistant", "content": text[start + len(shown) : end]}, "stop"
                    if failed in text:
                        return {"role": "assistant", "content": ""}, "stop"
            return None
        if prompt.startswith(COMPACTION_PROMPT) and index + 1 < len(calls):
            for message in calls[index + 1]["payload"]["messages"]:
                text = _text(message)
                if message.get("role") == "user" and text.startswith(COMPACTION_SHOWN) and "]\n" in text:
                    return {"role": "assistant", "content": text.split("]\n", 1)[1]}, "stop"
        return None
    if index + 1 < len(calls) and _force_act_of(calls[index + 1], messages):
        reasoning = _text(calls[index + 1]["payload"]["messages"][-1])[: -len(THINK_CLOSE)]
        tail = "\n\n" + FORCE_ACT_CLOSE
        if reasoning.endswith(tail):
            reasoning = reasoning[: -len(tail)]
        return {"role": "assistant", "content": "", "reasoning_content": reasoning}, "length"
    reply = reply_of(calls, index)
    if reply is None:
        return None
    return reply, "tool_calls" if reply.get("tool_calls") else "stop"


def tool_results(calls: list[dict[str, Any]]) -> dict[str, str]:
    """tool_call_id -> the recorded tool message content, over every request."""
    out: dict[str, str] = {}
    for call in calls:
        for message in call["payload"]["messages"]:
            if message.get("role") == "tool" and message.get("tool_call_id"):
                out.setdefault(str(message["tool_call_id"]), str(message.get("content")))
    return out


def sandbox_dir_names(calls: list[dict[str, Any]], results: dict[str, str]) -> list[tuple]:
    """(request index, sandbox directory name) at each name's first mention, in recording order."""
    out, seen = [], set()
    for index, call in enumerate(calls):
        for message in call["payload"]["messages"]:
            for tool_call in message.get("tool_calls") or [] if message.get("role") == "assistant" else []:
                for name in SANDBOX_DIR.findall(results.get(str(tool_call.get("id", ""))) or ""):
                    if name not in seen:
                        seen.add(name)
                        out.append((index, SANDBOX_DIR_PREFIX + name))
    return out


def memory_limit_bytes(texts: list[str | None]) -> int | None:
    """The sandbox memory limit recorded tool results show (the MemoryError cap is finer, so it wins)."""
    for pattern, unit in ((MEMORY_GIB, 1 << 30), (MEMORY_MIB, 1 << 20)):
        for text in texts:
            match = pattern.search(text or "")
            if match:
                return int(round(float(match.group(1)) * unit))
    return None


def reply_tokens(message: dict[str, Any]) -> int:
    """A recorded reply's generated length in tokens, estimated from its text."""
    n = len(str(message.get("content") or "")) + len(str(message.get("reasoning_content") or ""))
    for tool_call in message.get("tool_calls") or []:
        n += len(str((tool_call.get("function") or {}).get("arguments") or ""))
    return int(n / CHARS_PER_TOKEN)


def mask_identities(text: str) -> str:
    text = HEX_ADDRESS.sub("0x#", text)
    return ID_INTEGER.sub(lambda m: "#id#" if int(m.group(0)) % 16 == 0 else m.group(0), text)


def first_difference(got: list[dict[str, Any]], want: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Where two message lists first differ (None when equal): the message index, role and the text around it."""
    for i in range(max(len(got), len(want))):
        g = got[i] if i < len(got) else None
        w = want[i] if i < len(want) else None
        if g == w:
            continue
        gs = json.dumps(g, sort_keys=True, ensure_ascii=False) if g is not None else ""
        ws = json.dumps(w, sort_keys=True, ensure_ascii=False) if w is not None else ""
        c = next((j for j in range(min(len(gs), len(ws))) if gs[j] != ws[j]), min(len(gs), len(ws)))
        return {
            "message": i,
            "role": (g or w or {}).get("role"),
            "tool_call_id": (g or w or {}).get("tool_call_id"),
            "got": gs[max(0, c - 120) : c + 240],
            "want": ws[max(0, c - 120) : c + 240],
        }
    return None


def masked_equal(got: list[dict[str, Any]], want: list[dict[str, Any]]) -> bool:
    """Equal once object addresses and id() values are masked."""
    dump = lambda m: mask_identities(json.dumps(m, sort_keys=True, ensure_ascii=False))  # noqa: E731
    return dump(got) == dump(want)


def clock_pairs(text: str) -> list[tuple]:
    """Every (elapsed, remaining) run-clock pair the text prints, as strings."""
    return [(m.group(1), m.group(2)) for m in CLOCK.finditer(text or "")]


def without_clock(text: str) -> str:
    return CLOCK.sub("", text or "")
