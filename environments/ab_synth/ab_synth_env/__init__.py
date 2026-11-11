# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic workflow tasks on the AutomationBench app simulator: single-day, multi-day and multi-turn.

Tasks run against the AutomationBench app simulator (its world state, API routes and assertion checks). They come from
the dataset, not from the benchmark. Each row turns features on through keys in its `info`:

- Single-day: `initial_state` and `assertions` only.
- Multi-day (`period_end`, `events`, `wait_tool_description`, `wait_until_description`): a `wait` tool moves a
  per-rollout clock forward and applies the dated events in time order. Successful writes are logged with the
  simulated time, so two more assertion types can grade when the agent acted: `acted_on_day` (`record`, `day`,
  optional `body_contains`: the first write whose URL names `record` happened on `day`) and `not_acted` (`record`: no
  write named it; a guardrail).
- Multi-turn (`user_card`): when the agent ends a turn without tool calls, a simulated requester answers from its
  private notes. The requester is an OpenAI-compatible chat endpoint (`requester_base_url`, `requester_model`) that
  sees only the visible conversation. It answers at most `max_user_turns` times and replies DONE when it has nothing
  to add.
- Requester-owned apps (`requester_services`): those apps leave the agent's connected services, and the requester
  operates them itself with `api_search` and `api_fetch` when the agent asks, then reports back.

The reward is the guardrail-gated share of objectives over the world and day assertions. With
`unavailable_call_break`, a call to a tool the agent was not given, or to a service not connected for it, also breaks
a guardrail (reward 0). A requester call that still fails after `requester_max_tries` ends the episode and sets the
`requester_failed` metric, so training can mask the rollout instead of learning from it.

The dataset owns all text the policy reads: the `wait` tool's text comes from each row, and the requester's replies
come from the requester model.
"""

import asyncio
import copy
import json
import logging
import math
import os
import re
import typing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState
from automationbench.tools.api.fetch import api_fetch as upstream_api_fetch


logger = logging.getLogger(__name__)

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
DAY_ASSERTIONS = {"acted_on_day", "not_acted"}
OUT_OF_SCOPE = re.compile(r'"code": 401, "message": "No (\w+) account is connected')
THINK = re.compile(r"<think>.*?(</think>|\Z)", re.S)  # an unclosed block runs to the end of the message
# A world assertion on a date or time field: upstream routes and schema defaults stamp created records with the wall
# clock (datetime.now()), not simulated time, so such an assertion would change from run to run. Refused at load.
TIME_FIELD = re.compile(r"(?:^|_)(?:date|time|datetime|timestamp|ts|at|created|updated|modified)(?:_|$)")
FIELD_KEYS = {"field", "property", "column"}
REQUESTER_RESULT_CHARS = 30000  # one requester tool result, cut: a whole sheet runs to 70k+ characters

# Instructions to the requester model only. The policy sees the requester's replies, never this text.
REQUESTER_SYSTEM = """You play the person who sent a work request to an assistant that operates business apps for you.
Your request, word for word:
{request}

Your private notes (facts you know but did not write in the request):
{card}

Reply to the assistant's latest message as that person, in one to three plain sentences.
- If it asks something your notes answer, give that answer exactly as your notes state it.
- If it asks something your notes do not cover, say you do not know and that it should use its judgment. Then add
  every one of your notes that the assistant has not mentioned yet, since the work needs them.
- Never invent facts beyond the request and the notes.
- If the assistant says the work is done, or asks nothing and needs nothing from you, reply with exactly: DONE"""
ACTING_REQUESTER_SYSTEM = """You are a person who sent a work request to an AI assistant that operates business apps for
you. You are not the assistant. Every message you receive comes from that assistant, and you answer it as the person.

Your request, word for word:
{request}

Your private notes (facts you know but did not write in the request):
{card}

These apps are yours, and only you can use them: {services}. The assistant cannot reach them. You use them yourself
through the api_search tool (find the endpoint) and the api_fetch tool (call it). What they hold (ids and names):
{records}
- When the assistant's latest message asks you to do or look up something in your apps, do it now with the tools,
  using the exact values the assistant gave (names, dates, amounts, recipients, wording). Do only what it asked. If
  it left out a value you need, do not guess: ask for it.
- Then reply as the person, in one to three plain sentences: what you did, or what you saw.
- If it asks you to do something in any other app, tell it to do that part itself.
- If it asks something your notes answer, give that answer exactly as your notes state it. If your notes do not cover
  it, say you do not know and that it should use its judgment.
- Never invent facts beyond the request, the notes and what your apps show.
- If the assistant says the work is done, or asks nothing and needs nothing from you, reply with exactly: DONE"""


def _time(value) -> datetime:
    """Naive UTC time. A time with an offset is converted to UTC first, a naive one is taken as UTC."""
    t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def _as_json_text(value) -> str:
    # The api_fetch schema declares params and body as JSON objects, upstream's fetch takes JSON strings.
    return value if isinstance(value, str) else json.dumps(value or {})


def _text(content) -> str:
    """Visible text of a message: list content joined, reasoning dropped."""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return THINK.sub("", content or "").split("</think>")[-1].strip()


def _get(msg, key):
    return msg.get(key) if isinstance(msg, dict) else getattr(msg, key, None)


def apply_patch(world: WorldState, patch: dict) -> None:
    """Apply one dated event: set fields on a record (`app`, `collection`, `id`, `set`) or add one (`add`)."""
    app = getattr(world, patch["app"])
    collection = getattr(app, patch["collection"])
    if "add" in patch:
        # The record type comes from the collection's declared schema (list[Contact] -> Contact), so a record added
        # to an empty collection is the same typed object as one loaded with the world.
        (model,) = typing.get_args(type(app).model_fields[patch["collection"]].annotation)
        collection.append(model.model_validate(patch["add"]))
        return
    record = next((r for r in collection if str(r.id) == str(patch["id"])), None)
    if record is None:
        raise ValueError(f"event patch targets no record {patch['id']!r} in {patch['app']}.{patch['collection']}")
    for key, value in patch["set"].items():
        setattr(record, key, value)


class Clock:
    """Per-rollout simulated time: the world, its pending events, and the agent's writes keyed by simulated time."""

    def __init__(self, world: WorldState, events: list[dict], end: str) -> None:
        self.world = world
        self.end = _time(end)
        self.events = sorted(events, key=lambda e: _time(e["at"]))
        self.log: list[dict] = []

    def now(self) -> datetime:
        return _time(self.world.meta.current_time.isoformat())

    def wait(self, until: str) -> str:
        """Move to `until` (capped at the end of the task's period), applying every event due by then."""
        target = min(_time(until), self.end)
        if target <= self.now():
            return json.dumps({"now": self.now().isoformat(), "advanced": False})
        while self.events and _time(self.events[0]["at"]) <= target:
            apply_patch(self.world, self.events.pop(0)["patch"])
        # Keep the seed's awareness: upstream date queries read a naive time as host-local time.
        aware = self.world.meta.current_time.tzinfo is not None
        self.world.meta.current_time = target.replace(tzinfo=timezone.utc) if aware else target
        return json.dumps({"now": target.isoformat(), "advanced": True})

    def fetch(self, method: str, url: str, params=None, body=None) -> str:
        params, body = _as_json_text(params), _as_json_text(body)
        out = upstream_api_fetch(self.world, method, url, params, body)
        if method.upper() in WRITE_METHODS and not out.lstrip().startswith('{"error"'):
            self.log.append({"at": self.now().isoformat(), "method": method.upper(), "url": url, "body": body})
        return out


def names(record: str, url: str) -> bool:
    """Whether `record` appears in the decoded URL as a whole identifier.

    Row "6" matches "Sheet!I6:J6" but not "I16" or "I60", and "FD-101" does not match "FD-1010": an edge digit may not
    touch another digit, an edge letter may not touch a letter or digit.
    """
    url = unquote(url)
    for match in re.finditer(re.escape(record), url):
        before, after = url[match.start() - 1 : match.start()], url[match.end() : match.end() + 1]
        if not (_joins(record[0], before) or _joins(record[-1], after)):
            return True
    return False


def _joins(edge: str, neighbor: str) -> bool:
    return bool(neighbor) and (neighbor.isdigit() if edge.isdigit() else edge.isalpha() and neighbor.isalnum())


def day_check(log: list[dict], assertion: dict) -> bool:
    hits = [
        e
        for e in log
        if names(assertion["record"], e["url"])
        and (not assertion.get("body_contains") or assertion["body_contains"] in e["body"])
    ]
    if assertion["type"] == "not_acted":
        return not hits
    days = sorted(e["at"][:10] for e in hits)
    return bool(days) and days[0] == assertion["day"]


def headline(initial_state: dict, world: WorldState, log: list[dict], assertions: list[dict]) -> float:
    """Guardrail-gated share of objectives over world and day assertions.

    Same split as AutomationBench's headline: an assertion with `scored: false` or `excluded: true` is skipped, one
    already true before the agent acts is a guardrail unless `excluded: false` forces it to score. Every `not_acted`
    is a guardrail. A broken guardrail scores the task 0.
    """
    initial = WorldState(**copy.deepcopy(initial_state))
    passed = total = 0
    for a in assertions:
        if a.get("scored") is False or a.get("excluded") is True:
            continue
        if a["type"] in DAY_ASSERTIONS:
            result, guardrail = day_check(log, a), a["type"] == "not_acted"
        else:
            result = AssertionRegistry.check(world, a)
            guardrail = AssertionRegistry.check(initial, a) and a.get("excluded") is not False
        if guardrail:
            if not result:
                return 0.0
            continue
        total += 1
        passed += int(result)
    return passed / total if total else 0.0


def _is_time_field(name: str) -> bool:
    return bool(TIME_FIELD.search(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()))  # closeDate -> close_date


def time_fields(assertion: dict) -> list[str]:
    """Date or time fields a world assertion names, as a key or as the `field`/`property`/`column` it checks."""
    if assertion.get("type") in DAY_ASSERTIONS:
        return []
    found = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if _is_time_field(key):
                    found.append(key)
                elif key in FIELD_KEYS and isinstance(value, str) and _is_time_field(value):
                    found.append(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(assertion)
    return found


def replay(info: dict, calls: list[dict]) -> float:
    """Score a scripted plan against a task: each call is {"wait": iso time} or {"method", "url", "params", "body"}.

    Useful to check that a task is solvable (a reference plan scores 1) before training on it.
    """
    world = WorldState(**copy.deepcopy(info["initial_state"]))
    clock = Clock(world, copy.deepcopy(info.get("events", [])), info.get("period_end") or world.meta.current_time)
    for call in calls:
        if "wait" in call:
            clock.wait(call["wait"])
        else:
            clock.fetch(call["method"], call["url"], call.get("params"), call.get("body"))
    return headline(info["initial_state"], world, clock.log, info["assertions"])


def requester_tool(world: WorldState, services: list[str], name: str, arguments: str) -> str:
    """Run one requester tool call: api_search, or api_fetch with only the requester's own services connected."""
    from automationbench.tools.api import api_search

    try:
        args = json.loads(arguments or "{}")
        if name == "api_search":
            return api_search(str(args.get("query", "")), int(args.get("top_k") or 5))
        if name != "api_fetch":
            return json.dumps({"error": f"no tool named {name}"})
        saved, world.meta.allowed_services = world.meta.allowed_services, list(services)
        try:
            return upstream_api_fetch(
                world, str(args.get("method", "GET")), str(args.get("url", "")), args.get("params"), args.get("body")
            )
        finally:
            world.meta.allowed_services = saved
    except Exception as e:  # noqa: BLE001  the requester sees the error the way the agent would
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def own_records(world: WorldState, services: list[str], budget: int = 4000) -> str:
    """The requester's index of its own apps: per collection the ids and names of up to 30 records, else a count."""

    def label(record: dict) -> str:
        return next((str(record[k]) for k in ("title", "name", "subject", "email") if record.get(k)), "")

    lines = []
    for service in services:
        for name, items in getattr(world, service).model_dump(mode="json").items():
            if not isinstance(items, list) or not items:
                continue
            if len(items) > 30 or not isinstance(items[0], dict):
                lines.append(f"{service}.{name}: {len(items)} records")
                continue
            lines.append(f"{service}.{name}: " + "; ".join(f"{x.get('id')} ({label(x)})" for x in items))
    text = "\n".join(lines) or "(nothing)"
    return text if len(text) <= budget else text[:budget] + " ..."


async def reward(state, **kwargs) -> float:
    """The headline over world and day assertions, 0 after a broken unavailable-call guardrail."""
    if state.get("violation"):
        return 0.0
    return headline(state["initial_state"], state["world"], state["clock"].log, state["info"]["assertions"])


class RequesterUnavailable(RuntimeError):
    """The requester endpoint gave no usable answer within its tries, or a dated event could not be applied.

    Either way the episode ends with `requester_failed` set, so the rollout is masked instead of scored."""


def _state_metric(key: str):
    def metric(state, **kwargs) -> float:
        return float(state.get(key, 0))

    metric.__name__ = key
    return metric


METRICS = (
    "user_turns",
    "requester_failed",
    "empty_turn",
    "unknown_tool_calls",
    "out_of_scope_calls",
    "requester_service_calls",
    "requester_tool_calls",
    "violation",
)


def load_environment(
    tasks_path: str,
    max_turns: int = 80,
    requester_base_url: str | None = None,
    requester_model: str | None = None,
    requester_body: dict | None = None,
    requester_api_key_env: str | None = None,
    requester_api_key_file: str | None = None,
    max_user_turns: int = 3,
    max_requester_tool_calls: int = 8,
    requester_max_tries: int = 4,
    requester_timeout_s: float = 300.0,
    unavailable_call_break: bool = False,
    **kwargs,
):
    """Build the environment over the Gym rows in `tasks_path` (`responses_create_params.input` and `info`).

    `requester_base_url` and `requester_model` name the OpenAI-compatible chat endpoint that plays the requester on
    multi-turn rows (`user_card` or `requester_services`). Rows without those keys never call it. `requester_body` is
    merged into every requester request after the built-in sampling values, so its keys win (for example
    `{"chat_template_kwargs": {"thinking": false}, "max_tokens": 4096}`). An API key, if the endpoint needs one, is read
    from the environment variable named by `requester_api_key_env` or from the file at `requester_api_key_file` and
    sent as a Bearer token. Each requester try ends after `requester_timeout_s`. After `requester_max_tries` failed
    tries the episode ends with `requester_failed` set.
    """
    if not (0 < requester_timeout_s < math.inf):
        raise ValueError(f"requester_timeout_s must be positive and finite, got {requester_timeout_s!r}")
    if requester_max_tries < 1:
        raise ValueError(f"requester_max_tries must be at least 1, got {requester_max_tries!r}")
    if requester_api_key_env:
        api_key = os.environ.get(requester_api_key_env)
        if not api_key:
            raise ValueError(f"requester_api_key_env names {requester_api_key_env!r}, which is not set")
    else:
        api_key = Path(requester_api_key_file).read_text().strip() if requester_api_key_file else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    import verifiers as vf
    from automationbench_env import AutomationBenchEnv
    from datasets import Dataset

    class ABSynthEnv(AutomationBenchEnv):
        async def setup_state(self, state, **setup_kwargs):
            state = await super().setup_state(state, **setup_kwargs)
            info = state["info"]
            end = info.get("period_end") or state["world"].meta.current_time.isoformat()
            state["clock"] = Clock(state["world"], copy.deepcopy(info.get("events", [])), end)
            if info.get("period_end"):
                until = {"type": "string", "description": info["wait_until_description"]}
                wait_tool = vf.Tool(
                    name="wait",
                    description=info["wait_tool_description"],
                    parameters={"type": "object", "properties": {"until": until}, "required": ["until"]},
                )
                state["tool_defs"] = [*state["tool_defs"], wait_tool]
            own = set(info.get("requester_services") or [])
            if own:
                meta = state["world"].meta
                meta.allowed_services = [s for s in meta.allowed_services or [] if s not in own]
            return state

        def _unavailable(self, state, key: str) -> None:
            state[key] = state.get(key, 0) + 1
            if unavailable_call_break:
                state["violation"] = 1

        def update_tool_args(self, tool_name, tool_args, messages, state, **update_kwargs):
            if tool_name not in {t.name for t in state["tool_defs"]}:
                self._unavailable(state, "unknown_tool_calls")
            args = super().update_tool_args(tool_name, tool_args, messages, state, **update_kwargs)
            if tool_name in ("wait", "api_fetch"):
                args.pop("world", None)
                args["clock"] = state["clock"]
            return args

        async def env_response(self, messages, state, **response_kwargs):
            reply = state.pop("pending_user_reply", None)
            if reply is not None:
                return [{"role": "user", "content": reply}]
            try:
                out = await super().env_response(messages, state, **response_kwargs)
            except vf.ToolCallError as e:
                # A dated event that cannot be applied is a dataset failure, not the policy's: end the rollout masked.
                if isinstance(e.__cause__, RequesterUnavailable):
                    self._requester_failed(state, e.__cause__)
                raise
            own = set(state["info"].get("requester_services") or [])
            for m in out:
                hit = OUT_OF_SCOPE.search(str(_get(m, "content") or ""))
                if hit:
                    self._unavailable(state, "out_of_scope_calls")
                    state["requester_service_calls"] = state.get("requester_service_calls", 0) + (hit.group(1) in own)
            return out

        def _requester_failed(self, state, error: Exception) -> None:
            state["requester_failed"] = 1
            logger.warning("ab_synth: episode ends unscored: %s", str(error)[:200])

        @vf.stop
        async def no_tools_called(self, state) -> bool:
            """The agent ended its turn: on multi-turn rows let the requester answer, and stop when it has nothing to add."""
            if not await super().no_tools_called(state):
                return False
            info = state["info"]
            if not (info.get("user_card") or info.get("requester_services")):
                return True
            if not _text(_get(state["trajectory"][-1]["completion"][-1], "content")):
                # A turn with neither text nor tool call ends the episode: a reply to it would make the next model
                # request invalid (an assistant message with no content and no tool calls).
                state["empty_turn"] = 1
                return True
            if state.get("user_turns", 0) >= max_user_turns:
                return True
            try:
                reply = await (self._acting_reply(state) if info.get("requester_services") else self._reply(state))
            except RequesterUnavailable as e:
                self._requester_failed(state, e)
                return True
            if not reply or reply.rstrip(".").upper() == "DONE":
                return True
            state["user_turns"] = state.get("user_turns", 0) + 1
            state["pending_user_reply"] = reply
            return False

        def _conversation(self, state, system: str, **fields) -> list[dict]:
            """The requester's view: its instructions, then the visible conversation (no reasoning, no tool traffic)."""
            prompt = list(state["prompt"])
            first = next(i for i, m in enumerate(prompt) if _get(m, "role") == "user")
            msgs = [
                {"role": "system", "content": system.format(request=_text(_get(prompt[first], "content")), **fields)}
            ]
            last = state["trajectory"][-1]
            for m in prompt[first + 1 :] + list(last["prompt"])[len(prompt) :] + list(last["completion"]):
                role, text = _get(m, "role"), _text(_get(m, "content"))
                if role in ("assistant", "user") and text:
                    msgs.append({"role": "user" if role == "assistant" else "assistant", "content": text})
            return msgs

        async def _chat(self, messages: list[dict], **body) -> dict:
            """One requester call through Gym's shared aiohttp client, raising RequesterUnavailable after the last try.
            This loop owns the retries: each try (request and body read) ends after `requester_timeout_s`, so a refused
            or stalled requester cannot hang the rollout. A reply cut at max_tokens or with neither text nor tool calls
            counts as a failed try."""
            from nemo_gym.client import request

            url = requester_base_url.rstrip("/") + "/chat/completions"
            payload = {"model": requester_model, "messages": messages, **body, **(requester_body or {})}

            async def attempt_once() -> dict:
                response = await request("POST", url, json=payload, headers=headers or {})
                response.raise_for_status()
                return (await response.json())["choices"][0]

            error = None
            for attempt in range(requester_max_tries):
                if attempt:
                    await asyncio.sleep(min(60, 5 * 3**attempt))
                try:
                    choice = await asyncio.wait_for(attempt_once(), requester_timeout_s)
                    message = choice["message"]
                    if choice.get("finish_reason") != "length" and (
                        _text(message.get("content")) or message.get("tool_calls")
                    ):
                        return message
                    error = f"no usable answer (finish_reason={choice.get('finish_reason')})"
                except Exception as e:  # noqa: BLE001  retried, then reported as RequesterUnavailable
                    error = f"{type(e).__name__}: {e}"
            raise RequesterUnavailable(error)

        async def _reply(self, state) -> str:
            card = state["info"]["user_card"]
            message = await self._chat(
                self._conversation(state, REQUESTER_SYSTEM, card=card), temperature=0.7, max_tokens=400
            )
            return _text(message.get("content"))

        async def _acting_reply(self, state) -> str:
            """The requester does what the agent asked in its own apps, then replies."""
            info, world = state["info"], state["world"]
            services = info["requester_services"]
            msgs = self._conversation(
                state,
                ACTING_REQUESTER_SYSTEM,
                card=info.get("user_card") or "(none)",
                services=", ".join(services),
                records=own_records(world, services),
            )
            tools = [t for t in self._all_oai_tools if t["function"]["name"] in ("api_search", "api_fetch")]
            # The budget counts tool calls, not model responses: one response may carry several calls.
            remaining = max_requester_tool_calls
            for _ in range(max_requester_tool_calls + 1):
                choice = "auto" if remaining > 0 else "none"
                message = await self._chat(msgs, temperature=0.7, max_tokens=8192, tools=tools, tool_choice=choice)
                if not message.get("tool_calls"):
                    return _text(message.get("content"))
                msgs.append(
                    {"role": "assistant", "content": message.get("content") or "", "tool_calls": message["tool_calls"]}
                )
                for call in message["tool_calls"]:
                    fn = call["function"]
                    if remaining > 0:
                        result = requester_tool(world, services, fn["name"], fn.get("arguments"))
                        remaining -= 1
                        state["requester_tool_calls"] = state.get("requester_tool_calls", 0) + 1
                    else:
                        result = json.dumps({"error": "requester tool-call limit reached; reply without tools"})
                    msgs.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": result[:REQUESTER_RESULT_CHARS]}
                    )
            raise RequesterUnavailable("the requester kept calling tools after tool_choice=none")

    def wait(until: str, clock: Clock) -> str:
        try:
            return clock.wait(until)
        except Exception as e:  # noqa: BLE001  a broken event row, reported as a requester failure and masked
            raise RequesterUnavailable(f"dated event failed: {type(e).__name__}: {e}") from e

    def api_fetch(method: str, url: str, params=None, body=None, clock: Clock = None) -> str:
        return clock.fetch(method, url, params, body)

    with open(tasks_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not (requester_base_url and requester_model) and any(
        r["info"].get("user_card") or r["info"].get("requester_services") for r in rows
    ):
        raise ValueError(
            "multi-turn rows (user_card or requester_services) need requester_base_url and requester_model"
        )
    for i, r in enumerate(rows):
        for a in r["info"].get("assertions", []):
            if fields := time_fields(a):
                raise ValueError(
                    f"row {i}: assertion {a.get('type')!r} reads date or time fields {fields}. Records the agent creates "
                    "carry the wall-clock time, not simulated time, so grade timing with acted_on_day or not_acted"
                )
    dataset = Dataset.from_list(
        [
            {
                "example_id": r.get("example_id", i),
                "prompt": r["responses_create_params"]["input"],
                "answer": "",
                "info": json.dumps(r["info"]),
            }
            for i, r in enumerate(rows)
        ]
    )
    funcs = [reward, *(_state_metric(k) for k in METRICS)]
    env = ABSynthEnv(
        dataset=dataset,
        rubric=vf.Rubric(funcs=funcs, weights=[1.0] + [0.0] * len(METRICS)),
        max_turns=max_turns,
        toolset="api",
        stop_errors=[RequesterUnavailable],
        **kwargs,
    )
    env.tool_map["api_fetch"] = api_fetch
    env.tool_map["wait"] = wait
    env.skipped_args["api_fetch"] = ["world", "clock"]
    env.skipped_args["wait"] = ["clock"]
    return env
