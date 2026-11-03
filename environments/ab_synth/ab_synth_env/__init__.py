# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-day workflow environment with simulated time.

Tasks run against the AutomationBench app simulator (its world state, API routes and assertion checks) with a clock
the agent controls. A `wait` tool moves the clock forward and applies the task's dated events in time order. Every
successful state-changing `api_fetch` call is logged with the simulated time, so a task can grade when the agent
acted, not only what the world holds at the end. Two assertion types use that log:

- `acted_on_day` (`record`, `day`, optional `body_contains`): the first write whose URL names `record` happened on
  `day` (YYYY-MM-DD).
- `not_acted` (`record`): no write named `record`. It is a guardrail.

The reward is the guardrail-gated share of objectives over the world assertions plus these day assertions. Tasks
come from the dataset, not from the benchmark. The `wait` tool's description and the description of its `until`
parameter come from each task (`info.wait_tool_description`, `info.wait_until_description`), so the dataset owns all
text the policy reads.
"""

import copy
import json
import re
from datetime import datetime, timezone
from urllib.parse import unquote

from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState
from automationbench.tools.api.fetch import api_fetch as upstream_api_fetch


WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
DAY_ASSERTIONS = {"acted_on_day", "not_acted"}


def _time(value) -> datetime:
    """Naive UTC time. A time with an offset is converted to UTC first, a naive one is taken as UTC."""
    t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def _as_json_text(value) -> str:
    # The api_fetch schema declares params and body as JSON objects, upstream's fetch takes JSON strings.
    return value if isinstance(value, str) else json.dumps(value or {})


def apply_patch(world: WorldState, patch: dict) -> None:
    """Apply one dated event: set fields on a record (`app`, `collection`, `id`, `set`) or add one (`add`)."""
    collection = getattr(getattr(world, patch["app"]), patch["collection"])
    if "add" in patch:
        model = type(collection[0]) if collection else None
        collection.append(model(**patch["add"]) if model else patch["add"])
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
        self.world.meta.current_time = target
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

    A world assertion already true before the agent acts is a guardrail, and so is every `not_acted`. A broken
    guardrail scores the task 0.
    """
    initial = WorldState(**copy.deepcopy(initial_state))
    passed = total = 0
    for a in assertions:
        if a["type"] in DAY_ASSERTIONS:
            result, guardrail = day_check(log, a), a["type"] == "not_acted"
        else:
            result, guardrail = AssertionRegistry.check(world, a), AssertionRegistry.check(initial, a)
        if guardrail:
            if not result:
                return 0.0
            continue
        total += 1
        passed += int(result)
    return passed / total if total else 0.0


def replay(info: dict, calls: list[dict]) -> float:
    """Score a scripted plan against a task: each call is {"wait": iso time} or {"method", "url", "params", "body"}.

    Useful to check that a task is solvable (a reference plan scores 1) before training on it.
    """
    world = WorldState(**copy.deepcopy(info["initial_state"]))
    clock = Clock(world, copy.deepcopy(info.get("events", [])), info["period_end"])
    for call in calls:
        if "wait" in call:
            clock.wait(call["wait"])
        else:
            clock.fetch(call["method"], call["url"], call.get("params"), call.get("body"))
    return headline(info["initial_state"], world, clock.log, info["assertions"])


def load_environment(tasks_path: str, max_turns: int = 80, **kwargs):
    """Build the environment over the Gym rows in `tasks_path` (`responses_create_params.input` and `info`)."""
    import verifiers as vf
    from automationbench_env import AutomationBenchEnv
    from datasets import Dataset

    class TimeEnv(AutomationBenchEnv):
        async def setup_state(self, state, **setup_kwargs):
            state = await super().setup_state(state, **setup_kwargs)
            info = state["info"]
            state["clock"] = Clock(state["world"], copy.deepcopy(info.get("events", [])), info["period_end"])
            until = {"type": "string", "description": info["wait_until_description"]}
            wait_tool = vf.Tool(
                name="wait",
                description=info["wait_tool_description"],
                parameters={"type": "object", "properties": {"until": until}, "required": ["until"]},
            )
            state["tool_defs"] = [*state["tool_defs"], wait_tool]
            return state

        def update_tool_args(self, tool_name, tool_args, messages, state, **update_kwargs):
            args = super().update_tool_args(tool_name, tool_args, messages, state, **update_kwargs)
            if tool_name in ("wait", "api_fetch"):
                args.pop("world", None)
                args["clock"] = state["clock"]
            return args

    def wait(until: str, clock: Clock) -> str:
        return clock.wait(until)

    def api_fetch(method: str, url: str, params=None, body=None, clock: Clock = None) -> str:
        return clock.fetch(method, url, params, body)

    async def day_headline(state, **score_kwargs) -> float:
        info = state["info"]
        return headline(state["initial_state"], state["world"], state["clock"].log, info["assertions"])

    with open(tasks_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
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
    env = TimeEnv(
        dataset=dataset,
        rubric=vf.Rubric(funcs=[day_headline], weights=[1.0]),
        max_turns=max_turns,
        toolset="api",
        **kwargs,
    )
    env.tool_map["api_fetch"] = api_fetch
    env.tool_map["wait"] = wait
    env.skipped_args["api_fetch"] = ["world", "clock"]
    env.skipped_args["wait"] = ["clock"]
    return env
