# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recoverable obstacles (off by default): one web call of the policy's turn fails the way the row's harness records
such failures, so that giving up on the first error costs reward (docs/design.md#recoverable-obstacles).

A share of rows is armed, by a hash of the seed and the row's input, so every rollout of a row meets the same
obstacle. An armed row fails the first web call its obstacle applies to among the turn's first web calls that its
kind can fail (page fetches for a block, searches for an empty search, either for a transient error):

- `transient`: that call fails; the same call made again later is answered as usual.
- `blocked_route`: that URL (never one the user gave) fails for the rest of the turn.
- `empty_search`: that query finds nothing for the rest of the turn; a reworded query is answered as usual.

A web call in a reply that also calls a stop tool is never the one that fails first: the turn ends with that reply, so
the policy could not react to the failure. The agent still runs it, as the harness would.

The injected result is the harness's own failure text, in the format of that tool's latest result the conversation shows
where the simulator's format varies (the sandbox's searches). Everything follows from the row, the config and the turn
so far, and calls are told apart by their position in the turn (parsers can repeat call ids across steps), so /verify
recomputes it without the session.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, NonNegativeFloat

from nemo_gym.reward_profile import MASK_SAMPLE_FIELD
from resources_servers.agentic_turn_simulation.harnesses.base import Harness, ObstacleText, ObstacleTexts
from resources_servers.agentic_turn_simulation.items import Items, content_text, parse_args


ObstacleKind = Literal["transient", "blocked_route", "empty_search"]
CallKind = Literal["fetch", "search"]
KINDS: tuple[ObstacleKind, ...] = ("transient", "blocked_route", "empty_search")
# What the simulator sees in place of an injected failure, so that it answers the rest of the turn consistently:
# without it, "stay consistent" and "pages fail sometimes" in its prompt can fail the retry too.
SIMULATOR_NOTES: dict[ObstacleKind, str] = {
    "transient": (
        "[environment: this call hit a transient outage and had no effect. The service is healthy again: the same "
        "call made now succeeds.]"
    ),
    "blocked_route": (
        "[environment: {target} is unreachable for the rest of this turn. Every request to it fails the same way "
        "(HTTP 403), from any tool, curl and wget included. Other URLs are unaffected.]"
    ),
    "empty_search": (
        "[environment: this search found nothing. The same query finds nothing again this turn; a reworded query is "
        "answered normally.]"
    ),
}


class ObstacleInjectionConfig(BaseModel):
    """`obstacle_injection` in the server config."""

    enabled: bool = False
    # Share of rows armed. A row's `verifier_metadata.obstacle` replaces the draw: false never arms it, {kind: ...}
    # arms that kind.
    row_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    # Relative weights of the kinds an armed row draws from (among those its harness and tools can fail).
    kinds: dict[ObstacleKind, NonNegativeFloat] = Field(
        default_factory=lambda: {"transient": 0.25, "blocked_route": 0.5, "empty_search": 0.25}
    )
    # The web calls that may fail: page fetches and/or searches.
    call_kinds: list[CallKind] = Field(default_factory=lambda: ["fetch", "search"])
    # Harnesses whose rows may be armed; None: every harness with obstacle texts.
    harnesses: list[str] | None = None
    # The failure hits a web call at most this far into the turn's web calls its kind can fail (0-based): 2 is one of
    # the first three.
    max_call_index: int = Field(default=2, ge=0)
    seed: int = 0


class ObstacleReport(BaseModel):
    """/verify's `obstacle` field: whether the row was armed, and what the turn did after the failure. `retried`,
    `other_route` and `ended_within_1` count only calls that ran (a stop tool's sibling calls run too) after the policy
    saw the failure, except that `other_route` also counts the sources the failed call's reply got elsewhere. A turn
    that ended before the policy replied to the failure (on time_cap, max_steps, context_overflow, or a reply cut off
    or failed) did nothing about it: `seen` is false and `ended_within_1` None."""

    armed: bool = False
    fired: bool = False
    kind: ObstacleKind | None = None
    # The failed call: its tool and its index among the turn's web calls its kind can fail.
    tool: str | None = None
    call_index: int | None = None
    # Later web calls to the failed URL or query / to another one, plus the web calls elsewhere made beside the failed
    # one in its reply that got a result (a parallel fetch of another page).
    retried: int | None = None
    other_route: int | None = None
    # The policy replied after the failed call's reply, so its turn shows what it did about the failure.
    seen: bool | None = None
    # At most one call ran after the failure (None when the policy never saw it).
    ended_within_1: bool | None = None


def normalize_url(url: str) -> str:
    """A URL as a route: the host (lowercase, without `www.` or a default port), the path without a trailing slash,
    and the query. The scheme and the fragment do not matter."""
    url = url.strip()
    try:
        parts = urlsplit(url if "://" in url else f"http://{url}")
    except ValueError:  # e.g. an unclosed IPv6 bracket
        return url
    host = parts.netloc.lower().removeprefix("www.").removesuffix(":80").removesuffix(":443")
    return host + parts.path.rstrip("/") + (f"?{parts.query}" if parts.query else "")


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def answered_calls(items: Items) -> dict[int, int]:
    """Position of each answered call -> position of its result. A result answers the earliest unanswered call with its
    id, so ids repeated across steps pair up correctly."""
    pending: dict[Any, list[int]] = defaultdict(list)
    answered: dict[int, int] = {}
    for i, item in enumerate(items):
        if item.get("type") == "function_call":
            pending[item.get("call_id")].append(i)
        elif item.get("type") == "function_call_output" and pending[item.get("call_id")]:
            answered[pending[item.get("call_id")].pop(0)] = i
    return answered


def _steps(items: Items) -> list[int]:
    """Each item's step: a step is the model's items, then the results of its calls."""
    steps, step, after_result = [], 0, False
    for item in items:
        result = item.get("type") == "function_call_output"
        if after_result and not result:
            step += 1
        after_result = result
        steps.append(step)
    return steps


def _render(text: ObstacleText, value: str, shown: str | None) -> str:
    return text if isinstance(text, str) else text(value, shown)


def _output(item: Mapping[str, Any]) -> str:
    return content_text(item.get("output"))


@dataclass(frozen=True)
class _WebCall:
    position: int
    tool: str
    texts: ObstacleTexts
    # The URL or query as given, and what it is compared by: (is a search, normalized).
    value: str
    key: tuple[bool, str]


@dataclass(frozen=True)
class _Failures:
    first: _WebCall | None
    call_index: int | None
    # Position of each failed call -> its injected result.
    outputs: dict[int, str]


@dataclass(frozen=True, kw_only=True)
class Obstacle:
    """A row's armed obstacle."""

    kind: ObstacleKind
    # The harness's web tools the row offers (all are web calls), and the call kinds that may fail.
    tools: Mapping[str, ObstacleTexts]
    call_kinds: frozenset[CallKind]
    # The row's user messages, casefolded: a URL that appears in them is never blocked.
    user_text: str
    # The latest result of each web tool in the row's input, which an injected text can follow (ObstacleText).
    input_results: Mapping[str, str]
    max_call_index: int
    # The harness's stop tools: a reply that calls one ends the turn, so its web calls never fail first.
    stop_tools: frozenset[str]

    def _shown(self, items: Items, answered: Mapping[int, int], position: int, tool: str) -> str | None:
        """The latest result of `tool` the conversation shows before `position`: in the turn, else in the input."""
        results = [r for c, r in answered.items() if r < position and items[c].get("name") == tool]
        return _output(items[max(results)]) if results else self.input_results.get(tool)

    def _web_call(self, items: Items, position: int) -> _WebCall | None:
        item = items[position]
        texts = self.tools.get(item.get("name"))
        args = parse_args(item) if texts is not None else None
        value = args.get(texts.target) if args else None
        if not isinstance(value, str) or not value.strip():
            return None
        key = normalize_query(value) if texts.search else normalize_url(value)
        return _WebCall(position, item["name"], texts, value, (texts.search, key))

    def _can_fail(self, call: _WebCall) -> bool:
        """Whether the obstacle's kind can fail this call's tool (of an allowed call kind). Only such calls count
        toward `max_call_index`, so searching first does not push a page fetch out of a block's window."""
        call_kind: CallKind = "search" if call.texts.search else "fetch"
        return call_kind in self.call_kinds and getattr(call.texts, self.kind) is not None

    def _applies(self, call: _WebCall) -> bool:
        return self.kind != "blocked_route" or call.key[1].casefold() not in self.user_text

    def _failures(self, items: Items, calls: Sequence[int]) -> _Failures:
        """The failed calls among `calls`, the positions of the calls that ran, in order: the first web call the
        obstacle applies to, if it is among the turn's web calls 0..max_call_index that its kind can fail and its reply
        calls no stop tool; then the same call again in that reply (transient: it hit the same outage), or every later
        call to that URL or query. `items` holds each reply whole, as the agent sends it, so a stop call after the
        call being answered counts."""
        steps = _steps(items)
        answered = answered_calls(items)
        # Replies that end the turn on a stop tool: the policy never sees a failure there.
        ending = {
            steps[i]
            for i, x in enumerate(items)
            if x.get("type") == "function_call" and x.get("name") in self.stop_tools
        }
        first: _WebCall | None = None
        index = eligible = 0
        outputs: dict[int, str] = {}
        for position in calls:
            call = self._web_call(items, position)
            if call is None:
                continue
            text = getattr(call.texts, self.kind)
            if first is None:
                if not self._can_fail(call) or steps[position] in ending:
                    continue
                if eligible > self.max_call_index:
                    break
                if self._applies(call):
                    first, index = call, eligible
                    outputs[position] = _render(text, call.value, self._shown(items, answered, position, call.tool))
                eligible += 1
                continue
            same_reply = steps[position] == steps[first.position]
            if call.key == first.key and text is not None and (self.kind != "transient" or same_reply):
                outputs[position] = _render(text, call.value, self._shown(items, answered, position, call.tool))
        return _Failures(first, index if first is not None else None, outputs)

    def at_call(self, turn: Items, call: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
        """For the call to answer (the last call in `turn` with its id, else one right after it): its injected result,
        or None to answer it as usual; and `turn` as the simulator sees it, each failed call's result replaced by a
        note on what happened."""
        ids = [
            i for i, x in enumerate(turn) if x.get("type") == "function_call" and x.get("call_id") == call["call_id"]
        ]
        planned = list(turn) if ids else [*turn, call]
        current = ids[-1] if ids else len(turn)
        calls = [i for i, x in enumerate(planned[: current + 1]) if x.get("type") == "function_call"]
        failures = self._failures(planned, calls)
        view = [dict(x) for x in turn]
        for call_position, result_position in answered_calls(turn).items():
            if call_position in failures.outputs:
                target = self._web_call(turn, call_position).value
                view[result_position]["output"] = SIMULATOR_NOTES[self.kind].replace("{target}", target)
        return failures.outputs.get(current), view

    def outcome(self, turn: Items, *, replied_after: bool = False) -> tuple[ObstacleReport, frozenset[int]]:
        """From a finished turn: the report, and the positions of the failed calls' results. `replied_after`: the turn
        ends on a reply of the policy's that `turn` does not show (one holding only reasoning)."""
        answered = answered_calls(turn)
        ran = sorted(answered)
        failures = self._failures(turn, ran)
        failed_results = frozenset(answered[p] for p in failures.outputs)
        first = failures.first
        if first is None:
            return ObstacleReport(armed=True, kind=self.kind), failed_results
        steps = _steps(turn)
        # A model item after the failed call's reply is the policy's next reply: it saw the failure.
        seen = replied_after or steps[-1] > steps[first.position]
        later = [p for p in ran if steps[p] > steps[first.position]]
        web = [c for c in (self._web_call(turn, p) for p in later) if c is not None]
        retried = sum(c.key == first.key for c in web)
        # Calls beside the failed one in its reply ran before the policy saw the failure: a web call elsewhere that got
        # a result is a source the turn had; the same call again is neither a retry nor another source.
        beside = [self._web_call(turn, p) for p in ran if steps[p] == steps[first.position] and p != first.position]
        sources = sum(
            c is not None and c.key != first.key and not c.texts.failed(_output(turn[answered[c.position]]))
            for c in beside
        )
        report = ObstacleReport(
            armed=True,
            fired=True,
            kind=self.kind,
            tool=first.tool,
            call_index=failures.call_index,
            retried=retried,
            other_route=len(web) - retried + sources,
            seen=seen,
            ended_within_1=len(later) <= 1 if seen else None,
        )
        return report, failed_results


def _draws(seed: int, input_items: Items) -> tuple[float, float]:
    """Two uniform draws in [0, 1), fixed by the seed and the row's input."""
    row = hashlib.sha256(json.dumps(list(input_items), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    digest = hashlib.sha256(f"{seed}:{row}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64, int.from_bytes(digest[8:16], "big") / 2**64


def arm(
    config: ObstacleInjectionConfig,
    *,
    harness: Harness,
    tool_names: frozenset[str],
    input_items: Items,
    metadata: Mapping[str, Any],
) -> Obstacle | None:
    """The row's obstacle, or None: injection is off, the harness is not listed or has no texts for the row's tools,
    or the row is not drawn (or its `verifier_metadata.obstacle` is false, or names a kind its tools cannot fail)."""
    if not config.enabled or (config.harnesses is not None and harness.name not in config.harnesses):
        return None
    tools = {name: texts for name, texts in harness.obstacle_texts.items() if name in tool_names}
    allowed = [t for t in tools.values() if ("search" if t.search else "fetch") in config.call_kinds]
    available = [k for k in KINDS if any(getattr(t, k) is not None for t in allowed)]
    override = metadata.get("obstacle")
    if override is False:
        return None
    if override is not None:
        kind = override.get("kind") if isinstance(override, Mapping) else None
        if kind not in KINDS:
            raise ValueError(f"verifier_metadata.obstacle must be false or {{kind: one of {KINDS}}}, got {override!r}")
    else:
        armed, pick = _draws(config.seed, input_items)
        weights = [(k, config.kinds.get(k, 0.0)) for k in available if config.kinds.get(k, 0.0) > 0]
        if armed >= config.row_rate or not weights:
            return None
        point = pick * sum(w for _, w in weights)
        kind = weights[-1][0]
        for candidate, weight in weights:
            if point < weight:
                kind = candidate
                break
            point -= weight
    if kind not in available:
        return None
    user_text = "\n".join(content_text(x.get("content")) for x in input_items if x.get("role") == "user")
    prior = harness.to_tool_items(input_items)
    input_results: dict[str, str] = {}
    for call, result in sorted(answered_calls(prior).items(), key=lambda pair: pair[1]):
        if prior[call].get("name") in tools:
            input_results[prior[call]["name"]] = _output(prior[result])
    return Obstacle(
        kind=kind,
        tools=tools,
        call_kinds=frozenset(config.call_kinds),
        user_text=user_text.casefold(),
        input_results=input_results,
        max_call_index=config.max_call_index,
        stop_tools=frozenset(harness.stop_tools),
    )


def obstacle_metrics(rollouts: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Over the rollouts that report an obstacle (injection on), masked ones included: the share armed, the share of
    those whose turn met the failure, and of the turns whose policy saw it (replied after it), the share that retried
    it, the share that quit (no retry and no other source: no web call after the failure and none beside it that got a
    result; and at most one call of any kind after it) and the mean reward of those not masked. A turn that ended
    before the policy's next reply did nothing about the failure and counts in none of those. Persisting turns hit the
    wall-clock cap (masked) more often than quitting ones, and Gym's metrics see measured rollouts only, so take the
    rates from the rollouts file."""
    reported = [r for r in rollouts if isinstance(r.get("obstacle"), Mapping)]
    if not reported:
        return {}
    armed = [r for r in reported if r["obstacle"].get("armed")]
    fired = [r for r in armed if r["obstacle"].get("fired")]
    metrics = {"obstacle_armed_rate": len(armed) / len(reported)}
    if armed:
        metrics["obstacle_fired_rate"] = len(fired) / len(armed)
    seen = [r for r in fired if r["obstacle"].get("seen")]
    if seen:
        after = [r["obstacle"] for r in seen]
        quit_ = [o for o in after if o.get("ended_within_1") and not o.get("retried") and not o.get("other_route")]
        metrics["obstacle_retried_rate"] = sum(bool(o.get("retried")) for o in after) / len(seen)
        metrics["obstacle_quit_rate"] = len(quit_) / len(seen)
    scored = [r for r in seen if not r.get(MASK_SAMPLE_FIELD)]
    if scored:
        metrics["obstacle_turn_score"] = sum(r.get("reward") == 1.0 for r in scored) / len(scored)
    return metrics
