# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Capability sweep: profile how the environment attributes reward across policies.

Environment sanity check, run before any training: drive seeded episodes over
the real HTTP surface for scripted anchor policies with a known partial
order (relief beats noop and random-valid; both beat catastrophic play) and,
optionally, ranked OpenAI-compatible chat-completions models. The scripted
anchor check is a deterministic correctness gate. The model ladder is an empirical
gate: if a declared frontier model does not beat a smaller model on complete
paired profiles, investigate the prompt, task coverage, and reward before
training rather than assuming either the environment or model is correct.

Anchors need no model server or API key; the sweep runs fully offline on the
replay backend. LLM policies are described in a JSON file (see --models):

    [{"label": "frontier", "model": "<frontier-model>", "base_url": "https://api.example/v1",
      "api_key_env": "FRONTIER_API_KEY", "temperature": 0.2, "capability_rank": 2}]

Each model receives every task row's own messages (system prompt + task
prompt), the current rendered observation as the latest user message, and the
row's tool schemas converted to chat-completions format. Each decision is
single-turn on
purpose: each step stands alone, so models are compared on state-reading, not
context management. A reply without a parseable tool call is counted as a
parse failure and sent to the environment as a terminal protocol violation; a
parseable call the environment rejects as an unknown tool is counted as
invalid; and a dead endpoint drops the episode as an infrastructure error.
Dropped episodes are drained server-side, so one model's failures never starve
the pool for the next. Unparseable replies are never upgraded to valid noops.

Usage:
    python resources_servers/openair_congestion/model_sweep.py
    python resources_servers/openair_congestion/model_sweep.py --models sweep_models.json --out sweep.json
    python resources_servers/openair_congestion/model_sweep.py --compliance-profile --models sweep_models.json
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
import numpy as np
from openair_congestion.schemas import SUPPORTED_REGIMES

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.openair_congestion.app import _strict_json_object
from resources_servers.openair_congestion.client import (
    _start_local_server,
    _tool_response,
    choose_action,
)


_EXAMPLE_JSONL = Path(__file__).parent / "data" / "example.jsonl"
_SCHEDULERS = ("PF", "RR", "MaxCI")


def _load_example_rows(path: Path = _EXAMPLE_JSONL) -> list[dict[str, Any]]:
    """Load every checked-in task row; never silently profile only row one."""

    rows = [_strict_json_object(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 5:
        raise ValueError(f"reward profiling requires at least five example rows, got {len(rows)} from {path}")
    for row in rows:
        row.pop("agent_ref", None)
    return rows


def _validate_compliance_rows(rows: list[dict[str, Any]]) -> None:
    """Require one correctly labeled, one-hot task for every supported regime."""

    expected = set(SUPPORTED_REGIMES)
    counts = Counter(row.get("scenario_id") for row in rows)
    if set(counts) != expected or any(counts[regime] != 1 for regime in expected):
        raise ValueError(
            "compliance-profile example rows must cover every supported scenario_id exactly once; "
            f"expected {sorted(expected)}, got {dict(counts)}"
        )
    for index, row in enumerate(rows):
        scenario_id = row["scenario_id"]
        regime_mix = row.get("regime_mix")
        weight = regime_mix.get(scenario_id) if isinstance(regime_mix, dict) else None
        if (
            not isinstance(regime_mix, dict)
            or set(regime_mix) != {scenario_id}
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) != 1.0
        ):
            raise ValueError(
                f"compliance-profile row {index} ({scenario_id!r}) must have a one-hot regime_mix "
                f"of {{{scenario_id!r}: 1.0}}, got {regime_mix!r}"
            )


def _profile_task_rows(base_rows: list[dict[str, Any]], task_count: int) -> list[dict[str, Any]]:
    """Expand representative rows into deterministic prompts across all regimes."""

    if not base_rows:
        raise ValueError("at least one base task row is required")
    if task_count < 1:
        raise ValueError(f"task_count must be >= 1, got {task_count}")

    rows: list[dict[str, Any]] = []
    for task_index in range(task_count):
        base_index = task_index % len(base_rows)
        cycle = task_index // len(base_rows)
        row = json.loads(json.dumps(base_rows[base_index]))
        row["seed"] = int(row.get("seed", base_index)) + cycle * 1_000_000
        row["_profile_prompt_index"] = task_index
        rows.append(row)
    return rows


def _repeat_task_rows(prompt_rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    """Create repeated responses without changing the prompt/environment seed."""

    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompt_rows):
        for response_index in range(repeats):
            row = json.loads(json.dumps(prompt))
            row["_profile_prompt_index"] = prompt_index
            row["_profile_response_index"] = response_index
            rows.append(row)
    return rows


# --- Policies ----------------------------------------------------------------
# A policy maps (rendered observation, step_idx, rng) -> tool call dict, plus
# an async LLM variant. Anchors mirror the reward-oracle test policies but
# read the rendered text, exactly like an LLM would.


def _noop(observation: str, step_idx: int, rng: random.Random) -> dict[str, Any]:
    return {"name": "noop", "arguments": {}}


def _catastrophic(observation: str, step_idx: int, rng: random.Random) -> dict[str, Any]:
    # max_prb=0 starves the target and is guardrail-rejected every step.
    return {"name": "set_prb_cap", "arguments": {"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 0}}


_CELL_HEADER = re.compile(r"- Cell (\d+):")
_UE_LINE = re.compile(r"UE (\d+) \(")


def _parse_topology(observation: str) -> dict[int, list[int]]:
    # cell_id -> UE ids, read from the rendered per-cell / per-UE lines --
    # the same text an LLM policy sees.
    cells: dict[int, list[int]] = {}
    current: int | None = None
    for line in (observation or "").splitlines():
        header = _CELL_HEADER.search(line)
        if header:
            current = int(header.group(1))
            cells[current] = []
        elif current is not None:
            ue = _UE_LINE.search(line)
            if ue:
                cells[current].append(int(ue.group(1)))
    return cells or {0: [0]}


def _make_random_valid() -> Callable[[str, int, random.Random], dict[str, Any]]:
    # Mirrors the reward-oracle test policy (tests/test_reward_correctness.py):
    # uniform over the five guardrail-valid tool families with the same argument
    # ranges.  "valid" means accepted by the guardrail, not beneficial: under
    # the persistent synthetic setpoint dynamics this deliberately unguided policy
    # is expected to score below standing pat on the fixed ladder tasks.
    # deduplicated against the last two actions so the identical-action rate
    # limit never fires -- but reading the rendered text, like an LLM would.
    recent: list[str] = []

    def _sample(observation: str, rng: random.Random) -> dict[str, Any]:
        cells = _parse_topology(observation)
        cell_id = rng.choice(sorted(cells))
        choice = rng.randrange(5)
        if choice == 0:
            return {
                "name": "set_scheduler_policy",
                "arguments": {"cell_id": cell_id, "policy": rng.choice(_SCHEDULERS)},
            }
        if choice == 1:
            ue_id = rng.choice(cells[cell_id] or [0])
            return {
                "name": "set_prb_cap",
                "arguments": {
                    "cell_id": cell_id,
                    "target": "ue",
                    "target_id": ue_id,
                    "max_prb": rng.randrange(10, 273),
                },
            }
        if choice == 2:
            return {
                "name": "set_mcs_bounds",
                "arguments": {"cell_id": cell_id, "mcs_min": 0, "mcs_max": rng.randrange(5, 28), "target_bler": 0.1},
            }
        if choice == 3:
            return {
                "name": "set_admission_policy",
                "arguments": {
                    "cell_id": cell_id,
                    "accept_threshold_pct": rng.randrange(10, 100),
                    "slice_reservation": {},
                },
            }
        return {
            "name": "set_ul_power_control",
            "arguments": {
                "cell_id": cell_id,
                "p0_dbm": rng.randrange(-120, 20),
                "alpha": rng.choice([0.4, 0.7, 0.8, 1.0]),
            },
        }

    def policy(observation: str, step_idx: int, rng: random.Random) -> dict[str, Any]:
        action = _sample(observation, rng)
        for _ in range(20):
            key = json.dumps(action, sort_keys=True)
            if key not in recent:
                break
            action = _sample(observation, rng)
        recent.append(json.dumps(action, sort_keys=True))
        del recent[:-2]
        return action

    return policy


# Factories: random-valid keeps per-episode dedupe state, so every policy is
# constructed fresh per episode.
_ANCHORS: dict[str, Callable[[], Callable[[str, int, random.Random], dict[str, Any]]]] = {
    "anchor:relief": lambda: (lambda obs, i, rng: choose_action(obs, i)),
    "anchor:random-valid": _make_random_valid,
    "anchor:noop": lambda: _noop,
    "anchor:catastrophic": lambda: _catastrophic,
}
# Display order for the four scripted anchors.
_ANCHOR_ORDER = ("anchor:relief", "anchor:noop", "anchor:random-valid", "anchor:catastrophic")
# This is intentionally a partial order. In a genuinely congested
# environment, random guardrail-valid control can occasionally help by
# accident, so requiring noop > random-valid would reward an inert simulator.
# Intentional relief must beat both; catastrophic rejected play must lose to
# both.
_ANCHOR_CONSTRAINTS = (
    ("anchor:relief", "anchor:noop"),
    ("anchor:relief", "anchor:random-valid"),
    ("anchor:noop", "anchor:catastrophic"),
    ("anchor:random-valid", "anchor:catastrophic"),
)


@dataclass
class ModelSpec:
    label: str
    model: str
    base_url: str
    api_key_env: str = ""
    temperature: float = 0.2
    max_tokens: int = 512
    # Increasing integers encode the expected capability order. When two or
    # more models are configured, every model must have a unique rank.
    capability_rank: int | None = None


def _require_compliance_models(specs: list[ModelSpec]) -> None:
    """Prevent the named compliance profile from passing on anchors alone."""

    if len(specs) < 2:
        raise ValueError("--compliance-profile requires at least two real models")
    if any(not spec.label.strip() or not spec.model.strip() or not spec.base_url.strip() for spec in specs):
        raise ValueError("--compliance-profile requires non-empty label, model, and base_url values")
    identities = [spec.model.strip() for spec in specs]
    if len(identities) != len(set(identities)):
        raise ValueError("--compliance-profile requires distinct model identities")
    if any(spec.capability_rank is None for spec in specs):
        raise ValueError("--compliance-profile requires capability_rank for every model")
    ranks = [int(spec.capability_rank) for spec in specs if spec.capability_rank is not None]
    if len(ranks) != len(set(ranks)):
        raise ValueError("--compliance-profile requires unique capability_rank values")


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    """Population Pearson correlation, or None when either side is constant."""

    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denominator


def _paired_bootstrap_ci(
    deltas: list[float],
    *,
    seed: int = 0,
    draws: int = 10_000,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap interval for paired mean deltas."""

    if not deltas:
        raise ValueError("at least one paired delta is required")
    if draws < 1:
        raise ValueError("draws must be positive")
    values = np.asarray(deltas, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("paired deltas must be finite")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    # Keep peak index memory bounded for the 8,000-episode compliance gate.
    chunk_size = max(1, min(256, 2_000_000 // len(values)))
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        indices = rng.integers(
            0,
            len(values),
            size=(count, len(values)),
        )
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


@dataclass
class PolicyStats:
    returns: list[float] = field(default_factory=list)
    steps: int = 0
    rejected: int = 0
    noop_steps: int = 0
    invalid_calls: int = 0
    parse_failures: int = 0
    infra_errors: int = 0
    tool_rewards: dict[str, list[float]] = field(default_factory=dict)
    tool_rejections: Counter[str] = field(default_factory=Counter)
    episode_records: list[dict[str, Any]] = field(default_factory=list)

    def record_step(self, tool_name: str, reward: float, *, rejected: bool) -> None:
        self.steps += 1
        self.tool_rewards.setdefault(tool_name, []).append(reward)
        if rejected:
            self.rejected += 1
            self.tool_rejections[tool_name] += 1
        if tool_name == "noop":
            self.noop_steps += 1

    def finish_episode(
        self,
        episode_return: float,
        *,
        scenario_id: str,
        tool_counts: Counter[str],
        rejected_steps: int,
        episode_steps: int,
        pair_key: str,
        parse_failures: int,
        invalid_calls: int,
    ) -> None:
        usable = parse_failures == 0 and invalid_calls == 0
        self.returns.append(episode_return)
        self.episode_records.append(
            {
                "return": episode_return,
                "pair_key": pair_key,
                "usable": usable,
                "parse_failures": parse_failures,
                "invalid_calls": invalid_calls,
                "scenario_id": scenario_id,
                "tool_counts": dict(tool_counts),
                "rejection_rate": rejected_steps / episode_steps if episode_steps else 0.0,
                "noop_rate": tool_counts.get("noop", 0) / episode_steps if episode_steps else 0.0,
                "steps": episode_steps,
            }
        )

    def row(self, label: str) -> dict[str, Any]:
        std = statistics.pstdev(self.returns) if len(self.returns) > 1 else 0.0
        distribution = {
            "min": min(self.returns) if self.returns else None,
            "p05": _quantile(self.returns, 0.05),
            "p25": _quantile(self.returns, 0.25),
            "median": _quantile(self.returns, 0.50),
            "p75": _quantile(self.returns, 0.75),
            "p95": _quantile(self.returns, 0.95),
            "max": max(self.returns) if self.returns else None,
        }
        tool_metrics = {}
        for tool_name, rewards in sorted(self.tool_rewards.items()):
            calls = len(rewards)
            tool_metrics[tool_name] = {
                "calls": calls,
                "call_rate": round(calls / self.steps, 6) if self.steps else 0.0,
                "mean_step_reward": round(statistics.fmean(rewards), 6),
                "rejection_rate": round(self.tool_rejections[tool_name] / calls, 6),
            }

        episode_returns = [record["return"] for record in self.episode_records]
        tools = sorted({tool for record in self.episode_records for tool in record["tool_counts"]})
        correlations: dict[str, float | None] = {
            "rejection_rate": _correlation(
                [record["rejection_rate"] for record in self.episode_records], episode_returns
            ),
            "noop_rate": _correlation([record["noop_rate"] for record in self.episode_records], episode_returns),
        }
        for tool_name in tools:
            correlations[f"tool_rate:{tool_name}"] = _correlation(
                [
                    record["tool_counts"].get(tool_name, 0) / record["steps"] if record["steps"] else 0.0
                    for record in self.episode_records
                ],
                episode_returns,
            )

        returns_by_scenario = {}
        scenarios = sorted({record["scenario_id"] for record in self.episode_records})
        for scenario_id in scenarios:
            values = [record["return"] for record in self.episode_records if record["scenario_id"] == scenario_id]
            returns_by_scenario[scenario_id] = {
                "episodes": len(values),
                "mean_return": round(statistics.fmean(values), 6),
                "std_return": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
            }

        return {
            "policy": label,
            "episodes": len(self.returns),
            "mean_return": round(statistics.fmean(self.returns), 4) if self.returns else None,
            "std_return": round(std, 4),
            "return_distribution": {
                key: round(value, 6) if value is not None else None for key, value in distribution.items()
            },
            "rejection_rate": round(self.rejected / self.steps, 4) if self.steps else 0.0,
            "noop_rate": round(self.noop_steps / self.steps, 4) if self.steps else 0.0,
            "invalid_calls": self.invalid_calls,
            "parse_failures": self.parse_failures,
            "infra_errors": self.infra_errors,
            "usable_episodes": sum(bool(record["usable"]) for record in self.episode_records),
            "episode_records": list(self.episode_records),
            "tool_metrics": tool_metrics,
            "episode_return_correlations": {
                key: round(value, 6) if value is not None else None for key, value in correlations.items()
            },
            "returns_by_scenario": returns_by_scenario,
        }


def _chat_tools(row: dict) -> list[dict]:
    # Responses-API tool schemas -> chat-completions tool schemas.
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in row["responses_create_params"].get("tools", [])
        if t.get("type") == "function"
    ]


def _parse_tool_call(message: dict) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    if calls:
        # Preserve the environment's exactly-one-call contract. Selecting the
        # first item would silently turn a multi-call protocol violation into
        # a valid action and inflate that model's compliance profile.
        if len(calls) != 1:
            return None
        fn = calls[0].get("function", {})
        try:
            arguments = _strict_json_object(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            return None
        if not isinstance(fn.get("name"), str) or not isinstance(arguments, dict):
            return None
        return {"name": fn["name"], "arguments": arguments}
    # Fallback: a bare JSON object {"name": ..., "arguments": {...}} in content.
    content = message.get("content") or ""
    start, end = content.find("{"), content.rfind("}")
    if 0 <= start < end:
        try:
            obj = _strict_json_object(content[start : end + 1])
        except (TypeError, ValueError):
            return None
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments", {}), dict):
            return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
    return None


async def _llm_action(
    session: aiohttp.ClientSession, spec: ModelSpec, row: dict, observation: str, step_idx: int
) -> dict[str, Any] | None:
    # The row's own input messages (system prompt + task prompt), then the
    # current rendered observation. Single-turn on purpose: each step stands
    # alone, so models are compared on state-reading, not context management.
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in row["responses_create_params"]["input"]
        if m.get("role") in ("system", "user")
    ]
    messages.append({"role": "user", "content": observation})
    headers = {}
    if spec.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[spec.api_key_env]}"
    payload = {
        "model": spec.model,
        "messages": messages,
        "tools": _chat_tools(row),
        # The environment contract requires exactly one tool call every turn.
        # Enforce that protocol at the model endpoint as well as validating the
        # returned call; otherwise an API may legitimately emit prose under
        # its default "auto" policy and turn endpoint defaults into apparent
        # model-capability failures.
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    async with session.post(
        f"{spec.base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=90),
    ) as response:
        response.raise_for_status()
        try:
            body = await response.json()
        except (json.JSONDecodeError, aiohttp.ContentTypeError):
            return None  # 200 with a non-JSON body: a parse failure, not a crash
    # Tolerate non-conforming 200 bodies ({"error": ...}, empty choices, null
    # message) the same way as unparseable content: infra_errors stays
    # reserved for transport-level failures.
    choices = body.get("choices") or [] if isinstance(body, dict) else []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
    return _parse_tool_call(message) if isinstance(message, dict) else None


# --- Episode driver ----------------------------------------------------------


def _unparseable_response(step_idx: int) -> dict[str, Any]:
    """Build a model response with no tool call so the server applies its protocol penalty."""

    return NeMoGymResponse(
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{step_idx}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text="Model response did not contain a parseable tool call.",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        id="r",
        created_at=0.0,
        model="capability-sweep",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    ).model_dump()


async def _run_episode(
    base_url: str,
    row: dict,
    action_fn: Callable[[str, int], Awaitable[dict[str, Any] | None]],
    stats: PolicyStats,
) -> None:
    async with aiohttp.ClientSession(base_url=base_url, cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:

        async def post(url_path: str, payload: dict) -> dict:
            async with session.post(url_path, json=payload) as response:
                response.raise_for_status()
                return await response.json()

        async def step_env(action: dict[str, Any] | None, step_idx: int) -> dict:
            response = (
                _unparseable_response(step_idx)
                if action is None
                else _tool_response(action["name"], action["arguments"], step_idx)
            )
            return await post(
                "/step",
                {
                    "responses_create_params": row["responses_create_params"],
                    "response": response,
                },
            )

        reset = await post("/reset", row)
        observation = reset["observation"]
        episode_return, step_idx = 0.0, 0
        terminated = truncated = False
        tool_counts: Counter[str] = Counter()
        episode_rejected = 0
        episode_parse_failures = 0
        episode_invalid_calls = 0
        # The env terminates at the row's max_steps; the 4x margin only guards
        # against a served instance that never sets terminated/truncated.
        step_cap = 4 * int(row.get("max_steps", 16))
        try:
            while not (terminated or truncated) and step_idx < step_cap:
                action = await action_fn(observation, step_idx)
                if action is None:
                    stats.parse_failures += 1
                    episode_parse_failures += 1
                step = await step_env(action, step_idx)
                reward = float(step["reward"])
                episode_return += reward
                tool_name = action["name"] if action is not None else "<parse_failure>"
                rejected = not step["info"].get("guardrail_accepted", True)
                stats.record_step(tool_name, reward, rejected=rejected)
                tool_counts[tool_name] += 1
                episode_rejected += int(rejected)
                if step["info"].get("error") == "invalid_tool_call":
                    stats.invalid_calls += 1
                    episode_invalid_calls += 1
                observation = step["observation"]
                terminated, truncated = bool(step["terminated"]), bool(step["truncated"])
                step_idx += 1
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # The episode is already reset server-side; abandoning it here
            # would leak its replay-pool slot for the rest of the sweep (the
            # reaper only reclaims sessions the server no longer reports as
            # live). Drain to termination with noops, then let the caller
            # count the drop.
            while not (terminated or truncated) and step_idx < step_cap:
                step = await step_env({"name": "noop", "arguments": {}}, step_idx)
                terminated, truncated = bool(step["terminated"]), bool(step["truncated"])
                step_idx += 1
            raise
        stats.finish_episode(
            episode_return,
            scenario_id=str(row.get("scenario_id") or "unspecified"),
            tool_counts=tool_counts,
            rejected_steps=episode_rejected,
            episode_steps=sum(tool_counts.values()),
            pair_key=(f"{row.get('_profile_prompt_index', row.get('seed'))}:{row.get('_profile_response_index', 0)}"),
            parse_failures=episode_parse_failures,
            invalid_calls=episode_invalid_calls,
        )


async def _scripted_action(
    policy: Callable[[str, int, random.Random], dict[str, Any]],
    rng: random.Random,
    observation: str,
    step_idx: int,
) -> dict[str, Any]:
    return policy(observation, step_idx, rng)


def _evaluate_model_ordering(
    profile: list[dict[str, Any]],
    specs: list[ModelSpec],
    *,
    expected_episodes: int,
    compliance: bool = False,
    failure_rate_ceiling: float = 0.0,
) -> dict[str, Any]:
    """Evaluate usable, paired small-to-frontier return improvements."""

    if not math.isfinite(failure_rate_ceiling) or not 0.0 <= failure_rate_ceiling <= 1.0:
        raise ValueError(f"failure_rate_ceiling must be a finite number between 0 and 1, got {failure_rate_ceiling!r}")
    effective_failure_ceiling = 0.0 if compliance else failure_rate_ceiling

    if len(specs) < 2:
        return {"status": "NOT_CONFIGURED", "expected": [], "observed": {}, "reason": "fewer than two models"}
    if any(spec.capability_rank is None for spec in specs):
        return {
            "status": "NOT_EVALUABLE",
            "expected": [],
            "observed": {},
            "reason": "every model needs capability_rank",
        }

    ordered_specs = sorted(specs, key=lambda spec: int(spec.capability_rank or 0))
    expected = [f"model:{spec.label}" for spec in ordered_specs]
    by_policy = {row["policy"]: row for row in profile}
    if any(label not in by_policy for label in expected):
        return {"status": "NOT_EVALUABLE", "expected": expected, "observed": {}, "reason": "missing model row"}

    observed = {label: by_policy[label]["mean_return"] for label in expected}
    complete = all(
        by_policy[label]["episodes"] + by_policy[label]["infra_errors"] == expected_episodes
        and by_policy[label]["mean_return"] is not None
        for label in expected
    )
    if not complete:
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "reason": "one or more model profiles are incomplete",
        }

    failure_counts = {
        label: {
            "parse_failures": int(by_policy[label].get("parse_failures", 0)),
            "invalid_calls": int(by_policy[label].get("invalid_calls", 0)),
            "infra_errors": int(by_policy[label].get("infra_errors", 0)),
        }
        for label in expected
    }
    failure_rates = {
        label: min(1.0, sum(counts.values()) / expected_episodes) for label, counts in failure_counts.items()
    }
    if any(rate > effective_failure_ceiling for rate in failure_rates.values()):
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "failure_rate_ceiling": effective_failure_ceiling,
            "reason": (
                "one or more model profiles have parse, invalid-call, or "
                "infrastructure failures above the configured "
                f"failure-rate ceiling ({effective_failure_ceiling:.3f})"
            ),
        }

    paired_returns: dict[str, dict[str, float]] = {}
    for label in expected:
        records = by_policy[label].get("episode_records") or []
        if len(records) != by_policy[label]["episodes"]:
            return {
                "status": "NOT_EVALUABLE",
                "expected": expected,
                "observed": observed,
                "reason": f"{label} lacks complete paired episode records",
            }
        all_keys = [str(record.get("pair_key")) for record in records]
        if len(set(all_keys)) != len(all_keys) or "None" in all_keys:
            return {
                "status": "NOT_EVALUABLE",
                "expected": expected,
                "observed": observed,
                "reason": f"{label} has missing or duplicate pair keys",
            }
        paired_returns[label] = {
            str(record["pair_key"]): float(record["return"]) for record in records if record.get("usable", False)
        }

    reference_keys = set.intersection(*(set(paired_returns[label]) for label in expected))
    if not reference_keys:
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "failure_rate_ceiling": effective_failure_ceiling,
            "reason": "model profiles have no common usable prompt/repeat pairs",
        }
    if compliance and len(reference_keys) != expected_episodes:
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "failure_rate_ceiling": effective_failure_ceiling,
            "reason": "compliance mode requires every prompt/repeat pair to be usable",
        }
    for label in expected:
        unusable_count = by_policy[label]["episodes"] - len(paired_returns[label])
        reported_failures = failure_counts[label]["parse_failures"] + failure_counts[label]["invalid_calls"]
        if unusable_count != reported_failures:
            return {
                "status": "NOT_EVALUABLE",
                "expected": expected,
                "observed": observed,
                "failure_counts": failure_counts,
                "failure_rates": failure_rates,
                "failure_rate_ceiling": effective_failure_ceiling,
                "reason": (f"{label} usable flags disagree with its parse/invalid failure counts"),
            }

    comparisons: list[dict[str, Any]] = []
    passed = True
    sorted_keys = sorted(reference_keys)
    for comparison_index, (weaker, stronger) in enumerate(zip(expected, expected[1:], strict=False)):
        deltas = [paired_returns[stronger][key] - paired_returns[weaker][key] for key in sorted_keys]
        mean_delta = statistics.fmean(deltas)
        ci95_low, ci95_high = _paired_bootstrap_ci(
            deltas,
            seed=comparison_index,
        )
        comparison_passed = mean_delta > 0.0 and ci95_low > 0.0
        passed = passed and comparison_passed
        comparisons.append(
            {
                "weaker": weaker,
                "stronger": stronger,
                "pairs": len(deltas),
                "mean_delta": round(mean_delta, 6),
                "ci95_low": round(ci95_low, 6),
                "ci95_high": round(ci95_high, 6),
                "paired_episodes": len(deltas),
                "status": "PASS" if comparison_passed else "FAIL",
            }
        )

    return {
        "status": "PASS" if passed else "FAIL",
        "expected": expected,
        "observed": observed,
        "failure_counts": failure_counts,
        "failure_rates": failure_rates,
        "failure_rate_ceiling": effective_failure_ceiling,
        "valid_paired_episodes": len(reference_keys),
        "comparisons": comparisons,
        "reason": (
            None if passed else "paired 95% bootstrap interval did not show a positive adjacent-model improvement"
        ),
    }


async def _sweep(
    task_rows: list[dict[str, Any]],
    repeats: int,
    specs: list[ModelSpec],
    concurrency: int = 8,
    *,
    compliance_profile: bool = False,
    failure_rate_ceiling: float = 0.0,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if not task_rows:
        raise ValueError("at least one task row is required")
    if not 1 <= concurrency <= 32:
        raise ValueError(f"concurrency must be between 1 and 32, got {concurrency}")
    if not math.isfinite(failure_rate_ceiling) or not 0.0 <= failure_rate_ceiling <= 1.0:
        raise ValueError(f"failure_rate_ceiling must be a finite number between 0 and 1, got {failure_rate_ceiling!r}")
    if compliance_profile and failure_rate_ceiling != 0.0:
        raise ValueError("compliance profiles require failure_rate_ceiling=0")
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(
            f"duplicate model labels in --models: {sorted(set(x for x in labels if labels.count(x) > 1))}"
        )
    for spec in specs:
        if spec.api_key_env and not os.environ.get(spec.api_key_env):
            raise ValueError(f"model '{spec.label}': environment variable {spec.api_key_env} is not set")
    ranks = [spec.capability_rank for spec in specs if spec.capability_rank is not None]
    if ranks and len(ranks) != len(specs):
        raise ValueError("every model must declare capability_rank when sweeping two or more models")
    if len(ranks) != len(set(ranks)):
        raise ValueError("model capability_rank values must be unique")

    base_url = _start_local_server()
    rows = _repeat_task_rows(task_rows, repeats)
    results: dict[str, PolicyStats] = {}
    semaphore = asyncio.Semaphore(concurrency)

    for label, factory in _ANCHORS.items():
        stats = results[label] = PolicyStats()

        async def run_anchor(row: dict[str, Any]) -> None:
            rng_seed = int(row["seed"]) * 1_000_003 + int(row["_profile_response_index"])
            action_fn = functools.partial(_scripted_action, factory(), random.Random(rng_seed))
            async with semaphore:
                await _run_episode(base_url, row, action_fn, stats)

        await asyncio.gather(*(run_anchor(row) for row in rows))

    for spec in specs:
        stats = results[f"model:{spec.label}"] = PolicyStats()

        async def run_model(row: dict[str, Any], llm_session: aiohttp.ClientSession) -> None:
            async with semaphore:
                action_fn = functools.partial(_llm_action, llm_session, spec, row)
                try:
                    await _run_episode(base_url, row, action_fn, stats)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    stats.infra_errors += 1
                    print(f"model:{spec.label} seed={row['seed']}: episode dropped ({type(e).__name__}: {e})")

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency)) as llm_session:
            await asyncio.gather(*(run_model(row, llm_session) for row in rows))

    relief_mean = results["anchor:relief"].row("anchor:relief")["mean_return"]
    if relief_mean is None:
        raise RuntimeError("relief anchor produced no completed episodes")
    table = []
    for label, stats in results.items():
        entry = stats.row(label)
        entry["vs_relief"] = round(entry["mean_return"] - relief_mean, 4) if entry["mean_return"] is not None else None
        table.append(entry)
    table.sort(key=lambda r: r["mean_return"] if r["mean_return"] is not None else -math.inf, reverse=True)

    anchor_rows = {label: results[label].row(label) for label in _ANCHOR_ORDER}
    anchor_returns = {
        label: {str(record["pair_key"]): float(record["return"]) for record in row["episode_records"]}
        for label, row in anchor_rows.items()
    }
    anchor_comparisons: list[dict[str, Any]] = []
    for comparison_index, (better, worse) in enumerate(_ANCHOR_CONSTRAINTS):
        pair_keys = sorted(set(anchor_returns[better]) & set(anchor_returns[worse]))
        deltas = [anchor_returns[better][key] - anchor_returns[worse][key] for key in pair_keys]
        mean_delta = statistics.fmean(deltas)
        ci95_low, ci95_high = _paired_bootstrap_ci(
            deltas,
            seed=10_000 + comparison_index,
        )
        passed = mean_delta > 0.0 and ci95_low > 0.0
        anchor_comparisons.append(
            {
                "better": better,
                "worse": worse,
                "pairs": len(deltas),
                "mean_delta": round(mean_delta, 6),
                "ci95_low": round(ci95_low, 6),
                "ci95_high": round(ci95_high, 6),
                "status": "PASS" if passed else "FAIL",
            }
        )
    ordered = all(comparison["status"] == "PASS" for comparison in anchor_comparisons)
    model_ordering = _evaluate_model_ordering(
        table,
        specs,
        expected_episodes=len(rows),
        compliance=compliance_profile,
        failure_rate_ceiling=failure_rate_ceiling,
    )
    return {
        "tasks": [
            {
                "prompt_index": row.get("_profile_prompt_index"),
                "seed": row.get("seed"),
                "difficulty": row.get("difficulty"),
                "scenario_id": row.get("scenario_id"),
            }
            for row in task_rows
        ],
        "prompts": len(task_rows),
        "responses_per_prompt": repeats,
        "episodes_per_policy": len(rows),
        "concurrency": concurrency,
        "compliance_profile": compliance_profile,
        "failure_rate_ceiling": 0.0 if compliance_profile else failure_rate_ceiling,
        "profile": table,
        "anchor_ordering_ok": ordered,
        "anchor_order_expected": list(_ANCHOR_ORDER),
        "anchor_order_constraints": [list(pair) for pair in _ANCHOR_CONSTRAINTS],
        "anchor_ordering_comparisons": anchor_comparisons,
        "model_ordering": model_ordering,
        "model_ordering_ok": (
            model_ordering["status"] == "PASS" if model_ordering["status"] in {"PASS", "FAIL"} else None
        ),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"\ncapability sweep: {report['episodes_per_policy']} episodes/policy over "
        f"{report['prompts']} prompts x {report['responses_per_prompt']} responses\n"
    )
    header = (
        f"{'policy':<24} {'mean':>9} {'std':>7} {'vs relief':>10} {'reject%':>8} "
        f"{'noop%':>7} {'invalid':>8} {'parse-fail':>11} {'infra':>6} {'eps':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in report["profile"]:
        mean = f"{r['mean_return']:>9.3f}" if r["mean_return"] is not None else f"{'--':>9}"
        vs = f"{r['vs_relief']:>+10.3f}" if r["vs_relief"] is not None else f"{'--':>10}"
        print(
            f"{r['policy']:<24} {mean} {r['std_return']:>7.3f} {vs} "
            f"{r['rejection_rate'] * 100:>7.1f}% {r['noop_rate'] * 100:>6.1f}% "
            f"{r['invalid_calls']:>8} {r['parse_failures']:>11} {r['infra_errors']:>6} {r['episodes']:>4}"
        )
    verdict = "PASS" if report["anchor_ordering_ok"] else "FAIL"
    print(
        f"\nanchor partial order (relief > noop, relief > random-valid, noop/random-valid > catastrophic): {verdict}"
    )
    if not report["anchor_ordering_ok"]:
        print("a broken anchor ordering means reward attribution is suspect -- investigate before training.")
    model_ordering = report["model_ordering"]
    if model_ordering["status"] != "NOT_CONFIGURED":
        print(
            f"declared model capability ordering: {model_ordering['status']} "
            f"({model_ordering.get('reason') or 'strictly increasing mean return'})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="JSON file with a list of model specs (see sweep_models.example.json)")
    parser.add_argument("--task-count", type=int, default=5, help="number of deterministic prompts (default 5)")
    parser.add_argument("--repeats", type=int, default=2, help="responses per prompt (default 2)")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrent episodes, 1-32 (default 8)")
    parser.add_argument(
        "--compliance-profile",
        action="store_true",
        help="run the contribution-guide minimum: 500 prompts x 16 responses",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.0,
        help=("maximum parse/invalid/infrastructure failure fraction per model outside compliance mode (default 0)"),
    )
    parser.add_argument("--out", help="write the full report as JSON to this path")
    args = parser.parse_args()

    specs = []
    if args.models:
        with open(args.models) as f:
            specs = [ModelSpec(**spec) for spec in json.load(f)]

    try:
        base_rows = _load_example_rows()
        if args.compliance_profile:
            _require_compliance_models(specs)
            _validate_compliance_rows(base_rows)
        task_count = 500 if args.compliance_profile else args.task_count
        repeats = 16 if args.compliance_profile else args.repeats
        task_rows = _profile_task_rows(base_rows, task_count)
        report = asyncio.run(
            _sweep(
                task_rows,
                repeats,
                specs,
                concurrency=args.concurrency,
                compliance_profile=args.compliance_profile,
                failure_rate_ceiling=args.max_failure_rate,
            )
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    _print_report(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.out}")
    # Nonzero exit on a broken anchor ladder or a failed/incomplete declared
    # multi-model hierarchy, so the command can be used as a pre-training gate.
    model_status = report["model_ordering"]["status"]
    if not report["anchor_ordering_ok"] or (len(specs) >= 2 and model_status != "PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
