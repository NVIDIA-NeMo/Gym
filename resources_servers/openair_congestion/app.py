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

"""5G RAN congestion control, gymnasium style.

Multi-turn: the model observes rolling 5s cell/UE KPIs each turn and issues
exactly one tool call from an 8-tool action space (7 actuators + noop; tool
schemas ride in each task row's responses_create_params.tools). /step applies
the action through the selected Backend and returns the next KPIs plus the
per-step reward computed inside the env (rewards.compute_breakdown), passed
through unchanged; the shared gymnasium_agent sums step rewards into the
episode return, like blackjack.

Backends (backends.py): ``replay`` is the causal, deterministic training
environment. ``dataset_replay`` serves recorded transitions for diagnostics
only because policy actions cannot change a pre-recorded next state.

The ``openair_congestion`` domain package is colocated with this resource
server, so a clean NeMo Gym checkout is self-contained.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict
from typing import Any, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import Field, ValidationInfo, field_validator

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from resources_servers.gymnasium import GymnasiumServer

# Load the backend layer before the colocated domain imports so an incomplete
# checkout fails with the backend's targeted diagnostic.
from resources_servers.openair_congestion.backends import Backend, select_backend


# isort: split
from openair_congestion.render import T2_OBSERVATION_RENDER, to_policy_text
from openair_congestion.reward_profiles import select_reward_profile
from openair_congestion.rewards import DEFAULT_WEIGHTS
from openair_congestion.schemas import ToolCall
from openair_congestion.tools import TOOL_SCHEMA_BY_NAME


_GUARDRAIL_VALIDATION_KEYWORDS = {
    "const",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "maxItems",
    "maxLength",
    "maxProperties",
    "minimum",
    "minItems",
    "minLength",
    "minProperties",
    "multipleOf",
    "pattern",
}


def _structural_tool_schema(value: Any) -> Any:
    """Keep JSON shape/type checks here and leave value policy to guardrail."""

    if isinstance(value, dict):
        return {
            key: _structural_tool_schema(item)
            for key, item in value.items()
            if key not in _GUARDRAIL_VALIDATION_KEYWORDS
        }
    if isinstance(value, list):
        return [_structural_tool_schema(item) for item in value]
    return value


_TOOL_ARGUMENT_VALIDATORS = {
    name: Draft202012Validator(_structural_tool_schema(spec["function"]["parameters"]))
    for name, spec in TOOL_SCHEMA_BY_NAME.items()
}

_DEFAULT_OBSERVATION_RENDER = "openair_natural_language_v1"


def _episode_contract(
    capabilities: dict[str, Any],
    tier: str,
    *,
    n_cells: int,
    max_target_id: int,
) -> dict[str, Any]:
    """Return the explicit contract consumed by external rollout trainers."""

    profile = select_reward_profile(tier)
    contract = {
        **capabilities,
        "reward_profile": profile.version,
        "reward_weights": asdict(DEFAULT_WEIGHTS),
        "observation_render": (T2_OBSERVATION_RENDER if tier.upper() == "T2" else _DEFAULT_OBSERVATION_RENDER),
    }
    if tier.upper() == "T2":
        # The shared Gymnasium agent consumes this optional contract before
        # the first model turn. It makes the model-facing tool schemas match
        # the narrow service-safe T2 runtime guardrail instead of advertising
        # six actions that T2 deliberately rejects.
        contract["tool_contract"] = {
            "allowed_names": ["noop", "set_prb_cap"],
            "parameter_overrides": {
                "set_prb_cap": {
                    "cell_id": {
                        "minimum": 0,
                        "maximum": max(0, int(n_cells) - 1),
                    },
                    "target": {"enum": ["ue"]},
                    "target_id": {
                        "minimum": 0,
                        "maximum": max(0, int(max_target_id)),
                    },
                    "max_prb": {"minimum": 200, "maximum": 273},
                }
            },
        }
    return contract


class OpenAirCongestionResourcesServerConfig(BaseResourcesServerConfig):
    # Which Backend drives episodes: 'replay' (default, causal/CI-safe) or
    # 'dataset_replay' (recorded, diagnostic-only). The
    # OPENAIR_CONGESTION_BACKEND env var overrides. Extra YAML keys bind here
    # because the config node type uses ConfigDict(extra='allow').
    backend: str = "replay"
    # Replay-backend knobs; defaults match openair_congestion.replay_env.ReplayEnv.
    replay_root: str = "data/replay"
    pool_size: int = Field(default=32, ge=1)
    max_steps_default: int = Field(default=60, ge=1)
    # dataset_replay knobs: replay a recorded dataset (KPI snapshots or GRPO
    # rollout traces; see dataset_backend.py) instead of synthesizing
    # trajectories. cell_capacity_mbps feeds the reward's throughput
    # normalizer; trace episodes recording cell_capacity_mbps_total override it.
    dataset_path: str = "data/fixtures/sample_provided.jsonl"
    cell_capacity_mbps: float = 60.0
    # Truncation-budget fallback for task rows that omit max_steps. Must not
    # exceed the gymnasium_agent's max_steps in the yaml: the agent truncates
    # client-side without notifying the env, so a larger server budget would
    # strand the backend episode slot.
    agent_max_steps: int = Field(default=16, ge=1)
    # A hard client/process crash cannot send /close. Expired cookie-scoped
    # sessions are reclaimed before a later reset attempts to allocate a slot.
    session_ttl_s: float = Field(default=3600.0, gt=0.0)
    # Terminal penalty for violating the exactly-one-tool-call protocol. It
    # must be finite and negative so refusal or malformed output cannot evade
    # congestion cost and outscore a valid noop merely by not advancing time.
    protocol_violation_penalty: float = -1.0

    @field_validator("pool_size", "max_steps_default", "agent_max_steps", mode="before")
    @classmethod
    def _strict_positive_integer_config(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{info.field_name} must be a positive integer, got {value!r}")
        return value

    @field_validator("session_ttl_s", mode="before")
    @classmethod
    def _strict_numeric_session_ttl(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"session_ttl_s must be a positive finite number, got {value!r}")
        return value


# Returned when a model turn contains no tool call.
_NO_TOOL_CALL_MSG = (
    "No tool call detected. Issue exactly one tool call per turn from the "
    "configured action space (use `noop` to stand pat). Telemetry unchanged."
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_object(raw: str) -> dict[str, Any]:
    parsed = json.loads(
        raw,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"arguments must be a JSON object, got {type(parsed).__name__}")
    return parsed


class OpenAirCongestionEnv(GymnasiumServer):
    """GymnasiumServer subclass: /reset + /step, driven by gymnasium_agent."""

    config: OpenAirCongestionResourcesServerConfig

    # Backend built once at startup so a bad replay_root / unknown backend
    # name fails at boot, not on the first rollout. Pydantic private attr.
    _backend: Optional[Backend] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        protocol_penalty = self.config.protocol_violation_penalty
        if not math.isfinite(protocol_penalty) or protocol_penalty >= 0.0:
            raise ValueError("protocol_violation_penalty must be finite and negative")
        if not math.isfinite(self.config.session_ttl_s):
            raise ValueError("session_ttl_s must be finite and positive")
        self._backend = select_backend(self.config)

    @property
    def backend(self) -> Backend:
        assert self._backend is not None, "Backend not initialized (model_post_init)"
        return self._backend

    def _live_episode_ids(self) -> set[str]:
        """Episode ids currently owned by live sessions (for the leak reaper)."""
        return {state["episode_id"] for state in self.session_state.values()}

    async def _reap_expired_sessions(self) -> None:
        """Release state left behind by clients that can no longer call /close."""

        now = time.monotonic()
        expired = [
            session_id
            for session_id, state in self.session_state.items()
            if now - float(state.get("last_activity_monotonic", now)) > self.config.session_ttl_s
        ]
        for session_id in expired:
            state = self.session_state.pop(session_id, None)
            if state is None:
                continue
            try:
                await asyncio.to_thread(self.backend.close, state["episode_id"])
            except KeyError:
                pass

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        if session_id is None:
            raise ValueError("session_id must not be None")
        requested_seed = metadata.get("seed")
        if requested_seed is not None:
            if isinstance(requested_seed, bool) or not isinstance(requested_seed, int):
                raise TypeError("seed must be a non-negative integer")
            if requested_seed < 0:
                raise ValueError("seed must be a non-negative integer")

        requested_max_steps = metadata.get("max_steps")
        if requested_max_steps is not None:
            if isinstance(requested_max_steps, bool) or not isinstance(requested_max_steps, int):
                raise TypeError("max_steps must be a positive integer")
            if requested_max_steps < 1:
                raise ValueError("max_steps must be a positive integer")

        await self._reap_expired_sessions()

        # A client retry can POST /reset twice with the same session cookie.
        # Close the previous episode first or its backend slot leaks forever.
        stale = self.session_state.pop(session_id, None)
        if stale is not None:
            try:
                await asyncio.to_thread(self.backend.close, stale["episode_id"])
            except KeyError:
                pass  # already closed inside the env

        # `metadata` = extra task-row fields forwarded by gymnasium_agent.
        task_params = {
            key: metadata[key]
            for key in ("seed", "difficulty", "regime_mix", "scenario_id", "tier", "max_steps")
            if metadata.get(key) is not None
        }
        first_obs, meta = await asyncio.to_thread(
            self.backend.reset,
            task_params,
            live_episode_ids=self._live_episode_ids(),
        )
        contract = _episode_contract(
            self.backend.capabilities(),
            meta.tier,
            n_cells=first_obs.global_.n_cells,
            max_target_id=max(
                (ue.ue_id for cell in first_obs.cells for ue in cell.ues),
                default=0,
            ),
        )
        self.session_state[session_id] = {
            "episode_id": meta.episode_id,
            "contract": contract,
            "cumulative_reward": 0.0,
            "n_steps": 0,
            # agent_steps counts model turns, n_steps env steps; a turn with
            # no tool call consumes a turn without advancing the env.
            "agent_steps": 0,
            "last_activity_monotonic": time.monotonic(),
            # Cap at the agent's turn budget so the server truncates no later
            # than the agent and the episode slot is freed via close_session().
            "max_agent_steps": min(
                int(task_params.get("max_steps") or self.config.max_steps_default),
                self.config.agent_max_steps,
            ),
        }
        # Observation appended as a user message after the dataset prompt.
        return to_policy_text(first_obs), {
            "episode_id": meta.episode_id,
            "seed": meta.seed,
            "scenario_id": meta.scenario_id,
            "tier": meta.tier,
            **contract,
        }

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        if session_id is None:
            raise ValueError("session_id must not be None")
        state = self.session_state.get(session_id)
        if state is None:
            # /step without /reset (defensive; gymnasium_agent always resets).
            return (
                None,
                0.0,
                False,
                True,
                {
                    "error": "no_active_episode",
                    "training_eligible": False,
                    "rollout_usable": False,
                    "training_usable": False,
                },
            )

        state["last_activity_monotonic"] = time.monotonic()
        state["agent_steps"] += 1
        out_of_budget = state["agent_steps"] >= state["max_agent_steps"]

        calls = [item for item in action.output if getattr(item, "type", None) == "function_call"]

        # Protocol failures terminate and release the episode. They must not
        # exploit the negative congestion objective by avoiding an env step.
        if not calls:
            return await self._standard_protocol_violation(
                session_id=session_id,
                state=state,
                error="no_tool_call",
                message=_NO_TOOL_CALL_MSG,
                tool_outputs=[],
            )

        if len(calls) != 1:
            message = "Exactly one tool call is required; the episode was terminated."
            return await self._standard_protocol_violation(
                session_id=session_id,
                state=state,
                error="multiple_tool_calls",
                message=message,
                tool_outputs=[self.tool_output(call, {"accepted": False, "error": message}) for call in calls],
            )

        call: NeMoGymResponseFunctionToolCall = calls[0]
        tool_outputs: list[dict[str, Any]] = []

        # Normalise to the env's ToolCall. Unknown tool names, malformed JSON,
        # and structurally invalid arguments terminate with the configured
        # protocol penalty; the backend is not stepped. Numeric/enum/runtime
        # bounds deliberately remain guardrail decisions so they receive the
        # standard auditable rejection reward without ending the episode.
        try:
            raw_args = _strict_json_object(call.arguments) if (call.arguments or "").strip() else {}
            tool_call = ToolCall(name=call.name, arguments=raw_args)
            _TOOL_ARGUMENT_VALIDATORS[tool_call.name].validate(raw_args)
        except (ValueError, JSONSchemaValidationError) as exc:
            tool_outputs.insert(0, self.tool_output(call, {"accepted": False, "error": str(exc)}))
            return await self._standard_protocol_violation(
                session_id=session_id,
                state=state,
                error="invalid_tool_call",
                message="Invalid tool call rejected; the episode was terminated.",
                tool_outputs=tool_outputs,
            )

        # One env step. In-range-but-rejected actions (guardrail) come back as
        # accepted=False with the env's own penalty reward, never an exception.
        next_obs, reward, done, step_info = await asyncio.to_thread(
            self.backend.step,
            state["episode_id"],
            tool_call,
        )
        step_info.update(state["contract"])

        # The server returns the per-step reward; gymnasium_agent sums the
        # episode return.
        state["cumulative_reward"] += float(reward)
        state["n_steps"] += 1

        accepted = bool(step_info.get("guardrail_accepted", True))
        rejection_reason = step_info.get("rejection_reason")
        step_idx = step_info.get("step_idx", state["n_steps"])
        tool_outputs.insert(
            0,
            self.tool_output(
                call,
                {"accepted": accepted, "rejection_reason": rejection_reason, "step_idx": step_idx},
            ),
        )

        terminated = bool(done)
        truncated = (not terminated) and out_of_budget
        observation = None if (terminated or truncated) else to_policy_text(next_obs)

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            {
                # Preserve the backend's auditable transition provenance and
                # reward decomposition. Explicit server-owned keys below win
                # if a backend ever emits a colliding name.
                **step_info,
                "tool_outputs": tool_outputs,
                "guardrail_accepted": accepted,
                "rejection_reason": rejection_reason,
                "step_idx": step_idx,
                "episode_id": state["episode_id"],
                "n_steps": state["n_steps"],
                "cumulative_reward": state["cumulative_reward"],
            },
        )

    async def _standard_protocol_violation(
        self,
        *,
        session_id: Optional[str],
        state: dict[str, Any],
        error: str,
        message: str,
        tool_outputs: list[dict[str, Any]],
    ) -> tuple[None, float, bool, bool, dict[str, Any]]:
        """Penalize one invalid model turn and eagerly release its episode."""

        penalty = float(self.config.protocol_violation_penalty)
        cumulative_reward = float(state["cumulative_reward"]) + penalty
        episode_id = state["episode_id"]
        release = await self._release_session(session_id)
        return (
            None,
            penalty,
            True,
            False,
            {
                **state["contract"],
                "error": error,
                "message": message,
                "protocol_violation": True,
                "protocol_rejection": True,
                "guardrail_accepted": False,
                "rejection_reason": message,
                "tool_outputs": tool_outputs,
                "episode_id": episode_id,
                "n_steps": state["n_steps"],
                "cumulative_reward": cumulative_reward,
                "training_eligible": False,
                "rollout_usable": False,
                "training_usable": False,
                "release": release,
            },
        )

    async def _release_session(self, session_id: Optional[str]) -> dict[str, Any]:
        """Free one backend slot exactly once and retain no stale session state."""

        state = self.session_state.pop(session_id, None)
        if state is None:
            return {"ok": True, "already_closed": True, "summary": {}}
        try:
            summary = await asyncio.to_thread(
                self.backend.close,
                state["episode_id"],
            )
        except KeyError:
            # The underlying env can close an episode on a terminal step.  It
            # is still safe to consume our session state exactly once.
            summary = {"ok": True, "already_closed_by_backend": True}
        return {"ok": True, "already_closed": False, "summary": summary}

    async def close_session(self, session_id: Optional[str]) -> None:
        # Framework calls this when a step returns terminated or truncated.
        if session_id is None:
            raise ValueError("session_id must not be None")
        await self._release_session(session_id)

    async def explicit_close(self, session_id: Optional[str]) -> dict[str, Any]:
        """Cookie-scoped, idempotent cleanup for stateful clients."""

        if session_id is None:
            raise ValueError("session_id must not be None")
        return await self._release_session(session_id)


if __name__ == "__main__":
    OpenAirCongestionEnv.run_webserver()
