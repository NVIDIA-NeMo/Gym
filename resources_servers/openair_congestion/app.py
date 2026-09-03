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
from typing import Any, Optional

from fastapi import Request
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import Field, PrivateAttr, ValidationInfo, field_validator

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.gymnasium import GymnasiumServer

# Load the backend layer before the colocated domain imports so an incomplete
# checkout fails with the backend's targeted diagnostic.
from resources_servers.openair_congestion.backends import Backend, select_backend


# isort: split
from openair_congestion.render import to_policy_text
from openair_congestion.schemas import AgentAux, LastActionEcho, ToolCall
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
    reward_contract: dict[str, Any],
) -> dict[str, Any]:
    """Return the explicit contract consumed by external rollout trainers."""

    return {
        **capabilities,
        **reward_contract,
        "observation_render": _DEFAULT_OBSERVATION_RENDER,
        "supports_explicit_close": True,
        "supports_step_idempotency": True,
    }


class OpenAirCongestionResourcesServerConfig(BaseResourcesServerConfig):
    # Which Backend drives episodes: 'replay' (default, causal/CI-safe) or
    # 'dataset_replay' (recorded, diagnostic-only). The
    # OPENAIR_CONGESTION_BACKEND env var overrides. Extra YAML keys bind here
    # because the config node type uses ConfigDict(extra='allow').
    backend: str = "replay"
    # Replay-backend knobs; defaults match openair_congestion.replay_env.ReplayEnv.
    pool_size: int = Field(default=32, ge=1)
    max_steps_default: int = Field(default=60, ge=1)
    # dataset_replay knobs: replay nested KPI snapshot JSONL instead of
    # synthesizing trajectories. cell_capacity_mbps feeds the reward's
    # throughput normalizer.
    dataset_path: str = "data/fixtures/sample_provided.jsonl"
    cell_capacity_mbps: float = 60.0
    reward_weights: Optional[dict[str, float]] = None
    # Truncation-budget fallback for task rows that omit max_steps. Must not
    # exceed the gymnasium_agent's max_steps in the yaml: the agent truncates
    # client-side without notifying the env, so a larger server budget would
    # strand the backend episode slot.
    agent_max_steps: int = Field(default=16, ge=1)
    # A hard client/process crash cannot send /close. Expired cookie-scoped
    # sessions are reclaimed before a later reset attempts to allocate a slot.
    session_ttl_s: float = Field(default=3600.0, gt=0.0)
    # Surcharge added to a noop transition when the model violates the
    # exactly-one-tool-call protocol. It must be finite and negative.
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
    "configured action space (use `noop` to stand pat). Applied penalized noop."
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

    # Backend built once at startup so an unknown backend fails at boot, not
    # on the first rollout. Pydantic private attr.
    _backend: Optional[Backend] = None
    # Allocation and session registration must be one atomic operation.  The
    # backend leak reaper treats any allocation absent from session_state as
    # orphaned, so concurrent resets cannot safely overlap that interval.
    _reset_lock: Optional[asyncio.Lock] = None
    _step_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _step_response_cache: dict[str, tuple[str, tuple, float]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        protocol_penalty = self.config.protocol_violation_penalty
        if not math.isfinite(protocol_penalty) or protocol_penalty >= 0.0:
            raise ValueError("protocol_violation_penalty must be finite and negative")
        if not math.isfinite(self.config.session_ttl_s):
            raise ValueError("session_ttl_s must be finite and positive")
        self._backend = select_backend(self.config)
        self._reset_lock = asyncio.Lock()

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
            except Exception:
                self.session_state.setdefault(session_id, state)
                raise

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        if session_id is None:
            raise ValueError("session_id must not be None")
        self._step_response_cache.pop(session_id, None)
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

        # `metadata` = extra task-row fields forwarded by gymnasium_agent.
        task_params = {
            key: metadata[key]
            for key in ("seed", "difficulty", "regime_mix", "scenario_id", "tier", "max_steps")
            if metadata.get(key) is not None
        }
        # The paired agent can drive at most ``agent_max_steps`` turns. Pass
        # that same effective budget into the backend so replay does not
        # precompute unreachable observations from an omitted or oversized
        # task-row value.
        effective_max_steps = min(
            int(requested_max_steps or self.config.max_steps_default),
            self.config.agent_max_steps,
        )
        task_params["max_steps"] = effective_max_steps
        assert self._reset_lock is not None
        async with self._reset_lock:
            await self._reap_expired_sessions()

            # A client retry can POST /reset twice with the same session cookie.
            # Close the previous episode first or its backend slot leaks forever.
            stale = self.session_state.pop(session_id, None)
            if stale is not None:
                try:
                    await asyncio.to_thread(self.backend.close, stale["episode_id"])
                except KeyError:
                    pass  # already closed inside the env
                except Exception:
                    self.session_state.setdefault(session_id, stale)
                    raise

            first_obs, meta = await asyncio.to_thread(
                self.backend.reset,
                task_params,
                live_episode_ids=self._live_episode_ids(),
            )
            try:
                contract = _episode_contract(
                    self.backend.capabilities(),
                    self.backend.reward_contract(meta.tier),
                )
                self.session_state[session_id] = {
                    "episode_id": meta.episode_id,
                    "contract": contract,
                    "cumulative_reward": 0.0,
                    "n_steps": 0,
                    # Structural protocol violations consume both counters
                    # because they advance through a noop backend transition.
                    "agent_steps": 0,
                    "last_activity_monotonic": time.monotonic(),
                    # Cap at the agent's turn budget so the server truncates no later
                    # than the agent and the episode slot is freed via close_session().
                    "max_agent_steps": effective_max_steps,
                }
            except BaseException:
                try:
                    await asyncio.to_thread(self.backend.close, meta.episode_id)
                except KeyError:
                    pass
                raise
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
        """Apply one turn, deduplicating transport retries from the paired agent."""

        if session_id is None:
            raise ValueError("session_id must not be None")
        request_id = metadata.get("_ng_step_request_id")
        if request_id is None:
            return await self._step_once(action, metadata, session_id)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("_ng_step_request_id must be a non-empty string of at most 128 characters")

        lock = self._step_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            cached = self._step_response_cache.get(session_id)
            if cached is not None and cached[0] == request_id:
                return cached[1]

            result = await self._step_once(action, metadata, session_id)
            now = time.monotonic()
            self._step_response_cache[session_id] = (request_id, result, now)
            cutoff = now - self.config.session_ttl_s
            for cached_session_id, (_, _, cached_at) in list(self._step_response_cache.items()):
                if cached_at < cutoff:
                    self._step_response_cache.pop(cached_session_id, None)
                    if cached_session_id not in self.session_state:
                        self._step_locks.pop(cached_session_id, None)
            return result

    async def _step_once(
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

        # Protocol failures advance the same backend with noop plus a negative
        # surcharge, so malformed output cannot end a costly episode early.
        if not calls:
            return await self._standard_protocol_violation(
                state=state,
                error="no_tool_call",
                message=_NO_TOOL_CALL_MSG,
                tool_outputs=[],
            )

        if len(calls) != 1:
            message = "Exactly one tool call is required; applied penalized noop."
            return await self._standard_protocol_violation(
                state=state,
                error="multiple_tool_calls",
                message=message,
                tool_outputs=[self.tool_output(call, {"accepted": False, "error": message}) for call in calls],
            )

        call: NeMoGymResponseFunctionToolCall = calls[0]
        tool_outputs: list[dict[str, Any]] = []

        # Normalise to the env's ToolCall. Unknown tool names, malformed JSON,
        # and structurally invalid arguments receive a penalized noop fallback.
        # Numeric/enum/runtime bounds deliberately remain guardrail decisions
        # so they receive the standard auditable rejection reward.
        try:
            raw_args = _strict_json_object(call.arguments) if (call.arguments or "").strip() else {}
            tool_call = ToolCall(name=call.name, arguments=raw_args)
            _TOOL_ARGUMENT_VALIDATORS[tool_call.name].validate(raw_args)
        except (ValueError, JSONSchemaValidationError) as exc:
            tool_outputs.insert(0, self.tool_output(call, {"accepted": False, "error": str(exc)}))
            return await self._standard_protocol_violation(
                state=state,
                error="invalid_tool_call",
                message="Invalid tool call rejected; applied penalized noop.",
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
        # Gymnasium terminal transitions still return the observation reached
        # by the action.  It is the after-state used to compute this reward
        # and must remain available to trace/evaluation consumers.
        observation = to_policy_text(next_obs)

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
        state: dict[str, Any],
        error: str,
        message: str,
        tool_outputs: list[dict[str, Any]],
    ) -> tuple[Optional[str], float, bool, bool, dict[str, Any]]:
        """Advance one invalid model turn as noop plus a negative surcharge."""

        penalty = float(self.config.protocol_violation_penalty)
        next_obs, base_reward, done, step_info = await asyncio.to_thread(
            self.backend.step,
            state["episode_id"],
            ToolCall(name="noop", arguments={}),
        )
        step_info.update(state["contract"])

        reward = float(base_reward) + penalty
        state["cumulative_reward"] += reward
        state["n_steps"] += 1
        step_idx = step_info.get("step_idx", state["n_steps"])

        reward_terms = dict(step_info.get("reward_terms") or {})
        reward_terms["protocol_violation"] = penalty
        reward_terms["total"] = reward
        step_info["reward_terms"] = reward_terms

        next_obs = next_obs.model_copy(
            update={
                "agent_aux": AgentAux(
                    last_action=LastActionEcho(name="noop", arguments={}),
                    last_reward=reward,
                    last_rejection=message,
                    step_idx=step_idx,
                )
            }
        )
        terminated = bool(done)
        truncated = (not terminated) and state["agent_steps"] >= state["max_agent_steps"]
        return (
            to_policy_text(next_obs),
            reward,
            terminated,
            truncated,
            {
                **step_info,
                **state["contract"],
                "error": error,
                "message": message,
                "protocol_violation": True,
                "protocol_rejection": True,
                "guardrail_accepted": False,
                "rejection_reason": message,
                "applied_fallback_action": {"name": "noop", "arguments": {}},
                "tool_outputs": tool_outputs,
                "step_idx": step_idx,
                "episode_id": state["episode_id"],
                "n_steps": state["n_steps"],
                "cumulative_reward": state["cumulative_reward"],
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
        except Exception:
            # Keep retry ownership unless a newer reset already replaced it.
            self.session_state.setdefault(session_id, state)
            raise
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

    async def _close_endpoint(self, request: Request) -> dict[str, Any]:
        """Release the cookie-scoped OpenAir episode without resetting it."""

        return await self.explicit_close(request.session.get(SESSION_ID_KEY))

    def setup_webserver(self):
        """Add the OpenAir-only cleanup route to the shared Gymnasium API."""

        app = super().setup_webserver()
        app.post("/close")(self._close_endpoint)
        return app


if __name__ == "__main__":
    OpenAirCongestionEnv.run_webserver()
