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
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, Optional, TypeVar

from fastapi import HTTPException, Request
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import Field, PrivateAttr, ValidationInfo, field_validator

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.gymnasium import (
    EnvResetRequest,
    EnvResetResponse,
    EnvStepRequest,
    EnvStepResponse,
    GymnasiumServer,
)

# Load the backend layer before the colocated domain imports so an incomplete
# checkout fails with the backend's targeted diagnostic.
from resources_servers.openair_congestion.backends import Backend, select_backend


# isort: split
from openair_congestion.render import to_policy_text
from openair_congestion.schemas import SUPPORTED_REGIMES, AgentAux, LastActionEcho, ToolCall
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
_MAX_STEP_RESPONSE_CACHE_ENTRIES = 128
_MAX_COMPLETED_STEP_RESPONSE_CACHE_ENTRIES = 128
_MIN_PROTOCOL_PENALTY_MAGNITUDE = 1e-6
_MAX_PROTOCOL_PENALTY_MAGNITUDE = 1e6
_UNSUPPORTED_TOOL_CALL_TYPES = frozenset(
    {
        "code_interpreter_call",
        "computer_call",
        "custom_tool_call",
        "file_search_call",
        "image_generation_call",
        "local_shell_call",
        "mcp_approval_request",
        "mcp_call",
        "mcp_list_tools",
        "web_search_call",
    }
)
_T = TypeVar("_T")


class _IdempotencyRequestError(ValueError):
    """A client-correctable idempotency-key or payload error."""


class _ResetRequestError(ValueError):
    """A client-correctable reset task-input error."""


async def _finish_despite_cancellation(operation: Awaitable[_T]) -> _T:
    """Let accepted backend work finish atomically before propagating cancellation."""

    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        # ``asyncio.to_thread`` cannot stop a backend call that is already
        # running. Wait for server bookkeeping to catch up before releasing
        # lifecycle ownership. Keep shielding because callers may cancel the
        # outer request more than once while that bookkeeping is in flight.
        while not task.done():
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            task.result()
        except BaseException:
            pass
        raise cancellation


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
    # Additive penalty for violating the exactly-one-tool-call protocol. The
    # server advances the environment with a noop fallback, then adds this
    # finite negative surcharge so malformed output cannot benefit by ending
    # the episode early. Bounds keep the surcharge visible at the environment's
    # reward scale without overflowing episode totals.
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
    "configured action space (use `noop` to stand pat). Applied a noop "
    "fallback with the protocol penalty."
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


def _validation_error_message(exc: BaseException) -> str:
    """Render validation failures without recursively formatting user JSON."""

    if isinstance(exc, JSONSchemaValidationError):
        return exc.message
    try:
        return str(exc)
    except RecursionError:
        return f"{type(exc).__name__}: arguments are nested too deeply"


def _reset_payload_fingerprint(payload: dict[str, Any]) -> str:
    """Canonicalize one reset payload without its transport idempotency key."""

    payload = dict(payload)
    payload.pop("_ng_reset_request_id", None)
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise _IdempotencyRequestError("reset payload must contain only finite JSON data for idempotency") from exc


def _reset_request_id(metadata: dict[str, Any]) -> Optional[str]:
    request_id = metadata.get("_ng_reset_request_id")
    if request_id is None:
        return None
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise _IdempotencyRequestError("_ng_reset_request_id must be a non-empty string of at most 128 characters")
    return request_id


def _validate_reset_task_input(metadata: dict[str, Any]) -> None:
    """Validate task-owned reset fields before allocating a backend slot."""

    seed = metadata.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise _ResetRequestError("seed must be a non-negative integer")

    max_steps = metadata.get("max_steps")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise _ResetRequestError("max_steps must be a positive integer")

    difficulty = metadata.get("difficulty")
    if difficulty is not None:
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)):
            raise _ResetRequestError("difficulty must be a finite number between 0 and 1")
        parsed_difficulty = float(difficulty)
        if not math.isfinite(parsed_difficulty) or not 0.0 <= parsed_difficulty <= 1.0:
            raise _ResetRequestError("difficulty must be a finite number between 0 and 1")

    tier = metadata.get("tier")
    if tier is not None and tier != "replay":
        raise _ResetRequestError("tier must be 'replay'")

    scenario_id = metadata.get("scenario_id")
    if scenario_id is not None and not isinstance(scenario_id, str):
        raise _ResetRequestError("scenario_id must be a string")

    regime_mix = metadata.get("regime_mix")
    if regime_mix is None or regime_mix == {}:
        return
    if not isinstance(regime_mix, dict):
        raise _ResetRequestError("regime_mix must be an object")
    total = 0.0
    for name, weight in regime_mix.items():
        if not isinstance(name, str) or name not in SUPPORTED_REGIMES:
            raise _ResetRequestError(f"unknown regime_mix key {name!r}; valid: {SUPPORTED_REGIMES}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise _ResetRequestError(f"regime_mix weight must be numeric: {weight!r}")
        parsed_weight = float(weight)
        if not math.isfinite(parsed_weight) or not 0.0 <= parsed_weight <= 1.0:
            raise _ResetRequestError(f"regime_mix weight out of [0,1]: {weight}")
        total += parsed_weight
    if total <= 0.0:
        raise _ResetRequestError("regime_mix must contain at least one positive weight")
    if abs(total - 1.0) > 1e-3:
        raise _ResetRequestError(f"regime_mix must sum to 1.0 (got {total})")


def _step_payload_fingerprint(action: NeMoGymResponse, metadata: dict[str, Any]) -> str:
    """Canonicalize the payload bound to one step idempotency key."""

    step_metadata = dict(metadata)
    step_metadata.pop("_ng_step_request_id", None)
    try:
        payload = {
            "response": action.model_dump(mode="json"),
            "metadata": step_metadata,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, RecursionError) as exc:
        raise _IdempotencyRequestError("step payload must contain only finite JSON data for idempotency") from exc


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
    _lifecycle_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _lifecycle_users: dict[str, int] = PrivateAttr(default_factory=dict)
    _next_lifecycle_generation: int = PrivateAttr(default=0)
    _reset_request_owners: dict[str, str] = PrivateAttr(default_factory=dict)
    _completed_step_response_cache: OrderedDict[tuple[str, str], tuple[str, tuple, float]] = PrivateAttr(
        default_factory=OrderedDict
    )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        protocol_penalty = self.config.protocol_violation_penalty
        if (
            not math.isfinite(protocol_penalty)
            or abs(protocol_penalty) < _MIN_PROTOCOL_PENALTY_MAGNITUDE
            or abs(protocol_penalty) > _MAX_PROTOCOL_PENALTY_MAGNITUDE
            or protocol_penalty >= 0.0
        ):
            raise ValueError(
                "protocol_violation_penalty must be finite and between "
                f"-{_MAX_PROTOCOL_PENALTY_MAGNITUDE:g} and "
                f"-{_MIN_PROTOCOL_PENALTY_MAGNITUDE:g}"
            )
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

    @asynccontextmanager
    async def _session_lifecycle(self, session_id: str) -> AsyncIterator[None]:
        """Serialize every owner-changing operation for one cookie session."""

        lock = self._lifecycle_locks.setdefault(session_id, asyncio.Lock())
        self._lifecycle_users[session_id] = self._lifecycle_users.get(session_id, 0) + 1
        try:
            await lock.acquire()
        except BaseException:
            self._release_lifecycle_user(session_id, lock)
            raise
        try:
            yield
        finally:
            lock.release()
            self._release_lifecycle_user(session_id, lock)

    def _release_lifecycle_user(self, session_id: str, lock: asyncio.Lock) -> None:
        users = self._lifecycle_users[session_id] - 1
        if users:
            self._lifecycle_users[session_id] = users
            return
        self._lifecycle_users.pop(session_id, None)
        if session_id not in self.session_state and self._lifecycle_locks.get(session_id) is lock:
            self._lifecycle_locks.pop(session_id, None)

    def _prune_completed_step_responses(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        cutoff = now - self.config.session_ttl_s
        for key, (_, _, cached_at) in list(self._completed_step_response_cache.items()):
            if cached_at < cutoff:
                self._completed_step_response_cache.pop(key, None)
        while len(self._completed_step_response_cache) > _MAX_COMPLETED_STEP_RESPONSE_CACHE_ENTRIES:
            self._completed_step_response_cache.popitem(last=False)

    def _completed_step_response(
        self,
        session_id: str,
        request_id: str,
        request_fingerprint: str,
    ) -> Optional[tuple]:
        self._prune_completed_step_responses()
        key = (session_id, request_id)
        cached = self._completed_step_response_cache.get(key)
        if cached is None:
            return None
        cached_fingerprint, result, _ = cached
        if cached_fingerprint != request_fingerprint:
            raise _IdempotencyRequestError("_ng_step_request_id was already used with a different step payload")
        self._completed_step_response_cache[key] = (cached_fingerprint, result, time.monotonic())
        self._completed_step_response_cache.move_to_end(key)
        return result

    def _cache_completed_step_response(
        self,
        session_id: str,
        request_id: str,
        request_fingerprint: str,
        result: tuple,
    ) -> None:
        key = (session_id, request_id)
        self._completed_step_response_cache[key] = (request_fingerprint, result, time.monotonic())
        self._completed_step_response_cache.move_to_end(key)
        self._prune_completed_step_responses()

    async def _reap_expired_sessions(self) -> None:
        """Release state left behind by clients that can no longer call /close."""

        now = time.monotonic()
        expired = [
            session_id
            for session_id, state in self.session_state.items()
            if now - float(state.get("last_activity_monotonic", now)) > self.config.session_ttl_s
        ]
        for session_id in expired:
            async with self._session_lifecycle(session_id):
                state = self.session_state.get(session_id)
                if state is None:
                    continue
                if now - float(state.get("last_activity_monotonic", now)) <= self.config.session_ttl_s:
                    continue
                await self._release_session_locked(session_id)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        if session_id is None:
            raise ValueError("session_id must not be None")
        observation, info, _ = await self._reset_owned(
            metadata,
            session_id,
            request_fingerprint=_reset_payload_fingerprint(metadata),
        )
        return observation, info

    async def _reset_endpoint(self, body: EnvResetRequest, request: Request) -> EnvResetResponse:
        """Recover the original cookie owner when a committed reset is retried."""

        metadata = dict(body.model_extra or {})
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id is None:
            raise ValueError("session_id must not be None")
        try:
            observation, info, owner_session_id = await self._reset_owned(
                metadata,
                session_id,
                request_fingerprint=_reset_payload_fingerprint(body.model_dump(mode="json")),
            )
        except (_IdempotencyRequestError, _ResetRequestError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        # SessionMiddleware signs this owner back into Set-Cookie. A retry that
        # lost the first response therefore resumes the original episode rather
        # than transferring or allocating state under a second cookie.
        request.session[SESSION_ID_KEY] = owner_session_id
        return EnvResetResponse(observation=observation, info=info)

    async def _reset_owned(
        self,
        metadata: dict[str, Any],
        session_id: str,
        *,
        request_fingerprint: str,
    ) -> tuple[Optional[str], dict, str]:
        _validate_reset_task_input(metadata)
        request_id = _reset_request_id(metadata)
        requested_max_steps = metadata.get("max_steps")

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
        return await _finish_despite_cancellation(
            self._reset_with_lifecycle(
                task_params,
                effective_max_steps,
                session_id,
                request_id=request_id,
                request_fingerprint=request_fingerprint,
            )
        )

    async def _reset_with_lifecycle(
        self,
        task_params: dict[str, Any],
        effective_max_steps: int,
        session_id: str,
        *,
        request_id: Optional[str],
        request_fingerprint: str,
    ) -> tuple[Optional[str], dict, str]:
        assert self._reset_lock is not None
        async with self._reset_lock:
            await self._reap_expired_sessions()

            if request_id is not None:
                owner_session_id = self._reset_request_owners.get(request_id)
                if owner_session_id is not None:
                    async with self._session_lifecycle(owner_session_id):
                        owner_state = self.session_state.get(owner_session_id)
                        if owner_state is None or owner_state.get("reset_request_id") != request_id:
                            self._reset_request_owners.pop(request_id, None)
                        else:
                            if owner_state["reset_request_fingerprint"] != request_fingerprint:
                                raise _IdempotencyRequestError(
                                    "_ng_reset_request_id was already used with a different reset payload"
                                )
                            if session_id != owner_session_id and session_id in self.session_state:
                                raise _IdempotencyRequestError(
                                    "_ng_reset_request_id belongs to a different active cookie session"
                                )
                            observation, info = owner_state["reset_response"]
                            return observation, info, owner_session_id

            async with self._session_lifecycle(session_id):
                # A client retry can POST /reset twice with the same session cookie.
                # Close the previous episode first or its backend slot leaks forever.
                await self._release_session_locked(session_id)

                try:
                    first_obs, meta = await asyncio.to_thread(
                        self.backend.reset,
                        task_params,
                        live_episode_ids=self._live_episode_ids(),
                    )
                except KeyError as exc:
                    detail = exc.args[0] if exc.args else str(exc)
                    is_unknown_dataset_scenario = self.backend.capabilities().get(
                        "backend"
                    ) == "dataset_replay" and str(detail).startswith("scenario_id ")
                    if is_unknown_dataset_scenario:
                        raise _ResetRequestError(str(detail)) from exc
                    raise
                try:
                    contract = _episode_contract(
                        self.backend.capabilities(),
                        self.backend.reward_contract(meta.tier),
                    )
                    contract.update(
                        {
                            "protocol_violation_mode": "penalized_noop_v1",
                            "protocol_violation_penalty": float(self.config.protocol_violation_penalty),
                        }
                    )
                    observation = to_policy_text(first_obs)
                    info = {
                        "episode_id": meta.episode_id,
                        "seed": meta.seed,
                        "scenario_id": meta.scenario_id,
                        "tier": meta.tier,
                        **contract,
                    }
                    self._next_lifecycle_generation += 1
                    self.session_state[session_id] = {
                        "episode_id": meta.episode_id,
                        "generation": self._next_lifecycle_generation,
                        "contract": contract,
                        "cumulative_reward": 0.0,
                        "n_steps": 0,
                        # agent_steps counts model turns and n_steps counts backend
                        # transitions. Structural protocol violations consume both
                        # because they advance with a noop backend transition
                        # (causal in synthetic replay).
                        "agent_steps": 0,
                        "protocol_violation_count": 0,
                        "terminal": False,
                        "step_response_cache": OrderedDict(),
                        "reset_request_id": request_id,
                        "reset_request_fingerprint": request_fingerprint,
                        "reset_response": (observation, info),
                        "last_activity_monotonic": time.monotonic(),
                        # Cap at the agent's turn budget so the server truncates no later
                        # than the agent and the episode slot is freed via close_session().
                        "max_agent_steps": effective_max_steps,
                    }
                    if request_id is not None:
                        self._reset_request_owners[request_id] = session_id
                except BaseException:
                    try:
                        await asyncio.to_thread(self.backend.close, meta.episode_id)
                    except KeyError:
                        pass
                    raise
        # Observation appended as a user message after the dataset prompt.
        return observation, info, session_id

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        """Apply one turn, deduplicating transport retries from the paired agent."""

        if session_id is None:
            raise ValueError("session_id must not be None")
        request_id = metadata.get("_ng_step_request_id")
        if request_id is None:
            return await _finish_despite_cancellation(
                self._step_with_lifecycle(
                    action,
                    metadata,
                    session_id,
                    request_id=None,
                    request_fingerprint=None,
                )
            )
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise _IdempotencyRequestError("_ng_step_request_id must be a non-empty string of at most 128 characters")
        request_fingerprint = _step_payload_fingerprint(action, metadata)

        return await _finish_despite_cancellation(
            self._step_with_lifecycle(
                action,
                metadata,
                session_id,
                request_id=request_id,
                request_fingerprint=request_fingerprint,
            )
        )

    async def _step_with_lifecycle(
        self,
        action: NeMoGymResponse,
        metadata: dict,
        session_id: str,
        request_id: Optional[str],
        request_fingerprint: Optional[str],
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        """Serialize one episode transition and cache completed requests."""

        async with self._session_lifecycle(session_id):
            if request_id is not None:
                assert request_fingerprint is not None
                completed = self._completed_step_response(session_id, request_id, request_fingerprint)
                if completed is not None:
                    return completed
            state = self.session_state.get(session_id)
            if request_id is not None and state is not None:
                cached = state["step_response_cache"].get(request_id)
                if cached is not None:
                    cached_fingerprint, cached_result = cached
                    if cached_fingerprint != request_fingerprint:
                        raise _IdempotencyRequestError(
                            "_ng_step_request_id was already used with a different step payload"
                        )
                    return cached_result
            result = await self._step_once(action, metadata, session_id)
            if state is not None and self.session_state.get(session_id) is state:
                if result[2] or result[3]:
                    # Seal the generation before releasing its lifecycle lock.
                    # Endpoint cleanup happens just after this method returns;
                    # a queued distinct step must not advance the terminal
                    # backend in that small interval.
                    state["terminal"] = True
                result[4]["_ng_lifecycle_generation"] = state["generation"]
                if request_id is not None:
                    assert request_fingerprint is not None
                    cache = state["step_response_cache"]
                    cache[request_id] = (request_fingerprint, result)
                    cache.move_to_end(request_id)
                    while len(cache) > _MAX_STEP_RESPONSE_CACHE_ENTRIES:
                        cache.popitem(last=False)
                    self._cache_completed_step_response(
                        session_id,
                        request_id,
                        request_fingerprint,
                        result,
                    )
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
        if state.get("terminal"):
            return (
                None,
                0.0,
                False,
                True,
                {
                    "error": "episode_complete",
                    "episode_id": state["episode_id"],
                    "n_steps": state["n_steps"],
                    "cumulative_reward": state["cumulative_reward"],
                    "training_eligible": False,
                    "rollout_usable": False,
                    "training_usable": False,
                },
            )

        state["last_activity_monotonic"] = time.monotonic()
        next_agent_steps = state["agent_steps"] + 1
        out_of_budget = next_agent_steps >= state["max_agent_steps"]

        calls = [item for item in action.output if getattr(item, "type", None) == "function_call"]
        unsupported_calls = [
            item for item in action.output if getattr(item, "type", None) in _UNSUPPORTED_TOOL_CALL_TYPES
        ]

        # A protocol failure advances through one noop backend transition
        # (causal in synthetic replay), then adds a negative surcharge. This
        # prevents malformed output from improving return by ending a
        # negative-reward episode early.
        if unsupported_calls:
            message = (
                "Only configured function tools are supported; applied a noop fallback with the protocol penalty."
            )
            return await self._standard_protocol_violation(
                state=state,
                next_agent_steps=next_agent_steps,
                out_of_budget=out_of_budget,
                error="unsupported_tool_call",
                message=message,
                tool_outputs=[self.tool_output(call, {"accepted": False, "error": message}) for call in calls],
            )

        if not calls:
            return await self._standard_protocol_violation(
                state=state,
                next_agent_steps=next_agent_steps,
                out_of_budget=out_of_budget,
                error="no_tool_call",
                message=_NO_TOOL_CALL_MSG,
                tool_outputs=[],
            )

        if len(calls) != 1:
            message = "Exactly one tool call is required; applied a noop fallback with the protocol penalty."
            return await self._standard_protocol_violation(
                state=state,
                next_agent_steps=next_agent_steps,
                out_of_budget=out_of_budget,
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
            raw_args = _strict_json_object(call.arguments or "")
            tool_call = ToolCall(name=call.name, arguments=raw_args)
            _TOOL_ARGUMENT_VALIDATORS[tool_call.name].validate(raw_args)
        except (ValueError, RecursionError, JSONSchemaValidationError) as exc:
            tool_outputs.insert(
                0,
                self.tool_output(
                    call,
                    {"accepted": False, "error": _validation_error_message(exc)},
                ),
            )
            return await self._standard_protocol_violation(
                state=state,
                next_agent_steps=next_agent_steps,
                out_of_budget=out_of_budget,
                error="invalid_tool_call",
                message="Invalid tool call rejected; applied a noop fallback with the protocol penalty.",
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
        state["agent_steps"] = next_agent_steps
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
                "protocol_violation_count": state["protocol_violation_count"],
                "had_protocol_violation": state["protocol_violation_count"] > 0,
            },
        )

    async def _standard_protocol_violation(
        self,
        *,
        state: dict[str, Any],
        next_agent_steps: int,
        out_of_budget: bool,
        error: str,
        message: str,
        tool_outputs: list[dict[str, Any]],
    ) -> tuple[Optional[str], float, bool, bool, dict[str, Any]]:
        """Advance one invalid model turn as noop plus a negative surcharge."""

        penalty = float(self.config.protocol_violation_penalty)
        fallback = ToolCall(name="noop", arguments={})
        next_obs, base_reward, done, step_info = await asyncio.to_thread(
            self.backend.step,
            state["episode_id"],
            fallback,
        )
        step_info.update(state["contract"])

        reward = float(base_reward) + penalty
        state["agent_steps"] = next_agent_steps
        state["cumulative_reward"] += reward
        state["n_steps"] += 1
        state["protocol_violation_count"] += 1

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
        truncated = (not terminated) and out_of_budget
        return (
            to_policy_text(next_obs),
            reward,
            terminated,
            truncated,
            {
                **step_info,
                "error": error,
                "message": message,
                "protocol_violation": True,
                "protocol_rejection": True,
                "protocol_accepted": False,
                # The submitted call never reached the guardrail. The actual
                # fallback did, and the backend accepted it.
                "guardrail_accepted": True,
                "fallback_guardrail_accepted": True,
                "rejection_reason": message,
                "applied_fallback_action": {"name": "noop", "arguments": {}},
                "tool_outputs": tool_outputs,
                "step_idx": step_idx,
                "episode_id": state["episode_id"],
                "n_steps": state["n_steps"],
                "cumulative_reward": state["cumulative_reward"],
                "protocol_violation_count": state["protocol_violation_count"],
                "had_protocol_violation": True,
            },
        )

    async def _release_session_locked(
        self,
        session_id: str,
        generation: Optional[int] = None,
    ) -> dict[str, Any]:
        """Free the locked generation once, retaining ownership on failure."""

        state = self.session_state.get(session_id)
        if state is None:
            return {"ok": True, "already_closed": True, "summary": {}}
        if generation is not None and state["generation"] != generation:
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
            # The same lifecycle lock prevents replacement while close runs;
            # leave state intact so explicit close or the reaper can retry.
            raise
        if self.session_state.get(session_id) is state:
            reset_request_id = state.get("reset_request_id")
            if reset_request_id is not None and self._reset_request_owners.get(reset_request_id) == session_id:
                self._reset_request_owners.pop(reset_request_id, None)
            state["step_response_cache"].clear()
            self.session_state.pop(session_id, None)
        return {"ok": True, "already_closed": False, "summary": summary}

    async def _release_session(
        self,
        session_id: str,
        generation: Optional[int] = None,
    ) -> dict[str, Any]:
        async with self._session_lifecycle(session_id):
            return await self._release_session_locked(session_id, generation)

    async def close_session(self, session_id: Optional[str], generation: Optional[int] = None) -> None:
        # Framework calls this when a step returns terminated or truncated.
        if session_id is None:
            raise ValueError("session_id must not be None")
        try:
            await _finish_despite_cancellation(self._release_session(session_id, generation))
        except Exception:
            # The terminal transition is already complete and scored. Cleanup
            # failure must not replace that response; retained state gives
            # explicit close and TTL reaping durable retry ownership.
            return

    async def explicit_close(self, session_id: Optional[str]) -> dict[str, Any]:
        """Cookie-scoped, idempotent cleanup for stateful clients."""

        if session_id is None:
            raise ValueError("session_id must not be None")
        return await _finish_despite_cancellation(self._release_session(session_id))

    async def _step_endpoint(self, body: EnvStepRequest, request: Request) -> EnvStepResponse:
        """Close exactly the episode generation that produced a terminal step."""

        session_id = request.session.get(SESSION_ID_KEY)
        try:
            obs, reward, terminated, truncated, info = await self.step(
                body.response,
                body.model_extra or {},
                session_id,
            )
        except _IdempotencyRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        generation = info.get("_ng_lifecycle_generation")
        if (terminated or truncated) and generation is not None:
            await self.close_session(session_id, generation=generation)
        public_info = {key: value for key, value in info.items() if key != "_ng_lifecycle_generation"}
        return EnvStepResponse(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=public_info,
        )

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
