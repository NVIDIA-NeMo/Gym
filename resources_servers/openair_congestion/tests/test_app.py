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
#
# Unit tests for the openair_congestion gymnasium-style resources_server,
# modeled on resources_servers/blackjack/tests/test_app.py (direct
# reset()/step() calls with a mock ServerClient).
#
import asyncio
import json
import math
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCustomToolCall,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.openair_congestion.app import (
    OpenAirCongestionEnv,
    OpenAirCongestionResourcesServerConfig,
)
from resources_servers.openair_congestion.backends import (
    ReplayBackend,
    select_backend,
)


def _make_env(**config_overrides) -> OpenAirCongestionEnv:
    config = OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="", **config_overrides)
    return OpenAirCongestionEnv(config=config, server_client=MagicMock(spec=ServerClient))


_RESPONSE_KWARGS = dict(
    id="r",
    created_at=0.0,
    model="m",
    object="response",
    parallel_tool_calls=True,
    tool_choice="auto",
    tools=[],
)


def _text_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        **_RESPONSE_KWARGS,
    )


def _tool_response(name: str, arguments: dict) -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(arguments),
                call_id="call_0",
                name=name,
                type="function_call",
                id="fc_0",
                status="completed",
            )
        ],
        **_RESPONSE_KWARGS,
    )


def _raw_tool_response(name: str, arguments: str) -> NeMoGymResponse:
    """Build one deliberately malformed function call for protocol tests."""

    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=arguments,
                call_id="call_0",
                name=name,
                type="function_call",
                id="fc_0",
                status="completed",
            )
        ],
        **_RESPONSE_KWARGS,
    )


def _multi_tool_response(*actions: tuple[str, dict]) -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(arguments),
                call_id=f"call_{index}",
                name=name,
                type="function_call",
                id=f"fc_{index}",
                status="completed",
            )
            for index, (name, arguments) in enumerate(actions)
        ],
        **_RESPONSE_KWARGS,
    )


def _mixed_custom_and_function_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseCustomToolCall(
                call_id="custom_0",
                input="ignored",
                name="unsupported",
                type="custom_tool_call",
                id="custom_0",
            ),
            NeMoGymResponseFunctionToolCall(
                arguments="{}",
                call_id="call_0",
                name="noop",
                type="function_call",
                id="fc_0",
                status="completed",
            ),
        ],
        **_RESPONSE_KWARGS,
    )


_TASK_METADATA = {
    "seed": 7001,
    "difficulty": 0.6,
    "regime_mix": {"prb_exhaustion": 1.0},
    "scenario_id": "prb_exhaustion",
    "tier": "replay",
    "max_steps": 16,
}
_SNAPSHOT_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "sample_provided.jsonl"


class TestReset:
    @pytest.mark.parametrize(
        "penalty",
        [0.0, 1.0, math.nan, math.inf, -math.inf, -1e-12, -1e9],
    )
    def test_protocol_violation_penalty_must_be_finite_and_negative(self, penalty):
        with pytest.raises(ValueError, match="protocol_violation_penalty"):
            _make_env(protocol_violation_penalty=penalty)

    @pytest.mark.parametrize("field", ["pool_size", "max_steps_default", "agent_max_steps"])
    @pytest.mark.parametrize("value", [0, -1, True])
    def test_episode_budget_config_must_be_positive(self, field, value):
        with pytest.raises(ValueError):
            _make_env(**{field: value})

    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf, True])
    def test_session_ttl_must_be_finite_positive_number(self, value):
        with pytest.raises(ValueError, match="session_ttl_s"):
            _make_env(session_ttl_s=value)

    @pytest.mark.asyncio
    async def test_reset_populates_state_and_renders_kpis(self):
        env = _make_env()
        obs, info = await env.reset(dict(_TASK_METADATA), session_id="sid")
        assert "sid" in env.session_state
        state = env.session_state["sid"]
        assert state["episode_id"] == info["episode_id"]
        assert state["cumulative_reward"] == 0.0
        assert state["n_steps"] == 0
        assert "5G RAN telemetry" in obs  # render.to_user_text output
        assert info["seed"] == 7001
        assert info["scenario_id"] == "prb_exhaustion"
        assert info["dynamics_mode"] == "synthetic_action_effect_v6_shared_capacity"
        assert info["causal_action_effects"] is True
        assert info["training_usable"] is True
        assert info["supports_explicit_close"] is True
        assert info["supports_step_idempotency"] is True

    @pytest.mark.asyncio
    async def test_reset_rejects_deferred_t2_tier(self):
        env = _make_env()
        metadata = dict(
            _TASK_METADATA,
            tier="T2",
            regime_mix={"prb_exhaustion": 1.0},
        )
        with pytest.raises(ValueError, match="tier"):
            await env.reset(metadata, session_id="t2-policy")

    @pytest.mark.asyncio
    async def test_dataset_reset_advertises_the_effective_reward_configuration(self):
        env = _make_env(
            backend="dataset_replay",
            dataset_path=str(_SNAPSHOT_FIXTURE),
            reward_weights={"w_sla": 0.25, "w_reject": 0.75},
        )

        _, info = await env.reset({"scenario_id": "lab_run_a"}, session_id="dataset")

        assert info["backend"] == "dataset_replay"
        assert info["reward_profile"] == "openair_v1"
        assert info["reward_weights"]["w_sla"] == pytest.approx(0.25)
        assert info["reward_weights"]["w_reject"] == pytest.approx(0.75)
        assert info["prb_pressure_threshold"] == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_none_session_is_rejected_before_reset(self):
        env = _make_env()
        with pytest.raises(ValueError, match="session_id"):
            await env.reset(dict(_TASK_METADATA), session_id=None)

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self):
        env = _make_env()
        _, info_a = await env.reset(dict(_TASK_METADATA), session_id="a")
        _, info_b = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="b")
        assert info_a["episode_id"] != info_b["episode_id"]
        assert env.session_state["a"]["episode_id"] != env.session_state["b"]["episode_id"]

    @pytest.mark.asyncio
    async def test_concurrent_resets_do_not_reap_an_unregistered_allocation(self, monkeypatch):
        env = _make_env(pool_size=1)
        original_reset = env.backend.reset
        first_allocated = threading.Event()
        second_attempted = threading.Event()
        call_lock = threading.Lock()
        call_count = 0

        def interleaved_reset(*args, **kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
            if current_call == 2:
                assert first_allocated.wait(timeout=1.0)
            result = original_reset(*args, **kwargs)
            if current_call == 1:
                first_allocated.set()
                second_attempted.wait(timeout=0.1)
            else:
                second_attempted.set()
            return result

        monkeypatch.setattr(env.backend, "reset", interleaved_reset)
        results = await asyncio.gather(
            env.reset(dict(_TASK_METADATA), session_id="a"),
            env.reset(dict(_TASK_METADATA, seed=7002), session_id="b"),
            return_exceptions=True,
        )

        successes = [result for result in results if not isinstance(result, BaseException)]
        failures = [result for result in results if isinstance(result, BaseException)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], RuntimeError)
        assert "pool exhausted" in str(failures[0])
        assert len(env.session_state) == 1

        surviving_session = next(iter(env.session_state))
        _, reward, terminated, truncated, _ = await env.step(
            _tool_response("noop", {}),
            {},
            session_id=surviving_session,
        )
        assert math.isfinite(reward)
        assert terminated is False
        assert truncated is False

    @pytest.mark.asyncio
    async def test_re_reset_same_session_closes_old_episode(self):
        # A client retry POSTing /reset twice with the same session cookie
        # must not leak the first episode's backend pool slot.
        env = _make_env()
        _, info_old = await env.reset(dict(_TASK_METADATA), session_id="sid")
        _, info_new = await env.reset(dict(_TASK_METADATA), session_id="sid")
        assert info_new["episode_id"] != info_old["episode_id"]
        assert env.session_state["sid"]["episode_id"] == info_new["episode_id"]
        # The old episode was closed during the second reset: closing it again
        # must raise KeyError (unknown episode_id) inside the backend.
        with pytest.raises(KeyError):
            env.backend.close(info_old["episode_id"])

    @pytest.mark.asyncio
    async def test_expired_session_is_reaped_after_hard_client_crash(self):
        # A hard client/process crash sends no /close and leaves its cookie
        # state resident on the server. A bounded lease must reclaim both the
        # session and backend slot without tests manually deleting state.
        env = _make_env(pool_size=1, session_ttl_s=1.0)
        _, info_dead = await env.reset(dict(_TASK_METADATA), session_id="dead")
        env.session_state["dead"]["last_activity_monotonic"] = time.monotonic() - 2.0

        _, info_new = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="new")
        assert info_new["episode_id"] != info_dead["episode_id"]
        assert "dead" not in env.session_state
        assert env.session_state["new"]["episode_id"] == info_new["episode_id"]

    @pytest.mark.asyncio
    async def test_missing_max_steps_falls_back_to_agent_budget(self):
        # Rows lacking max_steps must NOT fall back to the env default (60):
        # the agent truncates client-side at agent_max_steps (16), and a
        # larger server budget would strand the episode slot.
        env = _make_env()
        metadata = {k: v for k, v in _TASK_METADATA.items() if k != "max_steps"}
        await env.reset(metadata, session_id="sid")
        assert env.session_state["sid"]["max_agent_steps"] == 16

    @pytest.mark.asyncio
    async def test_requested_max_steps_is_capped_at_agent_budget(self):
        # A dataset row can request a longer episode than the paired agent is
        # configured to drive.  The server must still end and free its backend
        # slot no later than the agent's own turn budget.
        env = _make_env(pool_size=1, agent_max_steps=2)
        await env.reset(dict(_TASK_METADATA, max_steps=17), session_id="sid")

        assert env.session_state["sid"]["max_agent_steps"] == 2
        for turn in range(2):
            _, _, terminated, truncated, _ = await env.step(_tool_response("noop", {}), {}, session_id="sid")
            if turn == 0:
                assert terminated is False and truncated is False
        assert terminated or truncated
        await env.close_session("sid")

        # The only replay slot is immediately reusable.
        _, info = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="next")
        assert info["episode_id"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("requested_max_steps", [None, 1_000_000])
    async def test_backend_reset_uses_the_agent_step_budget(self, monkeypatch, requested_max_steps):
        env = _make_env(agent_max_steps=2)
        metadata = dict(_TASK_METADATA)
        if requested_max_steps is None:
            metadata.pop("max_steps")
        else:
            metadata["max_steps"] = requested_max_steps
        captured = {}
        original_reset = env.backend.reset

        def recording_reset(task_params, **kwargs):
            captured.update(task_params)
            assert task_params.get("max_steps") == 2
            return original_reset(task_params, **kwargs)

        monkeypatch.setattr(env.backend, "reset", recording_reset)
        await env.reset(metadata, session_id="sid")

        assert captured["max_steps"] == 2
        assert env.session_state["sid"]["max_agent_steps"] == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_steps", [0, -1, 1.5, True])
    async def test_reset_rejects_invalid_task_max_steps_without_opening_slot(self, max_steps):
        env = _make_env(pool_size=1)

        with pytest.raises((TypeError, ValueError), match="max_steps"):
            await env.reset(dict(_TASK_METADATA, max_steps=max_steps), session_id="bad")

        assert "bad" not in env.session_state
        _, info = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="good")
        assert info["episode_id"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [-1, 1.5, True])
    async def test_reset_rejects_invalid_seed_without_opening_slot(self, seed):
        env = _make_env(pool_size=1)

        with pytest.raises((TypeError, ValueError), match="seed"):
            await env.reset(dict(_TASK_METADATA, seed=seed), session_id="bad")

        assert "bad" not in env.session_state
        _, info = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="good")
        assert info["episode_id"]

    @pytest.mark.asyncio
    async def test_same_session_reset_waits_for_in_flight_step(self, monkeypatch):
        env = _make_env(pool_size=1, agent_max_steps=3)
        _, first_info = await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        original_step = env.backend.step
        step_started = threading.Event()
        release_step = threading.Event()

        def blocking_step(*args, **kwargs):
            step_started.set()
            assert release_step.wait(timeout=2.0)
            return original_step(*args, **kwargs)

        monkeypatch.setattr(env.backend, "step", blocking_step)
        step_task = asyncio.create_task(env.step(_tool_response("noop", {}), {}, session_id="sid"))
        assert await asyncio.to_thread(step_started.wait, 2.0)
        reset_task = asyncio.create_task(env.reset(dict(_TASK_METADATA, seed=7002), session_id="sid"))
        try:
            await asyncio.sleep(0.02)
            assert not reset_task.done()
        finally:
            release_step.set()

        step_result = await step_task
        _, replacement_info = await reset_task
        assert step_result[4]["episode_id"] == first_info["episode_id"]
        assert replacement_info["episode_id"] != first_info["episode_id"]
        assert env.session_state["sid"]["episode_id"] == replacement_info["episode_id"]
        assert env.session_state["sid"]["n_steps"] == 0

    @pytest.mark.asyncio
    async def test_repeated_cancellation_keeps_lease_until_backend_reset_finishes(self, monkeypatch):
        env = _make_env(pool_size=1)
        original_reset = env.backend.reset
        allocated = threading.Event()
        release_reset = threading.Event()
        calls = 0

        def blocking_reset(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original_reset(*args, **kwargs)
            if calls == 1:
                allocated.set()
                assert release_reset.wait(timeout=2.0)
            return result

        monkeypatch.setattr(env.backend, "reset", blocking_reset)
        cancelled = asyncio.create_task(env.reset(dict(_TASK_METADATA), session_id="sid"))
        assert await asyncio.to_thread(allocated.wait, 2.0)
        cancelled.cancel()
        await asyncio.sleep(0)
        cancelled.cancel()
        replacement = asyncio.create_task(env.reset(dict(_TASK_METADATA, seed=7002), session_id="sid"))
        try:
            await asyncio.sleep(0.02)
            assert not cancelled.done()
            assert not replacement.done()
        finally:
            release_reset.set()

        with pytest.raises(asyncio.CancelledError):
            await cancelled
        _, replacement_info = await replacement
        assert env.session_state["sid"]["episode_id"] == replacement_info["episode_id"]
        assert env.backend._open_episode_ids == {replacement_info["episode_id"]}


class TestStep:
    @pytest.mark.asyncio
    async def test_blocking_backend_step_is_offloaded_from_event_loop(self, monkeypatch):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        original = env.backend.step
        release = threading.Event()
        timed_out = threading.Event()

        def blocking_step(*args, **kwargs):
            if not release.wait(timeout=0.5):
                timed_out.set()
            return original(*args, **kwargs)

        monkeypatch.setattr(env.backend, "step", blocking_step)

        async def let_backend_continue():
            await asyncio.sleep(0.01)
            release.set()

        await asyncio.gather(
            env.step(_tool_response("noop", {}), {}, session_id="sid"),
            let_backend_continue(),
        )

        assert not timed_out.is_set()

    @pytest.mark.asyncio
    async def test_same_session_steps_without_request_ids_are_serialized(self, monkeypatch):
        env = _make_env(agent_max_steps=4)
        await env.reset(dict(_TASK_METADATA, max_steps=4), session_id="sid")
        original = env.backend.step
        first_started = threading.Event()
        release_first = threading.Event()
        call_lock = threading.Lock()
        call_count = 0

        def overlapping_step(*args, **kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)
            return original(*args, **kwargs)

        monkeypatch.setattr(env.backend, "step", overlapping_step)
        first = asyncio.create_task(env.step(_tool_response("noop", {}), {}, session_id="sid"))
        assert await asyncio.to_thread(first_started.wait, 2.0)
        second = asyncio.create_task(env.step(_tool_response("noop", {}), {}, session_id="sid"))
        await asyncio.sleep(0.02)
        release_first.set()
        results = await asyncio.gather(first, second)

        assert sorted(result[4]["step_idx"] for result in results) == [1, 2]
        assert env.session_state["sid"]["agent_steps"] == 2
        assert env.session_state["sid"]["n_steps"] == 2

    @pytest.mark.asyncio
    async def test_explicit_close_waits_for_in_flight_step(self, monkeypatch):
        env = _make_env(agent_max_steps=3)
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        original_step = env.backend.step
        step_started = threading.Event()
        release_step = threading.Event()

        def blocking_step(*args, **kwargs):
            step_started.set()
            assert release_step.wait(timeout=2.0)
            return original_step(*args, **kwargs)

        monkeypatch.setattr(env.backend, "step", blocking_step)
        step_task = asyncio.create_task(env.step(_tool_response("noop", {}), {}, session_id="sid"))
        assert await asyncio.to_thread(step_started.wait, 2.0)
        close_task = asyncio.create_task(env.explicit_close("sid"))
        try:
            await asyncio.sleep(0.02)
            assert not close_task.done()
        finally:
            release_step.set()

        await step_task
        close_result = await close_task
        assert close_result["already_closed"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_ttl_reaper_waits_for_in_flight_step(self, monkeypatch):
        env = _make_env(pool_size=2, session_ttl_s=1.0, agent_max_steps=3)
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="expired")
        original_step = env.backend.step
        step_started = threading.Event()
        release_step = threading.Event()

        def blocking_step(*args, **kwargs):
            step_started.set()
            assert release_step.wait(timeout=2.0)
            return original_step(*args, **kwargs)

        monkeypatch.setattr(env.backend, "step", blocking_step)
        step_task = asyncio.create_task(env.step(_tool_response("noop", {}), {}, session_id="expired"))
        assert await asyncio.to_thread(step_started.wait, 2.0)
        env.session_state["expired"]["last_activity_monotonic"] = time.monotonic() - 2.0
        reset_task = asyncio.create_task(env.reset(dict(_TASK_METADATA, seed=7002), session_id="next"))
        try:
            await asyncio.sleep(0.02)
            assert not reset_task.done()
        finally:
            release_step.set()

        step_result = await step_task
        _, reset_info = await reset_task
        assert step_result[4]["step_idx"] == 1
        assert "expired" not in env.session_state
        assert env.session_state["next"]["episode_id"] == reset_info["episode_id"]

    @pytest.mark.asyncio
    async def test_none_session_is_rejected_before_step(self):
        env = _make_env()
        with pytest.raises(ValueError, match="session_id"):
            await env.step(
                _tool_response("noop", {}),
                {},
                session_id=None,
            )

    @pytest.mark.asyncio
    async def test_noop_step_returns_finite_reward_and_tool_output(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(_tool_response("noop", {}), {}, session_id="sid")
        assert math.isfinite(reward)
        assert term is False
        assert trunc is False
        assert "5G RAN telemetry" in obs
        assert info["guardrail_accepted"] is True
        assert info["causal_action_effects"] is True
        assert info["training_usable"] is True
        assert info["diagnostic_only"] is False
        # The applied call gets a matching function_call_output for the agent.
        assert info["tool_outputs"][0]["call_id"] == "call_0"
        assert env.session_state["sid"]["n_steps"] == 1

    @pytest.mark.asyncio
    async def test_out_of_range_action_is_rejected_not_crashed(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        # cell_id=99 is in-schema-type but out of range: the env guardrail
        # rejects and applies its own penalty; the server must not raise.
        obs, reward, term, trunc, info = await env.step(
            _tool_response("set_scheduler_policy", {"cell_id": 99, "policy": "PF"}), {}, session_id="sid"
        )
        assert math.isfinite(reward)
        assert info["guardrail_accepted"] is False
        assert info["rejection_reason"]
        assert term is False and trunc is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("action", "expected_error"),
        [
            pytest.param(_text_response("No action."), "no_tool_call", id="missing"),
            pytest.param(_raw_tool_response("noop", ""), "invalid_tool_call", id="blank-arguments"),
            pytest.param(_raw_tool_response("noop", "   "), "invalid_tool_call", id="whitespace-arguments"),
            pytest.param(_raw_tool_response("noop", "["), "invalid_tool_call", id="malformed"),
            pytest.param(
                _raw_tool_response("noop", '{"x":' * 1_200 + "0" + "}" * 1_200),
                "invalid_tool_call",
                id="deeply-nested-json",
            ),
            pytest.param(
                _raw_tool_response(
                    "set_qos_weights",
                    '{"cell_id": 0, "weights": {"1": NaN}}',
                ),
                "invalid_tool_call",
                id="non-finite-json",
            ),
            pytest.param(
                _raw_tool_response(
                    "set_scheduler_policy",
                    '{"cell_id": 0, "cell_id": 1, "policy": "PF"}',
                ),
                "invalid_tool_call",
                id="duplicate-json-key",
            ),
            pytest.param(
                _raw_tool_response(
                    "set_prb_cap",
                    '{"cell_id": false, "target": "ue", "target_id": false, "max_prb": true}',
                ),
                "invalid_tool_call",
                id="boolean-integer-arguments",
            ),
            pytest.param(
                _raw_tool_response("noop", '{"unexpected": 1}'),
                "invalid_tool_call",
                id="extra-argument",
            ),
            pytest.param(
                _raw_tool_response("set_scheduler_policy", '{"cell_id": 0}'),
                "invalid_tool_call",
                id="missing-required-argument",
            ),
            pytest.param(
                _tool_response("open_pod_bay_doors", {}),
                "invalid_tool_call",
                id="unknown",
            ),
            pytest.param(
                _multi_tool_response(("noop", {}), ("noop", {})),
                "multiple_tool_calls",
                id="multiple",
            ),
            pytest.param(
                _mixed_custom_and_function_response(),
                "unsupported_tool_call",
                id="mixed-custom-and-function",
            ),
        ],
    )
    async def test_protocol_violation_applies_penalized_noop_transition(self, action, expected_error):
        noop_env = _make_env(pool_size=1, protocol_violation_penalty=-1.0)
        await noop_env.reset(dict(_TASK_METADATA), session_id="noop")
        _, noop_reward, _, _, _ = await noop_env.step(_tool_response("noop", {}), {}, session_id="noop")
        await noop_env.explicit_close("noop")

        env = _make_env(pool_size=1, protocol_violation_penalty=-1.0)
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(action, {}, session_id="sid")

        assert obs is not None
        assert reward == pytest.approx(noop_reward - 1.0)
        assert term is False and trunc is False
        assert info["error"] == expected_error
        assert info["protocol_violation"] is True
        assert info["protocol_rejection"] is True
        assert info["protocol_accepted"] is False
        assert info["guardrail_accepted"] is True
        assert info["fallback_guardrail_accepted"] is True
        assert info["protocol_violation_count"] == 1
        assert info["rejection_reason"]
        assert info["backend"] == "replay"
        assert info["action_affects_observation"] is True
        assert info["reward_profile"] == "openair_v1"
        assert info["reward_weights"]
        assert info["reward_terms"]["protocol_violation"] == pytest.approx(-1.0)
        assert info["reward_terms"]["total"] == pytest.approx(reward)
        assert info["observation_render"] == "openair_natural_language_v1"
        assert info["training_usable"] is True
        assert env.session_state["sid"]["n_steps"] == 1
        assert env.session_state["sid"]["cumulative_reward"] == pytest.approx(reward)

        # The malformed model turn advances as a penalized noop, so the slot
        # remains owned until normal completion or explicit cleanup.
        await env.explicit_close("sid")
        _, reset_info = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="next")
        assert reset_info["episode_id"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("violation_turn", [0, 2, 3])
    async def test_protocol_violation_cannot_beat_complete_noop_episode(self, violation_turn):
        async def run(*, invalid_turn: int | None) -> tuple[float, dict]:
            env = _make_env(protocol_violation_penalty=-1.0, agent_max_steps=4)
            await env.reset(dict(_TASK_METADATA, max_steps=4), session_id="sid")
            total = 0.0
            final_info: dict = {}
            for turn in range(4):
                response = _text_response("No action.") if turn == invalid_turn else _tool_response("noop", {})
                _, reward, terminated, truncated, final_info = await env.step(response, {}, session_id="sid")
                total += reward
                if terminated or truncated:
                    break
            await env.close_session("sid")
            return total, final_info

        noop_total, _ = await run(invalid_turn=None)
        invalid_total, info = await run(invalid_turn=violation_turn)

        assert invalid_total == pytest.approx(noop_total - 1.0)
        assert invalid_total < noop_total
        assert info["protocol_violation_count"] == 1

    @pytest.mark.asyncio
    async def test_duplicate_protocol_violation_is_charged_once(self):
        env = _make_env(protocol_violation_penalty=-1.0)
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        metadata = {"_ng_step_request_id": "invalid-turn-1"}

        first = await env.step(_text_response("No action."), metadata, session_id="sid")
        second = await env.step(_text_response("No action."), metadata, session_id="sid")

        assert second == first
        assert env.session_state["sid"]["n_steps"] == 1
        assert env.session_state["sid"]["protocol_violation_count"] == 1

    @pytest.mark.asyncio
    async def test_failed_backend_step_does_not_consume_the_agent_budget(self, monkeypatch):
        env = _make_env(protocol_violation_penalty=-1.0, agent_max_steps=2)
        await env.reset(dict(_TASK_METADATA, max_steps=2), session_id="sid")
        original_step = env.backend.step
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("backend step failed")
            return original_step(*args, **kwargs)

        monkeypatch.setattr(env.backend, "step", fail_once)
        metadata = {"_ng_step_request_id": "retry-after-error"}
        with pytest.raises(RuntimeError, match="backend step failed"):
            await env.step(_text_response("No action."), metadata, session_id="sid")

        state = env.session_state["sid"]
        assert state["agent_steps"] == 0
        assert state["n_steps"] == 0

        _, _, terminated, truncated, info = await env.step(_text_response("No action."), metadata, session_id="sid")
        assert terminated is False and truncated is False
        assert info["step_idx"] == 1
        assert state["agent_steps"] == 1
        assert state["n_steps"] == 1

    @pytest.mark.asyncio
    async def test_cancelled_request_finishes_and_caches_the_backend_transition(self, monkeypatch):
        env = _make_env(protocol_violation_penalty=-1.0, agent_max_steps=2)
        await env.reset(dict(_TASK_METADATA, max_steps=2), session_id="sid")
        original_step = env.backend.step
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def slow_step(*args, **kwargs):
            started.set()
            assert release.wait(timeout=2.0)
            try:
                return original_step(*args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(env.backend, "step", slow_step)
        metadata = {"_ng_step_request_id": "cancelled-turn"}
        task = asyncio.create_task(env.step(_text_response("No action."), metadata, session_id="sid"))
        assert await asyncio.to_thread(started.wait, 2.0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(finished.wait, 2.0)

        retry = await env.step(_text_response("No action."), metadata, session_id="sid")
        info = retry[4]
        assert info["step_idx"] == 1
        assert env.session_state["sid"]["agent_steps"] == 1
        assert env.session_state["sid"]["n_steps"] == 1
        assert env.session_state["sid"]["protocol_violation_count"] == 1

    @pytest.mark.asyncio
    async def test_dataset_protocol_fallback_remains_diagnostic_only(self):
        env = _make_env(
            backend="dataset_replay",
            dataset_path=str(_SNAPSHOT_FIXTURE),
            protocol_violation_penalty=-1.0,
        )
        await env.reset({"scenario_id": "lab_run_a"}, session_id="dataset")

        _, _, _, _, info = await env.step(_text_response("No action."), {}, session_id="dataset")

        assert info["protocol_violation"] is True
        assert info["training_usable"] is False
        assert info["diagnostic_only"] is True
        await env.explicit_close("dataset")

    @pytest.mark.asyncio
    async def test_reward_accumulates_per_step_like_blackjack(self):
        # The server returns PER-STEP rewards (the agent sums them); the
        # session's cumulative bookkeeping must equal that sum.
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        total = 0.0
        for _ in range(3):
            _, reward, term, trunc, _ = await env.step(_tool_response("noop", {}), {}, session_id="sid")
            total += reward
            assert not term and not trunc
        assert env.session_state["sid"]["cumulative_reward"] == pytest.approx(total)
        assert env.session_state["sid"]["n_steps"] == 3

    @pytest.mark.asyncio
    async def test_episode_terminates_at_env_max_steps_and_session_closes(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        term = trunc = False
        for _ in range(3):
            _, _, term, trunc, _ = await env.step(_tool_response("noop", {}), {}, session_id="sid")
        assert term or trunc  # episode ended within the 3-step budget
        # Mirror the framework: /step calls close_session on terminated/truncated.
        await env.close_session("sid")
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_terminal_step_preserves_the_scored_after_observation(self):
        env = _make_env(agent_max_steps=1)
        before, _ = await env.reset(dict(_TASK_METADATA, max_steps=1), session_id="sid")

        after, reward, terminated, truncated, info = await env.step(
            _tool_response("noop", {}),
            {},
            session_id="sid",
        )

        assert before is not None
        assert math.isfinite(reward)
        assert terminated or truncated
        assert after is not None
        assert after != before
        assert info["step_idx"] == 1
        assert info["reward_measurements"]
        assert info["reward_terms"]["total"] == pytest.approx(reward)

    @pytest.mark.asyncio
    async def test_duplicate_step_request_returns_cached_transition(self):
        env = _make_env()
        _, reset_info = await env.reset(dict(_TASK_METADATA), session_id="sid")
        metadata = {"_ng_step_request_id": "turn-1"}

        first = await env.step(_tool_response("noop", {}), metadata, session_id="sid")
        second = await env.step(_tool_response("noop", {}), metadata, session_id="sid")

        assert second == first
        assert env.session_state["sid"]["episode_id"] == reset_info["episode_id"]
        assert env.session_state["sid"]["n_steps"] == 1

        await env.close_session("sid")
        assert await env.step(_tool_response("noop", {}), metadata, session_id="sid") == first

    @pytest.mark.asyncio
    async def test_active_step_request_id_rejects_a_different_action_without_advancing(self):
        env = _make_env(agent_max_steps=3)
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        metadata = {"_ng_step_request_id": "turn-1"}

        first = await env.step(_tool_response("noop", {}), metadata, session_id="sid")
        with pytest.raises(ValueError, match="different step payload"):
            await env.step(_text_response("No action."), metadata, session_id="sid")

        state = env.session_state["sid"]
        assert state["n_steps"] == 1
        assert state["protocol_violation_count"] == 0
        assert await env.step(_tool_response("noop", {}), metadata, session_id="sid") == first

    @pytest.mark.asyncio
    async def test_active_step_request_id_rejects_different_metadata_without_advancing(self):
        env = _make_env(agent_max_steps=3)
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")

        first_metadata = {
            "_ng_step_request_id": "turn-1",
            "trace_context": {"attempt": 1, "source": "agent"},
        }
        reordered_metadata = {
            "trace_context": {"source": "agent", "attempt": 1},
            "_ng_step_request_id": "turn-1",
        }
        conflicting_metadata = {
            "_ng_step_request_id": "turn-1",
            "trace_context": {"attempt": 2, "source": "agent"},
        }

        first = await env.step(_tool_response("noop", {}), first_metadata, session_id="sid")
        assert await env.step(_tool_response("noop", {}), reordered_metadata, session_id="sid") == first
        with pytest.raises(ValueError, match="different step payload"):
            await env.step(_tool_response("noop", {}), conflicting_metadata, session_id="sid")
        assert env.session_state["sid"]["n_steps"] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_value", [math.nan, math.inf, object()])
    async def test_step_request_id_rejects_non_json_metadata_before_advancing(self, invalid_value):
        env = _make_env(agent_max_steps=3)
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        metadata = {
            "_ng_step_request_id": "turn-1",
            "trace_context": invalid_value,
        }

        with pytest.raises(ValueError, match="finite JSON"):
            await env.step(_tool_response("noop", {}), metadata, session_id="sid")

        assert env.session_state["sid"]["n_steps"] == 0
        valid = await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "turn-1"},
            session_id="sid",
        )
        assert valid[4]["step_idx"] == 1

    @pytest.mark.asyncio
    async def test_step_request_id_rejects_a_nonfinite_action_before_advancing(self):
        env = _make_env(agent_max_steps=3)
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        action = _tool_response("noop", {}).model_copy(update={"created_at": math.nan})

        with pytest.raises(ValueError, match="finite JSON"):
            await env.step(
                action,
                {"_ng_step_request_id": "turn-1"},
                session_id="sid",
            )

        assert env.session_state["sid"]["n_steps"] == 0

    @pytest.mark.asyncio
    async def test_step_request_id_rejects_recursive_metadata_before_advancing(self):
        env = _make_env(agent_max_steps=3)
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        recursive: dict = {}
        recursive["self"] = recursive

        with pytest.raises(ValueError, match="finite JSON"):
            await env.step(
                _tool_response("noop", {}),
                {"_ng_step_request_id": "turn-1", "trace_context": recursive},
                session_id="sid",
            )

        assert env.session_state["sid"]["n_steps"] == 0

    @pytest.mark.asyncio
    async def test_nonadjacent_duplicate_step_request_returns_cached_transition(self):
        env = _make_env(agent_max_steps=4)
        await env.reset(dict(_TASK_METADATA, max_steps=4), session_id="sid")

        first = await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "turn-1"},
            session_id="sid",
        )
        await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "turn-2"},
            session_id="sid",
        )
        retry = await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "turn-1"},
            session_id="sid",
        )

        assert retry == first
        assert env.session_state["sid"]["n_steps"] == 2

    @pytest.mark.asyncio
    async def test_step_request_cache_is_bounded(self):
        env = _make_env(agent_max_steps=130)
        await env.reset(dict(_TASK_METADATA, max_steps=130), session_id="sid")

        for index in range(129):
            await env.step(
                _tool_response("noop", {}),
                {"_ng_step_request_id": f"turn-{index}"},
                session_id="sid",
            )

        assert len(env.session_state["sid"]["step_response_cache"]) == 128

    @pytest.mark.asyncio
    async def test_completed_step_retry_cache_is_bounded_after_sessions_close(self):
        env = _make_env(pool_size=1, agent_max_steps=1)
        first_result = last_result = None

        for index in range(129):
            session_id = f"session-{index}"
            request_id = f"turn-{index}"
            await env.reset(
                dict(_TASK_METADATA, seed=7001 + index, max_steps=1),
                session_id=session_id,
            )
            result = await env.step(
                _tool_response("noop", {}),
                {"_ng_step_request_id": request_id},
                session_id=session_id,
            )
            await env.close_session(session_id)
            if index == 0:
                first_result = result
            last_result = result

        evicted = await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "turn-0"},
            session_id="session-0",
        )
        retained = await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "turn-128"},
            session_id="session-128",
        )

        assert first_result is not None and last_result is not None
        assert evicted[4]["error"] == "no_active_episode"
        assert retained == last_result

    @pytest.mark.asyncio
    async def test_completed_step_retry_cache_expires_with_the_session_ttl(self):
        env = _make_env(pool_size=1, agent_max_steps=1, session_ttl_s=1.0)
        await env.reset(dict(_TASK_METADATA, max_steps=1), session_id="sid")
        await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "terminal-turn"},
            session_id="sid",
        )
        await env.close_session("sid")

        cache_key = ("sid", "terminal-turn")
        cached_fingerprint, cached_result, _ = env._completed_step_response_cache[cache_key]
        env._completed_step_response_cache[cache_key] = (
            cached_fingerprint,
            cached_result,
            time.monotonic() - 2.0,
        )
        expired = await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "terminal-turn"},
            session_id="sid",
        )

        assert expired[4]["error"] == "no_active_episode"

    @pytest.mark.asyncio
    async def test_completed_step_request_id_rejects_a_different_payload(self):
        env = _make_env(pool_size=1, agent_max_steps=1)
        await env.reset(dict(_TASK_METADATA, max_steps=1), session_id="sid")
        metadata = {"_ng_step_request_id": "terminal-turn"}
        first = await env.step(_tool_response("noop", {}), metadata, session_id="sid")
        await env.close_session("sid")

        with pytest.raises(ValueError, match="different step payload"):
            await env.step(_text_response("No action."), metadata, session_id="sid")

        assert await env.step(_tool_response("noop", {}), metadata, session_id="sid") == first

    @pytest.mark.asyncio
    async def test_close_clears_episode_cache_and_lifecycle_registry(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        await env.step(
            _tool_response("noop", {}),
            {"_ng_step_request_id": "turn-1"},
            session_id="sid",
        )
        state = env.session_state["sid"]
        assert state["step_response_cache"]

        await env.explicit_close("sid")

        assert not state["step_response_cache"]
        assert "sid" not in env._lifecycle_locks
        assert "sid" not in env._lifecycle_users

    @pytest.mark.asyncio
    async def test_step_without_reset_truncates_gracefully(self):
        env = _make_env()
        obs, reward, term, trunc, info = await env.step(_tool_response("noop", {}), {}, session_id="ghost")
        assert reward == 0.0
        assert trunc is True
        assert info["error"] == "no_active_episode"
        assert info["training_eligible"] is False
        assert info["rollout_usable"] is False
        assert info["training_usable"] is False

    @pytest.mark.asyncio
    async def test_none_session_is_rejected_before_close(self):
        env = _make_env()
        with pytest.raises(ValueError, match="session_id"):
            await env.explicit_close(session_id=None)

    @pytest.mark.asyncio
    async def test_failed_close_can_be_retried(self, monkeypatch):
        env = _make_env(pool_size=1)
        _, info = await env.reset(dict(_TASK_METADATA), session_id="session-a")
        original_close = env.backend.close
        calls = 0

        def flaky_close(episode_id):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("close failure")
            return original_close(episode_id)

        monkeypatch.setattr(env.backend, "close", flaky_close)
        with pytest.raises(RuntimeError, match="close failure"):
            await env.explicit_close("session-a")

        assert env.session_state["session-a"]["episode_id"] == info["episode_id"]
        result = await env.explicit_close("session-a")
        assert result["already_closed"] is False
        assert "session-a" not in env.session_state


class TestRoutes:
    def test_gymnasium_routes_registered(self):
        env = _make_env()
        routes = {r.path for r in env.setup_webserver().routes}
        assert {"/reset", "/step", "/close", "/aggregate_metrics"}.issubset(routes)

    @pytest.mark.asyncio
    async def test_explicit_close_route_is_cookie_scoped_and_idempotent(self):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()
        async with _http_client(app) as client:
            reset = await client.post("/reset", json=_reset_body())
            assert reset.status_code == 200
            first = await client.post("/close", json={})
            second = await client.post("/close", json={})

        assert first.status_code == 200
        assert first.json()["already_closed"] is False
        assert second.status_code == 200
        assert second.json()["already_closed"] is True

    @pytest.mark.asyncio
    async def test_duplicate_reset_request_without_cookie_reuses_the_original_episode(self):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()
        body = _reset_body(_ng_reset_request_id="reset-1")

        async with _http_client(app) as first_client, _http_client(app) as retry_client:
            first = await first_client.post("/reset", json=body)
            retry = await retry_client.post("/reset", json=body)
            first_owner_step = await first_client.post("/step", json=_step_body("noop", {}))
            await retry_client.post("/close", json={})

        assert first.status_code == 200
        assert retry.status_code == 200
        assert retry.json() == first.json()
        assert first_owner_step.status_code == 200
        assert first_owner_step.json()["info"].get("error") is None
        assert not env.session_state

    @pytest.mark.asyncio
    async def test_duplicate_reset_request_rejects_a_conflicting_payload(self):
        env = _make_env(pool_size=2)
        app = env.setup_webserver()

        async with (
            _http_client(app) as first_client,
            _http_client(
                app,
                raise_app_exceptions=False,
            ) as retry_client,
        ):
            first = await first_client.post(
                "/reset",
                json=_reset_body(seed=7001, _ng_reset_request_id="reset-conflict"),
            )
            conflict = await retry_client.post(
                "/reset",
                json=_reset_body(seed=7002, _ng_reset_request_id="reset-conflict"),
            )

        assert first.status_code == 200
        assert conflict.status_code == 400
        assert conflict.json() == {"detail": "_ng_reset_request_id was already used with a different reset payload"}
        assert len(env.session_state) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_id", ["", "x" * 129])
    async def test_invalid_reset_request_id_is_an_http_400(self, request_id):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()

        async with _http_client(app, raise_app_exceptions=False) as client:
            response = await client.post(
                "/reset",
                json=_reset_body(_ng_reset_request_id=request_id),
            )

        assert response.status_code == 400
        assert response.json() == {
            "detail": "_ng_reset_request_id must be a non-empty string of at most 128 characters"
        }
        assert not env.session_state

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overrides", "detail_fragment"),
        [
            ({"seed": "bad"}, "seed"),
            ({"seed": -1}, "seed"),
            ({"max_steps": 0}, "max_steps"),
            ({"difficulty": 2.0}, "difficulty"),
            ({"tier": "T2"}, "tier"),
            ({"regime_mix": {"unknown": 1.0}}, "regime_mix"),
        ],
    )
    async def test_invalid_reset_task_input_is_an_http_400(self, overrides, detail_fragment):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()

        async with _http_client(app, raise_app_exceptions=False) as client:
            response = await client.post("/reset", json=_reset_body(**overrides))

        assert response.status_code == 400
        assert detail_fragment in response.json()["detail"]
        assert not env.session_state
        assert not env.backend._open_episode_ids

    @pytest.mark.asyncio
    async def test_unknown_dataset_episode_is_an_http_400(self):
        env = _make_env(
            backend="dataset_replay",
            dataset_path=str(_SNAPSHOT_FIXTURE),
            pool_size=1,
        )
        app = env.setup_webserver()

        async with _http_client(app, raise_app_exceptions=False) as client:
            response = await client.post(
                "/reset",
                json=_reset_body(scenario_id="not-in-the-dataset"),
            )

        assert response.status_code == 400
        assert "scenario_id" in response.json()["detail"]
        assert not env.session_state
        assert not env.backend._episodes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_id", ["", "x" * 129])
    async def test_invalid_step_request_id_is_an_http_400(self, request_id):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()

        async with _http_client(app, raise_app_exceptions=False) as client:
            reset = await client.post("/reset", json=_reset_body(max_steps=2))
            response = await client.post(
                "/step",
                json={**_step_body("noop", {}), "_ng_step_request_id": request_id},
            )

        assert reset.status_code == 200
        assert response.status_code == 400
        assert response.json() == {
            "detail": "_ng_step_request_id must be a non-empty string of at most 128 characters"
        }
        assert next(iter(env.session_state.values()))["n_steps"] == 0

    @pytest.mark.asyncio
    async def test_conflicting_step_request_payload_is_an_http_400_without_advancing(self):
        env = _make_env(pool_size=1, agent_max_steps=3)
        app = env.setup_webserver()
        metadata = {"_ng_step_request_id": "turn-1"}

        async with _http_client(app, raise_app_exceptions=False) as client:
            reset = await client.post("/reset", json=_reset_body(max_steps=3))
            first = await client.post("/step", json={**_step_body("noop", {}), **metadata})
            conflict = await client.post("/step", json={**_step_body("set_prb_cap", {"cap_percent": 90}), **metadata})

        assert reset.status_code == 200
        assert first.status_code == 200
        assert conflict.status_code == 400
        assert conflict.json() == {"detail": "_ng_step_request_id was already used with a different step payload"}
        assert next(iter(env.session_state.values()))["n_steps"] == 1

    @pytest.mark.asyncio
    async def test_closed_reset_request_id_can_start_a_new_episode(self):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()
        body = _reset_body(_ng_reset_request_id="reset-reused-after-close")

        async with _http_client(app) as first_client, _http_client(app) as later_client:
            first = await first_client.post("/reset", json=body)
            closed = await first_client.post("/close", json={})
            later = await later_client.post("/reset", json=body)

        assert first.status_code == 200
        assert closed.status_code == 200
        assert later.status_code == 200
        assert later.json()["info"]["episode_id"] != first.json()["info"]["episode_id"]
        assert len(env.session_state) == 1

    @pytest.mark.asyncio
    async def test_terminal_step_retry_after_automatic_close_returns_the_original_transition(self):
        env = _make_env(pool_size=1, agent_max_steps=1)
        app = env.setup_webserver()
        step_body = {
            **_step_body("noop", {}),
            "_ng_step_request_id": "terminal-turn",
        }

        async with _http_client(app) as client:
            reset = await client.post("/reset", json=_reset_body(max_steps=1))
            first = await client.post("/step", json=step_body)
            retry = await client.post("/step", json=step_body)

        assert reset.status_code == 200
        assert first.status_code == 200
        assert first.json()["terminated"] or first.json()["truncated"]
        assert retry.status_code == 200
        assert retry.json() == first.json()
        assert not env.session_state

    @pytest.mark.asyncio
    async def test_concurrent_distinct_steps_cannot_advance_past_terminal_transition(self, monkeypatch):
        env = _make_env(pool_size=1, agent_max_steps=1)
        app = env.setup_webserver()
        original_step = env.backend.step
        first_started = threading.Event()
        release_first = threading.Event()
        backend_calls = 0

        def blocking_step(*args, **kwargs):
            nonlocal backend_calls
            backend_calls += 1
            if backend_calls == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)
            return original_step(*args, **kwargs)

        monkeypatch.setattr(env.backend, "step", blocking_step)

        async with _http_client(app) as client:
            reset = await client.post("/reset", json=_reset_body(max_steps=1))
            first_task = asyncio.create_task(
                client.post(
                    "/step",
                    json={**_step_body("noop", {}), "_ng_step_request_id": "terminal-a"},
                )
            )
            assert await asyncio.to_thread(first_started.wait, 2.0)
            second_task = asyncio.create_task(
                client.post(
                    "/step",
                    json={**_step_body("noop", {}), "_ng_step_request_id": "terminal-b"},
                )
            )
            await asyncio.sleep(0.02)
            release_first.set()
            first, second = await asyncio.gather(first_task, second_task)

        assert reset.status_code == 200
        assert first.status_code == 200
        assert second.status_code == 200
        assert backend_calls == 1
        assert first.json()["terminated"] or first.json()["truncated"]
        assert second.json()["terminated"] or second.json()["truncated"]
        assert second.json()["info"]["error"] in {"episode_complete", "no_active_episode"}
        assert second.json()["reward"] == 0.0
        assert not env.session_state

    @pytest.mark.asyncio
    async def test_delayed_terminal_retry_does_not_close_the_replacement_generation(self):
        env = _make_env(pool_size=1, agent_max_steps=2)
        app = env.setup_webserver()
        old_step_body = {
            **_step_body("noop", {}),
            "_ng_step_request_id": "old-terminal-turn",
        }

        async with _http_client(app) as client:
            await client.post(
                "/reset",
                json=_reset_body(max_steps=1, _ng_reset_request_id="old-reset"),
            )
            terminal = await client.post("/step", json=old_step_body)
            replacement = await client.post(
                "/reset",
                json=_reset_body(seed=7002, max_steps=2, _ng_reset_request_id="replacement-reset"),
            )
            delayed_retry = await client.post("/step", json=old_step_body)

        assert terminal.status_code == 200
        assert terminal.json()["terminated"] or terminal.json()["truncated"]
        assert replacement.status_code == 200
        assert delayed_retry.json() == terminal.json()
        session_state = next(iter(env.session_state.values()))
        assert session_state["episode_id"] == replacement.json()["info"]["episode_id"]
        assert session_state["n_steps"] == 0

    @pytest.mark.asyncio
    async def test_old_terminal_step_cannot_close_replacement_episode(self, monkeypatch):
        env = _make_env(pool_size=1, agent_max_steps=1)
        app = env.setup_webserver()
        original_close_session = type(env).close_session
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        async def delayed_close_session(self, session_id, **kwargs):
            close_started.set()
            await release_close.wait()
            return await original_close_session(self, session_id, **kwargs)

        monkeypatch.setattr(type(env), "close_session", delayed_close_session)
        async with _http_client(app) as client:
            reset_response = await client.post("/reset", json=_reset_body(max_steps=1))
            session_id = next(iter(env.session_state))
            terminal_step = asyncio.create_task(client.post("/step", json=_step_body("noop", {})))
            await asyncio.wait_for(close_started.wait(), timeout=2.0)
            _, replacement_info = await env.reset(
                dict(_TASK_METADATA, seed=7002, max_steps=1),
                session_id=session_id,
            )
            release_close.set()
            response = await terminal_step

        assert reset_response.status_code == 200
        assert response.status_code == 200
        assert response.json()["terminated"] or response.json()["truncated"]
        assert env.session_state[session_id]["episode_id"] == replacement_info["episode_id"]

    @pytest.mark.asyncio
    async def test_terminal_response_survives_automatic_close_failure(self, monkeypatch):
        env = _make_env(pool_size=1, agent_max_steps=1)
        app = env.setup_webserver()
        original_close = env.backend.close

        def failing_close(episode_id):
            raise RuntimeError("close failure")

        monkeypatch.setattr(env.backend, "close", failing_close)
        async with _http_client(app) as client:
            await client.post("/reset", json=_reset_body(max_steps=1))
            session_id = next(iter(env.session_state))
            response = await client.post("/step", json=_step_body("noop", {}))

        assert response.status_code == 200
        assert response.json()["terminated"] or response.json()["truncated"]
        assert session_id in env.session_state

        monkeypatch.setattr(env.backend, "close", original_close)
        result = await env.explicit_close(session_id)
        assert result["already_closed"] is False
        assert session_id not in env.session_state


def _http_client(app, *, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
    # In-process ASGI transport (as in aviary's tests): real /reset and /step
    # requests through routing, request parsing, and the session middleware,
    # no live socket. Each AsyncClient keeps its own cookie jar, so each
    # client is one session.
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions),
        base_url="http://testserver",
    )


def _reset_body(**overrides) -> dict:
    # EnvResetRequest: responses_create_params plus task-row extras.
    body = {"responses_create_params": {"input": []}, **_TASK_METADATA}
    body.update(overrides)
    return body


def _step_body(name: str, arguments: dict) -> dict:
    # EnvStepRequest: responses_create_params plus the model's response.
    return {"responses_create_params": {"input": []}, "response": _tool_response(name, arguments).model_dump()}


class TestHTTPSurface:
    @pytest.mark.asyncio
    async def test_interleaved_sessions_step_independently_over_http(self):
        # Two clients (= two session cookies) with interleaved /step calls:
        # each must advance only its own episode, with no state bleed.
        env = _make_env()
        app = env.setup_webserver()
        async with _http_client(app) as client_a, _http_client(app) as client_b:
            episode_a = (await client_a.post("/reset", json=_reset_body())).json()["info"]["episode_id"]
            episode_b = (await client_b.post("/reset", json=_reset_body(seed=7002))).json()["info"]["episode_id"]
            assert episode_a != episode_b
            assert len(env.session_state) == 2
            expected = {id(client_a): episode_a, id(client_b): episode_b}
            steps_taken = {id(client_a): 0, id(client_b): 0}
            for client in (client_a, client_b, client_a, client_b, client_a):
                response = await client.post("/step", json=_step_body("noop", {}))
                assert response.status_code == 200
                info = response.json()["info"]
                steps_taken[id(client)] += 1
                assert info["episode_id"] == expected[id(client)]
                assert info["n_steps"] == steps_taken[id(client)]

    @pytest.mark.asyncio
    async def test_same_task_row_twice_yields_identical_episode_over_http(self):
        # Offline determinism at the HTTP surface: the same task row and
        # action sequence must reproduce the observation and reward sequence
        # exactly (fresh session each run; episode_ids differ, so info is
        # excluded from the comparison).
        env = _make_env()
        app = env.setup_webserver()
        actions = [
            ("set_ul_power_control", {"cell_id": 0, "p0_dbm": -90, "alpha": 0.8}),
            ("noop", {}),
            ("set_prb_cap", {"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 120}),
        ]

        async def run_episode() -> list:
            async with _http_client(app) as client:
                trace = [(await client.post("/reset", json=_reset_body())).json()["observation"]]
                for name, arguments in actions:
                    body = (await client.post("/step", json=_step_body(name, arguments))).json()
                    trace.append((body["observation"], body["reward"], body["terminated"], body["truncated"]))
                return trace

        assert await run_episode() == await run_episode()

    @pytest.mark.asyncio
    async def test_pool_exhaustion_reaps_orphans_over_http(self):
        # HTTP counterpart of the in-process reaper test: with a live session
        # holding the only slot, a second /reset fails pool-exhausted; once
        # that crashed session's lease expires, the reaper reclaims its slot
        # and the retry succeeds.
        env = _make_env(pool_size=1, session_ttl_s=1.0)
        app = env.setup_webserver()
        async with _http_client(app) as client_dead, _http_client(app) as client_new:
            episode_dead = (await client_dead.post("/reset", json=_reset_body())).json()["info"]["episode_id"]
            # The server registers no exception middleware, so the pool-exhausted
            # RuntimeError tunnels through the in-process ASGI transport; a
            # client on a real socket would see a 500 instead.
            with pytest.raises(RuntimeError, match="pool exhausted"):
                await client_new.post("/reset", json=_reset_body(seed=7002))
            dead_session_id = next(iter(env.session_state))
            env.session_state[dead_session_id]["last_activity_monotonic"] = time.monotonic() - 2.0
            response = await client_new.post("/reset", json=_reset_body(seed=7002))
            assert response.status_code == 200
            info = response.json()["info"]
            assert info["episode_id"] != episode_dead
            assert env.session_state[next(iter(env.session_state))]["episode_id"] == info["episode_id"]


class TestBackends:
    def test_close_failure_keeps_episode_tracked(self, monkeypatch):
        backend = ReplayBackend(pool_size=1, max_steps_default=2)
        _, meta = backend.reset(dict(_TASK_METADATA, max_steps=2))

        def _raise_close_error(episode_id):
            raise RuntimeError("close failure")

        monkeypatch.setattr(backend._env, "close", _raise_close_error)
        with pytest.raises(RuntimeError, match="close failure"):
            backend.close(meta.episode_id)

        assert meta.episode_id in backend._open_episode_ids

    def test_select_backend_defaults_to_replay(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="")
        assert isinstance(select_backend(config), ReplayBackend)

    def test_select_backend_rejects_unknown_name(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(
            host="", port=0, entrypoint="", name="", backend="flexric_dreams"
        )
        with pytest.raises(ValueError, match="unknown backend"):
            select_backend(config)

    def test_unimplemented_oai_collector_is_not_selectable(self):
        with pytest.raises(ValueError, match="not implemented.*supported backends"):
            select_backend(type("Config", (), {"backend": "oai_collector"})())
