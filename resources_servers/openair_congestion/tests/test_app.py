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
    @pytest.mark.parametrize("penalty", [0.0, 1.0, math.nan, math.inf, -math.inf])
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
            pytest.param(_raw_tool_response("noop", "["), "invalid_tool_call", id="malformed"),
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
        ],
    )
    async def test_protocol_violation_applies_penalized_noop_transition(self, action, expected_error):
        noop_env = _make_env(pool_size=1, protocol_violation_penalty=-1.0)
        await noop_env.reset(dict(_TASK_METADATA), session_id="noop")
        _, noop_reward, noop_term, noop_trunc, _ = await noop_env.step(
            _tool_response("noop", {}), {}, session_id="noop"
        )
        await noop_env.explicit_close("noop")

        env = _make_env(pool_size=1, protocol_violation_penalty=-1.0)
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(action, {}, session_id="sid")

        assert obs is not None
        assert reward == pytest.approx(noop_reward - 1.0)
        assert reward < noop_reward
        assert (term, trunc) == (noop_term, noop_trunc) == (False, False)
        assert info["error"] == expected_error
        assert info["protocol_violation"] is True
        assert info["protocol_rejection"] is True
        assert info["applied_fallback_action"] == {"name": "noop", "arguments": {}}
        assert info["rejection_reason"]
        assert info["n_steps"] == 1
        assert info["cumulative_reward"] == pytest.approx(reward)
        assert info["reward_terms"]["protocol_violation"] == pytest.approx(-1.0)
        assert info["reward_terms"]["total"] == pytest.approx(reward)
        assert info["backend"] == "replay"
        assert info["action_affects_observation"] is True
        assert info["reward_profile"] == "openair_v1"
        assert info["reward_weights"]
        assert info["observation_render"] == "openair_natural_language_v1"
        assert "sid" in env.session_state
        await env.explicit_close("sid")

    @pytest.mark.asyncio
    async def test_protocol_violation_cannot_beat_complete_noop_episode(self):
        async def episode_return(first_action: NeMoGymResponse) -> float:
            env = _make_env(protocol_violation_penalty=-1.0, agent_max_steps=4)
            await env.reset(dict(_TASK_METADATA, max_steps=4), session_id="sid")
            total = 0.0
            action = first_action
            while True:
                _, reward, term, trunc, _ = await env.step(action, {}, session_id="sid")
                total += reward
                if term or trunc:
                    break
                action = _tool_response("noop", {})
            await env.explicit_close("sid")
            return total

        noop_total = await episode_return(_tool_response("noop", {}))
        invalid_total = await episode_return(_text_response("I cannot choose an action."))

        assert invalid_total == pytest.approx(noop_total - 1.0)
        assert invalid_total < noop_total

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


def _http_client(app) -> httpx.AsyncClient:
    # In-process ASGI transport (as in aviary's tests): real /reset and /step
    # requests through routing, request parsing, and the session middleware,
    # no live socket. Each AsyncClient keeps its own cookie jar, so each
    # client is one session.
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


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
