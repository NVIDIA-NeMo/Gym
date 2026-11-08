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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from aiohttp import ClientResponseError
from fastapi import Response
from fastapi.testclient import TestClient

import responses_api_agents.scicode_agent.app as app
from nemo_gym.agents import AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.client import raise_for_status as real_raise_for_status
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.core import (
    EpisodeId,
    FailureBoundary,
    FailureHeader,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    RequestFailureLog,
    TaskId,
    failure_report_for,
)
from responses_api_agents.scicode_agent.app import (
    SCICODE_RESPONSE_KEY,
    TOKEN_USAGE_VERSION,
    ModelServerRef,
    ResourcesServerRef,
    ScicodeAgent,
    ScicodeAgentConfig,
    ScicodeAgentRunRequest,
)
from responses_api_agents.scicode_agent.step_utils import (
    PREFILLED_STEPS_CODE,
    extract_python_script,
    is_context_window_error,
    process_problem_steps,
)


_PROMPT_FPATH = "benchmarks/scicode/prompts/background.yaml"


def _config():
    return ScicodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="scicode_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="scicode"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        prompt_fpath=_PROMPT_FPATH,
    )


def _agent(observability_enabled=False):
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": observability_enabled}
    return ScicodeAgent(config=_config(), server_client=server_client)


def _model_json(code: str) -> dict:
    return {
        "id": "r",
        "created_at": 0.0,
        "model": "d",
        "object": "response",
        "output": [
            {
                "id": "m",
                "content": [{"annotations": [], "text": f"```python\n{code}\n```", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class _Resp:
    def __init__(self, payload, cookies=None):
        self._payload = payload
        self.cookies = cookies or {}

    async def json(self):
        return self._payload

    async def read(self):
        return orjson.dumps(self._payload)


class _FakeRequest:
    cookies: dict = {}


def _run_request(problem_id="1", n_steps=2, **extra):
    sub_steps = [
        {
            "step_number": f"{problem_id}.{i + 1}",
            "step_description_prompt": f"desc {i}",
            "step_background": f"bg {i}",
            "function_header": f"def f{i}():",
            "return_line": "return None",
            "test_cases": ["assert True"],
        }
        for i in range(n_steps)
    ]
    return ScicodeAgentRunRequest(
        responses_create_params={"input": []},
        problem_id=problem_id,
        sub_steps=sub_steps,
        required_dependencies="import numpy as np",
        uuid=problem_id,
        **extra,
    )


# ----------------------------
# step_utils helpers
# ----------------------------
def test_extract_python_script_python_fence_strips_imports():
    assert extract_python_script("pre\n```python\nimport numpy as np\nx = 1\n```\npost") == "\nx = 1\n"


def test_extract_python_script_generic_fence():
    assert extract_python_script("```\ny = 2\n```") == "\ny = 2\n"


def test_extract_python_script_no_fence():
    assert extract_python_script("z = 3") == "z = 3"


def test_process_problem_steps_with_and_without_background():
    sub_steps = [
        {
            "step_description_prompt": "D0",
            "step_background": "B0",
            "function_header": "def f0():",
            "return_line": "r0",
        },
        {
            "step_description_prompt": "D1",
            "step_background": "B1",
            "function_header": "def f1():",
            "return_line": "r1",
        },
    ]
    prev = ["code0", None]
    ps_bg, ns_bg, prevcode = process_problem_steps(sub_steps, 1, prev, with_background=True)
    assert "B0" in ps_bg and "B1" in ns_bg and "def f1()" in ns_bg and prevcode == "code0"
    ps_no, ns_no, _ = process_problem_steps(sub_steps, 1, prev, with_background=False)
    assert "B0" not in ps_no and "B1" not in ns_no


def test_is_context_window_error():
    assert is_context_window_error(Exception("... exceeds maximum input length ...")) is True
    assert is_context_window_error(Exception("some other error")) is False


def test_prefilled_steps_present():
    assert set(PREFILLED_STEPS_CODE.keys()) == {("13", 5), ("62", 0), ("76", 2)}


# ----------------------------
# agent
# ----------------------------
class TestApp:
    def test_sanity(self):
        _agent()

    def test_config_defaults(self):
        assert _config().with_background is True

    @pytest.mark.asyncio
    async def test_responses_forwards_to_model(self):
        agent = _agent()
        agent.server_client.post = AsyncMock(return_value=_Resp(_model_json("x = 1"), cookies={"sid": "abc"}))
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi", temperature=0.25)
        with patch.object(app, "raise_for_status", AsyncMock()):
            result = await agent.responses(_FakeRequest(), Response(), body)
        assert result.output_text == "```python\nx = 1\n```"
        request_json = agent.server_client.post.await_args.kwargs["json"]
        assert isinstance(request_json, dict)
        assert request_json["temperature"] == 0.25
        assert request_json["input"][0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_responses_propagates_rollout_path_to_model(self):
        agent = _agent(observability_enabled=True)
        agent.server_client.post = AsyncMock(return_value=_Resp(_model_json("x = 1")))
        request = _FakeRequest()
        request.path_params = {"rollout_id": "4-2"}
        with patch.object(app, "raise_for_status", AsyncMock()):
            await agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="hi"))
        assert agent.server_client.post.await_args.kwargs["url_path"] == "/ng-rollout/4-2/v1/responses"

    @pytest.mark.asyncio
    async def test_run_correlates_every_substep_with_complete_request_payload(self):
        agent = _agent(observability_enabled=True)
        captured = []

        def _post(server_name, url_path, json, cookies):
            if url_path.endswith("/v1/responses"):
                captured.append((url_path, json))
                return _Resp(_model_json("x = 1"))
            return _Resp({"reward": 1.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        body = _run_request(problem_id="1", n_steps=3, _ng_task_index=4, _ng_rollout_index=2)
        with patch.object(app, "raise_for_status", AsyncMock()):
            await agent.run(_FakeRequest(), body)

        assert len(captured) == 3
        for url_path, request_json in captured:
            assert url_path == "/ng-rollout/4-2/v1/responses"
            assert request_json["input"]
            assert request_json["tools"] == []
            assert request_json["model"] == "policy_model"

    @pytest.mark.asyncio
    async def test_run_builds_solutions_and_calls_verify(self):
        agent = _agent()
        captured = {"model": []}

        def _post(server_name, url_path, json, cookies):
            if url_path == "/v1/responses":
                captured["model"].append(json)
                return _Resp(_model_json("x = 1"))
            captured["verify"] = json
            return _Resp({"reward": 1.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        body = _run_request(problem_id="1", n_steps=2)
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[],
            max_output_tokens=4096,
            temperature=0.25,
            top_p=0.9,
        )
        with patch.object(app, "raise_for_status", AsyncMock()):
            result = await agent.run(_FakeRequest(), body)

        assert result == {"reward": 1.0}
        assert len(captured["model"]) == 2
        for request_json in captured["model"]:
            assert request_json["max_output_tokens"] == 4096
            assert request_json["temperature"] == 0.25
            assert request_json["top_p"] == 0.9
            assert request_json["input"][0]["role"] == "user"
            assert request_json["input"][0]["content"]
        verify = captured["verify"]
        assert "response" in verify  # /verify requires a response field
        solutions = verify["solutions"]
        assert set(solutions.keys()) == {"1.1", "1.2"}
        assert "x = 1" in solutions["1.1"]

    @pytest.mark.asyncio
    async def test_run_skips_prefilled_step(self):
        # Problem "62" has a prefilled step at index 0 -> no model call, no solution entry for it.
        agent = _agent()
        captured = {}
        model_calls = 0

        def _post(server_name, url_path, json, cookies):
            nonlocal model_calls
            if url_path == "/v1/responses":
                model_calls += 1
                return _Resp(_model_json("x = 1"))
            captured["verify"] = json
            return _Resp({"reward": 0.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        with patch.object(app, "raise_for_status", AsyncMock()):
            await agent.run(_FakeRequest(), _run_request(problem_id="62", n_steps=2))

        assert model_calls == 1  # only the non-prefilled step is generated
        assert set(captured["verify"]["solutions"].keys()) == {"62.2"}

    @pytest.mark.asyncio
    async def test_run_uses_row_prefill_before_legacy_fallback(self):
        agent = _agent()
        captured = {"model": []}

        def _post(server_name, url_path, json, cookies):
            if url_path == "/v1/responses":
                captured["model"].append(json)
                return _Resp(_model_json("x = 1") | {"usage": _usage(100)})
            captured["verify"] = json
            return _Resp({"reward": 1.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        body = _run_request(problem_id="62", n_steps=2)
        body.prefilled_steps_code = {"62.1": "class OfficialVerifiedBlock:\n    pass"}
        with patch.object(app, "raise_for_status", AsyncMock()):
            await agent.run(_FakeRequest(), body)

        assert len(captured["model"]) == 1
        prompt = captured["model"][0]["input"][0]["content"]
        assert "class OfficialVerifiedBlock" in prompt
        assert "self.operator_dict = operator_dict" not in prompt
        assert set(captured["verify"]["solutions"]) == {"62.2"}
        assert [step["status"] for step in captured["verify"]["step_usage"]] == ["prefilled", "generated"]
        assert captured["verify"]["step_usage"][1]["usage"] == _usage(100)

    @pytest.mark.asyncio
    async def test_run_context_window_sentinels_remaining_steps(self):
        agent = _agent()
        captured = {}

        def _post(server_name, url_path, json, cookies):
            if url_path == "/v1/responses":
                raise RuntimeError("... exceeds maximum input length ...")
            captured["verify"] = json
            return _Resp({"reward": 0.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        with patch.object(app, "raise_for_status", AsyncMock()):
            await agent.run(_FakeRequest(), _run_request(problem_id="1", n_steps=2))

        solutions = captured["verify"]["solutions"]
        assert solutions == {"1.1": "_ran_out_of_context_", "1.2": "_ran_out_of_context_"}

    @pytest.mark.asyncio
    async def test_run_reraises_non_context_error(self):
        agent = _agent()

        def _post(server_name, url_path, json, cookies):
            raise RuntimeError("boom")  # not a context-window error -> should propagate

        agent.server_client.post = AsyncMock(side_effect=_post)
        with patch.object(app, "raise_for_status", AsyncMock()), pytest.raises(RuntimeError, match="boom"):
            await agent.run(_FakeRequest(), _run_request(problem_id="1", n_steps=1))


def _usage(output_tokens, input_tokens=10, cached_tokens=2, reasoning_tokens=3):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


async def _collect_usage(usages, problem_id="1", n_steps=None, response_overrides=None):
    """Exercise the step loop and capture the payload sent to verification."""
    agent = _agent()
    responses = iter(usages)
    overrides = iter(response_overrides) if isinstance(response_overrides, list) else None

    def _post(server_name, url_path, json, cookies):
        if url_path == "/v1/responses":
            usage = next(responses)
            if isinstance(usage, Exception):
                raise usage
            extra = next(overrides) if overrides is not None else (response_overrides or {})
            return _Resp({**_model_json("x = 1"), "usage": usage, **extra})
        return _Resp({**json, "reward": 0.0})

    agent.server_client.post = AsyncMock(side_effect=_post)
    with patch.object(app, "raise_for_status", AsyncMock()):
        return await agent.run(
            _FakeRequest(), _run_request(problem_id=problem_id, n_steps=len(usages) if n_steps is None else n_steps)
        )


class TestTokenAccounting:
    @pytest.mark.asyncio
    async def test_usage_sent_to_verifier_sums_all_steps_and_details(self):
        row = await _collect_usage([_usage(100), _usage(300, input_tokens=20)])
        assert row["token_usage_version"] == TOKEN_USAGE_VERSION
        assert row["response"]["usage"] == _usage(400, input_tokens=30, cached_tokens=4, reasoning_tokens=6)
        assert [{k: s[k] for k in ("step_number", "status", "usage")} for s in row["step_usage"]] == [
            {"step_number": "1.1", "status": "generated", "usage": _usage(100)},
            {"step_number": "1.2", "status": "generated", "usage": _usage(300, input_tokens=20)},
        ]
        assert row["response"]["output"][0]["content"][0]["text"] == "```python\nx = 1\n```"

    @pytest.mark.asyncio
    async def test_prefilled_and_context_skipped_steps_are_not_generated(self):
        row = await _collect_usage(
            [_usage(100), RuntimeError("exceeds maximum input length")], problem_id="62", n_steps=4
        )
        assert [s["status"] for s in row["step_usage"]] == [
            "prefilled",
            "generated",
            "context_window_exceeded",
            "skipped",
        ]
        assert row["response"]["usage"]["output_tokens"] == 100
        for i in (0, 2, 3):
            assert row["step_usage"][i]["usage"] == _usage(0, input_tokens=0, cached_tokens=0, reasoning_tokens=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("context_rejected", [False, True])
    async def test_local_context_rejection_is_distinct_from_zero_token_truncation(self, context_rejected):
        zero = _usage(0, input_tokens=0, cached_tokens=0, reasoning_tokens=0)
        row = await _collect_usage(
            [_usage(100), zero],
            n_steps=3 if context_rejected else 2,
            response_overrides=[
                {},
                {
                    "output": [],
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "metadata": {"nemo_gym_context_window_exceeded": "true"} if context_rejected else {},
                },
            ],
        )
        assert row["step_usage"][1]["status"] == ("context_window_exceeded" if context_rejected else "generated")
        if context_rejected:
            assert row["step_usage"][2]["status"] == "skipped"
        assert row["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 2
        assert row["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 3
        assert row["response"]["usage"]["output_tokens"] == 100

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_index", [0, 1])
    async def test_missing_usage_stays_unknown_in_rollout(self, missing_index):
        usages = [_usage(100), _usage(300)]
        usages[missing_index] = None
        unknown = await _collect_usage(usages)
        assert unknown["response"]["usage"] is None
        assert unknown["step_usage"][missing_index]["usage"] is None

    @pytest.mark.asyncio
    async def test_aggregate_metrics_come_from_the_resources_server(self):
        # The resources server owns the SciCode metrics, so the relay path reports what an environment server does.
        agent = _agent()
        metrics = {"agent_metrics": {"subtask_accuracy": 0.5}, "key_metrics": {"subtask_accuracy": 0.5}}
        agent.server_client.post = AsyncMock(return_value=_Resp(metrics))
        body = AggregateMetricsRequest(verify_responses=[{"reward": 0.0, "num_steps_passed": 1, "num_steps_total": 2}])
        with patch.object(app, "raise_for_status", AsyncMock()):
            result = await agent.aggregate_metrics(body)
        assert result.key_metrics == {"subtask_accuracy": 0.5}
        call = agent.server_client.post.await_args
        assert (call.kwargs["server_name"], call.kwargs["url_path"]) == ("scicode", "/aggregate_metrics")
        assert call.kwargs["json"] == body

    @pytest.mark.asyncio
    async def test_unknown_reasoning_details_do_not_poison_known_output_count(self):
        row = await _collect_usage([_usage(100, reasoning_tokens=None), _usage(200)])
        assert row["response"]["usage"]["output_tokens"] == 300
        assert row["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["empty", "prefilled", "context"])
    async def test_no_generations_zero_usage_with_fixed_subproblem_denominator(self, kind):
        if kind == "prefilled":
            row = await _collect_usage([], problem_id="62", n_steps=1)
        elif kind == "context":
            row = await _collect_usage([RuntimeError("exceeds maximum input length")], n_steps=2)
        else:
            row = await _collect_usage([])
        assert row["response"]["usage"]["output_tokens"] == 0
        assert all(step["usage"]["total_tokens"] == 0 for step in row["step_usage"])

    @pytest.mark.asyncio
    async def test_truncated_generation_counts_all_usage(self):
        row = await _collect_usage(
            [_usage(262000, input_tokens=144)],
            response_overrides={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        )
        assert row["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
        assert row["step_usage"][0]["status"] == "generated"
        assert row["response"]["usage"] == _usage(262000, input_tokens=144)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["not Python code", "```python\nthis is ! invalid\n```", "```python\n", ""])
    async def test_unusable_code_does_not_stop_later_steps_or_drop_usage(self, text):
        output = _model_json("ignored")["output"]
        output[0]["content"][0]["text"] = text
        row = await _collect_usage([_usage(100), _usage(200)], response_overrides={"output": output})
        assert set(row["solutions"]) == {"1.1", "1.2"}
        assert [step["status"] for step in row["step_usage"]] == ["generated", "generated"]
        assert row["response"]["usage"]["output_tokens"] == 300


@pytest.mark.asyncio
@pytest.mark.parametrize("prefilled", [False, True])
async def test_collector_builds_perf_from_generated_substeps(prefilled):
    from nemo_gym.rollouts.rollout_perf import _attach_ng_perf
    from nemo_gym.rollouts.trajectory_builder import _attach_trajectory_record

    agent = _agent(observability_enabled=True)
    calls = []
    problem_id = "62" if prefilled else "1"
    body = _run_request(problem_id=problem_id, n_steps=3, _ng_task_index=0, _ng_rollout_index=0)

    def post(server_name, url_path, json, cookies):
        if url_path.endswith("/v1/responses"):
            response_id = f"r{len(calls)}"
            calls.append(
                {
                    "model_call_id": f"c{len(calls)}",
                    "response_id": response_id,
                    "model_ref": agent.config.model_server.model_dump(),
                    "tokens_in": 10,
                    "tokens_out": 20,
                }
            )
            return _Resp({**_model_json("x = 1"), "id": response_id})
        return _Resp({"reward": 0.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    with patch.object(app, "raise_for_status", AsyncMock()):
        result = await agent.run(_FakeRequest(), body)
    result["ng_model_call_capture"] = {"calls": calls}
    _attach_trajectory_record(body.model_dump(), result)
    _attach_ng_perf(result, observability_enabled=True, rollout_latency_ms=123)
    expected = 2 if prefilled else 3
    assert result["ng_perf"] == {
        "num_turns": expected,
        "num_tool_calls": 0,
        "token_observability_coverage": 1.0,
        "prompt_tokens": expected * 10,
        "completion_tokens": expected * 20,
        "total_latency_ms": 123,
    }
    assert [t["turn_no"] for t in result["ng_trajectory"]["turns"]] == list(range(1, expected + 1))
    # An unmatched capture remains a missing contribution, not an invented count.
    result["ng_trajectory"]["model_calls"].pop()
    _attach_ng_perf(result, observability_enabled=True)
    assert result["ng_perf"]["token_observability_coverage"] == (expected - 1) / expected
    assert result["ng_perf"]["completion_tokens"] == (expected - 1) * 20


class TestStepStopMetadata:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first",
        [
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            {"status": "completed"},
            {"status": "failed", "error": {"code": "server_error", "message": "failed"}},
            {},
        ],
    )
    async def test_earlier_stop_evidence_survives_completed_final_step(self, first):
        row = await _collect_usage(
            [_usage(100), _usage(200)],
            response_overrides=[{"id": "first", **first}, {"id": "last", "status": "completed"}],
        )
        assert row["response"]["status"] == "completed"
        assert row["response"]["id"] == "last"
        assert row["response"]["usage"]["output_tokens"] == 300
        step = row["step_usage"][0]
        assert step["response_id"] == "first"
        assert step["response_status"] == first.get("status")
        assert step["incomplete_details"] == first.get("incomplete_details")
        assert step["error"] == first.get("error")
        assert row["step_usage"][1]["response_status"] == "completed"

    @pytest.mark.asyncio
    async def test_non_generated_steps_have_no_response_metadata(self):
        row = await _collect_usage(
            [_usage(100), Exception("exceeds maximum input length")],
            problem_id="62",
            n_steps=4,
            response_overrides={"status": "completed"},
        )
        assert [step["status"] for step in row["step_usage"]] == [
            "prefilled",
            "generated",
            "context_window_exceeded",
            "skipped",
        ]
        for i in (0, 2, 3):
            assert "response_status" not in row["step_usage"][i]
            assert "response_id" not in row["step_usage"][i]


def _failed_verify_reply() -> MagicMock:
    """A 500 /verify reply whose failure report names the resources server's verifier error, as on the wire."""
    failure = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="verifier_error",
        failure_reason="Verification failed: ScicodeError",
        operation="verify",
        stage="verification",
        http_status=500,
    )
    request_info = SimpleNamespace(url="http://server", method="POST", headers={}, real_url="http://server")
    error = ClientResponseError(
        request_info,
        (),
        status=500,
        message="Internal Server Error",
        headers=FailureHeader.encode(failure_report_for(failure)),
    )
    reply = MagicMock(status=500, ok=False, cookies={})
    reply.content.read = AsyncMock(return_value=b"Internal Server Error")
    reply.raise_for_status = MagicMock(side_effect=error)
    return reply


@pytest.mark.asyncio
async def test_run_masks_the_row_when_verify_fails_and_keeps_the_answer():
    agent = _agent()
    failed = _failed_verify_reply()

    def _post(server_name, url_path, json, cookies):
        return _Resp(_model_json("x = 1")) if url_path == "/v1/responses" else failed

    async def _raise_for_status(response):
        if not isinstance(response, _Resp):
            await real_raise_for_status(response)

    agent.server_client.post = AsyncMock(side_effect=_post)
    with patch.object(app, "raise_for_status", _raise_for_status), RequestFailureLog.capture() as log:
        result = await agent.run(_FakeRequest(), _run_request(problem_id="1", n_steps=1))

    assert (result["reward"], result["mask_sample"], result["failure_kind"]) == (0.0, True, "judge_failed")
    assert set(result["solutions"]) == {"1.1"}
    assert "verifier_error" in {event.failure_kind for event in log.report.events.values()}


# ---------------------------------------------------------------------------
# Agent sessions: the same step loop inside one /v1/responses activation
# ---------------------------------------------------------------------------

_PROMPT_KEYS = ("step_number", "step_description_prompt", "step_background", "function_header", "return_line")


def _agent_context(body: ScicodeAgentRunRequest) -> dict:
    """The row's agent_context values: the prompt material of the problem, without its tests."""
    return {
        "problem_id": body.problem_id,
        "sub_steps": [{key: step[key] for key in _PROMPT_KEYS} for step in body.sub_steps],
        "required_dependencies": body.required_dependencies,
        "prefilled_steps_code": body.prefilled_steps_code,
    }


def _seed(context: dict | None) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="scicode-session",
        episode_id=EpisodeId(rollout_id="scicode-rollout", attempt=0),
        task_id=TaskId(taskset="scicode", task_id="1"),
        agent_context={"values": context} if context is not None else None,
    )


def _wire(value):
    """A request body as it crosses HTTP."""
    return orjson.loads(orjson.dumps(value, default=lambda item: item.model_dump(mode="json")))


class _ScriptedPolicy:
    """Answers the n-th model call with `x = n`; a prompt that holds `overflow` gets a context-window error."""

    def __init__(self):
        self.requests = []

    def reply(self, body: dict) -> _Resp:
        self.requests.append(body)
        index = len(self.requests) - 1
        if "overflow" in body["input"][0]["content"]:
            raise RuntimeError("exceeds maximum input length")
        return _Resp({**_model_json(f"x = {index}"), "id": f"r{index}", "usage": _usage(10 * (index + 1))})


class TestAgentSessions:
    @pytest.mark.parametrize("problem", ["missing", "with_tests", "sandbox"])
    def test_seed_refuses_a_session_it_cannot_run(self, problem):
        context = _agent_context(_run_request(n_steps=1))
        seed = _seed(None if problem == "missing" else context)
        if problem == "with_tests":
            # Test cases are the verifier's; they never reach the agent.
            seed = _seed(context | {"sub_steps": [context["sub_steps"][0] | {"test_cases": ["assert True"]}]})
        elif problem == "sandbox":
            seed = AgentSeedSessionRequest.model_validate(
                seed.model_dump(mode="json")
                | {
                    "sandbox_access": {
                        "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {}},
                        "workdir": "/workspace",
                    }
                }
            )
        agent = _agent()
        detail = check_agent_seed_rejected(agent.setup_webserver(), seed)
        assert detail
        assert not agent._session_records

    @pytest.mark.parametrize(
        ("problem_id", "n_steps", "overflow_step"),
        [("1", 3, None), ("62", 3, None), ("1", 4, 2)],
        ids=["three_steps", "prefilled_first_step", "context_window_exceeded"],
    )
    def test_session_sends_the_run_requests_and_verify_payload(self, problem_id, n_steps, overflow_step):
        body = _run_request(problem_id=problem_id, n_steps=n_steps)
        if overflow_step is not None:
            body.sub_steps[overflow_step]["step_description_prompt"] = "overflow"
        run_policy, session_policy = _ScriptedPolicy(), _ScriptedPolicy()

        # /run calls this agent's own /v1/responses for each sub-step, then /verify.
        run_agent = _agent()
        captured = {}

        async def run_post(server_name, url_path, json, cookies=None):
            if server_name == "policy_model":
                return run_policy.reply(_wire(json))
            if server_name == "scicode_agent":
                params = NeMoGymResponseCreateParamsNonStreaming.model_validate(_wire(json))
                reply = await run_agent.responses(_FakeRequest(), Response(), params)
                return _Resp(reply.model_dump(mode="json"))
            captured["verify"] = _wire(json)
            return _Resp({**captured["verify"], "reward": 0.0})

        run_agent.server_client.post = AsyncMock(side_effect=run_post)

        # The session: seeded with the row's agent_context, then one activation under the rollout path.
        session_agent = _agent()

        async def session_post(server_name, url_path, json, cookies=None):
            assert server_name == "policy_model", url_path
            return session_policy.reply(_wire(json))

        session_agent.server_client.post = AsyncMock(side_effect=session_post)
        seed = _seed(_agent_context(body))
        path = f"/ng-rollout/{seed.episode_id.capture_key}/v1/responses"
        params = body.responses_create_params.model_dump(mode="json", exclude_unset=True)
        with patch.object(app, "raise_for_status", AsyncMock()):
            import asyncio

            asyncio.run(run_agent.run(_FakeRequest(), body))
            with TestClient(session_agent.setup_webserver()) as client:
                assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
                assert client.post("/ng-rollout/wrong/v1/responses", json=params).status_code == 409
                activation = client.post(path, json=params)
                assert activation.status_code == 200, activation.text
                # The same activation again joins the first: the loop does not run twice.
                assert client.post(path, json=params).json() == activation.json()
                close = {"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump()}
                closed = client.post("/v1/agent_sessions/close", json=close)

        assert session_policy.requests == run_policy.requests
        response = activation.json()
        accounting = response.pop(SCICODE_RESPONSE_KEY)
        verify = captured["verify"]
        assert accounting == {
            "solutions": verify["solutions"],
            "step_usage": verify["step_usage"],
            "token_usage_version": verify["token_usage_version"],
        }
        same = NeMoGymResponse.model_validate
        assert same(response).model_dump(mode="json") == same(verify["response"]).model_dump(mode="json")
        # The activation's model calls are the session's observations, as /run's trajectory records them.
        observed = closed.json()["agent_observations"]
        generated = [step["response_id"] for step in verify["step_usage"] if step["status"] == "generated"]
        assert [call["response_id"] for call in observed["records"][0]["model_calls"]] == generated
        assert observed["records"][0]["status"] == ("incomplete" if overflow_step is not None else "completed")
