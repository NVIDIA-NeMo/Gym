# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.config import ConfigError, ModelServerRef
from nemo_gym.eval.rollout_reverification import InputRolloutPair, _build_verify_payload
from nemo_gym.server_runtime import ReverifyMode
from nemo_gym.verification import is_judge_failure
from nemo_gym.web import WebBenchmark, WebTask, compact_webvoyager_judge_evidence
from resources_servers.webvoyager_judge.app import (
    WebVoyagerJudgeResourcesServer,
    parse_gemini_verdict,
)
from resources_servers.webvoyager_judge.config import WebVoyagerJudgeConfig
from resources_servers.webvoyager_judge.models import (
    MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS,
    WebVoyagerJudgeRequest,
    WebVoyagerStandardVerifyRequest,
)


def _model_response(text: str) -> dict:
    return {
        "id": "judge-response",
        "created_at": 0.0,
        "model": "judge",
        "object": "response",
        "output": [
            {
                "id": "judge-message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class _FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    @property
    def content(self):
        payload = self.payload

        class _Body:
            async def read(self):
                return json.dumps(payload).encode()

        return _Body()

    async def read(self):
        return json.dumps(self.payload).encode()


def _server(**overrides):
    config_values = {
        "name": "webvoyager_judge",
        "host": "localhost",
        "port": 8002,
        "entrypoint": "app.py",
        "domain": "agent",
        "judge_model_server": ModelServerRef(type="responses_api_models", name="judge-model"),
        "judge_responses_create_params": {"input": "placeholder", "temperature": 0},
    }
    config_values.update(overrides)
    config = WebVoyagerJudgeConfig(
        **config_values,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return WebVoyagerJudgeResourcesServer(config=config, server_client=client)


def _request(answer="answer", *, screenshot_count=1, page_url_count=0):
    return WebVoyagerJudgeRequest(
        task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="site--0",
            intent="Find the requested fact",
            start_urls=["https://example.test"],
        ),
        final_answer=answer,
        screenshots=[f"data:image/png;base64,image-{index}" for index in range(screenshot_count)],
        page_urls=[f"https://example.test/step-{index}" for index in range(page_url_count)],
    )


def _standard_request(*, screenshot_count: int = 1) -> WebVoyagerStandardVerifyRequest:
    evidence_request = _request(screenshot_count=screenshot_count)
    return WebVoyagerStandardVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Solve"},
            "response": _model_response("final"),
            "web_task": evidence_request.task.model_dump(mode="json"),
            "final_answer": evidence_request.final_answer,
            "screenshots": evidence_request.screenshots,
            "page_urls": evidence_request.page_urls,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["", "answer"])
async def test_standard_verify_masks_missing_required_screenshots(answer: str, caplog) -> None:
    import httpx

    server = _server()
    server.server_client.post = AsyncMock(side_effect=AssertionError("missing evidence must not call the judge"))
    request = _standard_request(screenshot_count=0).model_copy(update={"final_answer": answer})
    transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)

    with caplog.at_level(logging.WARNING, logger="nemo_gym.resources_servers.webvoyager_judge"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/verify", json=request.model_dump(mode="json"))

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["reward"] == result["verifier_result"]["reward"] == 0.0
    assert result["mask_sample"] is True
    assert result["verifier_result"]["valid_sample"] is False
    assert result["failure_kind"] == result["verifier_result"]["failure_kind"] == "web:missing_judge_evidence"
    assert result["judge_text"] == ""
    assert result["response"] == request.response.model_dump(mode="json")
    assert "reason=missing_judge_evidence valid_sample=False" in caplog.text
    server.server_client.post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,reward", [("SUCCESS", 1.0), ("FAILURE", 0.0)])
async def test_standard_verify_supports_text_only_judging_when_configured(verdict: str, reward: float) -> None:
    import httpx

    server = _server(require_screenshot=False)
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(_model_response(json.dumps({"verdict": verdict})))
    )
    request = _standard_request(screenshot_count=0)
    transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/verify", json=request.model_dump(mode="json"))

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["reward"] == result["verifier_result"]["reward"] == reward
    assert result["mask_sample"] is False
    assert result["verifier_result"]["valid_sample"] is True
    assert result["failure_kind"] is None
    assert result["verifier_result"]["metadata"]["parsed"]["verdict"] == verdict
    assert result["verifier_result"]["metadata"]["screenshots_used"] == 0
    server.server_client.post.assert_awaited_once()
    params = server.server_client.post.await_args.kwargs["json"]
    assert [block["type"] for block in params.model_dump(mode="json")["input"][0]["content"]] == ["input_text"]


@pytest.mark.asyncio
async def test_committed_screenshotless_examples_match_missing_evidence_contract() -> None:
    import httpx

    data_dir = Path(__file__).resolve().parents[1] / "data"
    rows = [json.loads(line) for line in (data_dir / "example_rollouts.jsonl").read_text().splitlines()]
    metrics = json.loads((data_dir / "example_metrics.json").read_text())
    server = _server()
    server.server_client.post = AsyncMock(side_effect=AssertionError("screenshotless fixture must not call the judge"))
    transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for row in rows:
            response = await client.post("/verify", json=row)
            assert response.status_code == 200, response.text
            result = response.json()
            assert row["screenshots"] == []
            assert row["reward"] == result["reward"] == 0.0
            assert row["mask_sample"] is result["mask_sample"] is True
            assert row["failure_kind"] == result["failure_kind"] == "web:missing_judge_evidence"
            assert row["verifier_result"]["valid_sample"] is result["verifier_result"]["valid_sample"] is False
            assert row["verifier_result"] == result["verifier_result"]
            assert row["verifier_metadata"] == result["verifier_metadata"] == {}

    assert len(rows) == metrics["Number of examples"] == 5
    assert metrics["Masked examples"] == sum(row["mask_sample"] for row in rows) == 5
    assert metrics["Scored examples"] == sum(not row["mask_sample"] for row in rows) == 0
    server.server_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_judge_exposes_only_standard_stateless_verify_route():
    server = _server()
    paths = {route.path for route in server.setup_webserver().routes}

    assert "/verify" in paths
    assert "/verify_webvoyager" not in paths
    assert await server.get_reverify_mode() == ReverifyMode.STATELESS


@pytest.mark.asyncio
async def test_standard_route_classifies_judge_transport_failure_for_sidecar():
    server = _server()
    server.server_client.post = AsyncMock(side_effect=TimeoutError("judge timed out"))
    verify_endpoint = next(route.endpoint for route in server.setup_webserver().routes if route.path == "/verify")

    response = await verify_endpoint(body=_standard_request())
    payload = json.loads(response.body)

    assert payload["reward"] == 0.0
    assert payload["_ng_failure_class"] == "judge_failed"
    assert "TimeoutError" in payload["_ng_failure_judge_error"]
    assert payload["response"] == _standard_request().response.model_dump(mode="json")


def test_gemini_verdict_requires_json_success_or_failure():
    assert parse_gemini_verdict('{"thought":"ok","verdict":"SUCCESS"}') == (
        True,
        {"thought": "ok", "verdict": "SUCCESS"},
    )
    assert parse_gemini_verdict("SUCCESS") is None
    assert parse_gemini_verdict('{"verdict":"NOT SUCCESS"}') is None


def test_request_contract_accepts_a_full_100_step_trajectory():
    request = _request(screenshot_count=101, page_url_count=101)

    assert len(request.screenshots) == 101
    assert len(request.page_urls) == 101
    assert (
        WebVoyagerStandardVerifyRequest.model_json_schema()["properties"]["screenshots"]["maxItems"]
        == MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS
    )


def test_request_contract_rejects_evidence_above_the_configured_ceiling():
    with pytest.raises(ValidationError, match="List should have at most 200 items"):
        _request(screenshot_count=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS + 1)


def test_standard_request_recovers_evidence_from_saved_rollout_response():
    response = _model_response("final")
    response["webvoyager_judge_evidence"] = {
        "final_answer": "42",
        "screenshots": ["data:image/png;base64,evidence"],
        "page_urls": ["https://example.test/result"],
    }
    request = WebVoyagerStandardVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Solve"},
            "response": response,
            "web_task": _request().task.model_dump(mode="json"),
        }
    )

    assert request.final_answer == "42"
    assert request.screenshots == ["data:image/png;base64,evidence"]
    assert request.page_urls == ["https://example.test/result"]


@pytest.mark.asyncio
async def test_generic_reverify_reconstructs_compact_evidence_without_browser_replay():
    trajectory_image = "data:image/png;base64,trajectory"
    terminal_image = "data:image/png;base64,terminal"
    response = _model_response("final")
    response["output"].insert(
        0,
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": trajectory_image, "detail": "high"}],
        },
    )
    response["webvoyager_judge_evidence"] = compact_webvoyager_judge_evidence(
        response=response,
        final_answer="42",
        screenshots=[trajectory_image, terminal_image],
        page_urls=["https://example.test/start", "https://example.test/result"],
    )
    task = _request().task
    payload = _build_verify_payload(
        InputRolloutPair(
            input={
                "responses_create_params": {"input": "Solve"},
                "web_task": task.model_dump(mode="json"),
            },
            rollout={"response": response},
        )
    )
    request = WebVoyagerStandardVerifyRequest.model_validate(payload)
    assert request.final_answer == "42"
    assert request.screenshots == [trajectory_image, terminal_image]
    assert request.page_urls == ["https://example.test/start", "https://example.test/result"]

    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"retained evidence proves completion","verdict":"SUCCESS"}')
        )
    )
    result = await server.verify(request)

    assert result.reward == 1.0
    assert result.task_success is True
    server.server_client.post.assert_awaited_once()
    judge_call = server.server_client.post.await_args
    assert judge_call.kwargs["server_name"] == "judge-model"
    assert judge_call.kwargs["url_path"] == "/v1/responses"
    assert result.response.model_dump(mode="json") == request.response.model_dump(mode="json")


@pytest.mark.asyncio
async def test_standard_verify_response_does_not_echo_top_level_evidence():
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshot proves completion","verdict":"SUCCESS"}')
        )
    )

    result = await server.verify(_standard_request())
    dumped = result.model_dump(mode="json")

    assert result.reward == 1.0
    assert "screenshots" not in dumped
    assert "page_urls" not in dumped
    assert "final_answer" not in dumped


@pytest.mark.asyncio
async def test_judge_uses_a_full_100_step_trajectory():
    server = _server(max_screenshots=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS)
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(_model_response('{"thought":"enough evidence","verdict":"SUCCESS"}'))
    )

    response = await server._judge_evidence(_request(screenshot_count=101, page_url_count=101))

    assert response.result.valid_sample is True
    assert response.result.metadata["screenshots_used"] == 101
    params = server.server_client.post.await_args.kwargs["json"]
    assert len(params.input[0].content) == 203


@pytest.mark.asyncio
async def test_empty_answer_still_uses_visual_trajectory_evidence():
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshots do not prove completion","verdict":"FAILURE"}')
        )
    )

    response = await server._judge_evidence(_request(answer=""))

    assert response.result.valid_sample is True
    assert response.result.reward == 0.0
    assert response.result.failure_kind is None
    server.server_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_success_returns_binary_reward():
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshot proves completion","verdict":"SUCCESS"}')
        )
    )

    response = await server._judge_evidence(_request())

    assert response.result.valid_sample is True
    assert response.result.reward == 1.0
    assert response.result.task_success is True


@pytest.mark.asyncio
async def test_received_unparseable_verdict_is_a_valid_zero_reward():
    server = _server()
    server.server_client.post = AsyncMock(return_value=_FakeHttpResponse(_model_response("unclear")))

    response = await server._judge_evidence(_request())

    assert response.result.valid_sample is True
    assert response.result.reward == 0.0
    assert response.result.failure_kind == "judge_unparseable"


@pytest.mark.asyncio
async def test_judge_parse_retry_keeps_evidence_and_preserves_first_bad_verdict():
    server = _server()
    server.config.judge_parse_max_retries = 1
    server.server_client.post = AsyncMock(
        side_effect=[
            _FakeHttpResponse(_model_response("unclear")),
            _FakeHttpResponse(_model_response('{"thought":"evidence proves success","verdict":"SUCCESS"}')),
        ]
    )
    response = await server._judge_evidence(_request())
    assert response.result.task_success
    assert response.result.metadata["judge_attempts"][0] == "unclear"
    assert len(response.result.metadata["judge_attempts"]) == 2
    first, second = server.server_client.post.call_args_list
    assert first.kwargs["json"].model_dump() == second.kwargs["json"].model_dump()
    assert len(response.result.metadata["judge_evidence_sha256"]) == 64


@pytest.mark.asyncio
async def test_unparseable_judge_can_be_masked_without_changing_legacy_default():
    server = _server()
    server.config.unparseable_verdict_policy = "mask"
    server.config.judge_parse_max_retries = 1
    server.server_client.post = AsyncMock(return_value=_FakeHttpResponse(_model_response("unclear")))
    response = await server._judge_evidence(_request())
    assert not response.result.valid_sample
    assert response.result.failure_kind == "judge_unparseable"
    assert server.server_client.post.await_count == 2
    assert response.result.metadata["judge_attempts"] == ["unclear", "unclear"]
    assert _server().config.unparseable_verdict_policy == "zero"


@pytest.mark.asyncio
async def test_judge_parse_retries_share_one_overall_deadline():
    server = _server()
    server.config.judge_parse_max_retries = 1
    server.config.judge_call_timeout_secs = 0.05

    async def slow_judge(*_args, **_kwargs):
        await asyncio.sleep(0.03)
        return _FakeHttpResponse(_model_response("unclear"))

    server.server_client.post = AsyncMock(side_effect=slow_judge)
    with pytest.raises(Exception) as caught:
        await server._judge_evidence(_request())
    assert is_judge_failure(caught.value)
    assert server.server_client.post.await_count == 2


@pytest.mark.asyncio
async def test_judge_logs_lifecycle_without_screenshot_payload(caplog):
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshot proves completion","verdict":"SUCCESS"}')
        )
    )

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.webvoyager_judge"):
        response = await server._judge_evidence(_request())

    assert response.result.task_success is True
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=webvoyager_judge_start" in messages
    assert "event=webvoyager_judge_model_start" in messages
    assert "event=webvoyager_judge_model_complete" in messages
    assert "event=webvoyager_judge_complete" in messages
    assert "origins=none" in messages
    assert "data:image/png;base64,image-0" not in messages
    assert "The screenshot proves completion" not in messages


@pytest.mark.asyncio
async def test_verify_route_deadline_retains_evidence_for_reverification():
    import httpx

    server = _server()
    server.config.judge_call_timeout_secs = 0.02

    async def stalled(*args, **kwargs):
        await asyncio.sleep(10)

    server.server_client.post = AsyncMock(side_effect=stalled)
    request = _standard_request()
    request.response.webvoyager_judge_evidence = compact_webvoyager_judge_evidence(
        response=request.response.model_dump(mode="json"),
        final_answer=request.final_answer,
        screenshots=request.screenshots,
        page_urls=request.page_urls,
    )
    transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await client.post("/verify", json=request.model_dump(mode="json"))
        assert result.status_code == 200, result.text
        failure = result.json()
        assert failure["_ng_failure_class"] == "judge_failed"
        assert failure["response"] == request.response.model_dump(mode="json")
        payload = _build_verify_payload(InputRolloutPair(input=request.model_dump(mode="json"), rollout=failure))
        server.server_client.post = AsyncMock(return_value=_FakeHttpResponse(_model_response('{"verdict":"SUCCESS"}')))
        recovered = await client.post("/verify", json=payload)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["reward"] == 1.0


@pytest.mark.asyncio
async def test_judge_caller_cancellation_propagates():
    server = _server()
    entered = asyncio.Event()

    async def stalled(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    server.server_client.post = AsyncMock(side_effect=stalled)
    task = asyncio.create_task(server._judge_evidence(_request()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,reward", [("SUCCESS", 1.0), ("FAILURE", 0.0), (None, 0.0)])
async def test_public_rejudgment_persists_consistent_nested_grade(tmp_path, verdict, reward):
    import httpx

    from nemo_gym.rollouts import RolloutStore, RunManifest

    request = _standard_request(screenshot_count=0 if verdict is None else 1)
    inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}}]
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(inputs[0]) + "\n")
    manifest = RunManifest.create(source, inputs, {}, {"agent": {"responses_api_agents": {}}})
    output = tmp_path / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=False) as store:
        row = store.pending()[0]
        store.record_dispatch(row)
        failure = (
            row
            | request.model_dump(mode="json")
            | {
                "_ng_failure_class": "judge_failed",
                "_ng_failure_id": "saved-judge",
                "reward": 0.0,
                "mask_sample": True,
                "verifier_result": {"reward": 0.0, "valid_sample": False, "failure_kind": "judge_unparseable"},
            }
        )
        store.record_outcome(failure)
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(_model_response(json.dumps({"verdict": verdict})))
    )
    transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/verify", json=failure)
    assert response.status_code == 200, response.text
    with RolloutStore.open_for_rejudgment(output) as store:
        pending = store.admit_rejudgment(failure)
        result = response.json() | {
            key: failure[key]
            for key in ("_ng_task_index", "_ng_rollout_index", "agent_ref", "_ng_run_id", "_ng_attempt_index")
        }
        result.update(
            _ng_rejudged_failure_id="saved-judge",
            _ng_rejudge_operation_id=pending.operation_id,
            _ng_rejudge_admitted_at=pending.admitted_at,
        )
        if verdict is None:
            assert result["mask_sample"] is True
            assert result["verifier_result"]["valid_sample"] is False
            with pytest.raises(ConfigError, match="no scored result"):
                store.record_rejudgment(failure, result)
            store.withdraw_rejudgment("0-0", pending.operation_id, unsaved_judge_failure=True)
        else:
            assert store.record_rejudgment(failure, result)
    if verdict is None:
        assert RolloutStore.read(output).selected("success") == []
        assert RolloutStore.read(output).selected("failure") == [failure]
        server.server_client.post.assert_not_awaited()
        return
    (saved,) = RolloutStore.read(output).selected("success")
    assert saved["reward"] == saved["verifier_result"]["reward"] == reward
    assert saved["mask_sample"] is False
    assert saved["verifier_result"]["valid_sample"] is True
    assert saved["failure_kind"] == saved["verifier_result"]["failure_kind"] is None
    assert saved["task_success"] == saved["verifier_result"]["task_success"] == (reward == 1)
    assert saved["verifier_metadata"] == saved["verifier_result"]["metadata"]
    assert saved["response"] == failure["response"]
