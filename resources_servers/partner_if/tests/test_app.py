# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import (
    IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME,
    IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME,
    ServerClient,
    SimpleServer,
)
from resources_servers.partner_if.app import PartnerIFResourcesServer, PartnerIFResourcesServerConfig, VerifyRequest
from resources_servers.partner_if.grading_contract import GradingContract
from resources_servers.partner_if.grading_service import contract_hash


def contract(task_id="smoke-1", audit_signoff=True, data_kind="synthetic_smoke"):
    return {
        "task_id": task_id,
        "origin": "synthetic",
        "source": "authored smoke fixture",
        "rubric_version": "test-v1",
        "audit_signoff": audit_signoff,
        "data_kind": data_kind,
        "rubrics": [
            {
                "id": rid,
                "category": "always",
                "constraint": "The answer contains HELLO.",
                "verifier": {"id": "llm_judge_v1"},
                "taxonomy": ["IF-FORMAT"],
                "provenance": {"instruction_text": "Say HELLO.", "source_role": "user"},
            }
            for rid in ("R1", "R2")
        ],
    }


def response(text="HELLO", **overrides):
    return NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "model": "actor",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            **overrides,
        }
    )


def request(**contract_overrides):
    return VerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Say HELLO."}]},
            "response": response().model_dump(),
            "verifier_metadata": {"partner_if": contract(**contract_overrides)},
        }
    )


def server(**kwargs):
    return PartnerIFResourcesServer(
        config=PartnerIFResourcesServerConfig(
            name="partner_if",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            **kwargs,
        ),
        server_client=AsyncMock(spec=ServerClient),
    )


def verdict(passed=True):
    return {
        "items": [
            {
                "id": rid,
                "passed": passed,
                "reason": "Contains the requested word.",
                "evidence": "HELLO",
                "references": [{"source": "answer", "start_line": 1, "end_line": 1}],
            }
            for rid in ("R1", "R2")
        ]
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["rubrics"].append(copy.deepcopy(c["rubrics"][0])),
        lambda c: c.update(rubrics=[]),
        lambda c: c.update(evidence_mode="tool_trace"),
        lambda c: c["rubrics"][0].update(category="conditional"),
        lambda c: c["rubrics"][0].update(verifier={}),
        lambda c: c["rubrics"][0].update(constraint=""),
        lambda c: c.update(audit_signoff="false"),
    ],
)
def test_reject_invalid_contract(mutate):
    data = contract()
    mutate(data)
    with pytest.raises(ValidationError):
        GradingContract.model_validate(data)


def test_hash_is_stable_and_sensitive():
    data = contract()
    first = contract_hash(GradingContract.model_validate(data))
    assert first == contract_hash(GradingContract.model_validate(dict(reversed(list(data.items())))))
    data["rubrics"][0]["constraint"] = "A different instruction."
    assert first != contract_hash(GradingContract.model_validate(data))


@pytest.mark.parametrize("passed,expected", [(True, 1), (False, 0)])
async def test_fixed_denominator_and_prompt_isolation(passed, expected):
    body = request()
    body.response = response("<!-- scratchpad --> HELLO")
    before = body.model_dump()
    parsed = verdict()
    parsed["items"][1]["passed"] = passed
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(parsed)),
    ) as call:
        result = await server().verify(body)
    assert result.reward == expected
    assert result.if_result.total == 2
    assert result.if_result.passed == (2 if passed else 1)
    assert result.if_result.rubric_hash == contract_hash(body.verifier_metadata.partner_if)
    assert result.if_result.status == "scored"
    assert body.model_dump() == before
    judge_input = call.call_args.kwargs["json"]["input"]
    payload = json.loads(judge_input[1]["content"])
    assert payload["evidence_sources"]["context:0/content"]["lines"][0]["text"] == "Say HELLO."
    assert payload["evidence_sources"]["answer"]["lines"][0]["text"] == "<!-- scratchpad --> HELLO"
    assert payload["rubrics"][0]["constraint"] == "The answer contains HELLO."
    assert "actor" not in payload  # No model label or reference answer given to the judge.
    assert result.judge_response is not None


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '{"items": []}',
        json.dumps({"items": [verdict()["items"][0]] * 2}),
        json.dumps({"items": [{**verdict()["items"][0], "id": "unexpected"}, verdict()["items"][1]]}),
        json.dumps({"items": [{**verdict()["items"][0], "passed": "true"}, verdict()["items"][1]]}),
        json.dumps({"items": [{"id": "R1", "passed": True, "reason": "ok"}, verdict()["items"][1]]}),
    ],
)
async def test_invalid_judge_is_not_actor_failure(bad):
    with patch(
        "resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock, return_value=response(bad)
    ):
        output = await server().verify(request())
    assert output.mask_sample and output.if_result.status == "unscorable"
    assert output.judge_response is not None


async def test_judge_call_failure_reaches_gyms_judge_failsafe():
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        side_effect=JudgeError("unavailable"),
    ):
        with pytest.raises(JudgeError):
            await server().verify(request())
        row = TestClient(server().setup_webserver()).post("/verify", json=request().model_dump(mode="json")).json()
    assert row["_ng_failure_class"] == row["failure_kind"] == "judge_failed"
    assert row["mask_sample"] is True


@pytest.mark.parametrize(
    "output",
    [
        [],
        [
            {
                "type": "function_call",
                "id": "f",
                "call_id": "c",
                "name": "run",
                "arguments": "{}",
                "status": "completed",
            }
        ],
    ],
)
async def test_missing_or_tool_evidence_is_unscorable(output):
    body = request()
    body.response = response(output=output)
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock) as call:
        result = await server().verify(body)
    assert result.mask_sample is True
    assert result.if_result.status == "unscorable"
    call.assert_not_called()


async def test_incomplete_actor_not_scored():
    body = request()
    body.response = response(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    result = await server().verify(body)
    assert result.mask_sample and result.if_result.status == "unscorable"


async def test_unsigned_audit_requires_explicit_development_option():
    body = request(audit_signoff=False)
    result = await server().verify(body)
    assert result.mask_sample and result.if_result.status == "unscorable"
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(verdict())),
    ):
        provisional = await server(allow_provisional_rubrics=True).verify(body)
    assert provisional.reward == 1
    assert provisional.if_result.status == "provisional"


async def test_incomplete_judge_not_scored():
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(verdict()), status="incomplete"),
    ):
        output = await server().verify(request())
    assert output.mask_sample and output.if_result.status == "unscorable"
    assert output.judge_response.status == "incomplete"


@pytest.mark.parametrize("hidden", [{"conversation": "conv_123"}, {"prompt": {"id": "pmpt_123"}}])
async def test_hidden_context_is_unscorable(hidden):
    body = request()
    body.responses_create_params = body.responses_create_params.model_copy(update=hidden)
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(verdict())),
    ) as call:
        result = await server().verify(body)
    assert result.mask_sample and result.if_result.status == "unscorable"
    call.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n\t"])
async def test_empty_visible_text_never_earns_reward(text):
    body = request()
    body.response = response(text)
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(verdict())),
    ) as call:
        result = await server().verify(body)
    assert result.reward == 0 and result.mask_sample
    call.assert_not_called()


async def test_verifier_fixture_pins_both_reward_endpoints_without_a_judge():
    from nemo_gym.verifier_fixture import exercise_verifier_fixture
    from resources_servers.partner_if.app import VERIFIER_FIXTURE

    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock) as call:
        results = await exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
        )
    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed"}
    call.assert_not_called()


def test_uvicorn_workers_import_a_module_level_app(monkeypatch):
    # With num_workers > 1, each uvicorn worker imports `resources_servers.partner_if.app:app`.
    path, name = str(Path(__file__).parents[1] / "app.py"), "resources_servers.partner_if.app"
    started = MagicMock(return_value=object())
    monkeypatch.setattr(SimpleServer, "run_webserver", started)
    for key in (IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME):
        monkeypatch.delenv(key, raising=False)
    assert "app" not in runpy.run_path(path, run_name=name)
    started.assert_not_called()
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME, "resources_servers/partner_if/app.py")
    assert runpy.run_path(path, run_name=name)["app"] is started.return_value
    started.assert_called_once_with()
