# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.judge import JudgeError
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.vials.verifier_fixture import judge_http_reply

from ..app import VERIFIER_FIXTURE, VialsResourcesServer, VialsResourcesServerConfig, VialsVerifier, VialsVerifyRequest


def make_request(text: str, **fields: object) -> VialsVerifyRequest:
    return VialsVerifyRequest.model_validate(
        {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            *[
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64," + base64.b64encode(image).decode(),
                                    "detail": "auto",
                                }
                                for image in [b"first panel", b"second panel"]
                            ],
                            {"type": "input_text", "text": "How many cells are visible?"},
                        ],
                    }
                ]
            },
            "question": "How many cells are visible?",
            "expected_answer": "12",
            "response": {
                "id": "test-response",
                "created_at": 0,
                "model": "test-policy",
                "object": "response",
                "output": [
                    {
                        "id": "test-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
        }
        | fields
    )


def reply(equivalent: bool, reasoning: str, *, tokens: int = 10) -> MagicMock:
    return judge_http_reply(json.dumps({"equivalent": equivalent, "reasoning": reasoning}), tokens=tokens)


def make_verifier(*replies: MagicMock, **settings: object) -> VialsVerifier:
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(side_effect=list(replies))
    return VialsVerifier(client, **settings)


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
        )
    )


def test_extraction_bounds_and_gym_routing() -> None:
    verifier = make_verifier(reply(True, "Equivalent wording."))
    result = asyncio.run(
        verifier.verify(
            make_request(
                "ANSWER: wrong\nREASONING: Counted again.\n- **ANSWER:** **twelve**",
                answer_lower_bound=11,
                answer_upper_bound=13,
            )
        )
    )
    kwargs = verifier._judge.server_client.post.call_args.kwargs
    params = kwargs["json"].model_dump(mode="json", exclude_none=True)
    assert kwargs["server_name"] == "vials_judge_model" and kwargs["url_path"] == "/v1/responses"
    prompt = params["input"][1]["content"]
    assert "Ground-truth final answer:\n12" in prompt
    assert "lower=11.0, upper=13.0" in prompt
    assert "Candidate final answer:\ntwelve" in prompt
    assert params["text"]["format"]["schema"]["additionalProperties"] is False
    assert result.reward == 1 and not result.mask_sample and result.candidate_answer == "twelve"
    assert result.judge_model == "gpt-5-mini" and result.judge_input_tokens == 10


@pytest.mark.parametrize(
    "text", ["", "I counted twelve cells.", "ANSWER: ", "<think>ANSWER: twelve</think>", "<thinking>ANSWER: twelve"]
)
def test_missing_visible_answer_never_calls_judge(text: str) -> None:
    verifier = make_verifier()
    result = asyncio.run(verifier.verify(make_request(text)))
    verifier._judge.server_client.post.assert_not_called()
    assert result.reward == 0 and not result.mask_sample and result.candidate_answer is None


def test_visible_answer_after_thinking() -> None:
    verifier = make_verifier(reply(True, "Correct."))
    result = asyncio.run(verifier.verify(make_request("<think>ANSWER: eleven</think>\nANSWER: twelve")))
    assert result.candidate_answer == "twelve" and result.reward == 1


def test_incorrect_verdict() -> None:
    verifier = make_verifier(reply(False, "Wrong count."))
    result = asyncio.run(verifier.verify(make_request("ANSWER: eleven")))
    assert result.reward == 0 and not result.mask_sample


def test_multimodal_fallback_preserves_panels_and_usage() -> None:
    verifier = make_verifier(reply(False, "UNVERIFIABLE: labels."), reply(True, "Resolved.", tokens=20))
    result = asyncio.run(verifier.verify(make_request("ANSWER: twelve")))
    params = verifier._judge.server_client.post.call_args_list[1].kwargs["json"].model_dump(mode="json")
    blocks = params["input"][1]["content"]
    assert [base64.b64decode(block["image_url"].split(",", 1)[1]) for block in blocks[:-1]] == [
        b"first panel",
        b"second panel",
    ]
    assert "Use it ONLY to resolve" in params["input"][0]["content"]
    assert result.reward == 1 and result.judge_multimodal
    assert result.judge_input_tokens == 30 and result.judge_output_tokens == 10


def test_fallback_disabled() -> None:
    verifier = make_verifier(reply(False, "UNVERIFIABLE: labels."), multimodal_fallback=False)
    result = asyncio.run(verifier.verify(make_request("ANSWER: twelve")))
    assert verifier._judge.server_client.post.call_count == 1
    assert result.judge_unverifiable and result.reward == 0 and not result.mask_sample


def test_missing_fallback_evidence() -> None:
    verifier = make_verifier(reply(False, "UNVERIFIABLE: labels."))
    result = asyncio.run(
        verifier.verify(make_request("ANSWER: twelve", responses_create_params={"input": "question"}))
    )
    assert result.mask_sample and result.failure_kind == "verifier_error"


def test_failed_call_raises_gym_judge_error() -> None:
    verifier = make_verifier()
    verifier._judge.server_client.post.side_effect = RuntimeError("judge unavailable")
    with pytest.raises(JudgeError, match="judge unavailable"):
        asyncio.run(verifier.verify(make_request("ANSWER: twelve")))


def test_unparseable_judge_retries_without_masking() -> None:
    verifier = make_verifier(*[judge_http_reply("not JSON") for _ in range(3)])
    with patch("resources_servers.vials.grading.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(verifier.verify(make_request("ANSWER: twelve")))
    assert verifier._judge.server_client.post.call_count == 3
    assert result.reward == 0 and result.judge_unparseable and not result.mask_sample
    assert result.judge_input_tokens == 30


def test_parse_recovery_and_absent_usage() -> None:
    response = judge_http_reply('```json\n{"equivalent":true,"reasoning":"Correct."}\n```')
    payload = json.loads(response.read.return_value)
    payload.pop("usage")
    response.read.return_value = json.dumps(payload).encode()
    verifier = make_verifier(response)
    result = asyncio.run(verifier.verify(make_request("ANSWER: twelve")))
    assert result.reward == 1 and result.judge_input_tokens == 0


@pytest.mark.parametrize(
    "bounds", [{"answer_lower_bound": 13, "answer_upper_bound": 11}, {"answer_lower_bound": float("nan")}]
)
def test_invalid_bounds(bounds: dict) -> None:
    with pytest.raises(ValidationError):
        make_request("ANSWER: twelve", **bounds)


@pytest.mark.parametrize("failed", [False, True])
def test_http_verification_and_failure_sidecar(failed: bool) -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create({})
    client.checkpoint_enabled.return_value = False
    client.post = AsyncMock(return_value=reply(True, "Correct."))
    if failed:
        client.post.side_effect = RuntimeError("judge unavailable")
    server = VialsResourcesServer(
        config=VialsResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="vials_resources_server",
            judge_model_server={"type": "responses_api_models", "name": "vials_judge_model"},
        ),
        server_client=client,
    )
    with TestClient(server.setup_webserver()) as http:
        response = http.post("/verify", json=make_request("ANSWER: twelve").model_dump(mode="json"))
    assert response.status_code == 200
    result = response.json()
    if failed:
        assert result["mask_sample"] and result["_ng_failure_class"] == "judge_failed"
        assert result["response"]["id"] == "test-response"
    else:
        assert result["reward"] == 1 and not result["mask_sample"]
