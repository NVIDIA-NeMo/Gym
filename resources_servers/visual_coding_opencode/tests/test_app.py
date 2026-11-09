# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymResponse
from resources_servers.visual_coding_opencode.app import (
    JUDGE_SYSTEM,
    VisualCodingOpenCodeConfig,
    VisualCodingOpenCodeServer,
    VisualCodingOpenCodeVerifyRequest,
    parse_verdict,
)


def message(text: str) -> dict:
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def request(*texts: str, expected: str = "(4,7)") -> VisualCodingOpenCodeVerifyRequest:
    response = NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [message(text) for text in texts],
        }
    )
    return VisualCodingOpenCodeVerifyRequest(
        responses_create_params={"input": "q"}, response=response, expected_answer=expected, question="Which cell?"
    )


@pytest.fixture
def server() -> VisualCodingOpenCodeServer:
    config = VisualCodingOpenCodeConfig(
        name="visual_coding_opencode",
        host="localhost",
        port=1,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_model="gpt",
    )
    return VisualCodingOpenCodeServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ('{"verdict": "equivalent", "confidence": 1, "reason": "same"}', "equivalent"),
        ('Sure. {"verdict": "DIFFERENT", "reason": "x"}', "different"),
        ('{"verdict": "maybe"}', "unsure"),
        ("not json", "unsure"),
        ("", "unsure"),
    ],
)
def test_parse_verdict(reply: str, verdict: str) -> None:
    assert parse_verdict(reply)["verdict"] == verdict


@pytest.mark.asyncio
@pytest.mark.parametrize(("verdict", "reward"), [("equivalent", 1.0), ("different", 0.0), ("unsure", 0.0)])
async def test_reward_is_the_judge_verdict(server, verdict: str, reward: float) -> None:
    with patch.object(
        VisualCodingOpenCodeServer, "judge", AsyncMock(return_value={"verdict": verdict, "reason": ""})
    ) as judge:
        result = await server.verify(request("Looking at row 4.", "It is \\boxed{(4,7)}"))
    assert result.reward == reward and result.judge_verdict == verdict
    assert result.extracted_answer == "(4,7)" and result.string_match_reward == 1.0
    question, expected, extracted, text = judge.call_args.args
    assert (question, expected, extracted) == ("Which cell?", "(4,7)", "(4,7)")
    assert text == "Looking at row 4.\nIt is \\boxed{(4,7)}"


@pytest.mark.asyncio
async def test_unextracted_answer_still_judged(server) -> None:
    with patch.object(
        VisualCodingOpenCodeServer, "judge", AsyncMock(return_value={"verdict": "equivalent", "reason": ""})
    ) as judge:
        result = await server.verify(request("row 4, column 7"))
    assert result.reward == 1.0 and result.extracted_answer is None
    assert judge.call_args.args[2] is None


@pytest.mark.asyncio
async def test_empty_response_is_wrong_without_a_judge_call(server) -> None:
    with patch.object(VisualCodingOpenCodeServer, "judge", AsyncMock()) as judge:
        result = await server.verify(request(""))
    assert result.reward == 0.0 and result.failure_reason == "no_response"
    judge.assert_not_called()


@pytest.mark.asyncio
async def test_judge_reply_without_choices_is_unsure(server) -> None:
    """A received completion with no choices has no verdict: an ordinary, unmasked zero."""
    empty = NeMoGymChatCompletion.model_validate(
        {"id": "c", "created": 0, "model": "gpt", "object": "chat.completion", "choices": []}
    )
    with patch("resources_servers.visual_coding_opencode.app.call_judge", AsyncMock(return_value=empty)):
        result = await server.verify(request("\\boxed{(4,7)}"))
    assert (result.reward, result.judge_verdict, result.failure_reason) == (0.0, "unsure", "incorrect_answer")


def test_uvicorn_workers_import_a_module_level_app() -> None:
    """With num_workers > 1, each worker imports this entrypoint and serves its `app`."""
    import runpy

    from resources_servers.visual_coding_opencode import app as module

    served = object()
    # Run the module's source in a fresh namespace: the classes it defines inherit the patched run_webserver.
    with patch("nemo_gym.server_runtime.SimpleResourcesServer.run_webserver", return_value=served) as run_webserver:
        with patch("nemo_gym.utils.is_nemo_gym_fastapi_entrypoint", return_value=True):
            worker = runpy.run_path(module.__file__)
        plain = runpy.run_path(module.__file__)  # an ordinary import starts nothing
    assert worker["app"] is served and run_webserver.call_count == 1
    assert "app" not in plain


def _http_reply(status: int, body: bytes = b'{"error": "unavailable"}') -> MagicMock:
    """An aiohttp-shaped reply whose ``raise_for_status`` raises the ``ClientResponseError`` aiohttp raises."""
    from aiohttp import ClientResponseError, RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    url = URL("http://judge.example.invalid/v1/chat/completions")
    info = RequestInfo(url=url, method="POST", headers=CIMultiDictProxy(CIMultiDict()), real_url=url)
    reply = MagicMock(ok=status < 400, status=status)
    reply.read = AsyncMock(return_value=body)
    reply.content.read = AsyncMock(return_value=body)
    reply.raise_for_status = MagicMock(side_effect=ClientResponseError(info, (), status=status, message="judge error"))
    return reply


def test_judge_outage_masks_the_row_instead_of_scoring_it(server, monkeypatch) -> None:
    """Persistent 503s use the shared three attempts, then /verify returns a masked judge_failed row."""
    monkeypatch.setattr("nemo_gym.verification.judge._judge_retry_wait_s", lambda attempt: 0.0)
    server.server_client.global_config_dict = {}
    server.server_client.post = AsyncMock(side_effect=[_http_reply(503) for _ in range(3)])
    body = request("\\boxed{(4,7)}").model_dump(mode="json")
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        reply = client.post("/verify", json=body)
    row = reply.json()
    assert reply.status_code == 200 and server.server_client.post.await_count == 3
    assert (row["reward"], row["mask_sample"], row["failure_kind"]) == (0.0, True, "judge_failed")
    assert row["_ng_failure_class"] == "judge_failed"
    assert row["response"]["output"][0]["content"][0]["text"] == "\\boxed{(4,7)}"


@pytest.mark.asyncio
async def test_judge_request(server) -> None:
    completion = {
        "id": "c",
        "created": 0,
        "model": "gpt",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"verdict": "equivalent", "reason": "ok"}'},
            }
        ],
    }
    with patch("resources_servers.visual_coding_opencode.app.call_judge", AsyncMock()) as call:
        call.return_value = NeMoGymChatCompletion.model_validate(completion)
        judged = await server.judge("Q" * 3000, "(4,7)", None, "x" * 5000 + "end")
    params = call.call_args.kwargs["json"]
    assert judged["verdict"] == "equivalent"
    assert params["model"] == "gpt" and params["temperature"] == 0.0 and params["max_tokens"] == 1024
    assert params["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM}
    user = params["messages"][1]["content"]
    assert "Q" * 2000 + "\n" in user and "Q" * 2001 not in user
    assert "<nothing extracted>" in user
    assert "x" * 3997 + "end" in user and "x" * 3998 not in user
