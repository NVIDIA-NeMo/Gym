# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest
from responses_api_agents.multiple_choice.app import MultipleChoiceAgent, MultipleChoiceRunRequest
from responses_api_agents.simple_agent.app import SimpleAgentConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["string", "message", "invalid", "skip", "mismatch", "missing", "nan", "bool", "wrong", "tie"]
)
async def test_native_flow_scores_choices_and_preserves_cookies(monkeypatch, mode):
    calls = []
    verifier = MCQAResourcesServer.model_construct(
        config=MCQAResourcesServerConfig(name="resource", host="127.0.0.1", port=1235, entrypoint="app.py"),
        server_client=MagicMock(),
    )

    async def post(**kwargs):
        calls.append(kwargs)
        route = kwargs["url_path"]
        if route == "/seed_session":
            result = {}
        elif route == "/loglikelihood":
            assert kwargs["json"] == {
                "context": "Question: x\nAnswer:",
                "continuations": [" short", " much longer option"],
            }
            result = {
                "model": "test",
                "scores": [
                    {
                        "continuation": " short",
                        "logprob": -3.0,
                        "context_tokens": 7,
                        "continuation_token_ids": [1],
                        "token_logprobs": [-3.0],
                    },
                    {
                        "continuation": " much longer option",
                        "logprob": -1.0,
                        "context_tokens": 7,
                        "continuation_token_ids": [2],
                        "token_logprobs": [-1.0],
                    },
                ],
            }
        else:
            assert route == "/verify"
            result = (await verifier.verify(MCQAVerifyRequest.model_validate(kwargs["json"]))).model_dump(mode="json")
        if route == "/loglikelihood":
            if mode == "mismatch":
                result["scores"].reverse()
            elif mode == "missing":
                result["scores"].pop()
            elif mode == "nan":
                result["scores"][0]["logprob"] = float("nan")
            elif mode == "bool":
                result["scores"][0]["logprob"] = True
            elif mode == "tie":
                result["scores"][0]["logprob"] = -1.0
        response = MagicMock()
        response.cookies = {"session": "carried"}
        response.json = AsyncMock(return_value=result)
        return response

    import responses_api_agents.multiple_choice.app as module

    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", lambda result: result.json())
    config = SimpleAgentConfig(
        name="agent",
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resource"},
        model_server={"type": "responses_api_models", "name": "model"},
    )
    client = MagicMock()
    client.post = AsyncMock(side_effect=post)
    agent = MultipleChoiceAgent.model_construct(config=config, server_client=client)
    request = MagicMock(cookies={"original": "cookie"})
    inputs = "Question: x\nAnswer:"
    if mode == "message":
        inputs = [{"role": "user", "content": inputs}]
    elif mode == "invalid":
        inputs = [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]
    body = MultipleChoiceRunRequest(
        responses_create_params={"input": inputs},
        choices=[" short", " much longer option"],
        options=[{"A": "short"}, {"B": "much longer option"}],
        expected_answer="A" if mode in {"wrong", "tie"} else "B",
    )
    if mode in {"invalid", "mismatch", "missing", "nan", "bool"}:
        with pytest.raises(ValueError):
            await agent.run(request, body)
        return
    if mode == "skip":
        config.skip_verification = True
        config.skip_verification_reward = 1.0
    result = await agent.run(request, body)
    selected = 0 if mode == "tie" else 1
    assert result.reward == (0 if mode == "wrong" else 1)
    assert result.selected_choice == selected
    assert result.response.output[0].content[0].text == "\\boxed{" + "AB"[selected] + "}"
    assert len(result.likelihood["scores"]) == 2
    if mode != "skip":
        assert result.extracted_answer == "AB"[selected]
    assert result.response.metadata["not_a_generated_answer"] == "true"
    assert calls[1]["cookies"] == {"original": "cookie", "session": "carried"}
    assert [c["url_path"] for c in calls] == (
        ["/seed_session", "/loglikelihood"] if mode == "skip" else ["/seed_session", "/loglikelihood", "/verify"]
    )
