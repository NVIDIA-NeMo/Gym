# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the bridge's typed boundary against the original HTTP graders."""

from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.aga_visual_tasks.app import Verify
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
)
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig
from resources_servers.string_match.app import StringMatchResourcesServer, StringMatchResourcesServerConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("grader", ["math", "mcqa", "string"])
@pytest.mark.parametrize("correct", [True, False])
async def test_private_grader_fields_survive_bridge_http(grader, correct):
    answer = "A" if grader == "mcqa" else "4"
    output = answer if correct else "9"
    response = NeMoGymResponse(
        id="saved-response",
        created_at=1,
        model="unused",
        object="response",
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        output=[
            {
                "type": "message",
                "id": "final",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": rf"\boxed{{{output}}}"}],
            }
        ],
    )
    policy_input = {"input": [{"role": "user", "content": "Solve the public problem."}]}
    original = {
        "responses_create_params": policy_input,
        "response": response.model_dump(mode="json"),
        "image_tools_base_agent_ref": {"type": "responses_api_agents", "name": grader},
        "question": "What is two plus two?",
        "expected_answer": answer,
        "metadata": {"private_source_id": "original-row"},
    }
    if grader == "string":
        original.update(extraction_mode="boxed", case_sensitive=True)
    if grader == "mcqa":
        original.update(grading_mode="strict_single_letter_boxed", options=[{"A": "four"}])
    parsed = Verify.model_validate(original)
    forwarded = parsed.model_dump(mode="json")
    for key in set(original) - {"responses_create_params", "response"}:
        assert forwarded[key] == original[key]
    assert "extraction_mode" in forwarded if grader == "string" else "extraction_mode" not in forwarded
    assert forwarded["responses_create_params"]["input"][0]["content"] == "Solve the public problem."
    assert "expected_answer" not in forwarded["responses_create_params"]
    common = dict(name=grader, host="127.0.0.1", port=1, entrypoint="")
    client = MagicMock(spec=ServerClient)
    if grader == "math":
        server = LibraryJudgeMathResourcesServer(
            config=LibraryJudgeMathResourcesServerConfig(
                **common,
                should_use_judge=False,
                judge_model_server={"type": "responses_api_models", "name": "never-call"},
                judge_responses_create_params={"input": []},
            ),
            server_client=client,
        )
    elif grader == "mcqa":
        server = MCQAResourcesServer(config=MCQAResourcesServerConfig(**common), server_client=client)
    else:
        server = StringMatchResourcesServer(config=StringMatchResourcesServerConfig(**common), server_client=client)
    app = FastAPI()
    app.post("/verify")(server.verify)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://grader") as http:
        reply = await http.post("/verify", json=forwarded)
        if grader != "mcqa":
            missing = dict(forwarded)
            missing.pop("expected_answer")
            refused = await http.post("/verify", json=missing)
            assert refused.status_code == 422
    assert reply.status_code == 200, reply.text
    assert reply.json()["reward"] == float(correct)
    client.post.assert_not_called()
