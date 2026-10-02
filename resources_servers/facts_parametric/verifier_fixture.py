# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic in-process judge used by the FACTS Parametric verifier fixture."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.server_utils import ServerClient


class _FixtureHTTPResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _fixture_judgment(prompt: str) -> str:
    prediction = prompt.rsplit("Predicted answer: ", 1)[1]
    for marker in ("CORRECT", "MISTAKE", "NOT_ATTEMPTED"):
        if f"GRADE-AS-{marker}" in prediction:
            return f"Brief reasoning.\n```\nOutput: [{marker}]\n```"
    return "```\nOutput: [UNKNOWN]\n```"


def create_fixture_server():
    """Create a resources server whose three judge calls return deterministic fixture labels."""
    from resources_servers.facts_parametric.app import FACTSParametricConfig, FACTSParametricResourcesServer

    client = MagicMock(spec=ServerClient)

    async def _post(server_name, url_path, json=None, **kwargs):
        assert server_name == "facts_parametric_judge"
        assert url_path == "/v1/chat/completions"
        content = _fixture_judgment(json.messages[0]["content"])
        payload = {
            "id": f"chatcmpl-{hashlib.sha256(content.encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": 0,
            "model": "fixture-gemini-2.5-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        }
        return _FixtureHTTPResponse(payload)

    client.post = AsyncMock(side_effect=_post)
    config = FACTSParametricConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="facts_parametric",
        judge_model_server={"type": "responses_api_models", "name": "facts_parametric_judge"},
    )
    return FACTSParametricResourcesServer(config=config, server_client=client)
