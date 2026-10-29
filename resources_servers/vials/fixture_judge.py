# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A service-free judge for the verifier fixture and the unit tests.

It answers the real judge request the server builds: ``equivalent`` is true when the
candidate equals the GTFA after case and whitespace folding. A ``replies`` list overrides
that with raw judge texts, one per call, to exercise retries and the image re-grade.
"""

import re
from typing import Any, Optional
from unittest.mock import MagicMock

import orjson

from nemo_gym.server_utils import ServerClient


_FIELD = re.compile(
    r"Ground-truth final answer:\n(?P<gtfa>.*?)\n\n.*Candidate final answer:\n(?P<answer>.*?)\n\n", re.S
)


class _FakeHTTPResponse:
    ok = True

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = orjson.dumps(payload)

    async def read(self) -> bytes:
        return self._body


def judge_response(text: str) -> dict[str, Any]:
    """A minimal Responses API body whose output text is ``text``."""
    return {
        "id": "fixture_judge",
        "created_at": 0,
        "model": "fixture-judge",
        "object": "response",
        "output": [
            {
                "id": "fixture_judge_message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


class FixtureJudgeClient:
    def __init__(self, replies: Optional[list[str]] = None) -> None:
        self.replies = list(replies) if replies is not None else None
        self.requests: list[dict[str, Any]] = []

    async def post(self, *, server_name: str, url_path: str, json: Any) -> _FakeHTTPResponse:
        body = json.model_dump(exclude_none=True) if hasattr(json, "model_dump") else json
        self.requests.append(body)
        if self.replies is not None:
            return _FakeHTTPResponse(judge_response(self.replies.pop(0)))
        user = body["input"][-1]["content"]
        text = user if isinstance(user, str) else user[-1]["text"]
        fields = _FIELD.search(text)
        same = fields is not None and fields["gtfa"].strip().casefold() == fields["answer"].strip().casefold()
        verdict = {"reasoning": "fixture judge", "equivalent": same}
        return _FakeHTTPResponse(judge_response(orjson.dumps(verdict).decode()))


def create_fixture_server(client: Optional[FixtureJudgeClient] = None, **config: Any) -> Any:
    from resources_servers.vials.app import VialsResourcesServer, VialsResourcesServerConfig

    server = VialsResourcesServer(
        config=VialsResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="vials",
            judge_model_server={"type": "responses_api_models", "name": "judge_model"},
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    server.server_client.post = (client or FixtureJudgeClient()).post
    return server
