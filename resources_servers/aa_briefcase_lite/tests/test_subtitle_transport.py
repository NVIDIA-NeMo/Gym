# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from nemo_gym.server_utils import ServerClient
from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
)
from resources_servers.gdpval.judge_panel import ResolvedJudge


@pytest.mark.parametrize("media_mode", ["native_pdf", "images_and_text"])
async def test_binary_request_preserves_valid_and_malformed_subtitles(
    monkeypatch, tmp_path: Path, media_mode: str
) -> None:
    """Capture serialized SDK requests with real section and binary-call code."""
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    server = AABriefcaseLiteResourcesServer(
        config=AABriefcaseLiteResourcesServerConfig(
            host="127.0.0.1",
            port=18000,
            entrypoint="",
            dataset_dir=str(tmp_path),
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            preconvert_office_to_pdf=False,
            binary_formatting_retries=0,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    server._aa_binary_system = "Inspect the submitted files."
    server._aa_binary_user = "{task_markdown}\n{check_description}\n{score_1_criteria}\n{score_0_criteria}"
    judge = ResolvedJudge(
        name="judge",
        model="test-model",
        base_url="http://upstream.invalid/v1",
        handles_video=True,
        media_mode=media_mode,
    )
    check = {
        "check_description": "Are the subtitles valid SRT?",
        "score_1_criteria": "All cue timestamps use valid SRT syntax.",
        "score_0_criteria": "A cue timestamp has invalid syntax.",
    }
    subtitles = [
        "1\n00:00:01,250 --> 00:00:02,750\nHello, café.\n\n",
        "1\ninvalid timestamp -> missing end\nHello, café.\n\n",
    ]
    video = b"synthetic-video-payload"
    (tmp_path / "clip.mp4").write_bytes(video)
    requests = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-subtitle-transport",
                "object": "chat.completion",
                "created": 0,
                "model": judge.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"passed": true, "reasoning": "capture"}'},
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(capture)) as transport:
        monkeypatch.setattr(
            "resources_servers.aa_briefcase_lite.app.AsyncOpenAI", partial(AsyncOpenAI, http_client=transport)
        )
        for content in subtitles:
            (tmp_path / "captions.srt").write_text(content, encoding="utf-8")
            section = await server._section(tmp_path, judge, [])
            await server._binary_call(judge, "Provide a clip and captions.", check, section)

    assert len(requests) == 2
    assert requests[0] != requests[1]
    for request, content in zip(requests, subtitles, strict=True):
        blocks = request["messages"][1]["content"]
        assert {"type": "text", "text": content} in blocks
        media = [block for block in blocks if block["type"] != "text"]
        kind = "image_url" if media_mode == "native_pdf" else "video_url"
        assert media == [{"type": kind, kind: {"url": f"data:video/mp4;base64,{base64.b64encode(video).decode()}"}}]
