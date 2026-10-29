# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline GDP.pdf verifier checks with fixed replies at the judge transport boundary."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.gdp_pdf.app import GdpPdfResourcesServer, GdpPdfVerifyRequest, GdpPdfVerifyResponse


def create_server() -> "GdpPdfResourcesServer":
    from resources_servers.gdp_pdf.app import GdpPdfConfig, GdpPdfResourcesServer

    server_dir = Path(__file__).resolve().parent
    config = OmegaConf.to_container(
        OmegaConf.load(server_dir / "configs/gdp_pdf.yaml").gdp_pdf.resources_servers.gdp_pdf,
        resolve=True,
    )
    config.update(
        host="127.0.0.1",
        port=8080,
        judge_prompt_template_fpath=str(server_dir / config["judge_prompt_template_fpath"]),
    )
    return GdpPdfResourcesServer(
        config=GdpPdfConfig.model_validate(config),
        server_client=MagicMock(spec=ServerClient),
    )


async def invoke(server: "GdpPdfResourcesServer", body: "GdpPdfVerifyRequest") -> "GdpPdfVerifyResponse":
    """Exercise the real judge parser and reward calculation without model services or a sandbox."""
    completion = NeMoGymChatCompletion(
        id="fixture_judge",
        created=0,
        model="fixture",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": body.fixture_judge_text},
            }
        ],
    )
    reply = MagicMock(ok=True)
    reply.read = AsyncMock(return_value=completion.model_dump_json())
    server.server_client.post = AsyncMock(return_value=reply)

    result = await server.verify(Request({"type": "http", "session": {}}), body)

    # Empty answers and missing rubrics must not call the judge. Successful
    # grading must retain the parsed per-criterion evidence, not just a scalar.
    judged = [evaluation for evaluation in result.criterion_evaluations if evaluation.response is not None]
    assert server.server_client.post.await_count == len(judged)
    assert len(judged) == body.fixture_expected_judge_calls
    assert result.all_pass == body.fixture_expected_all_pass
    return result
