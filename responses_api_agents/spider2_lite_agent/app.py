# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spider2-Lite agent with row-local handling for oversized prompts."""

import logging
from time import time

from aiohttp import ClientResponseError
from fastapi import Request, Response

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.simple_agent.app import Body, SimpleAgent


logger = logging.getLogger(__name__)


class Spider2LiteAgent(SimpleAgent):
    """Score oversized Spider2 rows as zero without weakening the shared agent."""

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        try:
            return await super().responses(request, response, body)
        except ClientResponseError as error:
            if not _is_context_window_error(error):
                raise

            logger.warning(
                "Spider2-Lite prompt exceeded the model context window. "
                "Returning an empty response so this row scores zero."
            )
            # Preserve cookies seeded by the resources server for the subsequent
            # verification request made by SimpleAgent.run().
            for name, value in request.cookies.items():
                response.set_cookie(name, value)
            created_at = time()
            return NeMoGymResponse(
                id=f"spider2-lite-context-window-exceeded-{int(created_at * 1000)}",
                created_at=created_at,
                model=self.config.model_server.name,
                object="response",
                status="failed",
                output=[],
                tools=[],
                tool_choice="none",
                parallel_tool_calls=False,
            )


# Context-window detection helpers used by Spider2LiteAgent.responses.
# Extend this tuple as new providers surface new phrases for input-size limits.
_CONTEXT_ERROR_MARKERS = (
    "contextwindowexceedederror",
    "context_length_exceeded",
    "context window",
    "maximum context length",
    "max input tokens",
    "exceeds maximum input length",
    "input is too long",
    "prompt is too long",
)


def _is_context_window_error(error: ClientResponseError) -> bool:
    """Return whether an upstream HTTP 400 explicitly reports an input-size limit."""
    if error.status != 400:
        return False
    content = getattr(error, "response_content", b"")
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    text = str(content).lower()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)


if __name__ == "__main__":
    Spider2LiteAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Spider2LiteAgent.run_webserver()  # noqa: F401
