# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses API invocation adapters for the NOOA capability fixtures."""

import json
from typing import Any

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


async def invoke_calculate(agent: Any, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    """Invoke the calculate capability from its structured Responses input."""
    if not isinstance(request.input, list) or not request.input:
        raise ValueError("calculate capability input must contain a message")
    content = request.input[-1].content
    if not isinstance(content, str):
        raise ValueError("calculate capability input message must contain text")
    arguments = json.loads(content)
    return await agent.calculate(
        a=arguments["a"],
        b=arguments["b"],
        calculation=arguments["calculation"],
    )
