# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy Responses API agent with combined harness and rollout responsibilities."""

from fastapi import Body, FastAPI

from nemo_gym.agents.base import BaseResponsesAPIAgent, BaseResponsesAPIAgentConfig
from nemo_gym.processors.base import BaseProcessor


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, BaseProcessor):
    """Legacy agent that combines harness and rollout responsibilities."""

    def setup_webserver(self) -> FastAPI:
        app = BaseResponsesAPIAgent.setup_webserver(self)
        return self.register_processor_routes(app)


__all__ = [
    "BaseResponsesAPIAgent",
    "BaseResponsesAPIAgentConfig",
    "Body",
    "SimpleResponsesAPIAgent",
]
