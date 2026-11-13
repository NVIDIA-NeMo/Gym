# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for mini-SWE agent sessions alone, which need no Ray.

Gym starts Ray for a server whose class declares ``ray_enabled = True``, as ``app.py`` does for the SWE-Gym ``/run``
pipeline. Agent sessions run mini-SWE in the task sandbox, so this entrypoint serves the same agent without Ray and
refuses ``/run``. ``configs/mini_swe_agent_sandboxed.yaml`` starts it.
"""

from typing import ClassVar

from fastapi import HTTPException

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.mini_swe_agent.app import MiniSWEAgent, MiniSWEAgentRunRequest, MiniSWEAgentVerifyResponse


class MiniSWESessionsAgent(MiniSWEAgent):
    """The mini-SWE agent with agent sessions only."""

    # Gym reads both flags from this file's source before the server starts, so each is declared here, not inherited.
    ray_enabled = False
    calls_model_from_sandbox: ClassVar[bool] = True

    async def run(self, body: MiniSWEAgentRunRequest) -> MiniSWEAgentVerifyResponse:
        raise HTTPException(422, "mini-SWE /run needs Ray; start it from app.py (configs/mini_swe_agent.yaml)")


if __name__ == "__main__":
    MiniSWESessionsAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = MiniSWESessionsAgent.run_webserver()  # noqa: F401
