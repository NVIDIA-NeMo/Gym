# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for how an agent refuses an agent session seed it cannot serve.

:func:`check_agent_seed_rejected` posts the seed through the app's real ``/v1/agent_sessions`` route, in process, the
way an Environment Server calls a deployed agent. Call it from an agent's own tests with a seed that agent does not
support, such as a sandbox or a required grant it cannot use.
"""

from fastapi import FastAPI

from nemo_gym.agents.base_responses_api_agent import AgentSeedSessionRequest


def check_agent_seed_rejected(app: FastAPI, seed: AgentSeedSessionRequest) -> str:
    """Seed ``seed`` through ``app``, and raise AssertionError unless the agent refuses it as final.

    The refusal must be HTTP 422 with a string ``detail``, which is what ``AgentSeedRejectedError`` gives, so the
    Environment Server fails the episode once with the agent's reason. A plain ``ValueError`` gives 500, which is
    retried. Returns the refusal's detail for the caller to check.
    """
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
    assert response.status_code == 422, (
        f"unsupported seed must fail with a final 422 from AgentSeedRejectedError, got {response.status_code}: "
        f"{response.text}"
    )
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    assert isinstance(detail, str), f"the 422 refusal must carry a string detail, got: {response.text}"
    return detail
