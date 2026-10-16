# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for how an agent refuses an agent session seed it cannot serve.

:func:`check_agent_seed_rejected` posts the seed through the app's real ``/v1/agent_sessions`` route, in process, the
way an Environment Server calls a deployed agent. Call it from an agent's own tests with a seed that agent does not
support, such as a sandbox or a required grant it cannot use.
"""

from fastapi import FastAPI

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest


# An Environment Server retries these 4xx statuses as transient, like 5xx.
_RETRIED_CLIENT_ERRORS = frozenset({408, 425, 429})


def check_agent_seed_rejected(app: FastAPI, seed: AgentSeedSessionRequest) -> str:
    """Seed ``seed`` through ``app``, and raise AssertionError unless the agent refuses it as final.

    The refusal must be a 4xx status the Environment Server does not retry, so the episode fails once with the
    agent's reason. Raising ``AgentSeedRejectedError`` gives HTTP 422; a plain ``ValueError`` gives 500, which is
    retried. Returns the refusal's detail for the caller to check.
    """
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
    assert 400 <= response.status_code < 500 and response.status_code not in _RETRIED_CLIENT_ERRORS, (
        f"unsupported seed must fail with a final 4xx, got {response.status_code}: {response.text}"
    )
    return response.json()["detail"]
