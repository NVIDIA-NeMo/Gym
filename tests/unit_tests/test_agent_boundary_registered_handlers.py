# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An error the app has a handler for keeps that handler's answer; the agent boundary does not wrap it."""

import pytest
from fastapi import Body

from nemo_gym.failure_transport import FailureHeader
from nemo_gym.server_runtime._checkpoint.control import install_control_error_handler
from nemo_gym.server_runtime._checkpoint.errors import AdmissionClosedError, StaleAttemptError
from tests.unit_tests.server_runtime import test_session_routing
from tests.unit_tests.test_agent_seed_failure_boundary import _served


ROUTES = {
    "/run": ("run", {"responses_create_params": {"input": "hi"}}),
    "/v1/responses": ("responses", {"input": "hi"}),
    "/v1/agent_sessions": ("_seed_agent_session_state", None),
}


@pytest.mark.parametrize("error_type", [AdmissionClosedError, StaleAttemptError])
@pytest.mark.parametrize("path", list(ROUTES))
def test_a_checkpoint_control_error_keeps_its_409_answer(monkeypatch, path, error_type) -> None:
    method, body = ROUTES[path]

    async def refuse(self, body: dict = Body(default_factory=dict)):
        raise error_type("refused by the checkpoint")

    monkeypatch.setattr(test_session_routing._SessionAgent, method, refuse)
    worker = _served()
    install_control_error_handler(worker.app)
    with worker.client:
        if body is None:
            reply = worker.seed()
        else:
            reply = worker.client.post(path, json=body)

    assert reply.status_code == 409
    assert reply.json() == {"error": {"code": error_type.code, "detail": "refused by the checkpoint"}}
    failure = FailureHeader.decode(reply.headers)
    assert failure is None or not any(e.boundary == "agent" for e in (failure.events or {}).values())
