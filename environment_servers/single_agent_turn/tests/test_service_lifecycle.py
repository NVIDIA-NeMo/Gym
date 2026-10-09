# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from environment_servers.single_agent_turn.tests.test_app import _Client, _environment_server, _request, _Response


@pytest.mark.parametrize("verify_error", [False, True])
async def test_finish_verify_close_order_and_final_cookies(monkeypatch, verify_error: bool) -> None:
    environment, client = _environment_server()
    environment.config.finish_agent_before_verification = True
    original = _Client.post
    service_alive = False

    async def post(self, server_name, url_path, **kwargs):
        nonlocal service_alive
        if url_path == "/v1/agent_sessions":
            service_alive = True
        elif url_path == "/v1/agent_sessions/finish":
            assert service_alive
            response = await original(self, server_name, "/v1/agent_sessions/close", **kwargs)
            self.calls[-1] = (server_name, url_path, kwargs)
            return response
        elif url_path == "/verify":
            assert service_alive
            assert kwargs["cookies"] == {"session": "updated-cookie"}
            if verify_error:
                self.calls.append((server_name, url_path, kwargs))
                self.responses.pop(0)
                raise RuntimeError("verifier failed")
        elif url_path == "/v1/agent_sessions/close":
            service_alive = False
            self.calls.append((server_name, url_path, kwargs))
            return _Response({"agent_session_id": kwargs["json"]["agent_session_id"]})
        elif url_path == "/close_session":
            assert not service_alive
        return await original(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await environment.run_request(_request())
    assert [path for _, path, _ in client.calls][-4:] == [
        "/v1/agent_sessions/finish",
        "/verify",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert not service_alive
    if verify_error:
        assert result.failure.stage == "verification"
    else:
        assert result.result.reward == 1
    assert not client.responses
