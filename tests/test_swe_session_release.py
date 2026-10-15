# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture-abort cleanup on swe_rebench and scale_swe stops only the session-owned sandbox."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import ReleaseSessionRequest
from nemo_gym.server_utils import SESSION_ID_KEY


@pytest.fixture(params=[("swe_rebench", "SWERebench"), ("scale_swe", "ScaleSWE")])
def release_case(request):
    package, prefix = request.param
    module = import_module(f"resources_servers.{package}.app")
    sandbox = SimpleNamespace(serialize=AsyncMock(return_value={"sandbox_id": "owned"}), stop=AsyncMock())
    server = SimpleNamespace(
        _session_id_to_sandbox={"session": sandbox},
        _session_id_to_pristine_untracked={"session": frozenset({"base"})},
        _session_id_to_base_commit={"session": "commit"},
    )
    return (
        getattr(module, f"{prefix}ResourcesServer").release_session,
        getattr(module, f"{prefix}SeedSessionResponse"),
        server,
        sandbox,
    )


def test_advertises_capture_abort_release(release_case):
    _, response_type, _, _ = release_case
    assert response_type(sandbox_handle="owned").model_dump()["supports_session_release"] is True


async def test_releases_owned_session(release_case):
    release, _, server, sandbox = release_case
    response = await release(
        server, SimpleNamespace(session={SESSION_ID_KEY: "session"}), ReleaseSessionRequest(sandbox_handle="owned")
    )
    assert response.model_dump() == {"sandbox_handle": "owned", "released": True}
    sandbox.stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._session_id_to_pristine_untracked == {}


@pytest.mark.parametrize("session,handle", [("foreign", "owned"), ("session", "foreign")])
async def test_rejects_foreign_session_or_handle_without_stopping(release_case, session, handle):
    release, _, server, sandbox = release_case
    with pytest.raises(HTTPException) as error:
        await release(
            server, SimpleNamespace(session={SESSION_ID_KEY: session}), ReleaseSessionRequest(sandbox_handle=handle)
        )
    assert error.value.status_code == 409
    sandbox.stop.assert_not_awaited()
    assert server._session_id_to_sandbox == {"session": sandbox}
