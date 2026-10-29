# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess


def test_sandbox_access_round_trips_an_owner_managed_connection() -> None:
    access = SandboxAccess(
        connection=DirectSandboxConnection(
            provider_config_ref="sandbox",
            descriptor={"sandbox_id": "sandbox-1"},
        ),
        workdir="/workspace",
    )

    assert SandboxAccess.model_validate(access.model_dump()) == access


def test_independent_agent_sessions_can_receive_the_same_sandbox_access() -> None:
    payload = {
        "connection": {
            "kind": "direct",
            "provider_config_ref": "sandbox",
            "descriptor": {"sandbox_id": "sandbox-1"},
        },
        "workdir": "/workspace",
    }

    first = SandboxAccess.model_validate(payload)
    second = SandboxAccess.model_validate(payload)

    assert first == second
    assert first is not second


def test_sandbox_access_rejects_unspecified_connection_fields() -> None:
    with pytest.raises(ValidationError):
        SandboxAccess.model_validate(
            {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "sandbox",
                    "descriptor": {"sandbox_id": "sandbox-1"},
                    "pty_session_id": "not-part-of-sandbox-access",
                },
                "workdir": "/workspace",
            }
        )


def test_sandbox_access_carries_the_agent_user() -> None:
    """Harnesses run every command as the task's `[agent].user`; absent, the sandbox's default user applies."""
    connection = DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"sandbox_id": "sandbox-1"})
    assert SandboxAccess(connection=connection, workdir="/app").user is None
    named = SandboxAccess(connection=connection, workdir="/app", user="agent")
    numeric = SandboxAccess(connection=connection, workdir="/app", user=1000)
    assert SandboxAccess.model_validate(named.model_dump()).user == "agent"
    assert SandboxAccess.model_validate(numeric.model_dump()).user == 1000
    with pytest.raises(ValidationError):
        SandboxAccess(connection=connection, workdir="/app", user=["agent"])


def test_sandbox_access_env_defaults_to_nothing_and_round_trips() -> None:
    """The owner's resolved environment for the harness's commands (the oracle's ``[solution].env``)."""
    connection = DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"sandbox_id": "sandbox-1"})
    assert SandboxAccess(connection=connection, workdir="/app").env == {}
    access = SandboxAccess(connection=connection, workdir="/app", env={"GREETING": "hi"})
    assert SandboxAccess.model_validate(access.model_dump(mode="json")).env == {"GREETING": "hi"}
    with pytest.raises(ValidationError):
        SandboxAccess(connection=connection, workdir="/app", env={"GREETING": 1})


def test_sandbox_access_keeps_an_unset_user_and_env_off_the_wire() -> None:
    """A seed for a harness the task declares nothing for carries the two keys harnesses matched before."""
    connection = DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"sandbox_id": "sandbox-1"})
    bare = SandboxAccess(connection=connection, workdir="/app")
    assert set(bare.model_dump()) == {"connection", "workdir"}
    assert set(bare.model_dump(mode="json")) == {"connection", "workdir"}
    assert SandboxAccess.model_validate(bare.model_dump()) == bare
    # Set values travel, a numeric user of 0 included.
    full = SandboxAccess(connection=connection, workdir="/app", user=0, env={"GREETING": "hi"})
    assert full.model_dump()["user"] == 0 and full.model_dump()["env"] == {"GREETING": "hi"}
