# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess


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
