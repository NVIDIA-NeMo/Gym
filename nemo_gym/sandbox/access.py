# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable access to an owner-managed sandbox."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue


class DirectSandboxConnection(BaseModel):
    """Reconnect through a named top-level sandbox-provider configuration."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct"] = "direct"
    provider_config_ref: str
    descriptor: dict[str, JsonValue]


# TODO: Add a sandbox-server connection after that server and its lease contract exist.


class SandboxAccess(BaseModel):
    """Describe borrower access to an owner-managed sandbox."""

    model_config = ConfigDict(extra="forbid")

    connection: DirectSandboxConnection
    workdir: str
