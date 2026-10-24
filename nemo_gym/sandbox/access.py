# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable access to an owner-managed sandbox."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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
    # The user the harness runs every command as (Harbor's ``[agent].user``); ``None`` keeps the sandbox's default.
    # Both stay off the wire while unset, so a seed for a harness the task declares nothing for reads as before.
    user: str | int | None = Field(default=None, exclude_if=lambda value: value is None)
    # Environment the owner resolved for the harness's commands, over the sandbox's own (Harbor's ``[solution].env``
    # for the oracle, resolved on the owner's host); empty for a harness the task declares nothing for.
    env: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
