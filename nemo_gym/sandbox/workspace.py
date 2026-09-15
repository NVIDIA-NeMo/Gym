# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-owned workspace handoff returned by /seed_session."""

from typing import Any

from pydantic import BaseModel, Field


class SandboxWorkspace(BaseModel):
    """The environment owns cleanup; the agent borrows execution and file access.

    ``provider`` references a provider block in the merged Gym configuration.
    ``descriptor`` is produced by AsyncSandbox.serialize(), not a bare sandbox ID.
    The environment must accept an idempotent POST to ``/cleanup_session`` using
    the seed response's cookies, including when execution or verification fails.
    """

    provider: str = Field(min_length=1)
    descriptor: dict[str, Any] = Field(min_length=1)
