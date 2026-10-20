# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-configured execution identity; never supplied by the task or actor."""

from typing import Annotated

from pydantic import StringConstraints

from resources_servers.partner_if.grading_contract import StrictModel, Text


SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class ExecutionIdentity(StrictModel):
    run_id: Text
    policy_model: Text
    code_revision: GitRevision
    profile: Text
    configuration_sha256: SHA256
