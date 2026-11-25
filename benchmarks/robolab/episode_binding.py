# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab identity transferred from the environment server to the agent session."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class RoboLabEpisodeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: Annotated[str, Field(min_length=1)]
    seed: Annotated[int, Field(ge=0)]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
