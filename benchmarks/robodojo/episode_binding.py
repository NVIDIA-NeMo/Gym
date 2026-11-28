# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo episode identity transferred between the environment and agent servers."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class RoboDojoEpisodeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: Annotated[str, Field(min_length=1)]
    assigned_case_id: Annotated[str, Field(min_length=1)]
    layout_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
