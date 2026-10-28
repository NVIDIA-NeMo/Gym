# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    src_id: Optional[str] = None
    lang: Optional[str] = None
    tag: Optional[str] = None
    length: Optional[str] = None
    nvoice_want: Optional[int] = None
    bpm: Optional[int] = None
    meter: Optional[str] = None
