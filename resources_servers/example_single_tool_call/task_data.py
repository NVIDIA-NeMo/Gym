# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the example_single_tool_call server.

This example's verify() is a stub that scores every rollout 1.0, so its rows carry no task-owned
fields at all: the task is fully described by ``responses_create_params``. The empty model still
ships so tooling (``gym env schema``, collate validation) treats the server uniformly.
"""

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")
