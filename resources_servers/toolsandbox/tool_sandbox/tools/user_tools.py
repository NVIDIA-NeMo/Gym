# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# For licensing see accompanying LICENSE file.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. The original Apple-authored portions of this
# file remain subject to the Apple license referenced above.

"""A collection of tools dedicated for user access, mostly to support user simulation."""

import polars as pl

from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType, get_current_context
from tool_sandbox.common.utils import register_as_tool


@register_as_tool(visible_to=(RoleType.USER,))
def end_conversation() -> None:
    """Finish the conversation

    Trigger this tool when you think the agent have completed the task for you,
    or the agent is unable to complete the task. Either way this tool will stop the conversation

    Returns:

    Raises:
        ValueError: If conversation already ended
    """
    current_context = get_current_context()
    sandbox_database = current_context.get_database(DatabaseNamespace.SANDBOX)
    if not sandbox_database["conversation_active"][-1]:
        raise ValueError("Conversation already ended")
    current_context.update_database(
        DatabaseNamespace.SANDBOX,
        dataframe=sandbox_database.with_columns(~pl.col("conversation_active")),
    )
