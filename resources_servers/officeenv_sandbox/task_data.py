# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the officeenv_sandbox server: the OfficeEnv server's rows
(``resources_servers/officeenv/task_data.py``), and what a task needs in its sandbox."""

from typing import Optional

from pydantic import Field

from resources_servers.officeenv.task_data import TaskData as OfficeEnvTaskData


class TaskData(OfficeEnvTaskData):
    sandbox_needs: Optional[list[str]] = Field(
        default=None,
        description=(
            "What the task needs in its sandbox beyond the shell, curl and Python with pandas and openpyxl: "
            "documents, office (app.py's TOOL_SETS). The episode gets the first configured image that provides them "
            "all, and a row none provides is refused (400 unsupported_needs)."
        ),
    )
