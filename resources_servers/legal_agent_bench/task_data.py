# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the legal_agent_bench server.

legal_agent_bench serves its prepared LAB Harbor tasks through harbor_tasks, so a row names one task by
``harbor_dataset`` (``legal_agent_bench``) and ``task_name``, and carries the task's instruction.md as the user
message. The task-local Harbor verifier grades the sandbox; the instruction's ``lab_task_id`` marker tells the LAB
harness which task it is solving.
"""

from resources_servers.harbor_tasks.task_data import TaskData


__all__ = ["TaskData"]
