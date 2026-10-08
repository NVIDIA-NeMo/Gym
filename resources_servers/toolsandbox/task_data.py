# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the toolsandbox server.

Same pointer shape as aviary (imported, not redefined): the row's only task datum is
``task_idx``, here an index into the deterministic sorted registry of vendored ToolSandbox
scenarios built in-process at runtime — the task definition lives in code, OUT of the row
(wire-required by ToolSandboxSeedSessionRequest). verify() is a cached-reward lookup keyed by
env_id (milestone similarity computed at /close, defaulting to 0.0).
"""

from resources_servers.aviary.task_data import TaskData as AviaryTaskData


class TaskData(AviaryTaskData):
    """ToolSandbox pointer row; ``task_idx`` indexes the vendored scenario registry."""
