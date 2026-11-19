# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task input the UserSim episode request validates: the UserSim resources server's task data.

Collation and ``gym env validate`` load this module without the server's dependencies (see ``nemo_gym/data/task_input.py``).
"""

from resources_servers.nemo_user_sim.task_data import TaskData as TaskInput


__all__ = ["TaskInput"]
