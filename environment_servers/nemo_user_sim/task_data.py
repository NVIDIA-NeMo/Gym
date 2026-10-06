# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields that the ``nemo_user_sim`` environment server reads: its resources server's ``TaskData``.

The environment server runs the user and assistant agents itself and reads the same task fields its
resources server declares, so it shares that schema instead of keeping a copy.
"""

from resources_servers.nemo_user_sim.task_data import TaskData


__all__ = ["TaskData"]
