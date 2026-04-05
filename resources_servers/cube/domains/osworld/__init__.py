# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OSWorld domain: benchmark bootstrap + :class:`OSWorldEnvironment`."""

from resources_servers.cube.domains.osworld.bootstrap import ensure_osworld_tasks
from resources_servers.cube.domains.osworld.environment import OSWorldEnvironment


__all__ = ["OSWorldEnvironment", "ensure_osworld_tasks"]
