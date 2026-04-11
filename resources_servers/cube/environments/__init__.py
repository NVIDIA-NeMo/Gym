# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable CUBE ``environment`` implementations for the resources server.

Concrete implementations live in subpackages (e.g. ``environments.osworld``). This
package ``__init__`` only imports the ABC so ``import resources_servers.cube.environments``
does not pull in OSWorld (avoids import cycles with ``server`` / ``osworld.env``).
"""

from resources_servers.cube.environments.base import CubeEnvironmentBase


__all__ = ["CubeEnvironmentBase"]
