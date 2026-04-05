# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable CUBE ``env_domain`` implementations for the resources server.

Concrete implementations live in subpackages (e.g. ``domains.osworld``). This
package ``__init__`` only imports the ABC so ``import resources_servers.cube.domains``
does not pull in OSWorld (avoids import cycles with ``server`` / ``bootstrap``).
"""

from resources_servers.cube.domains.base import CubeEnvironmentBase


__all__ = ["CubeEnvironmentBase"]
