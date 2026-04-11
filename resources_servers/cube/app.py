# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUBE resources server entrypoint (YAML ``environment`` picks the adapter, like OpenEnv ``env_class``).

On ``python app.py`` under ``ng_run``, we install optional per-environment wheels **before** importing the
server stack so ``osworld_cube`` and similar deps are present when ``CubeResourcesServer`` loads.
"""

from __future__ import annotations


if __name__ == "__main__":
    import logging
    import os

    logging.basicConfig(level=os.environ.get("NEMO_GYM_CUBE_BOOTSTRAP_LOG_LEVEL", "INFO"))
    from resources_servers.cube.bootstrap import maybe_install_environment_extras

    maybe_install_environment_extras()

    from resources_servers.cube.server import CubeResourcesServer

    CubeResourcesServer.run_webserver()
