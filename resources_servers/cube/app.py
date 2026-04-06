# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUBE resources server entrypoint (YAML ``env_domain`` picks the adapter, like OpenEnv ``env_class``)."""

from resources_servers.cube.server import CubeResourcesServer


if __name__ == "__main__":
    CubeResourcesServer.run_webserver()
