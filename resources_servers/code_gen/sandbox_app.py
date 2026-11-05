# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run code verification through a sandbox provider without starting Ray."""

from app import CompCodingResourcesServer


class CompCodingSandboxResourcesServer(CompCodingResourcesServer):
    """Compatibility entrypoint for configs that previously selected sandbox_app.py."""

    ray_enabled = False


if __name__ == "__main__":
    CompCodingSandboxResourcesServer.run_webserver()
