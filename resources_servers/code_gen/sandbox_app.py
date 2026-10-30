# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run code verification through a sandbox provider without starting Ray."""

from app import CompCodingResourcesServer


class CompCodingSandboxResourcesServer(CompCodingResourcesServer):
    """Sandbox-only entrypoint whose startup does not require a Ray cluster."""

    ray_enabled = False

    def model_post_init(self, context: object) -> None:
        if self.config.sandbox_provider is None:
            raise ValueError("sandbox_app.py requires sandbox_provider; use app.py for Ray verification")
        super().model_post_init(context)


if __name__ == "__main__":
    CompCodingSandboxResourcesServer.run_webserver()
