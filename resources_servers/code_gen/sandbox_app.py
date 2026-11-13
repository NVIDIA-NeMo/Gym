# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run code verification through a sandbox provider without starting Ray."""

from app import CompCodingResourcesServer

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class CompCodingSandboxResourcesServer(CompCodingResourcesServer):
    """Compatibility entrypoint for configs that previously selected sandbox_app.py."""

    ray_enabled = False


if __name__ == "__main__":
    CompCodingSandboxResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = CompCodingSandboxResourcesServer.run_webserver()  # noqa: F401
