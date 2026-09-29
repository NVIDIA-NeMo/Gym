# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Route OpenHands model requests through the rollout-specific Gym URL.

The pinned nv-OpenHands fork calls the model through ``nemo_gym.server_utils.ServerClient``
(``openhands/agenthub/nemo_gym_client.py``), not through litellm and ``llm.base_url``.
That client resolves a bare host and port from the global config and knows nothing about
the ``/ng-rollout/<id>/training-token-capture`` prefix, so the prefixed ``llm.base_url``
written into the OpenHands config has no effect on model routing.

The fork pins ``nemo-gym`` from Gym ``main``, which predates the
``NEMO_GYM_MODEL_SERVER_BASE_URL`` override in ``ServerClient.request``. Python loads this
module at interpreter startup from the launcher-provided ``PYTHONPATH`` and patches only
``ServerClient.request`` to honor that variable. The base Miniforge interpreter has no
``nemo-gym`` installed and skips the patch.

Remove this module once nv-OpenHands re-locks ``nemo-gym`` on a Gym commit that includes the
env override natively.
"""

import os
from typing import Any


_MODEL_SERVER_NAME_ENV = "NEMO_GYM_MODEL_SERVER_NAME"
_MODEL_SERVER_BASE_URL_ENV = "NEMO_GYM_MODEL_SERVER_BASE_URL"


def _install_capture_route_patch() -> None:
    try:
        from nemo_gym import server_utils
    except ModuleNotFoundError as error:
        # The base Miniforge interpreter also loads this module without ``nemo-gym`` installed.
        # Model requests run in the OpenHands virtual environment.
        # That environment has ``nemo-gym`` installed and applies the patch.
        if error.name == "nemo_gym":
            return
        raise

    from pydantic import BaseModel

    server_client_cls = server_utils.ServerClient
    if getattr(server_client_cls.request, "_nemo_gym_capture_route_patch", False):
        return

    original_request = server_client_cls.request

    async def request_with_capture_route(
        self: Any, server_name: str, url_path: str, method: str, **kwargs: Any
    ) -> Any:
        model_server_name = os.getenv(_MODEL_SERVER_NAME_ENV)
        model_server_base_url = os.getenv(_MODEL_SERVER_BASE_URL_ENV)
        if not model_server_base_url or server_name != model_server_name:
            return await original_request(
                self,
                server_name=server_name,
                url_path=url_path,
                method=method,
                **kwargs,
            )

        if "json" in kwargs and isinstance(kwargs["json"], BaseModel):
            kwargs["json"] = kwargs["json"].model_dump(exclude_unset=True)

        return await server_utils.request(
            method=method,
            url=f"{model_server_base_url.rstrip('/')}{url_path}",
            _internal=True,
            **kwargs,
        )

    request_with_capture_route._nemo_gym_capture_route_patch = True  # type: ignore[attr-defined]
    server_client_cls.request = request_with_capture_route


_install_capture_route_patch()
