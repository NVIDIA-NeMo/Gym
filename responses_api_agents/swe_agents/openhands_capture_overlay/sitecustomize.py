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

"""Preserve correlated Gym routes in OpenHands' pinned Gym client.

OpenHands runs in a separate Python environment and intentionally pins its own
``nemo-gym`` version.  Importing the entire current Gym tree there can pull in
dependencies that environment does not provide.  Python loads this module at
startup from the launcher-provided ``PYTHONPATH``, allowing us to patch only
the outbound request routing contract while leaving the pinned package intact.
"""

import os
from typing import Any


_MODEL_SERVER_NAME_ENV = "NEMO_GYM_MODEL_SERVER_NAME"
_MODEL_SERVER_BASE_URL_ENV = "NEMO_GYM_MODEL_SERVER_BASE_URL"
_CAPABILITY_HEADER = "x-nemo-gym-capture-capability"
_CAPTURE_PATH_SEGMENT = "training-token-capture"


def _install_capture_route_patch() -> None:
    try:
        from nemo_gym import server_utils
    except ModuleNotFoundError as error:
        # The launcher also invokes tooling from its base Miniforge Python,
        # where nemo-gym is intentionally absent.  The OpenHands venv that
        # makes model calls does provide it and will install the patch.
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

        if _CAPTURE_PATH_SEGMENT in model_server_base_url:
            capability = os.getenv("OPENAI_API_KEY")
            if capability:
                headers = dict(kwargs.get("headers") or {})
                headers.setdefault(_CAPABILITY_HEADER, capability)
                kwargs["headers"] = headers

        return await server_utils.request(
            method=method,
            url=f"{model_server_base_url.rstrip('/')}{url_path}",
            _internal=True,
            **kwargs,
        )

    request_with_capture_route._nemo_gym_capture_route_patch = True  # type: ignore[attr-defined]
    server_client_cls.request = request_with_capture_route


_install_capture_route_patch()
