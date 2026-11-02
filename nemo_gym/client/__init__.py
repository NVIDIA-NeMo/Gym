# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Calling HTTP endpoints and Gym servers: the shared aiohttp client with retries, and `ServerClient`."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.client.http_client import (
        GlobalAIOHTTPAsyncClientConfig,
        format_exception_with_response_body,
        get_global_aiohttp_client,
        get_response_json,
        is_global_aiohttp_client_request_debug_enabled,
        is_global_aiohttp_client_setup,
        raise_for_status,
        request,
        set_global_aiohttp_client,
    )
    from nemo_gym.client.server_client import (
        SESSION_ID_KEY,
        ServerClient,
        apply_rollout_prefix,
        get_server_url,
        rollout_path_prefix,
        setup_server_client,
        trace_context_path_segment,
    )


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "GlobalAIOHTTPAsyncClientConfig": "http_client",
    "SESSION_ID_KEY": "server_client",
    "ServerClient": "server_client",
    "apply_rollout_prefix": "server_client",
    "format_exception_with_response_body": "http_client",
    "get_global_aiohttp_client": "http_client",
    "get_response_json": "http_client",
    "get_server_url": "server_client",
    "is_global_aiohttp_client_request_debug_enabled": "http_client",
    "is_global_aiohttp_client_setup": "http_client",
    "raise_for_status": "http_client",
    "request": "http_client",
    "rollout_path_prefix": "server_client",
    "set_global_aiohttp_client": "http_client",
    "setup_server_client": "server_client",
    "trace_context_path_segment": "server_client",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
