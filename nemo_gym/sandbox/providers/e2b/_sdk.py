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

"""Lazy E2B SDK loading, traffic attribution, and async HTTP adaptation."""

import threading
from typing import Any

from nemo_gym.package_info import __version__


E2B_SDK_CONSTRAINT = "e2b>=2.36.0,<3.0.0"
_INTEGRATION = f"nemo-gym/{__version__}"
_CONFIGURED_SDK_MODULES: dict[int, Any] = {}
_CONFIGURE_LOCK = threading.Lock()


def _configure_async_http() -> None:
    """Route the E2B SDK's httpx control-plane clients through Gym's aiohttp pool.

    E2B 2.x does not expose transport injection on its high-level async API,
    so its two module-level factories are the narrowest available seam. The
    ConnectRPC command streams use pyqwest and are unaffected.
    """
    from urllib.parse import urlsplit

    import httpx
    from e2b.api import client_async
    from e2b.sandbox_async import main as sandbox_async
    from httpx_aiohttp import AiohttpTransport

    from nemo_gym.server_utils import get_global_aiohttp_client

    class E2BAiohttpTransport(AiohttpTransport):
        async def aclose(self) -> None:
            # The shared session is owned and closed by server_utils.
            return None

    def build_transport(
        config: Any,
        http2: bool = True,
        *,
        for_streaming: bool = False,
    ) -> E2BAiohttpTransport:
        # aiohttp speaks HTTP/1.1; the E2B endpoints support it. Streamed and
        # regular requests share Gym's globally configured connection pool.
        del http2, for_streaming
        proxy = config.proxy
        if proxy is not None:
            proxy_url = str(proxy.url if isinstance(proxy, httpx.Proxy) else proxy)
            if urlsplit(proxy_url).scheme.lower() not in {"http", "https"}:
                raise ValueError("The E2B aiohttp integration requires an HTTP or HTTPS proxy URL")
            proxy = httpx.Proxy(proxy_url)
        return E2BAiohttpTransport(
            client=get_global_aiohttp_client,
            proxy=proxy,
        )

    client_async.get_transport = build_transport
    client_async.get_envd_transport = build_transport
    # E2B <2.46 imported this factory by value in AsyncSandbox. Newer versions
    # import get_envd_api, whose module global resolves the patched factory.
    if hasattr(sandbox_async, "get_transport"):
        sandbox_async.get_transport = build_transport


def require_e2b_sdk(feature: str) -> Any:
    """Import E2B lazily and attribute this process's NeMo Gym SDK traffic."""
    try:
        import e2b
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        if exc.name != "e2b":
            # Preserve the actual missing transitive module from a broken SDK
            # installation instead of incorrectly claiming E2B is absent.
            raise
        raise ImportError(
            f"{feature} requires the 'e2b' package. Install it with `pip install '{E2B_SDK_CONSTRAINT}'`."
        ) from exc

    module_id = id(e2b)
    with _CONFIGURE_LOCK:
        if _CONFIGURED_SDK_MODULES.get(module_id) is not e2b:
            e2b.ConnectionConfig.set_integration(_INTEGRATION)
            _configure_async_http()
            _CONFIGURED_SDK_MODULES[module_id] = e2b
    return e2b
