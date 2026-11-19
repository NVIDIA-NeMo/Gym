# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service URL overrides for clients that run inside a task sandbox."""

from collections.abc import Mapping
from urllib.parse import urlsplit

from nemo_gym.config.global_config import get_first_server_config_dict, maybe_get_global_config_dict


def sandbox_service_url(name: str, *, global_config: Mapping[str, object] | None = None) -> str | None:
    """Read a service's sandbox_url; None preserves its ordinary host and port.

    One override applies to all sandboxed clients in a Gym run. An override does
    not provision a listener or grant permission to call the service.
    """
    config = global_config if global_config is not None else maybe_get_global_config_dict()
    if config is None:
        return None
    if name not in config:
        raise ValueError(f"No configured Gym service {name!r}")
    server = get_first_server_config_dict(config, name)
    url = server.get("sandbox_url")
    if url is None:
        return None
    if not isinstance(url, str) or not url:
        raise ValueError(f"Service {name!r} sandbox_url must be an HTTP(S) base URL")
    parsed = urlsplit(url)
    # Access port to reject malformed authorities before handing a URL to a client.
    port = parsed.port
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port == 0
        or any(char.isspace() for char in url)
    ):
        raise ValueError(
            f"Service {name!r} sandbox_url must be an HTTP(S) base URL without credentials, query or fragment"
        )
    return url.rstrip("/")
