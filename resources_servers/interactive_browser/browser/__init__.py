# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser backends for the interactive-browser environment."""

from .base import (
    BrowserBackend,
    BrowserSessionError,
    BrowserSessionHandle,
    BrowserSessionProvider,
    BrowserSessionSpec,
    Element,
    Observation,
)
from .local_playwright import LocalPlaywrightBackend
from .page import PlaywrightConnectedBackend, PlaywrightPageDriver
from .registry import (
    create_backend,
    create_session_provider,
    list_backends,
    list_session_providers,
    register_backend,
    register_session_provider,
)
from .remote_cdp import RemoteCDPBackend, StaticCDPProvider


__all__ = [
    "BrowserBackend",
    "BrowserSessionError",
    "BrowserSessionHandle",
    "BrowserSessionProvider",
    "BrowserSessionSpec",
    "Element",
    "LocalPlaywrightBackend",
    "Observation",
    "PlaywrightConnectedBackend",
    "PlaywrightPageDriver",
    "RemoteCDPBackend",
    "StaticCDPProvider",
    "create_backend",
    "create_session_provider",
    "list_backends",
    "list_session_providers",
    "register_backend",
    "register_session_provider",
]
