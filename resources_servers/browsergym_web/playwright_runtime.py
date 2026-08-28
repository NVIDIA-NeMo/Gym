# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thread-local Playwright lifecycle for concurrent BrowserGym sessions."""

from __future__ import annotations

import importlib
import logging
import threading
from types import ModuleType
from typing import Any, Callable


LOG = logging.getLogger("nemo_gym.resources_servers.browsergym_web.playwright_runtime")

_INSTALL_LOCK = threading.Lock()
_THREAD_STATE = threading.local()
_INSTALLED = False


def _thread_local_playwright() -> Any:
    playwright = getattr(_THREAD_STATE, "playwright", None)
    if playwright is None:
        sync_api = importlib.import_module("playwright.sync_api")
        playwright = sync_api.sync_playwright().start()
        _THREAD_STATE.playwright = playwright
    return playwright


def close_thread_local_playwright() -> None:
    """Stop the Playwright driver owned by the calling session thread."""

    playwright = getattr(_THREAD_STATE, "playwright", None)
    if playwright is None:
        return
    try:
        playwright.stop()
    finally:
        del _THREAD_STATE.playwright


def _load_browsergym_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    return (
        importlib.import_module("browsergym.core"),
        importlib.import_module("browsergym.core.env"),
        importlib.import_module("browsergym.core.chat"),
    )


def install_thread_local_playwright() -> None:
    """Replace BrowserGym's process-global getter with a thread-local getter.

    BrowserGym 0.14.x imports the private getter into both ``env`` and ``chat``,
    so all three references must be replaced. Structural guards fail closed if
    a future BrowserGym release changes this private integration point.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        core, env, chat = _load_browsergym_modules()
        original_getter: Callable[[], Any] | None = getattr(core, "_get_global_playwright", None)
        if original_getter is None:
            raise RuntimeError("BrowserGym no longer exposes _get_global_playwright")
        if getattr(core, "_PLAYWRIGHT", None) is not None:
            raise RuntimeError("BrowserGym global Playwright was initialized before thread-local isolation")
        if getattr(env, "_get_global_playwright", None) is not original_getter:
            raise RuntimeError("BrowserGym env Playwright getter has an unsupported shape")
        if getattr(chat, "_get_global_playwright", None) is not original_getter:
            raise RuntimeError("BrowserGym chat Playwright getter has an unsupported shape")

        core._get_global_playwright = _thread_local_playwright
        env._get_global_playwright = _thread_local_playwright
        chat._get_global_playwright = _thread_local_playwright
        _INSTALLED = True
        LOG.info(
            "event=browsergym_thread_local_playwright_installed version=%s",
            getattr(core, "__version__", "unknown"),
        )
