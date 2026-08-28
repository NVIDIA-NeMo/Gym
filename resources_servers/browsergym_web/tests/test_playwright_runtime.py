# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import ModuleType, SimpleNamespace

from resources_servers.browsergym_web import playwright_runtime


def test_thread_local_playwright_patches_all_browsergym_getters_and_stops_per_thread(monkeypatch):
    original_getter = lambda: None
    core = SimpleNamespace(__version__="0.14.3", _PLAYWRIGHT=None, _get_global_playwright=original_getter)
    env = SimpleNamespace(_get_global_playwright=original_getter)
    chat = SimpleNamespace(_get_global_playwright=original_getter)
    created: list[SimpleNamespace] = []

    def start():
        instance = SimpleNamespace(owner=threading.get_ident(), stopped=False)
        instance.stop = lambda: setattr(instance, "stopped", True)
        created.append(instance)
        return instance

    sync_api = ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: SimpleNamespace(start=start)
    monkeypatch.setattr(playwright_runtime, "_INSTALLED", False)
    monkeypatch.setattr(playwright_runtime, "_THREAD_STATE", threading.local())
    monkeypatch.setattr(playwright_runtime, "_load_browsergym_modules", lambda: (core, env, chat))
    original_import_module = playwright_runtime.importlib.import_module
    monkeypatch.setattr(
        playwright_runtime.importlib,
        "import_module",
        lambda name: sync_api if name == "playwright.sync_api" else original_import_module(name),
    )

    playwright_runtime.install_thread_local_playwright()
    first = env._get_global_playwright()
    assert first is chat._get_global_playwright()
    playwright_runtime.close_thread_local_playwright()
    second = core._get_global_playwright()

    assert first.stopped is True
    assert second is not first
    assert second.owner == threading.get_ident()
