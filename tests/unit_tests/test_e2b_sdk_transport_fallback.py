# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharing Gym's aiohttp pool must never be able to fail sandbox creation."""

from __future__ import annotations

import sys
import types

import pytest

from nemo_gym.sandbox.providers.e2b import _sdk


@pytest.fixture(autouse=True)
def _fake_e2b(monkeypatch):
    """A stand-in SDK, so the test does not depend on e2b being installed."""

    module = types.ModuleType("e2b")

    class ConnectionConfig:
        integration = None

        @classmethod
        def set_integration(cls, value):
            cls.integration = value

    module.ConnectionConfig = ConnectionConfig
    monkeypatch.setitem(sys.modules, "e2b", module)
    monkeypatch.setattr(_sdk, "_CONFIGURED_SDK_MODULES", {})
    return module


@pytest.mark.parametrize(
    "raised",
    [
        # The one that actually happened: the lazy import lands in a Ray worker,
        # reaches Hydra's parser, which rejects Ray's argv and exits.
        SystemExit(2),
        RuntimeError("no running loop"),
        ImportError("httpx_aiohttp"),
    ],
    ids=["systemexit", "runtime", "import"],
)
def test_transport_setup_failure_does_not_stop_the_sdk(monkeypatch, caplog, raised):
    def explode() -> None:
        raise raised

    monkeypatch.setattr(_sdk, "_configure_async_http", explode)

    sdk = _sdk.require_e2b_sdk("test")

    assert sdk is sys.modules["e2b"]
    assert sdk.ConnectionConfig.integration is not None
    assert "Falling back to the e2b SDK's own HTTP transport" in caplog.text


def test_transport_setup_runs_when_it_can(monkeypatch):
    calls = []
    monkeypatch.setattr(_sdk, "_configure_async_http", lambda: calls.append(1))

    _sdk.require_e2b_sdk("test")
    _sdk.require_e2b_sdk("test")

    # Configured once per SDK module, not once per call.
    assert calls == [1]
