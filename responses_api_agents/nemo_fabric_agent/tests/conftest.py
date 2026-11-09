# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from responses_api_agents.nemo_fabric_agent import app


@pytest.fixture
def model_url(monkeypatch):
    """Give the agent a Model Server URL without a resolved server configuration."""
    monkeypatch.setattr(app.NeMoFabricAgent, "resolve_model_base_url", lambda *_: "http://model/v1")


@pytest.fixture(autouse=True)
def _restore_adapter_python():
    """Host runs and the in-sandbox worker set ADAPTER_PYTHON; it must not leak into later tests."""
    saved = os.environ.get("ADAPTER_PYTHON")
    yield
    if saved is None:
        os.environ.pop("ADAPTER_PYTHON", None)
    else:
        os.environ["ADAPTER_PYTHON"] = saved
