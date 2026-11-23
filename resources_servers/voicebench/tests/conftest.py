# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision the same IFEval tokenizer resources as server startup."""

import pytest

from resources_servers.voicebench.app import _ensure_nltk


def pytest_configure(config: pytest.Config) -> None:
    _ensure_nltk(None)


@pytest.fixture(autouse=True)
def _no_global_config(monkeypatch):
    """Servers read the judge endpoint from the global config, which would parse pytest's own argv."""
    monkeypatch.setattr("resources_servers.voicebench.app.get_global_config_dict", lambda: {})
