# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(autouse=True)
def _no_global_config(monkeypatch):
    """Servers read the judge endpoint from the global config, which would parse pytest's own argv."""
    monkeypatch.setattr("resources_servers.audio_eval.app.get_global_config_dict", lambda: {})
