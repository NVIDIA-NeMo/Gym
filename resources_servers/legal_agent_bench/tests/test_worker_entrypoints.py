# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker imports must expose an app without starting unrelated entrypoints."""

import runpy
from pathlib import Path
from unittest.mock import patch, sentinel

import pytest


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("entrypoint", ["resources_servers/legal_agent_bench/harbor_bridge.py"])
def test_worker_entrypoint_exports_app_only_when_selected(entrypoint: str, worker: bool) -> None:
    root = Path(__file__).resolve().parents[3]
    with (
        patch("nemo_gym.utils.is_nemo_gym_fastapi_entrypoint", return_value=worker),
        patch("nemo_gym.server_runtime.SimpleServer.run_webserver", return_value=sentinel.app) as start,
    ):
        namespace = runpy.run_path(str(root / entrypoint), run_name="lab_worker_test")
    if worker:
        start.assert_called_once_with()
        assert namespace["app"] is sentinel.app
    else:
        start.assert_not_called()
        assert "app" not in namespace
