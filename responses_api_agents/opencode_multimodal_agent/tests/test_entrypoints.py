# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker imports expose the app once; normal imports never start servers."""

import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from nemo_gym.server_runtime.base_server import SimpleServer


@pytest.mark.parametrize(
    "relative_path",
    [
        "responses_api_agents/opencode_multimodal_agent/app.py",
        "responses_api_agents/opencode_multimodal_agent/eval.py",
        "responses_api_agents/opencode_multimodal_agent/combined.py",
        "resources_servers/opencode_multimodal_benchmarks/app.py",
    ],
)
@pytest.mark.parametrize("worker", [False, True])
def test_worker_entrypoints(relative_path, worker, monkeypatch):
    entrypoint = Path(__file__).resolve().parents[3] / relative_path
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_ENTRYPOINT", relative_path)
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_WORKER", "1" if worker else "0")
    sentinel = object()
    start = Mock(return_value=sentinel)
    monkeypatch.setattr(SimpleServer, "run_webserver", start)
    namespace = runpy.run_path(str(entrypoint), run_name="worker_entrypoint_test")
    if worker:
        assert namespace["app"] is sentinel
        start.assert_called_once_with()
    else:
        assert "app" not in namespace
        start.assert_not_called()
