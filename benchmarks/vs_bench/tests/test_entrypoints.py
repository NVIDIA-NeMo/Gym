# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual subprocess and worker entry points for each adapter."""

import runpy
from pathlib import Path

import pytest

from nemo_gym.server_runtime.base_server import SimpleServer


@pytest.mark.parametrize(
    "path",
    [
        "resources_servers/vs_bench_reference/app.py",
        "responses_api_agents/vs_bench_reference/app.py",
        "responses_api_agents/vs_bench_gameplay/app.py",
    ],
)
def test_server_script_and_worker_entrypoints(monkeypatch, path):
    called = []
    sentinel = object()
    monkeypatch.setattr(
        SimpleServer, "run_webserver", classmethod(lambda cls: called.append(cls.__name__) or sentinel)
    )
    runpy.run_path(path, run_name="__main__")
    assert len(called) == 1
    monkeypatch.setattr(
        "nemo_gym.utils.is_nemo_gym_fastapi_entrypoint", lambda file: Path(file).resolve() == Path(path).resolve()
    )
    result = runpy.run_path(path, run_name="gym_worker_fixture")
    assert len(called) == 2
    assert result["app"] is sentinel
