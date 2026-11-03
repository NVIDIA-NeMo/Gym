# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.nemo_fabric_agent import sandbox_runner


async def test_worker_isolates_adapter_python_but_keeps_task_environment(tmp_path, monkeypatch):
    # A SWE image may set these for its own Python. They must not poison Fabric's
    # Python subprocess, and removing them globally would change the task's tools.
    monkeypatch.setenv("PYTHONHOME", "/task-python-not-the-agent")
    monkeypatch.setenv("PYTHONPATH", "/task-modules")
    monkeypatch.setenv("ADAPTER_PYTHON", "task-python")
    task = tmp_path / "task"
    task.mkdir()
    invocation = {
        "config": {
            "metadata": {"name": "test"},
            "harness": {"adapter_id": "nvidia.fabric.mini-swe-agent"},
            "environment": {"provider": "local", "workspace": str(task)},
            "models": {"default": {"provider": "openai", "model": "policy"}},
        },
        "input": "question",
        "request_id": "rollout",
        "workdir": str(task),
    }
    (tmp_path / "input.json").write_text(json.dumps(invocation))
    result = {"status": "failed", "error": {"code": "mini_swe_agent_incomplete"}, "output": {"output": "partial"}}

    async def invoke(config, *, base_dir, request):
        assert base_dir == str(task) and request.request_id == "rollout"
        program = "import json,os,sys; print(json.dumps([sys.flags.isolated,os.environ['PYTHONHOME'],os.environ['PYTHONPATH']]))"
        completed = subprocess.run(
            [os.environ["ADAPTER_PYTHON"], "-c", program], capture_output=True, text=True, check=True, timeout=10
        )
        assert json.loads(completed.stdout) == [1, "/task-python-not-the-agent", "/task-modules"]
        return SimpleNamespace(to_mapping=lambda: result)

    monkeypatch.setattr(sandbox_runner, "Fabric", lambda: SimpleNamespace(run=AsyncMock(side_effect=invoke)))
    await sandbox_runner.run(tmp_path)
    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert not (tmp_path / "result.tmp").exists()
    assert list(task.iterdir()) == []
    assert Path(os.environ["ADAPTER_PYTHON"]).is_relative_to(tmp_path)


async def test_worker_orders_relay_trajectories_and_survives_files_of_another_shape(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (tmp_path / "input.json").write_text(
        json.dumps(
            {
                "config": {
                    "metadata": {"name": "test"},
                    "harness": {"adapter_id": "nvidia.fabric.mini-swe-agent"},
                    "environment": {"provider": "local", "workspace": str(task)},
                    "models": {"default": {"provider": "openai", "model": "policy"}},
                },
                "input": "question",
                "request_id": "rollout-a2",
                "workdir": str(task),
            }
        )
    )
    contents = {
        "late": {"steps": [{"source": "agent", "message": "late", "timestamp": "2026-01-01T00:00:02Z"}]},
        "early": {"steps": [{"source": "agent", "message": "early", "timestamp": "2026-01-01T00:00:01Z"}]},
        "a_list": [],
        "null_step": {"steps": [None]},
    }
    artifacts = []
    for name, content in contents.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(content))
        artifacts.append({"kind": "atif", "path": str(tmp_path / f"{name}.json")})
    (tmp_path / "cut.json").write_bytes(b'{"steps": [{"source": "agent", "message": "\xe2\x82')
    artifacts += [{"kind": "atif", "path": str(tmp_path / "cut.json")}, {"kind": "otel", "path": "ignored"}, "x"]
    seen = {}

    async def invoke(config, *, base_dir, request):
        seen["context"], seen["request_id"] = request.context, request.request_id
        return SimpleNamespace(
            to_mapping=lambda: {"status": "succeeded", "output": {"response": "a", "relay_artifacts": artifacts}}
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sandbox_runner, "Fabric", lambda: SimpleNamespace(run=AsyncMock(side_effect=invoke)))
        await sandbox_runner.run(tmp_path)

    written = json.loads((tmp_path / "result.json").read_text())
    # Timestamped trajectories in time order, then the step-less one ("None" sorts last); the other two are dropped.
    assert written["relay_atif"] == [contents["early"], contents["late"], contents["null_step"]]
    # The rollout's capture key reaches Fabric both as the request id and as the context the model route is keyed by.
    assert seen == {"context": {"nemo_gym_rollout_id": "rollout-a2"}, "request_id": "rollout-a2"}
