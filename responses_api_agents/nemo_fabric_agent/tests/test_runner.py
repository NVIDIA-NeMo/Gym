# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
