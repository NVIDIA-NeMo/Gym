# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain actual mini-SWE checkpoints when the task sandbox disappears."""

import asyncio
import json
import os
import shutil
import signal
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import evaluate_record
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.terminal_bench_4.resource_limits import sandbox_resource_limit
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig, original_response_id
from responses_api_agents.miniswe_sandboxed_agent.runtime.checkpoint_agent import CheckpointAgent


@pytest.mark.parametrize("provider_failure", ["exception", "nonzero_exit"])
async def test_eviction_during_tool_keeps_native_decision_and_provider_failure(
    tmp_path, runner_factory, monkeypatch, provider_failure
):
    durable = tmp_path / "efs" / "episode" / "miniswe"
    evicted = False
    recovered_bytes = {}
    detail = "[SANDBOX::ENDED] Sandbox has ended (Evicted): Pod ephemeral local storage usage exceeds the total limit of containers 4Gi"

    async def query(params):
        return NeMoGymResponse.model_validate(
            {
                "id": "before-eviction",
                "created_at": 0,
                "object": "response",
                "model": "model",
                "parallel_tool_calls": False,
                "tools": [],
                "tool_choice": "auto",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "unfinished-tool",
                        "name": "bash",
                        "arguments": '{"command":"echo $$ > tool.pid; sleep 60"}',
                    }
                ],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    async def recover(directory):
        assert evicted and not root.exists()
        for name in ("trajectory.json", "agent.log"):
            recovered_bytes[name] = (durable / name).read_bytes()
            shutil.copyfile(durable / name, directory / name)
        return True

    harness = await runner_factory(
        context=HarnessContext(
            session_id="evicted", task_id="0", rollout_id="0-0", instruction="inspect", durable_agent_logs=True
        ),
        config=MiniSWEConfig(),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path / "retained",
        evidence_directory=durable,
        recover_artifacts=recover,
        observability_enabled=True,
    )
    sandbox = harness.sandbox
    root = sandbox.directory
    execute = sandbox.exec
    download = sandbox.download

    async def execute_until_evicted(command, **kwargs):
        nonlocal evicted
        if evicted:
            if provider_failure == "nonzero_exit":
                return SandboxExecResult("", "", 0)
            raise RuntimeError(detail)
        if not command.startswith("setsid --fork --wait"):
            return await execute(command, **kwargs)
        running = asyncio.create_task(execute(command, **kwargs))
        async with asyncio.timeout(60):
            while not (root / "tool.pid").exists():
                assert not running.done(), (durable / "agent.log").read_text()
                await asyncio.sleep(0.025)
        saved = json.loads((durable / "trajectory.json").read_text())
        assert saved["info"]["exit_status"] == ""
        assert original_response_id(saved["messages"][-1]["id"]) == "before-eviction"
        tool_pid = int((root / "tool.pid").read_text())
        os.killpg(sandbox.runners[-1].pid, signal.SIGKILL)
        os.killpg(tool_pid, signal.SIGKILL)
        result = await running
        shutil.rmtree(root)
        evicted = True
        if provider_failure == "exception":
            raise RuntimeError(detail)
        return SandboxExecResult(result.stdout, detail, result.return_code)

    async def download_after_eviction(remote, local):
        if evicted:
            raise RuntimeError(detail)
        return await download(remote, local)

    monkeypatch.setattr(sandbox, "exec", execute_until_evicted)
    monkeypatch.setattr(sandbox, "download", download_after_eviction)
    response, termination, extra = await harness.execute(90)
    assert termination.reason == "infrastructure_error"
    assert "ephemeral local storage" in termination.detail
    assert sandbox_resource_limit(termination.detail).resource == "storage"
    assert response.usage.total_tokens == 5
    assert [turn["model_calls"][0]["response_id"] for turn in extra["ng_trajectory"]["turns"]] == ["before-eviction"]
    assert extra["ng_agent_observations"]["records"][-1]["status"] == "incomplete"
    assert extra["mini_swe_artifact_recovery"] == {"recovered": True, "error": None}
    assert (harness.directory / "trajectory.json").read_bytes() == recovered_bytes["trajectory.json"]
    assert (harness.directory / "agent.log").read_bytes() == recovered_bytes["agent.log"]
    gaps = {gap["code"] for gap in extra["ng_trajectory"]["gaps"]}
    assert {"native_trajectory_partial", "turn_model_call_scope_incomplete"} <= gaps
    record = {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": response.model_dump(mode="json"), **extra}
    _attach_trajectory_record(record, record)
    health = evaluate_record(record, frozenset())
    assert health.verdict != "healthy"
    assert "rollout_token_count_mismatch" in health.unobserved


def test_atomic_save_failure_retains_previous_native_checkpoint(tmp_path, monkeypatch):
    class Serializable:
        def serialize(self):
            return {}

    agent = CheckpointAgent(Serializable(), Serializable(), system_template="system", instance_template="{{task}}")
    agent.messages = [{"role": "user", "content": "actual message"}]
    path = tmp_path / "trajectory.json"
    path.write_bytes(b"previous complete checkpoint")

    def fail_replace(source, destination):
        assert Path(source).is_file() and destination == path
        raise OSError("simulated interrupted checkpoint replacement")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted checkpoint"):
        agent.save(path)
    assert path.read_bytes() == b"previous complete checkpoint"
    assert list(tmp_path.iterdir()) == [path]
