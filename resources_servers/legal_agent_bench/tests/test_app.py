# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LAB policy adds to shared Harbor execution without replacing its lifecycle."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from environment_servers.harbor.app import HarborEnvironmentServer
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.sandbox.providers.base import SandboxExecResult
from resources_servers.legal_agent_bench.app import LegalAgentBenchEnvironmentConfig, LegalAgentBenchEnvironmentServer


def server(tmp_path, **overrides):
    mapping = tmp_path / "tasks.json"
    mapping.write_text(
        json.dumps({"legal_agent_bench": {"folder": str(tmp_path / "tasks"), "tasks": {"one": "digest"}}})
    )
    config = LegalAgentBenchEnvironmentConfig(
        name="lab",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        agent_server={"type": "responses_api_agents", "name": "agent"},
        prepared_tasksets_file=str(mapping),
        artifacts_dir=tmp_path / "artifacts",
        judges=[{"model_name": "gpt-5.5", "api_key": "test-only-judge-key"}],
        **overrides,
    )
    return LegalAgentBenchEnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_prepared_mapping_and_secrets_are_loaded_without_rewriting_task_files(tmp_path):
    app = server(tmp_path)
    assert app.config.tasksets["legal_agent_bench"].tasks == {"one": "digest"}
    assert "test-only-judge-key" in app.config.secrets["LAB_JUDGES"]
    assert "test-only-judge-key" not in (tmp_path / "tasks.json").read_text()
    assert app.config.reward_mode == "full_task"
    assert app.config.judge_parallelism == 6
    assert app._seed.__func__ is HarborEnvironmentServer._seed


@pytest.mark.parametrize("reward", [0.0, 0.5, 1.0])
async def test_successful_rewards_are_preserved(tmp_path, reward):
    app = server(tmp_path)
    result = app._score(
        SimpleNamespace(task_id="one"),
        SimpleNamespace(network_mode="public"),
        SandboxExecResult(return_code=0, stdout="", stderr=""),
        tmp_path,
        asyncio.get_running_loop().time(),
        timed_out=False,
        rewards={"reward": reward, "judge_error_count": 0},
        problem=None,
    )
    assert result["reward"] == reward
    assert not result.get("mask_sample", False)


@pytest.mark.parametrize("failure_field", ["verifier_error", "judge_error_count"])
async def test_grading_errors_are_masked_in_the_verifier_result(tmp_path, failure_field):
    app = server(tmp_path)
    result = app._score(
        SimpleNamespace(task_id="one"),
        SimpleNamespace(network_mode="public"),
        SandboxExecResult(return_code=1, stdout="", stderr=""),
        tmp_path,
        asyncio.get_running_loop().time(),
        timed_out=False,
        rewards={"reward": 0, failure_field: 1},
        problem=None,
    )
    assert result["mask_sample"] is True
    assert result["failure_kind"] == "lab_judge_failed"


async def test_masked_verification_uses_operational_failure_boundary(tmp_path, monkeypatch):
    app = server(tmp_path)
    monkeypatch.setattr(
        HarborEnvironmentServer,
        "_verified_outcome",
        AsyncMock(
            return_value={
                "reward": 0,
                "mask_sample": True,
                "failure_reason": "judge unavailable",
                "verifier_logs_dir": "/reports/task",
            }
        ),
    )
    with pytest.raises(HTTPException, match="judge unavailable") as error:
        await app._verified_outcome(None, None, None)
    assert error.value.status_code == 503
    assert "/reports/task" in error.value.detail


@pytest.mark.parametrize(
    ("rewards", "timed_out", "masked", "reward"),
    [
        (None, False, True, 0),
        ({}, False, True, 0),
        ({"judge_error_count": 0}, False, True, 0),
        (None, True, False, 0),
        ({"reward": 0.5}, True, False, 0.5),
    ],
)
async def test_missing_grades_fail_but_measured_timeouts_remain_scoreable(
    tmp_path, rewards, timed_out, masked, reward
):
    app = server(tmp_path)
    outcome = app._score(
        SimpleNamespace(task_id="one"),
        SimpleNamespace(network_mode="public", timeout_sec=30),
        SandboxExecResult(return_code=124 if timed_out else 1, stdout="", stderr=""),
        tmp_path,
        asyncio.get_running_loop().time(),
        timed_out=timed_out,
        rewards=rewards,
        problem="No reward written",
    )
    assert bool(outcome.get("mask_sample")) is masked
    assert outcome["reward"] == reward
    if masked:
        assert outcome["failure_kind"] == "lab_judge_failed"


async def test_staging_copies_only_public_inputs_using_shared_transfer(tmp_path):
    import io
    import tarfile

    from environment_servers.harbor.tests.conftest import FakeSandbox

    app = server(tmp_path)
    root = tmp_path / "task"
    for name, text in (
        ("environment/vdr/source.txt", "public document"),
        ("environment/skills/docx/SKILL.md", "public skill"),
        ("tests/task.json", "private criteria"),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    sandbox = FakeSandbox()
    task = SimpleNamespace(path=root, config=SimpleNamespace(environment=SimpleNamespace(skills_dir=None)))
    await app._upload_task_skills(sandbox, task, "/workspace")
    contents = {}
    for payload in sandbox.uploaded_archives:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            for member in archive:
                if member.isfile():
                    contents[member.name] = archive.extractfile(member).read().decode()
    assert sorted(contents.values()) == ["public document", "public skill"]
    assert all("private criteria" not in item for item in contents.values())
    commands = "\n".join(call["command"] for call in sandbox.execs)
    assert "/workspace/vdr" in commands and "/workspace/skills" in commands
    assert not sandbox.stopped
    await app.shutdown()


async def test_verifier_provenance_comes_from_server_settings(tmp_path, monkeypatch):
    from resources_servers.legal_agent_bench import app as module

    app = server(tmp_path)
    app._gym_commit = "tested-source-commit"
    monkeypatch.setattr(
        module,
        "get_global_config_dict",
        lambda: {"agent": {"responses_api_agents": {"legal_agent_bench_native": {"max_turns": 60}}}},
    )
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        status="completed",
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
        metadata={"lab_harness_provenance": '{"gym_commit":"forged"}', "lab_finish_reason": "no_tool_calls"},
    )
    parent = AsyncMock(return_value={"reward": 1.0})
    monkeypatch.setattr(HarborEnvironmentServer, "_run_verifier", parent)
    await app._run_verifier(None, b"trajectory", response=response)
    sent = parent.call_args.kwargs["response"]
    assert json.loads(sent.metadata["lab_harness_provenance"]) == {
        "agent_id": "legal_agent_bench_native",
        "harness_pin": {"max_turns": 60},
        "gym_commit": "tested-source-commit",
    }
    assert sent.metadata["lab_finish_reason"] == "no_tool_calls"
    assert "forged" in response.metadata["lab_harness_provenance"]
