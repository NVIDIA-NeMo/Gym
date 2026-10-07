# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import ServerClient
from resources_servers.competitive_coding_challenges import ccc_eval, sandbox_worker
from resources_servers.competitive_coding_challenges.app import (
    CompetitiveCodingChallengesResourcesServer,
    CompetitiveCodingChallengesResourcesServerConfig,
    CompetitiveCodingChallengesVerifyRequest,
)
from resources_servers.competitive_coding_challenges.sandbox_evaluator import SandboxCCCEvaluator


@pytest.mark.parametrize(
    "case",
    [
        "pass",
        "fail",
        "compile_error",
        "worker_error",
        "artifact_error",
        "invalid_result",
        "upload_error",
        "cancel_upload",
        "cancel",
    ],
)
async def test_ccc_grading_and_cleanup(monkeypatch, tmp_path: Path, case: str):
    problem = {
        "compile": "g++ grader.cpp",
        "run": "./grader",
        "grader_files": [["grader.cpp", "original grader"]],
        "all_tests": {"one": {"input": "\u00ff", "output": "2"}},
        "subtasks": {"all_tests": {"aggregation": "min", "score": 1, "test_names": ["one"]}},
    }
    metadata = tmp_path / "tests.jsonl"
    metadata.write_text(json.dumps({"metadata": {"p": problem}}) + "\n")
    events = []
    requests = []
    worker_result = {}

    class Shell(sandbox_worker.ProcessSandbox):
        async def execute_code(self, generated_code, *_args, **_kwargs):
            command = generated_code
            if case == "compile_error" and "./compile.sh" in command:
                return {"stdout": "", "stderr": "compile failed", "process_status": "error"}, None
            if command == "echo hello world":
                stdout = "hello world"
            elif "./run.sh" in command:
                stdout = "1" if case == "pass" else "0"
            else:
                stdout = ""
            return {"stdout": stdout, "stderr": "", "process_status": "completed"}, None

    def forbid_http(*_args, **_kwargs):
        raise AssertionError("CCC attempted to use the NeMo-Skills HTTP service")

    monkeypatch.setattr(sandbox_worker, "ProcessSandbox", Shell)
    monkeypatch.setattr(ccc_eval.LocalSandbox, "__init__", forbid_http)

    class Provider:
        async def create(self, spec):
            assert spec.ttl_s > 1800
            return SandboxHandle("box", "fake", None)

        async def upload_file(self, _handle, source, target):
            if case == "cancel_upload":
                raise asyncio.CancelledError
            if case == "upload_error":
                raise RuntimeError("upload failed")
            if target == "/tmp/ccc-request.json":
                requests.append(json.loads(source.read_text()))

        async def exec(self, _handle, command, **_kwargs):
            assert "sandbox_worker" in command
            if case == "cancel":
                raise asyncio.CancelledError
            if case in {"pass", "fail", "compile_error"}:
                worker_result.update(await sandbox_worker.evaluate(requests[-1], tmp_path))
            return SandboxExecResult("", "failed", 1 if case == "worker_error" else 0)

        async def download_file(self, _handle, _remote, local):
            if case == "invalid_result":
                worker_result.update(test_case_results={"all_tests": {"score": "garbled"}})
            local.write_text("broken" if case == "artifact_error" else json.dumps(worker_result))

        async def close(self, _handle):
            events.append("close")

        async def aclose(self):
            events.append("aclose")

    monkeypatch.setattr("nemo_gym.sandbox.api.create_provider", lambda _config: Provider())
    server = CompetitiveCodingChallengesResourcesServer(
        config=CompetitiveCodingChallengesResourcesServerConfig(
            name="ccc",
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            test_file=str(metadata),
            sandbox_provider={"fake": {}},
            sandbox_spec={"image": "python"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    evaluator = SandboxCCCEvaluator(
        {"test_file": str(metadata), "test_batch_size": 1},
        num_parallel_requests=1,
        provider={"fake": {}},
        spec=server.config.sandbox_spec,
        python="python",
        setup_command=None,
        timeout=1800,
        named_configs={},
    )
    monkeypatch.setattr(server, "_evaluator", evaluator)
    request = CompetitiveCodingChallengesVerifyRequest.model_validate(
        {
            "problem_id": "p",
            "subtask": "all_tests",
            "subtask_score": 1,
            "responses_create_params": {"input": [{"role": "user", "content": "Solve this"}]},
            "response": {
                "id": "r",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [],
            },
        }
    )
    if case in {"pass", "fail", "compile_error"}:
        result = await server.verify(request)
        assert result.reward == (1.0 if case == "pass" else 0.0)
    else:
        with pytest.raises(
            asyncio.CancelledError if case in {"cancel", "cancel_upload"} else (ValueError, RuntimeError)
        ):
            await server.verify(request)
    assert events == ["close", "aclose"]
    if requests:
        assert requests[0]["metadata"]["metadata"]["p"] == problem


@pytest.mark.parametrize(
    "command,timeout,status,stdout",
    [
        ("printf ok", 1, "completed", "ok"),
        ("printf failed; exit 7", 1, "error", "failed"),
        ("sleep 2", 0.02, "timeout", ""),
    ],
)
async def test_worker_shell_status(command, timeout, status, stdout):
    result, session = await sandbox_worker.ProcessSandbox().execute_code(command, timeout=timeout)
    assert session is None
    assert result["process_status"] == status
    assert result["stdout"] == stdout
