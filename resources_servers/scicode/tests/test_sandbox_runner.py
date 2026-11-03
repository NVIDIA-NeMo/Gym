# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import scicode_integration.sandbox_runner as runner

import resources_servers.scicode.app as app
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.scicode.tests.test_app import _request, _server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["pass", "fail", "timeout", "infra", "cancel", "create_failure", "sigkill", "malformed"]
)
async def test_remote_result_and_cleanup(outcome):
    result = {"passed": outcome == "pass", "error": "timeout" if outcome == "timeout" else ""}
    sandbox = MagicMock()
    sandbox.start = AsyncMock()
    sandbox.stop = AsyncMock()
    sandbox.exec = AsyncMock(return_value=SandboxExecResult(stdout=json.dumps(result), stderr="", return_code=0))
    if outcome == "sigkill":
        sandbox.exec.return_value = SandboxExecResult(
            stdout=json.dumps({**result, "returncode": -9}), stderr="", return_code=0
        )
    if outcome == "malformed":
        sandbox.exec.return_value = SandboxExecResult(stdout=json.dumps({"passed": "yes"}), stderr="", return_code=0)
    if outcome == "infra":
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="OOM", return_code=137)
    if outcome == "cancel":
        sandbox.exec.side_effect = asyncio.CancelledError()
    if outcome == "create_failure":
        sandbox.start.side_effect = RuntimeError("create failed")
    with (
        patch.object(runner, "get_global_config_dict", return_value={"sandbox": {"opensandbox": {}}}),
        patch.object(runner, "AsyncSandbox", return_value=sandbox) as factory,
    ):
        call = runner.run_sandbox_substep(
            "assert True",
            config=runner.ScicodeSandboxConfig(image="image@sha256:abc"),
            python_executable="/opt/2024/bin/python",
            timeout_secs=1800,
            test_data_path="/data/targets.h5",
            test_data_md5="abc",
            output_tail_chars=40,
        )
        if outcome in {"infra", "create_failure", "cancel", "sigkill", "malformed"}:
            with pytest.raises(asyncio.CancelledError if outcome == "cancel" else RuntimeError):
                await call
        else:
            assert await call == result
        spec = factory.call_args.kwargs["spec"]
        assert spec.resources.memory_mib == 8192
        assert spec.files["/tmp/scicode-program.py"] == "assert True"
        assert json.loads(spec.files["/tmp/scicode-request.json"])["timeout_secs"] == 1800
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_sandbox_preserves_or_scoring_and_sentinels_without_local_execution():
    server = _server(
        "/data/targets.h5",
        sandbox={"image": "test-image"},
        grading_interpreters=[
            {"name": "2024", "python_executable": "/opt/2024/bin/python"},
            {"name": "2025", "python_executable": "/opt/2025/bin/python"},
        ],
        required_grading_interpreters=2,
    )
    with (
        patch.object(app, "run_substep", side_effect=AssertionError("must not execute locally")),
        patch.object(app, "run_sandbox_substep", new_callable=AsyncMock) as remote,
    ):
        remote.side_effect = [{"passed": False, "error": "AssertionError"}, {"passed": True, "error": ""}]
        result = await server.verify(_request({"1.1": "solution", "1.2": "_ran_out_of_context_"}, n_steps=3))
    assert result.step_results == [True, False]
    assert result.num_steps_total == 2
    assert result.step_environment_results == [{"2024": False, "2025": True}, {}]
    assert remote.await_count == 2


@pytest.mark.asyncio
async def test_sandbox_concurrency_covers_entire_attempt():
    server = _server(
        "/data/targets.h5",
        sandbox={"image": "test-image"},
        num_processes=1,
        grading_interpreters=[{"name": "2024", "python_executable": "/opt/python"}],
    )
    active = peak = 0

    async def execute(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"passed": True, "error": ""}

    with patch.object(app, "run_sandbox_substep", side_effect=execute):
        result = await server.verify(_request({"1.1": "a", "1.2": "b"}))
    assert result.reward == 1
    assert peak == 1


@pytest.mark.asyncio
async def test_failed_step_cancels_and_awaits_sibling_cleanup():
    server = _server(
        "/data/targets.h5",
        sandbox={"image": "test-image"},
        num_processes=2,
        grading_interpreters=[{"name": "2024", "python_executable": "/opt/python"}],
    )
    sibling_started = asyncio.Event()
    sibling_cleaned = asyncio.Event()

    async def execute(program, **kwargs):
        if program.startswith("fail"):
            await sibling_started.wait()
            raise RuntimeError("sandbox infrastructure failure")
        try:
            sibling_started.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            sibling_cleaned.set()

    with patch.object(app, "run_sandbox_substep", side_effect=execute):
        with pytest.raises(RuntimeError, match="infrastructure failure"):
            await server.verify(_request({"1.1": "fail", "1.2": "wait"}))
    assert sibling_cleaned.is_set()
