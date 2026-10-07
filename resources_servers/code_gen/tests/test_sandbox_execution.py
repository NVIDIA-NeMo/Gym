# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import subprocess
import sys
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox.providers import SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import ServerClient
from resources_servers.code_gen import app


@pytest.mark.parametrize(
    "case,results,reward",
    [
        ("pass", [True, True], 1.0),
        ("wrong", [False], 0.0),
        ("compile", [-2], 0.0),
        ("timeout", [-1, -1], 0.0),
        ("worker_error", [], None),
        ("setup_error", [], None),
        ("artifact_error", [], None),
        ("empty_result", [], None),
        ("cancel_upload", [], None),
        ("upload_error", [], None),
        ("cancel", [], None),
        ("legacy", [True, True], 1.0),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
async def test_sandbox_grading_and_lifecycle(monkeypatch, case, results, reward, packed):
    specs = []
    events = []
    requests = []

    class Provider:
        async def create(self, spec):
            specs.append(spec)
            events.append("create")
            return SandboxHandle(sandbox_id="test", provider_name="fake", raw=None)

        async def upload_file(self, _handle, source_path, target_path):
            events.append("upload")
            if case == "cancel_upload":
                raise asyncio.CancelledError
            if case == "upload_error":
                raise RuntimeError("upload failed")
            if target_path == "/tmp/code-gen-request.json":
                requests.append(json.loads(source_path.read_text()))

        async def exec(self, _handle, command, **kwargs):
            events.append("exec")
            if case == "setup_error":
                assert command == "prepare"
                return SandboxExecResult(stdout="", stderr="setup failed", return_code=1)
            assert "sandbox_worker" in command
            assert kwargs["timeout_s"] == 57  # Two tests, ten seconds each, plus harness/transport margin.
            if case == "cancel":
                raise asyncio.CancelledError
            return SandboxExecResult(stdout="", return_code=1 if case == "worker_error" else 0, stderr="worker failed")

        async def download_file(self, _handle, _remote_path, target_path):
            events.append("download")
            target_path.write_text(
                "invalid" if case == "artifact_error" else json.dumps({"result": results, "metadata": None})
            )

        async def close(self, _handle):
            events.append("close")

        async def aclose(self):
            events.append("aclose")

    monkeypatch.setattr("nemo_gym.sandbox.api.create_provider", lambda _config: Provider())

    async def legacy(*args):
        requests.append(dict(sample=args[0], generation=args[1], timeout=args[2], debug=args[3]))
        return results, None

    remote = MagicMock(side_effect=legacy)
    monkeypatch.setattr(app.check_correctness_remote, "remote", remote)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"sandbox": {"fake": {}}}
    server = app.CompCodingResourcesServer(
        config=app.CompCodingResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="code_gen",
            num_processes=1,
            unit_test_timeout_secs=10,
            debug=False,
            sandbox_setup_command="prepare" if case == "setup_error" else None,
            sandbox_provider=None if case == "legacy" else "sandbox",
            sandbox_spec=None if case == "legacy" else {"image": "test@sha256:abc"},
        ),
        server_client=client,
    )
    tests = {"inputs": ["2", "3"], "outputs": ["4", "9"]}
    request = app.CompCodingVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Square n"}]},
        response=NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="test",
            object="response",
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            output=[
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "```python\nprint(int(input()) ** 2)\n```", "annotations": []}
                    ],
                }
            ],
        ),
        verifier_metadata={
            "unit_tests": {"packed": base64.b64encode(zlib.compress(json.dumps(tests).encode())).decode()}
            if packed
            else tests
        },
    )
    for _ in range(2):
        if reward is None:
            with pytest.raises(
                asyncio.CancelledError if case in ("cancel", "cancel_upload") else (RuntimeError, ValueError)
            ):
                await server.verify(request)
        else:
            response = await server.verify(request)
            assert response.reward == reward
            assert response.result == results
            assert response.extracted_model_code == "print(int(input()) ** 2)"
        if case != "legacy":
            assert events[-2:] == ["close", "aclose"]
    if case != "legacy":
        assert specs[0].metadata["instance_id"] != specs[1].metadata["instance_id"]
        remote.assert_not_called()
    else:
        assert not specs
        assert remote.call_count == 2
    if requests:
        assert json.loads(requests[0]["sample"]["input_output"]) == {**tests, "fn_name": None}
        assert requests[0]["generation"] == "print(int(input()) ** 2)"


@pytest.mark.parametrize(
    "generation,fn_name,expected",
    [
        ("print(int(input()) ** 2)", None, [True, True]),
        ("print(0)", None, [False]),
        ("def square(n): return n ** 2", "square", [True, True]),
        ("def broken(:", None, [-2]),
        ("while True: pass", None, [-1]),
    ],
)
def test_standalone_worker(tmp_path, generation, fn_name, expected):
    """The shipped checker runs with Python and NumPy, without importing Gym or Ray."""
    source = Path(app.__file__).parent
    package = tmp_path / "resources_servers" / "code_gen"
    (package / "lcb_integration").mkdir(parents=True)
    for directory in (package.parent, package, package / "lcb_integration"):
        (directory / "__init__.py").write_text("")
    for name in ("sandbox_worker.py", "lcb_integration/checker.py", "lcb_integration/testing_util.py"):
        (package / name).write_text((source / name).read_text())
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(
        json.dumps(
            {
                "sample": {
                    "input_output": json.dumps(
                        {
                            "inputs": ["2", "3"],
                            "outputs": ["4", "9"],
                            "fn_name": fn_name,
                        }
                    )
                },
                "generation": generation,
                "timeout": 1,
                "debug": False,
            }
        )
    )
    process = subprocess.run(
        [sys.executable, "-m", "resources_servers.code_gen.sandbox_worker", str(request), str(result)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(result.read_text())["result"] == expected
