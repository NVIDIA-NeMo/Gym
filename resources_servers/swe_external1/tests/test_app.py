# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.swe_external1 import app, task_data
from resources_servers.swe_external1.task_data import TaskFile, TaskMetadata
from resources_servers.swe_external1.verification import run_verification, upload_files


def asset(path="test.sh", data=b"#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n", compressed=False, **kwargs):
    return TaskFile(
        path=path,
        content_b64=base64.b64encode(gzip.compress(data, mtime=0) if compressed else data).decode(),
        encoding="gzip+base64" if compressed else "base64",
        sha256=hashlib.sha256(data).hexdigest(),
        **kwargs,
    )


def task(**kwargs):
    return TaskMetadata.model_validate(
        {
            "task_id": "public-example",
            "image_ref": "example/image:tag",
            "workdir": "/app/repo",
            "test_files": [asset()],
            "solution_files": [asset("solve.sh", b"echo fixed\n", mode=0o755)],
            **kwargs,
        }
    )


class RecordingSandbox:
    def __init__(self, reward="1\n", exit_code=0, error_type=None):
        self.reward = reward
        self.exit_code = exit_code
        self.error_type = error_type
        self.files = {"/logs/verifier/reward.txt": b"1", "/tests/stale": b"stale"}
        self.commands = []
        self.events = []
        self.stopped = False
        self.raise_on = None
        self.fail_on = None

    async def start(self, spec):
        self.spec = spec
        if self.raise_on == "start":
            raise RuntimeError("start failed")

    async def serialize(self):
        if self.raise_on == "serialize":
            raise RuntimeError("serialize failed")
        return {"sandbox_id": "test-sandbox", "workdir": self.spec.workdir}

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        self.events.append(command)
        if self.raise_on and self.raise_on in command:
            raise TimeoutError("timed out")
        if self.fail_on and self.fail_on in command:
            return SandboxExecResult("", "command failed", 2)
        if command.startswith("rm -rf -- /tests"):
            self.files = {k: v for k, v in self.files.items() if not k.startswith("/tests/")}
        if "rm -rf -- /logs/verifier" in command:
            self.files = {k: v for k, v in self.files.items() if not k.startswith("/logs/verifier/")}
        if command == "bash /tests/test.sh":
            assert "/tests/test.sh" in self.files
            if self.reward is not None:
                self.files["/logs/verifier/reward.txt"] = self.reward.encode()
            return SandboxExecResult("test stdout", "test stderr", self.exit_code, self.error_type)
        return SandboxExecResult("ok", "", 0)

    async def upload(self, local_path, remote_path):
        self.events.append("upload " + remote_path)
        self.files[remote_path] = Path(local_path).read_bytes()

    async def download(self, remote_path, local_path):
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        Path(local_path).write_bytes(self.files[remote_path])

    async def stop(self):
        self.stopped = True
        if self.raise_on == "stop":
            raise RuntimeError("stop failed")


def server(monkeypatch, sandbox, **kwargs):
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app, "AsyncSandbox", lambda *_: sandbox)
    config = app.SweExternal1ResourcesServerConfig(
        name="test",
        host="",
        port=0,
        entrypoint="",
        sandbox_provider="test",
        **kwargs,
    )
    return app.SweExternal1ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def request(key="one"):
    return SimpleNamespace(session={SESSION_ID_KEY: key})


def body(metadata=None):
    return app.SweExternal1VerifyRequest.model_validate(
        {
            "verifier_metadata": metadata or task(),
            "responses_create_params": {"input": []},
            "response": {
                "id": "test",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [],
                "tool_choice": "auto",
                "tools": [],
                "parallel_tool_calls": False,
            },
        }
    )


@pytest.mark.parametrize("path", ["", "/absolute", "../escape", "a/../b", "a//b", "a/./b", "a\\b", "a\nb"])
def test_reject_unsafe_asset_paths(path):
    with pytest.raises(ValidationError):
        asset(path)


@pytest.mark.parametrize("path", ["relative", "/a/../b", "/a\nb"])
def test_reject_unsafe_workdir(path):
    with pytest.raises(ValidationError):
        task(workdir=path)


def test_asset_round_trip_and_limits(monkeypatch):
    raw = b"\x00\xffbinary\x00"
    assert asset("with space/file", raw).decoded() == raw
    assert asset("nested/file", raw, compressed=True).decoded() == raw
    bad = asset().model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="checksum"):
        bad.decoded()
    with pytest.raises(ValueError):
        asset().model_copy(update={"content_b64": "?!"}).decoded()
    compressed = base64.b64encode(gzip.compress(raw) + gzip.compress(raw)).decode()
    with pytest.raises(ValueError, match="concatenated"):
        asset().model_copy(update={"encoding": "gzip+base64", "content_b64": compressed}).decoded()
    monkeypatch.setattr(task_data, "MAX_FILE_BYTES", 2)
    with pytest.raises(ValueError):
        asset(data=raw).decoded()
    with pytest.raises(ValueError):
        asset(data=raw, compressed=True).decoded()


def test_task_asset_validation(monkeypatch):
    for files in ([asset(), asset()], [asset(), asset("dir"), asset("dir/file")], [asset("other.sh")]):
        with pytest.raises(ValidationError):
            task(test_files=files)
    with pytest.raises(ValidationError):
        asset(mode=0o4755)
    monkeypatch.setattr(task_data, "MAX_TASK_BYTES", 1)
    with pytest.raises(ValidationError, match="size limit"):
        task()


@pytest.mark.asyncio
async def test_same_sandbox_uploads_bytes_and_modes_without_solution():
    sandbox = RecordingSandbox()
    extra = asset("nested/file with space", b"\x00\xff", mode=0o751)
    result = await run_verification(sandbox, task(test_files=[asset(), extra]))
    assert result.evaluation_completed and result.reward == 1
    assert result.test_output == "test stdout\ntest stderr"
    assert sandbox.files["/tests/nested/file with space"] == b"\x00\xff"
    assert "/tests/stale" not in sandbox.files
    assert not any(path.startswith("/solution/") for path in sandbox.files)
    assert any("chmod 751" in command for command, _ in sandbox.commands)
    assert sandbox.commands[-1][1] == {"cwd": "/app/repo", "timeout_s": 300}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reward,code,complete,expected",
    [
        ("0", 1, True, 0),
        ("0", 0, True, 0),
        ("1", 0, True, 1),
        (None, 0, False, 0),
        ("nan", 0, False, 0),
        ("inf", 0, False, 0),
        ("0.5", 0, False, 0),
        ("garbage", 0, False, 0),
        ("1", 1, False, 0),
    ],
)
async def test_reward_contract(reward, code, complete, expected):
    result = await run_verification(RecordingSandbox(reward, code), task())
    assert result.evaluation_completed == complete
    assert result.reward == expected
    assert bool(result.error) == (not complete)


@pytest.mark.asyncio
async def test_timeout_and_provider_error():
    sandbox = RecordingSandbox()
    sandbox.raise_on = "bash /tests/test.sh"
    result = await run_verification(sandbox, task(), timeout_cap_s=17)
    assert not result.evaluation_completed and "TimeoutError" in result.error
    assert sandbox.commands[-1][1]["timeout_s"] == 17
    result = await run_verification(RecordingSandbox(error_type="SandboxTimeout"), task())
    assert not result.evaluation_completed and "SandboxTimeout" in result.error


@pytest.mark.asyncio
async def test_golden_solution_order_and_failure():
    sandbox = RecordingSandbox()
    result = await run_verification(sandbox, task(), golden=True)
    assert result.reward == 1 and result.solution_output == "ok\n"
    assert sandbox.events.index("bash /solution/solve.sh") < sandbox.events.index("upload /tests/test.sh")
    sandbox = RecordingSandbox()
    sandbox.fail_on = "bash /solution/solve.sh"
    result = await run_verification(sandbox, task(), golden=True)
    assert not result.evaluation_completed and "golden solution failed" in result.error
    assert "bash /tests/test.sh" not in sandbox.events
    result = await run_verification(RecordingSandbox(), task(solution_files=[]), golden=True)
    assert "requires solution/solve.sh" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["rm -rf -- /tests", "mkdir -p -- /tests", "chmod", "test ! -L /logs"])
async def test_setup_errors_are_incomplete(stage):
    sandbox = RecordingSandbox()
    sandbox.fail_on = stage
    assert not (await run_verification(sandbox, task())).evaluation_completed
    with pytest.raises(ValueError, match="unsupported"):
        await upload_files(sandbox, [asset()], "/tmp", 1)


@pytest.mark.asyncio
async def test_seed_verify_lifecycle_and_configuration(monkeypatch):
    sandbox = RecordingSandbox()
    resource = server(monkeypatch, sandbox, sandbox_config={"env": {"TEST": "yes"}, "ttl_s": 900})
    metadata = task(setup_script="echo prepare", cpu=3)
    seed = await resource.seed_session(request(), app.SweExternal1SeedRequest(verifier_metadata=metadata))
    assert seed.sandbox_descriptor == {"sandbox_id": "test-sandbox", "workdir": "/app/repo"}
    assert sandbox.spec.image == "example/image:tag" and sandbox.spec.resources.cpu == 3
    assert sandbox.spec.ttl_s == 900 and sandbox.spec.env["TEST"] == "yes"
    assert sandbox.spec.files == {} and not any("upload" in event for event in sandbox.events)
    assert sandbox.commands[0][0].startswith("bash -lc")
    result = await resource.verify(request(), body(metadata))
    assert result.reward == 1 and result.evaluation_completed and result.failure_reason is None
    assert sandbox.stopped and not resource._sessions and not resource._busy


@pytest.mark.asyncio
async def test_missing_and_mismatched_session(monkeypatch):
    sandbox = RecordingSandbox()
    resource = server(monkeypatch, sandbox)
    assert "no active task" in (await resource.verify(request(), body())).error
    await resource.seed_session(request(), app.SweExternal1SeedRequest(verifier_metadata=task()))
    result = await resource.verify(request(), body(task(task_id="different")))
    assert not result.evaluation_completed and "differs" in result.error and sandbox.stopped


@pytest.mark.asyncio
async def test_golden_server_does_not_seed_an_unused_sandbox(monkeypatch):
    sandbox = RecordingSandbox()
    resource = server(monkeypatch, sandbox, is_verifying_golden_patch=True)
    assert (
        await resource.seed_session(request(), app.SweExternal1SeedRequest(verifier_metadata=task()))
    ).sandbox_handle is None
    assert not hasattr(sandbox, "spec")
    assert (await resource.verify(request(), body())).reward == 1
    assert sandbox.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "serialize", "echo prepare"])
async def test_seed_failure_cleans_up(monkeypatch, failure):
    sandbox = RecordingSandbox()
    sandbox.raise_on = failure
    resource = server(monkeypatch, sandbox)
    with pytest.raises((RuntimeError, TimeoutError)):
        await resource.seed_session(
            request(), app.SweExternal1SeedRequest(verifier_metadata=task(setup_script="echo prepare"))
        )
    assert sandbox.stopped and not resource._sessions and not resource._busy


@pytest.mark.asyncio
async def test_busy_rejected_and_cleanup_failure_reported(monkeypatch):
    sandbox = RecordingSandbox()
    resource = server(monkeypatch, sandbox)
    resource._busy.add("one")
    with pytest.raises(HTTPException) as exc:
        await resource.seed_session(request(), app.SweExternal1SeedRequest(verifier_metadata=task()))
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException):
        await resource.verify(request(), body())
    resource._busy.clear()
    await resource.seed_session(request(), app.SweExternal1SeedRequest(verifier_metadata=task()))
    sandbox.raise_on = "stop"
    result = await resource.verify(request(), body())
    assert result.reward == 1 and "stop failed" in result.cleanup_error


@pytest.mark.asyncio
async def test_expiry_cleans_state_and_cannot_remove_new_session(monkeypatch):
    sandbox = RecordingSandbox()
    resource = server(monkeypatch, sandbox)
    old = app.Session(sandbox, "old")
    resource._sessions["one"] = old
    await resource._expire("one", old, 0)
    assert not resource._sessions and sandbox.stopped
    newer = app.Session(RecordingSandbox(), "new")
    resource._sessions["one"] = newer
    await resource._expire("one", old, 0)
    assert resource._sessions["one"] is newer
    await resource._stop(newer)


def test_multiworker_rejected():
    with pytest.raises(ValidationError, match="num_workers"):
        app.SweExternal1ResourcesServerConfig(
            name="", host="", port=0, entrypoint="", sandbox_provider="", num_workers=2
        )
