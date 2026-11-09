# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Git and subprocess checks for source pins and interrupted RPC streams."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from resources_servers.claweval.process import TaskProcess
from resources_servers.claweval.worker import TaskRuntime, match_dispatch, validate_source


@pytest.mark.parametrize("mutation", [None, "tracked", "staged", "untracked"])
def test_revision_pin_admits_only_clean_source(tmp_path, mutation):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "--quiet")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    source = tmp_path / "source.py"
    source.write_text("original\n")
    git("add", "source.py")
    git("commit", "--quiet", "-m", "Initial source")
    revision = git("rev-parse", "HEAD")
    if mutation == "untracked":
        (tmp_path / "injected.py").write_text("not pinned\n")
    elif mutation is not None:
        source.write_text("changed\n")
        if mutation == "staged":
            git("add", "source.py")
    if mutation is None:
        validate_source(tmp_path, revision)
        with pytest.raises(ValueError, match="source must match"):
            validate_source(tmp_path, "0" * 40)
    else:
        with pytest.raises(ValueError, match="source must match"):
            validate_source(tmp_path, revision)


@pytest.mark.asyncio
async def test_rpc_serializes_requests_and_closes_its_process():
    worker = TaskProcess()
    child = (
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " p=json.loads(line)\n"
        " print(json.dumps({'result':p.get('value')}),flush=True)\n"
        " if p['operation']=='close': sys.exit(0)\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        child,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    worker.process = process
    try:
        assert await asyncio.gather(*(worker.call("echo", {"value": value}, timeout=5) for value in range(8))) == list(
            range(8)
        )
    finally:
        await worker.close()
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_cancelled_rpc_cannot_reuse_a_stale_reply():
    worker = TaskProcess()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        "import sys,time; sys.stdin.readline(); time.sleep(60)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    worker.process = process
    try:
        with pytest.raises(TimeoutError):
            await worker.call("wait", {}, timeout=0.05)
        with pytest.raises(RuntimeError, match="unavailable"):
            await worker.call("echo", {}, timeout=1)
    finally:
        await worker.close()
    assert process.returncode is not None


def test_a_content_or_infrastructure_failure_prohibits_native_grading():
    runtime = TaskRuntime()
    runtime.failure = "Unsupported native video"
    # The failure is checked before loading the native grader or touching any task state.
    with pytest.raises(RuntimeError, match="grading is prohibited"):
        runtime.verify({"conversation": [], "model_turns": 1})


def test_native_children_cannot_read_or_write_the_rpc_pipe():
    child = (
        "import json,subprocess,sys\n"
        "from resources_servers.claweval.worker import isolate_protocol_stdio\n"
        "incoming,outgoing=isolate_protocol_stdio()\n"
        "subprocess.run([sys.executable,'-c',\"import sys; assert sys.stdin.read() == ''; print('native log')\"],check=True)\n"
        "outgoing.write(json.dumps({'result':json.loads(incoming.readline())})+'\\n'); outgoing.flush()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child], input='{"actual":"request"}\n', text=True, capture_output=True, check=True
    )
    assert result.stdout == '{"result": {"actual": "request"}}\n'
    assert result.stderr == "native log\n"


class DispatchResult(BaseModel):
    tool_use_id: str
    content: list[dict]
    is_error: bool = False


def dispatch_result(identifier, text):
    return (
        SimpleNamespace(name="Read", input={"path": "/workspace/file"}),
        DispatchResult(tool_use_id=identifier, content=[{"type": "text", "text": text}]),
        SimpleNamespace(response_status=200, response_body={"text": text}),
        [],
    )


def test_identical_repeated_calls_preserve_every_execution_without_fabricating_order():
    records = [dispatch_result("one", "same result"), dispatch_result("two", "same result")]
    index, record, ambiguous = match_dispatch("claweval_read", {"path": "/workspace/file"}, records)
    assert index == 0 and record is records[0] and ambiguous
    records.pop(index)
    assert match_dispatch("Read", {"path": "/workspace/file"}, records)[2] is False


def test_distinguishable_repeated_results_cannot_be_arbitrarily_reassigned():
    records = [dispatch_result("one", "before"), dispatch_result("two", "after")]
    with pytest.raises(ValueError, match="distinguishable results"):
        match_dispatch("Read", {"path": "/workspace/file"}, records)
