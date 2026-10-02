# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real process barriers for native stream observation."""

import asyncio
import sys

import pytest

from nemo_gym.native_stream import PIPE_CHUNK_BYTES, NativeStreamObserver, communicate_native


async def test_live_fidelity_and_concurrent_pipe_drainage():
    # Every child floods stderr before writing a stdout marker. It cannot exit
    # until the observer has received BOTH streams and releases stdin.
    async def run(identity):
        program = r"""
import os, sys
identity = sys.argv[1].encode()
payload = bytes(range(256)) * 1024
sys.stderr.buffer.write(payload)
sys.stderr.buffer.flush()
os.write(1, identity + b"\n")
assert os.read(0, 1) == b"!"
os.write(1, b"\xf0\x9f")
assert os.read(0, 1) == b"!"
os.write(1, b"\x90\x89\x00\xff")
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            program,
            str(identity),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None
        observed = {"stdout": bytearray(), "stderr": bytearray()}
        ended = set()
        released = 0

        def consume(channel, chunk):
            nonlocal released
            assert len(chunk) <= PIPE_CHUNK_BYTES
            assert channel not in ended
            if not chunk:
                ended.add(channel)
                return
            observed[channel].extend(chunk)
            if released == 0 and observed["stdout"].endswith(b"\n") and len(observed["stderr"]) == 262144:
                assert process.returncode is None
                process.stdin.write(b"!")
                released = 1
            elif released == 1 and observed["stdout"].endswith(b"\xf0\x9f"):
                process.stdin.write(b"!")
                released = 2

        observer = NativeStreamObserver(consume)
        try:
            async with asyncio.timeout(15):
                stdout, stderr = await communicate_native(process, observer)
            assert not observer.failed
            assert released == 2
            assert stdout == str(identity).encode() + b"\n\xf0\x9f\x90\x89\x00\xff"
            assert stderr == bytes(range(256)) * 1024
            assert observed == {"stdout": stdout, "stderr": stderr}
            assert ended == {"stdout", "stderr"}
            assert process.returncode == 0
        finally:
            process.stdin.close()
            if process.returncode is None:
                process.kill()
            await process.wait()

    await asyncio.gather(*(run(identity) for identity in range(8)))


async def test_observer_failure_disables_callback_but_drains_process():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'x' * 1000000); sys.stderr.buffer.write(b'y' * 1000000)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    calls = 0

    def broken(channel, chunk):
        nonlocal calls
        calls += 1
        raise OSError("synthetic sink failure")

    observer = NativeStreamObserver(broken)
    try:
        async with asyncio.timeout(15):
            stdout, stderr = await communicate_native(process, observer)
        assert (stdout, stderr) == (b"x" * 1000000, b"y" * 1000000)
        assert observer.failed and calls == 1
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


async def test_cancellation_joins_readers_and_records_incomplete_capture():
    before = asyncio.all_tasks()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import os; os.write(1, b'ready'); os.read(0, 1)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ready = asyncio.Event()
    observer = NativeStreamObserver(lambda channel, chunk: ready.set() if chunk == b"ready" else None)
    task = asyncio.create_task(communicate_native(process, observer))
    try:
        async with asyncio.timeout(15):
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert observer.failed
            assert asyncio.all_tasks() == before
            assert process.returncode is None  # Execution is still the caller's responsibility.
    finally:
        assert process.stdin is not None
        process.stdin.close()
        if process.returncode is None:
            process.kill()
        await process.communicate()


async def test_finalization_survives_cancellation_and_runs_once():
    entered, release = asyncio.Event(), asyncio.Event()
    results = []

    async def finish(code, incomplete):
        entered.set()
        await release.wait()
        results.append((code, incomplete))

    observer = NativeStreamObserver(lambda *_: None, on_finish=finish)
    closing = asyncio.create_task(observer.close(returncode=-9, incomplete=True))
    await entered.wait()
    closing.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    await observer.close(returncode=0)
    assert results == [(-9, True)]
    assert observer.failed


async def test_failed_start_still_finalizes_and_disables_content():
    results = []

    def broken(*args):
        raise OSError("synthetic observer failure")

    async def finish(code, incomplete):
        results.append((code, incomplete))

    observer = NativeStreamObserver(lambda *_: pytest.fail("disabled consumer"), on_start=broken, on_finish=finish)
    await observer.start("prompt", "system")
    await observer.emit("stdout", b"unused")
    await observer.close(returncode=None)
    assert results == [(None, True)]


async def test_failed_finish_is_observation_failure():
    async def broken(*args):
        raise OSError("synthetic delivery failure")

    observer = NativeStreamObserver(lambda *_: None, on_finish=broken)
    await observer.close(returncode=0)
    assert observer.failed


def test_factory_validation_and_result(monkeypatch):
    from nemo_gym.native_stream import NativeStreamConfig

    with pytest.raises(ValueError):
        NativeStreamConfig(factory="not an import")
    monkeypatch.setattr(sys.modules[__name__], "bad_factory", lambda **kwargs: object(), raising=False)
    config = NativeStreamConfig(factory=__name__ + ":bad_factory")
    with pytest.raises(TypeError, match="NativeStreamObserver"):
        config.create(agent_name="agent", rollout_id="2-3-a1")


async def test_concurrent_request_scope_and_self_call_do_not_cross_contaminate(monkeypatch):
    from nemo_gym.native_stream import (
        NativeInvocationScope,
        NativeScopeMiddleware,
        NativeStreamConfig,
        native_stream_headers,
    )
    from nemo_gym.rollout_correlation import trajectory_identity

    entered = asyncio.Queue()
    release = asyncio.Event()
    observed = []

    def factory(*, options, agent_name, rollout_id, scope):
        assert scope.agent_name == agent_name
        assert scope.rollout_id == rollout_id
        observed.append(scope)
        return NativeStreamObserver(lambda *_: None)

    monkeypatch.setattr(sys.modules[__name__], "scoped_factory", factory, raising=False)
    config = NativeStreamConfig(factory=__name__ + ":scoped_factory")

    async def child(scope, receive, send):
        metadata = scope["expected"]
        assert native_stream_headers() == metadata.headers()
        config.create(agent_name=metadata.agent_name, rollout_id=metadata.rollout_id)

    async def app(scope, receive, send):
        metadata = scope["expected"]
        await entered.put(metadata)
        await release.wait()
        assert native_stream_headers() == metadata.headers()
        config.create(agent_name=metadata.agent_name, rollout_id=metadata.rollout_id)
        forwarded = [(key.encode(), value.encode()) for key, value in native_stream_headers().items()]
        await NativeScopeMiddleware(child)({**scope, "headers": forwarded}, receive, send)
        assert native_stream_headers() == metadata.headers()

    async def request(index):
        row = {
            "task_id": "task-🐉-" + str(index // 4),
            "_ng_task_index": index // 4,
            "_ng_rollout_index": index % 4,
            "_ng_attempt_index": index % 2,
            "agent_ref": {"name": "candidate-" + str(index % 2)},
        }
        metadata = NativeInvocationScope.from_row(row)
        assert (metadata.task_id, metadata.rollout_id) == trajectory_identity(row)
        assert metadata.attempt_index == index % 2
        headers = [(key.encode(), value.encode()) for key, value in metadata.headers().items()]
        await NativeScopeMiddleware(app)({"type": "http", "headers": headers, "expected": metadata}, None, None)
        assert native_stream_headers() == {}
        return metadata

    tasks = [asyncio.create_task(request(index)) for index in range(8)]
    async with asyncio.timeout(10):
        for _ in tasks:
            await entered.get()
        release.set()
        expected = await asyncio.gather(*tasks)
    assert len(observed) == 16
    for metadata in expected:
        assert observed.count(metadata) == 2
    assert native_stream_headers() == {}


@pytest.mark.parametrize("values", [[b"{"], [b"x" * 8193], [b"{}", b"{}"], [b'{"task_id":true}']])
async def test_scope_rejects_malformed_or_unbounded_headers(values):
    from nemo_gym.native_stream import NativeScopeMiddleware

    async def forbidden(*args):
        pytest.fail("invalid scope reached native execution")

    messages = []

    async def send(message):
        messages.append(message)

    await NativeScopeMiddleware(forbidden)(
        {"type": "http", "headers": [(b"x-nemo-gym-native-scope", value) for value in values]}, None, send
    )
    assert messages[0]["status"] == 400
    assert messages[1]["body"] == b"invalid native observation scope"


async def test_finalization_deadline_cancels_owned_callback():
    closed = asyncio.Event()

    async def stalled(*args):
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    observer = NativeStreamObserver(lambda *_: None, on_finish=stalled)
    async with asyncio.timeout(10):
        await observer.close(returncode=0)
    assert closed.is_set() and observer.failed


async def test_scope_resets_after_cancelled_request():
    from nemo_gym.native_stream import NativeInvocationScope, NativeScopeMiddleware, native_stream_headers

    metadata = NativeInvocationScope(task_id="t", rollout_id="0-0", agent_name="a", task_index=0, rollout_index=0)

    async def cancelled(*args):
        assert native_stream_headers() == metadata.headers()
        raise asyncio.CancelledError

    headers = [(key.encode(), value.encode()) for key, value in metadata.headers().items()]
    with pytest.raises(asyncio.CancelledError):
        await NativeScopeMiddleware(cancelled)({"type": "http", "headers": headers}, None, None)
    assert native_stream_headers() == {}


@pytest.mark.parametrize("cancel", [False, True])
async def test_async_start_deadline_and_cancellation_release_callback(cancel):
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    observer = NativeStreamObserver(lambda *_: None, on_start=blocked)
    task = asyncio.create_task(observer.start("prompt", "system"))
    await entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        async with asyncio.timeout(10):
            await task
    assert observer.failed and cleaned.is_set()
    await observer.close(returncode=None)
