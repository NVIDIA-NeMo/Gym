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
"""Backend tests for the Lean sandbox clients."""

import asyncio
from types import SimpleNamespace

import pytest

from resources_servers.math_formal_lean.sandbox_client import (
    GymSandboxLean4Client,
    Lean4SandboxClient,
)


PROVIDER = {
    "opensandbox": {
        "connection": {"domain": "http://sandbox.example", "api_key": "k", "use_server_proxy": True},
    }
}


class TestHttpClientDefaults:
    def test_default_url_is_unchanged(self):
        client = Lean4SandboxClient()
        assert client._get_execute_url() == "http://127.0.0.1:6000/execute"


class _FakeSandbox:
    """Stands in for AsyncSandbox; records lifecycle and returns a scripted exec result."""

    instances = []
    next_exec_result = None
    next_raise_on_exec = None

    def __init__(self, provider=None, spec=None):
        self.spec = spec
        self.exec_result = _FakeSandbox.next_exec_result or SimpleNamespace(return_code=0, stdout="ok", stderr="")
        self.raise_on_exec = _FakeSandbox.next_raise_on_exec
        self.exec_command = None
        self.exec_timeout_s = None
        self.stopped = False
        _FakeSandbox.instances.append(self)

    async def start(self):
        return self

    async def exec(self, command, **kwargs):
        self.exec_command = command
        self.exec_commands = getattr(self, "exec_commands", []) + [command]
        self.exec_timeout_s = kwargs.get("timeout_s")
        if self.raise_on_exec:
            raise self.raise_on_exec
        return self.exec_result

    async def upload(self, local_path, remote_path):
        self.uploaded = getattr(self, "uploaded", []) + [remote_path]

    async def stop(self):
        self.stopped = True


@pytest.fixture()
def fake_sandbox(monkeypatch):
    import resources_servers.math_formal_lean.sandbox_client as sandbox_client

    _FakeSandbox.instances = []
    _FakeSandbox.next_exec_result = None
    _FakeSandbox.next_raise_on_exec = None
    monkeypatch.setattr(sandbox_client, "AsyncSandbox", _FakeSandbox)
    return _FakeSandbox


def _client(**overrides) -> GymSandboxLean4Client:
    kwargs = dict(provider=PROVIDER, image="lean-img")
    kwargs.update(overrides)
    return GymSandboxLean4Client(**kwargs)


class TestGymSandboxLean4Client:
    def test_empty_image_is_a_hard_error(self):
        with pytest.raises(ValueError, match="image"):
            GymSandboxLean4Client(provider=PROVIDER, image="")

    def test_empty_creds_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "", "api_key": ""}}}
        with pytest.raises(ValueError, match="OPENSANDBOX"):
            GymSandboxLean4Client(provider=bad, image="img")

    def test_completed_maps_rc_zero(self, fake_sandbox):
        out = asyncio.run(_client().execute_lean4("theorem t : True := trivial", timeout=30.0))
        assert out == {"process_status": "completed", "stdout": "ok", "stderr": ""}
        box = fake_sandbox.instances[0]
        assert box.stopped, "per-verify pod must be destroyed in finally"
        assert "timeout -s KILL 30.0 lake env --dir /lean4/my_project lean" in box.exec_command
        assert box.exec_timeout_s == 90.0, "provider deadline must trail the in-sandbox timeout"
        assert list(box.spec.files.values()) == ["theorem t : True := trivial"]
        assert box.spec.entrypoint == ["sleep", "infinity"]

    def test_nonzero_rc_maps_to_failed(self, fake_sandbox):
        fake_sandbox.next_exec_result = SimpleNamespace(return_code=1, stdout="", stderr="error: x")
        out = asyncio.run(_client().execute_lean4("bad", timeout=30.0))
        assert out["process_status"] == "failed"
        assert out["stderr"] == "error: x"

    def test_timeout_rc_maps_to_the_ns_timeout_contract(self, fake_sandbox):
        fake_sandbox.next_exec_result = SimpleNamespace(return_code=137, stdout="partial", stderr="")
        out = asyncio.run(_client().execute_lean4("slow", timeout=30.0))
        assert out["process_status"] == "timeout"
        assert out["stdout"] == "partial", "partial stdout must survive, matching the NS server"
        assert out["stderr"].endswith("Execution timed out after 30.0 seconds\n")

    def test_output_truncation_matches_the_ns_contract(self, fake_sandbox):
        fake_sandbox.next_exec_result = SimpleNamespace(return_code=137, stdout="1234", stderr="abcd")
        out = asyncio.run(_client(max_output_characters=3).execute_lean4("slow", timeout=30.0))
        assert out == {
            "process_status": "timeout",
            "stdout": "123<output cut>",
            "stderr": "abc<output cut>",
        }

    def test_infra_failure_degrades_to_error_and_still_tears_down(self, fake_sandbox):
        fake_sandbox.next_raise_on_exec = RuntimeError("proxy exploded")
        out = asyncio.run(_client().execute_lean4("x", timeout=5.0))
        assert out["process_status"] == "error"
        assert "proxy exploded" in out["stderr"]
        assert fake_sandbox.instances[0].stopped

    def test_admission_timeout_collapses_to_client_timed_out(self, fake_sandbox):
        client = _client(max_concurrent=1, acquire_timeout_s=0.05)

        async def main():
            sem = client._semaphore
            await sem.acquire()  # exhaust admission
            try:
                return await client.execute_lean4("x", timeout=5.0)
            finally:
                sem.release()

        out = asyncio.run(main())
        assert out == {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}
        assert not fake_sandbox.instances, "no pod may be created past a failed admission"

    def test_fresh_pod_is_stopped_when_exec_is_cancelled(self, fake_sandbox, monkeypatch):
        entered = asyncio.Event()

        async def block(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(fake_sandbox, "exec", block)

        async def scenario():
            task = asyncio.create_task(_client().execute_lean4("theorem t : True := trivial"))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        assert fake_sandbox.instances[0].stopped


class TestPooledMode:
    def test_pool_reuses_pods_across_verifies(self, fake_sandbox):
        client = _client(pool_size=2)

        async def scenario():
            return [await client.execute_lean4("theorem t : True := trivial", timeout=5.0) for _ in range(3)]

        results = asyncio.run(scenario())
        assert [r["process_status"] for r in results] == ["completed"] * 3
        # 2 pool pods serve 3 verifies — no per-verify creates.
        assert len(fake_sandbox.instances) == 2
        pod = fake_sandbox.instances[0]
        # First exec on a pool pod is the olean prefetch; compiles clean up their proof file.
        assert pod.exec_commands[0].startswith("find ")
        assert "rm -f" in pod.exec_commands[-1] and "timeout -s KILL 5.0" in pod.exec_commands[-1]
        assert not pod.stopped

    def test_pool_replaces_failed_pod_and_retries(self, fake_sandbox):
        client = _client(pool_size=1, acquire_timeout_s=5.0)

        async def scenario():
            # Warm the pool, then arm the NEXT pod acquisition's exec to fail once.
            first = await client.execute_lean4("theorem t : True := trivial", timeout=5.0)
            bad = fake_sandbox.instances[0]
            bad.raise_on_exec = RuntimeError("pod lost")
            second = await client.execute_lean4("theorem t : True := trivial", timeout=5.0)
            return first, bad, second

        first, bad, second = asyncio.run(scenario())
        assert first["process_status"] == "completed"
        # The dead pod was stopped and replaced; the retry ran on the replacement.
        assert bad.stopped
        assert second["process_status"] == "completed"
        assert len(fake_sandbox.instances) >= 2

    def test_pool_retries_initial_create_failure(self, fake_sandbox, monkeypatch):
        client = _client(pool_size=1, acquire_timeout_s=1.0)
        create = client._create_pool_pod
        attempts = 0

        async def flaky_create():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("control plane unavailable")
            return await create()

        real_sleep = asyncio.sleep

        async def fast_sleep(_):
            await real_sleep(0)

        monkeypatch.setattr(client, "_create_pool_pod", flaky_create)
        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        async def scenario():
            out = await client.execute_lean4("theorem t : True := trivial", timeout=5.0)
            await client.close()
            return out

        out = asyncio.run(scenario())
        assert out["process_status"] == "completed"
        assert attempts == 2
        assert all(box.stopped for box in fake_sandbox.instances)

    @pytest.mark.parametrize("blocked_method", ["upload", "exec"])
    def test_cancelled_lease_is_stopped_and_replaced(self, fake_sandbox, blocked_method):
        client = _client(pool_size=1, acquire_timeout_s=1.0)

        async def scenario():
            client.start_pool()
            while client._pool is None or client._pool.qsize() == 0:
                await asyncio.sleep(0)
            leased = fake_sandbox.instances[0]
            entered = asyncio.Event()

            async def blocked_operation(*args, **kwargs):
                entered.set()
                await asyncio.Event().wait()

            setattr(leased, blocked_method, blocked_operation)
            task = asyncio.create_task(client.execute_lean4("theorem t : True := trivial", timeout=5.0))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            while client._pool is None or client._pool.qsize() == 0:
                await asyncio.sleep(0)
            replacement = fake_sandbox.instances[-1]
            await client.close()
            return leased, replacement

        leased, replacement = asyncio.run(scenario())
        assert leased.stopped
        assert replacement is not leased
        assert replacement.stopped

    def test_cancelled_close_finishes_cleanup(self, fake_sandbox):
        client = _client(pool_size=1)

        async def scenario():
            client.start_pool()
            while client._pool is None or client._pool.qsize() == 0:
                await asyncio.sleep(0)
            sandbox = fake_sandbox.instances[0]
            stop_started = asyncio.Event()
            finish_stop = asyncio.Event()
            original_stop = sandbox.stop

            async def blocked_stop():
                stop_started.set()
                await finish_stop.wait()
                await original_stop()

            sandbox.stop = blocked_stop
            closing = asyncio.create_task(client.close())
            await stop_started.wait()
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
            finish_stop.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            await client.close()

        asyncio.run(scenario())
        assert len(fake_sandbox.instances) == 1
        assert all(box.stopped for box in fake_sandbox.instances)


class TestLeanPoolRef:
    """pool_ref rides SandboxSpec.provider_options into the provider's SDK extensions."""

    def test_pool_ref_rides_provider_options(self, fake_sandbox):
        out = asyncio.run(_client(pool_ref="math-lean-warm").execute_lean4("theorem t : True := trivial"))
        assert out["process_status"] == "completed"
        box = fake_sandbox.instances[0]
        assert box.spec.provider_options == {"extensions": {"poolRef": "math-lean-warm"}}

    def test_no_pool_ref_means_no_extensions(self, fake_sandbox):
        asyncio.run(_client().execute_lean4("theorem t : True := trivial"))
        assert fake_sandbox.instances[0].spec.provider_options == {}

    def test_claim_failure_falls_back_to_direct_create(self, fake_sandbox, monkeypatch):
        calls = {"n": 0}

        async def flaky_start(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("pool exhausted")
            return self

        monkeypatch.setattr(fake_sandbox, "start", flaky_start)
        out = asyncio.run(_client(pool_ref="math-lean-warm").execute_lean4("theorem t : True := trivial"))
        assert out["process_status"] == "completed"
        # First attempt carried the claim; the fallback dropped it.
        assert fake_sandbox.instances[0].spec.provider_options == {"extensions": {"poolRef": "math-lean-warm"}}
        assert fake_sandbox.instances[1].spec.provider_options == {}
