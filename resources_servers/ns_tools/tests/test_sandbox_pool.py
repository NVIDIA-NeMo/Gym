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
"""Unit tests for the sandbox_pool sandbox backend. No network, no live sandbox service:
routing/eviction logic is driven directly, the transport via a fake aiohttp session."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sandbox_pool import SandboxPool  # noqa: E402


PROVIDER = {
    "opensandbox": {
        "connection": {"domain": "http://sandbox.example", "api_key": "k", "use_server_proxy": True},
        "create": {"timeout_s": 90, "retries": 3},
    }
}


def _pool(**overrides) -> SandboxPool:
    kwargs = dict(provider=PROVIDER, image="img", size=2)
    kwargs.update(overrides)
    return SandboxPool(**kwargs)


def _admit(pool: SandboxPool, index: int) -> None:
    slot = pool._slots[index]
    slot.base_url = f"http://sandbox.example/v1/sandboxes/sbx-{index}/proxy/6000"
    slot.headers = {"OPEN-SANDBOX-API-KEY": "k"}
    slot.healthy = True
    pool._started = True  # skip lazy start in route()


class TestPoolConfigValidation:
    def test_empty_domain_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "", "api_key": "k"}}}
        with pytest.raises(ValueError, match="OPENSANDBOX_BASE_URL"):
            SandboxPool(provider=bad, image="img")

    def test_empty_api_key_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "http://sandbox.example", "api_key": ""}}}
        with pytest.raises(ValueError, match="OPENSANDBOX_API_KEY"):
            SandboxPool(provider=bad, image="img")

    def test_empty_image_is_a_hard_error(self):
        with pytest.raises(ValueError, match="NS_SANDBOX_IMAGE"):
            SandboxPool(provider=PROVIDER, image="")

    def test_ctor_is_pure_no_event_loop_required(self):
        # Constructing outside any running loop must work (pure ctor rule).
        pool = _pool()
        assert pool.ready_count == 0


class TestRouting:
    def test_sessions_stick_to_their_assigned_slot(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            first = await pool.route("sess-a")
            for _ in range(5):
                again = await pool.route("sess-a")
                assert again == first

        asyncio.run(main())

    def test_new_sessions_go_to_the_least_loaded_slot(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            urls = {(await pool.route(f"sess-{i}"))[0] for i in range(4)}
            per_slot = [len(s.sessions) for s in pool._slots]
            assert per_slot == [2, 2], f"expected even spread, got {per_slot}"
            assert len(urls) == 2

        asyncio.run(main())

    def test_total_outage_raises_the_timeout_contract(self):
        pool = _pool()
        pool._started = True  # no healthy slots admitted

        async def main():
            with pytest.raises(httpx.TimeoutException):
                await pool.route("sess-a")

        asyncio.run(main())

    def test_dead_slot_reroutes_the_session_and_drops_the_old_pin(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            await pool.route("sess-a")
            index = pool._session_to_slot["sess-a"]
            pool._slots[index].healthy = False
            url, _ = await pool.route("sess-a")
            new_index = pool._session_to_slot["sess-a"]
            assert new_index != index
            assert "sess-a" not in pool._slots[index].sessions
            assert url == pool._slots[new_index].base_url

        asyncio.run(main())

    def test_release_unpins(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            await pool.route("sess-a")

        asyncio.run(main())
        pool.release("sess-a")
        assert "sess-a" not in pool._session_to_slot
        assert all("sess-a" not in s.sessions for s in pool._slots)


class _FakeAiohttpResponse:
    def __init__(self, status: int, text: str = ""):
        self.status = status
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAiohttpSession:
    """Stands in for aiohttp.ClientSession; scripts responses per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, dict(headers or {})))
        return self.responses.pop(0)

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, dict(headers or {})))
        return self.responses.pop(0)


class TestSandboxBackend:
    """The nemo_skills subclass; skipped when nemo_skills is not installed (per-server dep)."""

    @pytest.fixture()
    def backend(self):
        pytest.importorskip("nemo_skills")
        import gym_sandbox

        sandbox = gym_sandbox.GymSandbox(
            pool=dict(provider=PROVIDER, image="img", size=1),
            host="127.0.0.1",
            port="6000",
            disable_session_restore=True,
        )
        _admit(sandbox._pool, 0)
        return sandbox

    def test_backend_registers_with_the_nemo_skills_registry(self):
        pytest.importorskip("nemo_skills")
        import gym_sandbox
        from nemo_skills.code_execution.sandbox import sandboxes

        assert sandboxes["sandbox_pool"] is gym_sandbox.GymSandbox

    def test_send_request_routes_with_pool_headers_and_session(self, backend):
        ok = '{"process_status": "completed", "stdout": "", "stderr": ""}'
        backend._aiohttp = _FakeAiohttpSession([_FakeAiohttpResponse(200, ok)])
        result = asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "sess-a"}, timeout=10.0))
        assert result["process_status"] == "completed"
        method, url, headers = backend._aiohttp.calls[0]
        assert method == "POST" and url.endswith("/proxy/6000/execute")
        assert headers["OPEN-SANDBOX-API-KEY"] == "k"
        assert headers["X-Session-ID"] == "sess-a"

    def test_non_200_normalizes_to_the_timeout_contract(self, backend):
        backend._aiohttp = _FakeAiohttpSession([_FakeAiohttpResponse(500, "boom")])
        with pytest.raises(httpx.TimeoutException):
            asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "s"}, timeout=10.0))

    def test_502_retries_exactly_once_then_succeeds(self, backend):
        ok = '{"process_status": "completed", "stdout": "", "stderr": ""}'
        backend._aiohttp = _FakeAiohttpSession(
            [_FakeAiohttpResponse(502, "bad gateway"), _FakeAiohttpResponse(200, ok)]
        )
        result = asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "s"}, timeout=10.0))
        assert result["process_status"] == "completed"
        assert len(backend._aiohttp.calls) == 2

    def test_transport_errors_normalize_to_httpx_timeout(self, backend):
        import aiohttp as _aiohttp

        class _Raising:
            closed = False

            def post(self, *a, **k):
                raise _aiohttp.ClientConnectionError("conn reset")

        backend._aiohttp = _Raising()
        with pytest.raises(httpx.TimeoutException):
            asyncio.run(backend._send_request({"generated_code": "1", "session_id": "s"}, timeout=10.0))

    def test_delete_session_routes_to_the_pinned_pod_and_releases(self, backend):
        async def main():
            await backend._pool.route("sess-a")
            backend._aiohttp = _FakeAiohttpSession([_FakeAiohttpResponse(200)])
            await backend.delete_session("sess-a")

        asyncio.run(main())
        method, url, _ = backend._aiohttp.calls[0]
        assert method == "DELETE"
        assert url.endswith("/sessions/sess-a")
        assert "sess-a" not in backend._pool._session_to_slot


class TestHealWarmupRace:
    """The heal loop must never race duplicate creates into a slot warmup still owns —
    a late duplicate overwrites base_url under pinned sessions and breaks stickiness
    (observed under slow pod creation)."""

    def test_create_slot_is_single_flight(self):
        pool = _pool()
        calls = {"n": 0}

        async def fake_inner(slot):
            calls["n"] += 1
            await asyncio.sleep(0.05)

        pool._create_slot_inner = fake_inner

        async def main():
            await asyncio.gather(pool._create_slot(pool._slots[0]), pool._create_slot(pool._slots[0]))

        asyncio.run(main())
        assert calls["n"] == 1, "second concurrent create for the same slot must be a no-op"

    def test_heal_loop_waits_for_warmup(self):
        pool = _pool()
        assert pool._warmup_done is False
        healed = []
        pool._heal_slot = lambda slot: healed.append(slot.index)

        async def one_heal_pass():
            pool._health_interval_s = 0.01
            task = asyncio.create_task(pool._heal_loop())
            await asyncio.sleep(0.05)
            pool._closed = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(one_heal_pass())
        assert healed == [], "heal loop must not touch slots before warmup completes"


class TestEnvStringify:
    def test_env_values_are_stringified(self):
        from sandbox_pool import SandboxPool

        pool = SandboxPool(
            provider={"opensandbox": {"connection": {"domain": "http://sandbox.example", "api_key": "k"}}},
            image="img:tag",
            env={"NUM_WORKERS": 4, "FLAG": True},
        )
        # The create API's env map is string->string; ints/bools 422 server-side.
        assert pool._env == {"NUM_WORKERS": "4", "FLAG": "True"}


class TestPoolRefFallback:
    """pool_ref acquire semantics — SDK required (create is monkeypatched, no network)."""

    def _pool(self, **overrides):
        from sandbox_pool import SandboxPool

        kwargs = dict(
            provider={"opensandbox": {"connection": {"domain": "http://sandbox.example", "api_key": "k"}}},
            image="img:tag",
            pool_ref="warm-pool",
            size=1,
            service_command="sh -c 'start & echo ok'",
        )
        kwargs.update(overrides)
        return SandboxPool(**kwargs)

    def test_pool_full_falls_back_to_direct_create(self, monkeypatch):
        opensandbox = pytest.importorskip("opensandbox")
        calls = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            if "extensions" in kwargs:
                raise RuntimeError("pool exhausted")
            return object()

        monkeypatch.setattr(opensandbox.Sandbox, "create", staticmethod(fake_create))
        pool = self._pool()
        pool._connection_config = object()
        sandbox, from_pool = asyncio.run(pool._acquire_sandbox())
        assert from_pool is False and sandbox is not None
        assert calls[0]["extensions"] == {"poolRef": "warm-pool"}
        # The fallback create carries the full direct spec, not the pool claim shape.
        assert "extensions" not in calls[1] and calls[1]["image"] == "img:tag"

    def test_pool_failure_raises_when_fallback_disabled(self, monkeypatch):
        opensandbox = pytest.importorskip("opensandbox")

        async def fake_create(**kwargs):
            raise RuntimeError("pool exhausted")

        monkeypatch.setattr(opensandbox.Sandbox, "create", staticmethod(fake_create))
        pool = self._pool(pool_fallback=False)
        pool._connection_config = object()
        with pytest.raises(RuntimeError, match="pool exhausted"):
            asyncio.run(pool._acquire_sandbox())

    def test_pool_claim_skips_prepare_and_fallback_does_not(self, monkeypatch):
        opensandbox = pytest.importorskip("opensandbox")
        prepared = []

        async def fake_create(**kwargs):
            if "extensions" in kwargs and fake_create.pool_ok:
                return "pool-pod"
            if "extensions" in kwargs:
                raise RuntimeError("pool exhausted")
            return "direct-pod"

        monkeypatch.setattr(opensandbox.Sandbox, "create", staticmethod(fake_create))
        pool = self._pool()
        pool._connection_config = object()

        async def fake_prepare(sandbox):
            prepared.append(sandbox)

        async def fake_endpoint_flow(slot, sandbox):
            slot.sandbox = sandbox
            slot.healthy = True

        pool._prepare = fake_prepare

        async def run_inner(pool_ok):
            fake_create.pool_ok = pool_ok
            sandbox, from_pool = await pool._acquire_sandbox()
            if pool._needs_prepare and not from_pool:
                await pool._prepare(sandbox)
            return sandbox

        assert asyncio.run(run_inner(True)) == "pool-pod"
        assert prepared == []
        assert asyncio.run(run_inner(False)) == "direct-pod"
        assert prepared == ["direct-pod"]


class TestProxyAuthHeaders:
    """Mirror of the provider's proxy-mode auth (PR 2462). The SDK's execd-facing
    clients authenticate only via ConnectionConfig.headers, so without the key
    there the create ready gate's health ping 401s on servers that enforce auth
    on /proxy/* routes and every claim dies at ready_timeout."""

    def _captured_kwargs(self, monkeypatch, connection) -> dict:
        opensandbox = pytest.importorskip("opensandbox")
        import opensandbox.config.connection as osb_connection

        captured: dict = {}

        class FakeConnectionConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(osb_connection, "ConnectionConfig", FakeConnectionConfig)

        async def fake_create(**kwargs):
            # Warmup must not reach the network; a fast failure leaves the slot
            # empty and lets start() finish so we can inspect the config.
            raise RuntimeError("no network in tests")

        monkeypatch.setattr(opensandbox.Sandbox, "create", staticmethod(fake_create))
        pool = _pool(provider={"opensandbox": {"connection": connection}}, size=1)

        async def main():
            await pool.start()
            await pool.aclose()

        asyncio.run(main())
        return captured

    def test_proxy_mode_carries_the_api_key_as_a_header(self, monkeypatch):
        captured = self._captured_kwargs(
            monkeypatch,
            {
                "domain": "http://sandbox.example",
                "api_key": "key",
                "use_server_proxy": True,
            },  # pragma: allowlist secret
        )
        assert captured["headers"] == {"OPEN-SANDBOX-API-KEY": "key"}  # pragma: allowlist secret

    def test_direct_mode_never_injects_the_key(self, monkeypatch):
        # A direct sandbox endpoint runs untrusted code and must never see the key.
        captured = self._captured_kwargs(
            monkeypatch,
            {"domain": "http://sandbox.example", "api_key": "key"},  # pragma: allowlist secret
        )
        assert "headers" not in captured

    def test_caller_supplied_headers_survive_and_win(self, monkeypatch):
        captured = self._captured_kwargs(
            monkeypatch,
            {
                "domain": "http://sandbox.example",
                "api_key": "key",  # pragma: allowlist secret
                "use_server_proxy": True,
                "headers": {"X-Route": "r", "OPEN-SANDBOX-API-KEY": "explicit"},  # pragma: allowlist secret
            },
        )
        assert captured["headers"] == {
            "X-Route": "r",
            "OPEN-SANDBOX-API-KEY": "explicit",  # pragma: allowlist secret
        }

    def test_normalize_endpoint_adds_the_key_only_in_proxy_mode(self):
        from types import SimpleNamespace

        resolved = SimpleNamespace(endpoint="sandbox.example/v1/sandboxes/sbx/proxy/6000", headers={})

        proxied = _pool()  # PROVIDER sets use_server_proxy: True
        _, headers = proxied._normalize_endpoint(resolved)
        assert headers == {"OPEN-SANDBOX-API-KEY": "k"}

        # A direct endpoint terminates at the sandbox, which runs untrusted code.
        direct = _pool(
            provider={"opensandbox": {"connection": {"domain": "http://sandbox.example", "api_key": "k"}}},
        )
        _, headers = direct._normalize_endpoint(resolved)
        assert headers == {}
