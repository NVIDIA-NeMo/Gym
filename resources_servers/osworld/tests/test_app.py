# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the OSWorld resources server (nemo_gym.sandbox SDK edition).

The sandbox provider is a registered fake (SDK layer) and guest HTTP is mocked via
``app.gym_request``, so the full seed -> screenshot -> execute -> verify path is
exercised without a cluster. The setup/evaluate subprocess seam is covered separately
with a stub script. A live end-to-end test is guarded by ``OPENSANDBOX_DOMAIN``.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Cookies
from pytest import MonkeyPatch

import resources_servers.osworld.app as app_module
from nemo_gym.sandbox import (
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
    register_provider,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.osworld.app import OSWorldResourcesServer, OSWorldResourcesServerConfig


# --------------------------------------------------------------------------------------
# Fakes: an SDK sandbox provider + the guest :5000 HTTP surface.
# --------------------------------------------------------------------------------------


class FakeOSWorldProvider:
    """State is CLASS-level: the server builds a fresh provider instance per AsyncSandbox,
    so per-instance lists would never accumulate across seed attempts."""

    name = "fake-osworld"
    created_specs: List[SandboxSpec] = []
    closed_ids: List[str] = []

    @classmethod
    def reset_state(cls) -> None:
        cls.created_specs = []
        cls.closed_ids = []

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        type(self).created_specs.append(spec)
        return SandboxHandle(sandbox_id=f"sbx-{len(type(self).created_specs)}", provider_name=self.name, raw=object())

    async def get_endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        return SandboxEndpoint(
            url=f"http://osb.test/sandboxes/{handle.sandbox_id}/proxy/{port}",
            headers={"X-Route": "r1"},
            proxied=True,
        )

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        del handle, command, kwargs
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        del handle, source_path, target_path

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        del handle, source_path, target_path

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        del handle
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        type(self).closed_ids.append(handle.sandbox_id)

    async def aclose(self) -> None:
        return None


class _FakeResp:
    def __init__(self, status: int, content: bytes, content_type: str = "application/json"):
        self.status = status
        self._content = content
        self.headers = {"Content-Type": content_type}

    async def read(self) -> bytes:
        return self._content


async def fake_gym_request(method, url, json=None, headers=None):
    """Emulate the guest :5000 control API behind the path proxy."""
    del headers
    if url.endswith("/platform"):
        return _FakeResp(200, b"Linux", "text/plain")
    if url.endswith("/screenshot"):
        # Any bytes above the (test-lowered) threshold count as a rendered desktop.
        return _FakeResp(200, b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png")
    if url.endswith("/screen_size"):
        return _FakeResp(200, b'{"width":1280,"height":720}')
    if url.endswith("/execute"):
        del json
        return _FakeResp(200, b'{"output":"ok","returncode":0}')
    return _FakeResp(404, b"missing")


def _register_fake_provider() -> str:
    provider_name = f"fake-osworld-{uuid4().hex}"
    register_provider(provider_name, FakeOSWorldProvider)
    return provider_name


def _make_config(**overrides) -> OSWorldResourcesServerConfig:
    FakeOSWorldProvider.reset_state()
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        sandbox_provider={_register_fake_provider(): {}},
        boot_wait_s=2,
        poll_interval_s=0.01,
        screenshot_min_bytes=8,
        request_retries=1,
    )
    base.update(overrides)
    return OSWorldResourcesServerConfig(**base)


def _client(server: OSWorldResourcesServer) -> TestClient:
    app = server.setup_webserver()
    client = TestClient(app, raise_server_exceptions=False)

    class StatelessCookies(Cookies):
        def extract_cookies(self, response):
            pass

    client._cookies = StatelessCookies(client._cookies)
    return client


def _patch_task_phase(
    monkeypatch: MonkeyPatch,
    server: OSWorldResourcesServer,
    *,
    setup_ok: bool = True,
    score: float = 1.0,
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    async def fake_run_task_phase(phase, task_config, sandbox_state, *, action_history=None, timeout_s=0.0):
        calls.append(
            {
                "phase": phase,
                "task_id": task_config.get("id"),
                "control_url": sandbox_state["control_url"],
                "proxied": sandbox_state["proxied"],
                "action_history": action_history,
                "timeout_s": timeout_s,
            }
        )
        if phase == "setup":
            return {"ok": setup_ok, "score": None, "error": None if setup_ok else "setup_exception"}
        return {"ok": True, "score": score, "error": None}

    monkeypatch.setattr(server, "_run_task_phase", fake_run_task_phase)
    return calls


SPOTIFY_TASK = {
    "id": "94d95f96-9699-4208-98ba-3c3119edf9c2",
    "instruction": "I want to install Spotify on my current system. Could you please help me?",
    "config": [{"type": "execute", "parameters": {"command": ["python", "-c", "print('hi')"]}}],
    "evaluator": {
        "func": "check_include_exclude",
        "result": {"type": "vm_command_line", "command": "which spotify"},
        "expected": {"type": "rule", "rules": {"include": ["spotify"], "exclude": ["not found"]}},
    },
}


class TestApp:
    def test_sanity(self) -> None:
        server = OSWorldResourcesServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        assert server.session_id_to_sandbox == {}

    def test_seed_screenshot_execute_verify(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(app_module, "gym_request", fake_gym_request)
        server = OSWorldResourcesServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        phase_calls = _patch_task_phase(monkeypatch, server, score=1.0)
        client = _client(server)

        # SEED: allocate via the SDK + boot + official setup subprocess.
        seed = client.post(
            "/seed_session",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": SPOTIFY_TASK["instruction"]}]},
                "verifier_metadata": SPOTIFY_TASK,
            },
        )
        assert seed.status_code == 200
        assert seed.json()["sandbox_id"] == "sbx-1"
        assert seed.json()["screen"] == {"width": 1280, "height": 720}
        cookies = seed.cookies
        (stored,) = server.session_id_to_sandbox.values()
        assert stored["control_url"] == "http://osb.test/sandboxes/sbx-1/proxy/5000"
        assert stored["headers"] == {"X-Route": "r1"}
        assert stored["proxied"] is True
        # The pool spec carried the poolRef extension and no image.
        provider = FakeOSWorldProvider
        assert provider.created_specs[0].image is None
        assert provider.created_specs[0].provider_options["extensions"]["poolRef"] == server.config.pool_ref
        assert phase_calls[0]["phase"] == "setup"
        assert phase_calls[0]["proxied"] is True

        # OBSERVE: the screenshot tool returns a base64 PNG.
        shot = client.post("/screenshot", cookies=cookies)
        assert shot.status_code == 200
        assert "image_base64" in shot.json()

        # ACT: the execute tool proxies a shell command into the guest.
        ex = client.post("/execute", json={"command": "echo hi", "shell": True}, cookies=cookies)
        assert ex.status_code == 200
        assert ex.json()["output"] == "ok"

        # VERIFY: the official evaluator subprocess scores; the sandbox is released.
        verify = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": SPOTIFY_TASK["instruction"]}]},
                "response": _minimal_response("done"),
                "verifier_metadata": SPOTIFY_TASK,
                "action_history": ["pyautogui.click(1, 2)", "DONE"],
            },
            cookies=cookies,
        )
        assert verify.status_code == 200
        assert verify.json()["reward"] == 1.0
        assert phase_calls[-1]["phase"] == "evaluate"
        assert phase_calls[-1]["action_history"] == ["pyautogui.click(1, 2)", "DONE"]
        assert server.session_id_to_sandbox == {}
        assert provider.closed_ids == ["sbx-1"]

    def test_seed_retries_and_proceeds_when_setup_keeps_failing(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(app_module, "gym_request", fake_gym_request)
        server = OSWorldResourcesServer(
            config=_make_config(alloc_retries=2), server_client=MagicMock(spec=ServerClient)
        )
        phase_calls = _patch_task_phase(monkeypatch, server, setup_ok=False)
        client = _client(server)

        seed = client.post(
            "/seed_session",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
                "verifier_metadata": SPOTIFY_TASK,
            },
        )
        # Non-final setup failure retries with a FRESH sandbox; the final attempt proceeds
        # un-set-up (the task will likely score 0) instead of crashing the rollout.
        assert seed.status_code == 200
        provider = FakeOSWorldProvider
        assert len(provider.created_specs) == 2
        assert provider.closed_ids == ["sbx-1"]  # first attempt released, second kept
        assert [c["phase"] for c in phase_calls] == ["setup", "setup"]

    def test_seed_boot_failure_exhausts_retries(self, monkeypatch: MonkeyPatch) -> None:
        async def dead_guest(method, url, json=None, headers=None):
            del method, url, json, headers
            return _FakeResp(503, b"")

        monkeypatch.setattr(app_module, "gym_request", dead_guest)
        server = OSWorldResourcesServer(
            config=_make_config(alloc_retries=2, boot_wait_s=0), server_client=MagicMock(spec=ServerClient)
        )
        client = _client(server)

        seed = client.post(
            "/seed_session",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
                "verifier_metadata": SPOTIFY_TASK,
            },
        )
        assert seed.status_code == 500
        provider = FakeOSWorldProvider
        # Every allocated sandbox was released.
        assert provider.closed_ids == ["sbx-1", "sbx-2"]
        assert server.session_id_to_sandbox == {}

    def test_screenshot_without_seed_is_400(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(app_module, "gym_request", fake_gym_request)
        server = OSWorldResourcesServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        client = _client(server)
        resp = client.post("/screenshot")
        assert resp.status_code == 400

    def test_verify_without_evaluator_scores_zero_and_releases(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(app_module, "gym_request", fake_gym_request)
        server = OSWorldResourcesServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        _patch_task_phase(monkeypatch, server)
        client = _client(server)

        task = {"id": "x", "instruction": "do", "config": []}
        seed = client.post(
            "/seed_session",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
                "verifier_metadata": task,
            },
        )
        cookies = seed.cookies
        verify = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
                "response": _minimal_response("x"),
                "verifier_metadata": task,
            },
            cookies=cookies,
        )
        assert verify.status_code == 200
        assert verify.json()["reward"] == 0.0
        assert server.session_id_to_sandbox == {}


def _minimal_response(text: str) -> dict:
    return {
        "id": "resp_test",
        "created_at": 1753983920.0,
        "model": "nemotron",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class TestTaskPhaseSubprocess:
    """The subprocess seam, exercised against a stub script (no desktop_env import)."""

    async def _run(self, monkeypatch: MonkeyPatch, tmp_path: Path, stub_body: str, **kwargs: Any) -> Dict[str, Any]:
        stub = tmp_path / "stub_eval_task.py"
        stub.write_text(stub_body)
        monkeypatch.setattr(app_module, "_EVAL_TASK_SCRIPT", str(stub))
        server = OSWorldResourcesServer(
            config=_make_config(cache_dir=str(tmp_path / "cache")), server_client=MagicMock(spec=ServerClient)
        )
        sandbox_state = {
            "control_url": "http://osb.test/sandboxes/s1/proxy/5000",
            "headers": {"X-Route": "r1"},
            "proxied": True,
            "screen": {"width": 1920, "height": 1080},
        }
        return await server._run_task_phase("evaluate", {"id": "t"}, sandbox_state, **kwargs)

    async def test_sentinel_parsed(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        result = await self._run(
            monkeypatch,
            tmp_path,
            'print("__NEMO_GYM_OSWORLD__ {\\"ok\\": true, \\"score\\": 0.5, \\"error\\": null}")',
            timeout_s=30.0,
        )
        assert result == {"ok": True, "score": 0.5, "error": None}

    async def test_no_sentinel(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        result = await self._run(monkeypatch, tmp_path, 'print("nothing to see")', timeout_s=30.0)
        assert result["ok"] is False
        assert "no_sentinel" in result["error"]

    async def test_timeout(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        result = await self._run(monkeypatch, tmp_path, "import time; time.sleep(30)", timeout_s=0.5)
        assert result["ok"] is False
        assert "timeout" in result["error"]


class TestEvalTaskAddressing:
    """eval_task.py address-mode plumbing (no desktop_env import needed)."""

    def test_direct_mode_sets_remote_addr(self, monkeypatch: MonkeyPatch) -> None:
        from resources_servers.osworld import eval_task

        monkeypatch.delenv("OSWORLD_CONTROL_SERVER_URL", raising=False)
        monkeypatch.delenv("OSWORLD_REMOTE_ADDR", raising=False)
        eval_task._configure_remote_addressing("http://10.0.0.7:5000", False, {})
        assert os.environ["OSWORLD_CONTROL_SERVER_URL"] == "http://10.0.0.7:5000"
        assert os.environ["OSWORLD_REMOTE_ADDR"] == "10.0.0.7:5000:9222:8006:8080"

    def test_proxied_mode_requires_5000_suffix(self, monkeypatch: MonkeyPatch) -> None:
        from resources_servers.osworld import eval_task

        monkeypatch.delenv("OSWORLD_CONTROL_SERVER_URL", raising=False)
        monkeypatch.delenv("OSWORLD_REMOTE_ADDR", raising=False)
        with pytest.raises(ValueError, match="must end with /5000"):
            eval_task._configure_remote_addressing("http://osb.test/sandboxes/s1/proxy/9999", True, {})


class TestLocalForwarder:
    def test_forwards_and_injects_headers(self) -> None:
        import http.server
        import threading

        import requests as rq

        from resources_servers.osworld.local_forwarder import start_forwarder

        seen: Dict[str, Any] = {}

        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                del args

            def do_GET(self):
                seen["path"] = self.path
                seen["route_header"] = self.headers.get("X-Route")
                body = b'{"pong": true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{upstream.server_address[1]}/proxy/5000"
            forwarder, port = start_forwarder(base, {"X-Route": "r1"})
            try:
                resp = rq.get(f"http://127.0.0.1:{port}/platform", timeout=10)
                assert resp.status_code == 200
                assert resp.json() == {"pong": True}
                assert seen["path"] == "/proxy/5000/platform"
                assert seen["route_header"] == "r1"
            finally:
                forwarder.shutdown()
        finally:
            upstream.shutdown()


@pytest.mark.skipif(
    not os.environ.get("OPENSANDBOX_DOMAIN"),
    reason="Requires a live OpenSandbox cluster (set OPENSANDBOX_DOMAIN).",
)
class TestLiveE2E:
    async def test_live_seed_and_release(self) -> None:
        # Outside a gym launcher there is no Hydra context: bootstrap the global aiohttp
        # client directly so `server_utils.request` does not try to parse sys.argv.
        from nemo_gym.server_utils import (
            GlobalAIOHTTPAsyncClientConfig,
            is_global_aiohttp_client_setup,
            set_global_aiohttp_client,
        )

        if not is_global_aiohttp_client_setup():
            set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())

        config = OSWorldResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="osworld_resources_server",
            sandbox_provider={
                "opensandbox": {
                    "connection": {
                        "domain": os.environ["OPENSANDBOX_DOMAIN"],
                        "api_key": os.environ.get("OPENSANDBOX_API_KEY"),
                        "protocol": "http",
                        "use_server_proxy": True,
                        "request_timeout_s": 300,
                    },
                    "create": {"skip_health_check": True, "retries": 1, "timeout_s": 600},
                    "probe": {"command": None},
                }
            },
            pool_ref=os.environ.get("OSWORLD_POOL_REF", "osworld-kvm"),
        )
        server = OSWorldResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        sandbox_state = await server._allocate_sandbox()
        try:
            await server._wait_for_boot(sandbox_state)
            result = await server._guest_execute(sandbox_state, "echo OSWORLD_LIVE_OK", True)
            assert "OSWORLD_LIVE_OK" in str(result)
        finally:
            await server._release(sandbox_state)
