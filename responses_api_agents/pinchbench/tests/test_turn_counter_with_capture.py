# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""max_turns turn-proxy + rollout-scoped model-call capture must compose.

OpenClaw talks to the local turn-counter proxy (so turns are capped). The proxy's
upstream must be the capture-prefixed Gym model URL
(``/ng-rollout/<rollout_id>/v1``), otherwise model-call capture cannot attribute
the traffic.
"""

from __future__ import annotations

import pytest
from aiohttp import ClientSession, web

from responses_api_agents.pinchbench.tests.test_app import make_agent

ROLLOUT = "12-0"


@pytest.mark.asyncio
async def test_turn_proxy_forwards_to_rollout_scoped_capture_path_and_enforces_max_turns():
    seen = {"paths": [], "n": 0}

    async def chat(request: web.Request) -> web.Response:
        seen["paths"].append(request.path)
        seen["n"] += 1
        await request.read()
        return web.json_response(
            {
                "id": "chatcmpl-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }
        )

    app = web.Application()
    app.router.add_post("/ng-rollout/{rollout_id}/v1/chat/completions", chat)
    app.router.add_post("/v1/chat/completions", chat)  # bare path would mean capture is broken
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001

    agent = make_agent(max_turns=2, model_base_url=f"http://127.0.0.1:{port}/v1")
    upstream = agent._rollout_scoped_model_base_url(ROLLOUT)
    assert upstream == f"http://127.0.0.1:{port}/ng-rollout/{ROLLOUT}/v1"

    proxy = await agent._maybe_start_turn_proxy(upstream)
    try:
        assert proxy is not None
        # OpenClaw must see the proxy, not the capture-prefixed model URL.
        assert "/ng-rollout/" not in proxy.base_url

        async with ClientSession() as client:
            for _ in range(2):
                async with client.post(
                    f"{proxy.base_url}/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                ) as resp:
                    assert resp.status == 200

            async with client.post(
                f"{proxy.base_url}/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            ) as resp:
                assert resp.status == 429
                err = await resp.json()
                assert err["error"]["code"] == "session_budget_exhausted"
    finally:
        await proxy.stop()
        await site.stop()
        await runner.cleanup()

    assert seen["n"] == 2
    assert seen["paths"] == [
        f"/ng-rollout/{ROLLOUT}/v1/chat/completions",
        f"/ng-rollout/{ROLLOUT}/v1/chat/completions",
    ]


@pytest.mark.asyncio
async def test_run_in_sandbox_wires_proxy_to_capture_upstream(tmp_path, monkeypatch):
    agent = make_agent(
        max_turns=5,
        model_base_url="http://model:8000/v1",
        sandbox_provider={"apptainer": {}},
        sandbox_spec={"image": "/sif/pinchbench.sif"},
        work_root=str(tmp_path / "work"),
    )

    started = {}

    class FakeProxy:
        base_url = "http://127.0.0.1:5555/v1"

        async def stop(self):
            started["stopped"] = True

    async def fake_maybe_start(upstream_base_url: str):
        started["upstream"] = upstream_base_url
        return FakeProxy()

    class FakeSandbox:
        def __init__(self, _provider):
            pass

        async def start(self, spec):
            started["env_url"] = spec.env["MODEL_BASE_URL"]

        async def exec(self, _cmd, timeout_s=None):
            return None

        async def download(self, _src, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Minimal gzip tar so extractall succeeds.
            import tarfile

            with tarfile.open(dest, "w:gz") as tf:
                pass

        async def stop(self):
            return None

    monkeypatch.setattr(agent, "_maybe_start_turn_proxy", fake_maybe_start)
    monkeypatch.setattr("responses_api_agents.pinchbench.app.AsyncSandbox", FakeSandbox)

    await agent._run_in_sandbox("task_x", tmp_path / "out", ROLLOUT)

    assert started["upstream"] == f"http://model:8000/ng-rollout/{ROLLOUT}/v1"
    assert started["env_url"] == "http://127.0.0.1:5555/v1"
    assert started.get("stopped") is True


@pytest.mark.asyncio
async def test_without_max_turns_sandbox_gets_capture_url_directly(tmp_path, monkeypatch):
    agent = make_agent(
        model_base_url="http://model:8000/v1",
        sandbox_provider={"apptainer": {}},
        sandbox_spec={"image": "/sif/pinchbench.sif"},
        work_root=str(tmp_path / "work"),
    )

    started = {}

    class FakeSandbox:
        def __init__(self, _provider):
            pass

        async def start(self, spec):
            started["env_url"] = spec.env["MODEL_BASE_URL"]

        async def exec(self, _cmd, timeout_s=None):
            return None

        async def download(self, _src, dest):
            import tarfile

            dest.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(dest, "w:gz") as tf:
                pass

        async def stop(self):
            return None

    monkeypatch.setattr("responses_api_agents.pinchbench.app.AsyncSandbox", FakeSandbox)
    await agent._run_in_sandbox("task_x", tmp_path / "out", ROLLOUT)

    assert started["env_url"] == f"http://model:8000/ng-rollout/{ROLLOUT}/v1"
    assert await agent._maybe_start_turn_proxy(started["env_url"]) is None
