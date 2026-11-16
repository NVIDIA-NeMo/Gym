# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import BaseRunRequest
from responses_api_agents.gdpval_sandboxed_agent import app


def agent():
    return app.GDPValSandboxedAgent(
        config=app.GDPValSandboxedAgentConfig(
            name="gdpval_train",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            resources_server={"type": "resources_servers", "name": "gdpval_resources"},
            checkpoint="policy",
            remote_openhands_install_script_path="/runtime/install.sh",
            remote_openhands_archive_path="/runtime/runtime.tar.gz",
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )


def run_body():
    return BaseRunRequest.model_validate({"responses_create_params": {"input": [], "metadata": {"task_id": "task"}}})


async def test_run_keeps_seed_cookie_and_resources_ownership(monkeypatch):
    instance = agent()
    body = run_body()
    result = {
        "responses_create_params": body.responses_create_params.model_dump(),
        "response": app._incomplete_response("policy", "test").model_dump(),
        "reward": 0.0,
        "agent_finished": False,
    }
    replies = [
        SimpleNamespace(
            cookies={"session": "seeded"}, json=AsyncMock(return_value={"sandbox_descriptor": {"sandbox_id": "owned"}})
        ),
        SimpleNamespace(json=AsyncMock(return_value=result)),
        SimpleNamespace(),
    ]
    instance.server_client.post = AsyncMock(side_effect=replies)
    sandbox = SimpleNamespace(disconnect=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_attach", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_execute", AsyncMock(return_value={"agent_finished": False}))
    response = await instance.run(body, SimpleNamespace(cookies={"caller": "original"}))
    assert response.model_extra["agent_finished"] is False
    calls = instance.server_client.post.call_args_list
    assert [call.kwargs["url_path"] for call in calls] == ["/seed_session", "/verify", "/cleanup_session"]
    assert calls[1].kwargs["cookies"] == {"caller": "original", "session": "seeded"}
    assert calls[2].kwargs["cookies"] == calls[1].kwargs["cookies"]
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    assert instance._resource_load == {"gdpval_resources": 0}


async def test_run_cancellation_still_closes_resources_and_disconnects(monkeypatch):
    instance = agent()
    instance.server_client.post = AsyncMock(
        side_effect=[
            SimpleNamespace(cookies={}, json=AsyncMock(return_value={"sandbox_descriptor": {"sandbox_id": "owned"}})),
            SimpleNamespace(),
        ]
    )
    sandbox = SimpleNamespace(disconnect=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_attach", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_execute", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await instance.run(run_body(), SimpleNamespace(cookies={}))
    assert instance.server_client.post.call_args_list[-1].kwargs["url_path"] == "/cleanup_session"
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_provider_timeout_without_exit_code_is_unfinished(monkeypatch):
    instance = agent()
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=None, error_type="timeout", stdout=None, stderr=None)),
    )
    monkeypatch.setattr(app, "install_runtime", AsyncMock())
    monkeypatch.setattr(app, "get_server_url", lambda _: "http://policy:8080")
    monkeypatch.setattr(app.GDPValSandboxedAgent, "base_url_for_run", lambda self, url, body: url)
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_download_trace", AsyncMock())
    result = await instance._execute(sandbox, run_body())
    assert result["agent_finished"] is False
    assert result["incomplete_reason"] == "agent_exec_timeout"
    assert result["response"]["status"] == "incomplete"
    assert result["response"]["output"] == []
    assert sandbox.exec.call_args.args[0].startswith("timeout -k 10 3600 ")
    assert sandbox.exec.call_args.kwargs["timeout_s"] == 3630


async def test_inconclusive_trace_packaging_is_rejected(tmp_path):
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=None, error_type="timeout", stderr="expired")),
        download=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="Failed to package OpenHands trace: timeout"):
        await app.GDPValSandboxedAgent._download_trace(sandbox, tmp_path)
    sandbox.download.assert_not_awaited()


async def test_unsafe_trace_returns_masked_payload_and_releases_borrow(monkeypatch):
    instance = agent()
    body = run_body()
    verify_payloads = []

    async def post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return SimpleNamespace(
                cookies={}, json=AsyncMock(return_value={"sandbox_descriptor": {"sandbox_id": "owned"}})
            )
        if kwargs["url_path"] == "/verify":
            verify_payloads.append(kwargs["json"])
            return SimpleNamespace(json=AsyncMock(return_value={**kwargs["json"], "reward": 0.0, "mask_sample": True}))
        return SimpleNamespace()

    instance.server_client.post = AsyncMock(side_effect=post)
    sandbox = SimpleNamespace(disconnect=AsyncMock())
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_attach", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_execute", AsyncMock(side_effect=app.TraceExportError("too large")))
    result = await instance.run(body, SimpleNamespace(cookies={}))
    assert result.mask_sample is True
    assert verify_payloads[0]["agent_finished"] is False
    assert verify_payloads[0]["incomplete_reason"] == "agent_exec_incomplete"
    assert verify_payloads[0]["agent_error"] == "too large"
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("archive_name", ["trace/../escape.json", "trace/oversized.json"])
async def test_download_trace_rejects_paths_and_limits_before_extracting(monkeypatch, tmp_path, archive_name):
    import zipfile

    async def download(remote, destination):
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(archive_name, b"x" * 1024)

    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None)),
        download=download,
    )
    if "oversized" in archive_name:
        monkeypatch.setattr(app, "MAX_TRACE_FILE_BYTES", 100)
    with pytest.raises(ValueError, match="exceeds extraction limits|Unsafe OpenHands trace"):
        await app.GDPValSandboxedAgent._download_trace(sandbox, tmp_path)
    assert not (tmp_path / "trace").exists()
    assert not (tmp_path / "escape.json").exists()


async def test_trace_read_error_has_a_typed_boundary(monkeypatch):
    instance = agent()
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")),
    )
    monkeypatch.setattr(app, "install_runtime", AsyncMock())
    monkeypatch.setattr(app, "get_server_url", lambda _: "http://policy:8080")
    monkeypatch.setattr(app.GDPValSandboxedAgent, "base_url_for_run", lambda self, url, body: url)
    monkeypatch.setattr(
        app.GDPValSandboxedAgent, "_download_trace", AsyncMock(side_effect=ValueError("unsafe archive"))
    )
    with pytest.raises(app.TraceExportError, match="Cannot collect OpenHands trace: unsafe archive"):
        await instance._execute(sandbox, run_body())


def lost_sandbox(failure_kind: str) -> app.NgFailureEvent:
    return app.NgFailureEvent(boundary="sandbox", failure_kind=failure_kind, failure_reason="sandbox went away")


def verify_capturing_agent(monkeypatch, execute_error: BaseException):
    instance = agent()
    verify_payloads = []

    async def post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return SimpleNamespace(
                cookies={}, json=AsyncMock(return_value={"sandbox_descriptor": {"sandbox_id": "x"}})
            )
        if kwargs["url_path"] == "/verify":
            verify_payloads.append(kwargs["json"])
            return SimpleNamespace(json=AsyncMock(return_value={**kwargs["json"], "reward": 0.0, "mask_sample": True}))
        return SimpleNamespace()

    instance.server_client.post = AsyncMock(side_effect=post)
    sandbox = SimpleNamespace(disconnect=AsyncMock())
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_attach", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app.GDPValSandboxedAgent, "_execute", AsyncMock(side_effect=execute_error))
    return instance, verify_payloads, sandbox


async def test_typed_sandbox_loss_is_reported_as_a_masked_attempt(monkeypatch):
    instance, verify_payloads, sandbox = verify_capturing_agent(monkeypatch, lost_sandbox("provider_unavailable"))
    await instance.run(run_body(), SimpleNamespace(cookies={}))
    assert verify_payloads[0]["agent_finished"] is False
    assert verify_payloads[0]["incomplete_reason"] == "sandbox_ended"
    assert verify_payloads[0]["sandbox_error"].startswith("provider_unavailable: ")
    assert instance.server_client.post.call_args_list[-1].kwargs["url_path"] == "/cleanup_session"
    sandbox.disconnect.assert_awaited_once()


async def test_rejected_sandbox_credentials_fail_the_run(monkeypatch):
    instance, verify_payloads, sandbox = verify_capturing_agent(
        monkeypatch, lost_sandbox("provider_credentials_rejected")
    )
    with pytest.raises(app.NgFailureEvent):
        await instance.run(run_body(), SimpleNamespace(cookies={}))
    assert not verify_payloads
    assert instance.server_client.post.call_args_list[-1].kwargs["url_path"] == "/cleanup_session"


async def test_failed_runtime_install_is_a_masked_attempt(monkeypatch):
    error = app.RuntimeInstallError("OpenHands runtime installation failed after 4 attempts")
    instance, verify_payloads, _ = verify_capturing_agent(monkeypatch, error)
    await instance.run(run_body(), SimpleNamespace(cookies={}))
    assert verify_payloads[0]["incomplete_reason"] == "agent_exec_incomplete"
    assert verify_payloads[0]["agent_error"].startswith("OpenHands runtime installation failed")


async def test_failed_seed_still_cleans_up(monkeypatch):
    instance = agent()

    async def check(response):
        if getattr(response, "status", 200) >= 400:
            raise RuntimeError("seed failed")

    instance.server_client.post = AsyncMock(
        side_effect=[SimpleNamespace(cookies={"session": "seeded"}, status=500), SimpleNamespace(status=200)]
    )
    monkeypatch.setattr(app, "raise_for_status", check)
    with pytest.raises(RuntimeError, match="seed failed"):
        await instance.run(run_body(), SimpleNamespace(cookies={}))
    calls = instance.server_client.post.call_args_list
    assert [call.kwargs["url_path"] for call in calls] == ["/seed_session", "/cleanup_session"]
    assert calls[1].kwargs["cookies"] == {"session": "seeded"}


def test_missing_web_search_keys_fail_at_startup(tmp_path):
    empty = tmp_path / "keys.txt"
    empty.write_text("# no keys\n")
    with pytest.raises(ValueError, match="no web search keys"):
        app.GDPValSandboxedAgentConfig(
            name="gdpval_train",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            resources_server={"type": "resources_servers", "name": "gdpval_resources"},
            checkpoint="policy",
            remote_openhands_install_script_path="/runtime/install.sh",
            remote_openhands_archive_path="/runtime/runtime.tar.gz",
            web_tools_enabled=True,
            web_search_api_keys_file=str(empty),
        )


async def test_trace_is_packaged_with_the_runtime_python(tmp_path):
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=1, error_type=None, stderr="stop")),
        download=AsyncMock(),
    )
    with pytest.raises(RuntimeError):
        await app.GDPValSandboxedAgent._download_trace(sandbox, tmp_path, python="/opt/runtime/bin/python3")
    assert sandbox.exec.call_args.args[0].startswith("/opt/runtime/bin/python3 -c ")
