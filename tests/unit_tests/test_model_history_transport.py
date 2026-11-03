# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loopback HTTP transport, history reconstruction, SSE and native configuration tests."""

import json
import threading
from contextlib import ExitStack
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import requests

from nemo_gym.agents.utils.model_history_transport import (
    DeltaState,
    HistoryTransport,
    configure_transport,
    stage_cli_transport,
)
from nemo_gym.models.history_delta import HistoryBuffer, HistoryResync


@pytest.fixture
def model_server():
    """Local upstream with observable requests and controllable replies."""
    model = SimpleNamespace(
        buffer=HistoryBuffer(),
        wire=[],
        restored=[],
        upstream_clients=set(),
        status=200,
        stream=False,
        disconnect_before_history=False,
        disconnect_after_history=False,
        first=threading.Event(),
        release=threading.Event(),
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            model.upstream_clients.add(self.client_address)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model.wire.append((deepcopy(body), dict(self.headers)))
            if model.disconnect_before_history:
                model.disconnect_before_history = False
                self.close_connection = True
                return
            field = "input" if self.path.endswith("/responses") else "messages"
            try:
                restored = (
                    model.buffer.reconstruct("rollout", body, history_field=field)
                    if self.headers.get("x-gym-history-transport")
                    else body
                )
            except HistoryResync:
                data = b'{"error":{"code":"gym_history_resync"}}'
                self.send_response(409)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            model.restored.append(restored)
            if model.disconnect_after_history:
                model.disconnect_after_history = False
                self.close_connection = True
                return
            if model.stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                data = b'data: {"delta":"first"}\n\n'
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                model.first.set()
                if model.release.wait(5):
                    self.wfile.write(b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")
                    self.wfile.flush()
            else:
                data = b'{"unchanged":true}'
                self.send_response(model.status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    model.url = f"http://127.0.0.1:{server.server_port}/ng-rollout/test/v1"
    yield model
    model.release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture
def post_model(model_server):
    """Send ordinary HTTP requests through the sandbox-local transport."""
    with HistoryTransport(model_server.url, timeout=5) as url, requests.Session() as session:
        session.trust_env = False

        def post(*, endpoint="chat/completions", **kwargs):
            return session.post(f"{url}/{endpoint}", timeout=5, **kwargs)

        try:
            yield post
        finally:
            model_server.release.set()


@pytest.mark.parametrize("endpoint,field", [("chat/completions", "messages"), ("responses", "input")])
def test_full_delta_edit_and_lost_base_preserve_http_requests(post_model, model_server, endpoint, field):
    options = {
        "model": "unchanged",
        "stream": True,
        "tools": [{"type": "function", "name": "tool"}],
        "metadata": {"a": 1},
    }
    headers = {"x-session-id": "unchanged"}
    user = {"role": "user", "content": "task"}
    assistant = {"role": "assistant", "content": "answer"}
    compacted = {"role": "user", "content": "compacted"}

    first_request = {**options, field: [user]}
    appended_request = {**options, field: [user, assistant]}
    edited_request = {**options, field: [compacted]}
    after_restart_request = {**options, field: [compacted, assistant]}

    first_response = post_model(endpoint=endpoint, json=first_request, headers=headers)
    appended_response = post_model(endpoint=endpoint, json=appended_request, headers=headers)
    edited_response = post_model(endpoint=endpoint, json=edited_request, headers=headers)
    model_server.buffer = HistoryBuffer()
    after_restart_response = post_model(endpoint=endpoint, json=after_restart_request, headers=headers)

    assert first_response.status_code == 200
    assert appended_response.status_code == 200
    assert edited_response.status_code == 200
    assert after_restart_response.status_code == 200
    assert first_response.json() == {"unchanged": True}
    assert appended_response.json() == {"unchanged": True}
    assert edited_response.json() == {"unchanged": True}
    assert after_restart_response.json() == {"unchanged": True}
    assert model_server.restored == [first_request, appended_request, edited_request, after_restart_request]
    assert len(model_server.upstream_clients) == 1
    assert [body["_gym_history"]["mode"] for body, _ in model_server.wire] == [
        "full",
        "append",
        "full",
        "append",
        "full",
    ]
    assert model_server.wire[1][0][field] == [assistant]
    assert all(received_headers["x-session-id"] == "unchanged" for _, received_headers in model_server.wire)


def test_model_errors_are_not_retried(post_model, model_server):
    model_server.status = 500
    response = post_model(json={"messages": [{"role": "user", "content": "task"}]})

    assert response.status_code == 500
    assert len(model_server.wire) == 1


def test_lost_first_response_preserves_history_for_retry(post_model, model_server):
    request = {"input": [{"role": "user", "content": "task"}]}

    model_server.disconnect_after_history = True
    lost_response = post_model(endpoint="responses", json=request)
    assert lost_response.status_code == 502
    assert len(model_server.wire) == 1

    retry_response = post_model(endpoint="responses", json=request)
    assert retry_response.status_code == 200
    assert retry_response.json() == {"unchanged": True}
    assert model_server.restored == [request, request]
    first_wire, retry_wire = [body for body, _ in model_server.wire]
    assert first_wire["_gym_history"]["mode"] == "full"
    assert retry_wire["_gym_history"]["mode"] == "append"
    assert retry_wire["_gym_history"]["base_version"] == first_wire["_gym_history"]["request_id"]
    assert retry_wire["input"] == []


def test_lost_append_response_preserves_history_for_next_request(post_model, model_server):
    user = {"role": "user", "content": "task"}
    assistant = {"role": "assistant", "content": "answer"}
    follow_up = {"role": "user", "content": "continue"}
    first_request = {"input": [user]}
    appended_request = {"input": [user, assistant]}
    next_request = {"input": [user, assistant, follow_up]}

    first_response = post_model(endpoint="responses", json=first_request)
    assert first_response.status_code == 200

    model_server.disconnect_after_history = True
    lost_response = post_model(endpoint="responses", json=appended_request)
    assert lost_response.status_code == 502
    assert len(model_server.wire) == 2

    next_response = post_model(endpoint="responses", json=next_request)
    assert next_response.status_code == 200
    assert model_server.restored == [first_request, appended_request, next_request]
    first_wire, appended_wire, next_wire = [body for body, _ in model_server.wire]
    assert first_wire["_gym_history"]["mode"] == "full"
    assert appended_wire["input"] == [assistant]
    assert next_wire["_gym_history"]["mode"] == "append"
    assert next_wire["_gym_history"]["base_version"] == appended_wire["_gym_history"]["request_id"]
    assert next_wire["input"] == [follow_up]


def test_repeated_lost_responses_do_not_resend_full_history(post_model, model_server):
    request = {"input": [{"role": "user", "content": "task"}]}

    model_server.disconnect_after_history = True
    first_response = post_model(endpoint="responses", json=request)
    assert first_response.status_code == 502

    model_server.disconnect_after_history = True
    second_response = post_model(endpoint="responses", json=request)
    assert second_response.status_code == 502

    third_response = post_model(endpoint="responses", json=request)
    assert third_response.status_code == 200
    assert model_server.restored == [request, request, request]
    first_wire, second_wire, third_wire = [body for body, _ in model_server.wire]
    assert first_wire["_gym_history"]["mode"] == "full"
    assert second_wire["_gym_history"]["mode"] == "append"
    assert second_wire["input"] == []
    assert third_wire["_gym_history"]["mode"] == "append"
    assert third_wire["input"] == []
    assert third_wire["_gym_history"]["base_version"] == second_wire["_gym_history"]["request_id"]


def test_unstored_history_is_resent_only_after_explicit_resync(post_model, model_server):
    user = {"role": "user", "content": "task"}
    assistant = {"role": "assistant", "content": "answer"}
    first_request = {"input": [user]}
    appended_request = {"input": [user, assistant]}

    first_response = post_model(endpoint="responses", json=first_request)
    assert first_response.status_code == 200

    model_server.disconnect_before_history = True
    lost_response = post_model(endpoint="responses", json=appended_request)
    assert lost_response.status_code == 502
    assert len(model_server.wire) == 2

    retry_response = post_model(endpoint="responses", json=appended_request)
    assert retry_response.status_code == 200
    assert model_server.restored == [first_request, appended_request]
    first_wire, lost_wire, rejected_wire, full_wire = [body for body, _ in model_server.wire]
    assert first_wire["_gym_history"]["mode"] == "full"
    assert lost_wire["input"] == [assistant]
    assert rejected_wire["_gym_history"]["mode"] == "append"
    assert rejected_wire["_gym_history"]["base_version"] == lost_wire["_gym_history"]["request_id"]
    assert rejected_wire["input"] == []
    assert full_wire["_gym_history"]["mode"] == "full"
    assert full_wire["input"] == [user, assistant]


def test_chunked_upload_and_streaming_reply_are_relayed_incrementally(post_model, model_server):
    model_server.stream = True
    body = {"input": [{"role": "user", "content": "task"}], "stream": True}
    with post_model(endpoint="responses", data=iter([json.dumps(body).encode()]), stream=True) as response:
        assert response.status_code == 200
        assert model_server.first.wait(2)
        assert response.raw.readline() == b'data: {"delta":"first"}\n'
        assert not model_server.release.is_set()
        model_server.release.set()
        assert response.raw.read() == b"\ndata: [DONE]\n\n"


def test_stale_delta_is_rejected_and_full_resync_restores_request():
    history_state = DeltaState()
    upstream_history = HistoryBuffer()
    user = {"role": "user", "content": "task"}
    assistant = {"role": "assistant", "content": "answer"}
    first_request = {"input": [user]}
    appended_request = {"input": [user, assistant]}

    first_wire, first_identity, first_items = history_state.prepare(first_request, "input")
    assert upstream_history.reconstruct("rollout", json.loads(first_wire), history_field="input") == first_request
    history_state.submitted(first_identity, first_items)

    accepted_wire, _, _ = history_state.prepare(appended_request, "input")
    stale_wire, _, _ = history_state.prepare(appended_request, "input")
    assert (
        upstream_history.reconstruct("rollout", json.loads(accepted_wire), history_field="input") == appended_request
    )
    with pytest.raises(HistoryResync, match="Missing or stale history base"):
        upstream_history.reconstruct("rollout", json.loads(stale_wire), history_field="input")

    full_resync_wire, _, _ = history_state.prepare(appended_request, "input", full=True)
    assert (
        upstream_history.reconstruct("rollout", json.loads(full_resync_wire), history_field="input")
        == appended_request
    )


def test_disabled_codex_configuration_leaves_command_and_environment_unchanged(tmp_path):
    params = {
        "directory": str(tmp_path),
        "env": {"PRESERVED": "yes"},
        "command": ["node", "codex.js", "exec", "--", "-"],
    }
    original = deepcopy(params)

    with ExitStack() as stack:
        configure_transport(stack, params, harness="codex")

    assert params == original


def test_nonlist_responses_input_bypasses_deltas(post_model, model_server):
    body = {"input": "single text input", "stream": False}
    response = post_model(endpoint="responses", json=body)

    assert response.status_code == 200
    assert model_server.restored == [body]
    assert "x-gym-history-transport" not in model_server.wire[0][1]


@pytest.fixture
def sandbox():
    return AsyncMock()


@pytest.fixture
def upload(monkeypatch):
    upload = AsyncMock()
    monkeypatch.setattr("nemo_gym.sandbox.utils.upload_text", upload)
    return upload


async def test_disabled_bash_runner_uses_original_cli_without_uploads(tmp_path, sandbox, upload):
    params = {
        "directory": str(tmp_path),
        "command": ["node", "pi.js"],
        "env": {"PRESERVED": "yes"},
        "cwd": "/app",
    }
    original = deepcopy(params)

    command = await stage_cli_transport(sandbox, params, harness="pi")

    assert command == ["node", "pi.js"]
    assert params == original
    sandbox.upload.assert_not_awaited()
    upload.assert_not_awaited()


async def test_enabled_bash_runner_stages_transport_and_preserves_cli(tmp_path, sandbox, upload):
    params = {
        "directory": str(tmp_path),
        "command": ["node", "pi.js"],
        "env": {"PRESERVED": "yes"},
        "cwd": "/app",
        "history_transport": {"base_url": "http://127.0.0.1/v1", "timeout": 5},
    }
    original = deepcopy(params)

    command = await stage_cli_transport(sandbox, params, harness="pi")

    assert command[:2] == ["bash", "-c"]
    assert "gym-history-runner" in command[2]
    assert command[-1] == "pi"
    assert json.loads(upload.await_args.kwargs["text"]) == original
    assert params == original
    sandbox.upload.assert_awaited_once()


def test_opencode_configuration_redirects_provider_without_changing_model_or_credentials(tmp_path, model_server):
    config = {
        "model": "nemo_gym/unchanged",
        "provider": {"nemo_gym": {"options": {"baseURL": model_server.url, "apiKey": "unchanged"}}},
    }
    params = {
        "directory": str(tmp_path),
        "history_transport": {"base_url": model_server.url, "timeout": 5},
        "env": {"OPENCODE_CONFIG_CONTENT": json.dumps(config)},
        "command": ["opencode", "run"],
    }

    with ExitStack() as stack:
        configure_transport(stack, params, harness="opencode")
        updated = json.loads(params["env"]["OPENCODE_CONFIG_CONTENT"])
        options = updated["provider"]["nemo_gym"]["options"]

        assert options["baseURL"] != model_server.url
        assert options["baseURL"].startswith("http://127.0.0.1:")
        assert options["apiKey"] == "unchanged"
        assert updated["model"] == "nemo_gym/unchanged"
        assert params["command"] == ["opencode", "run"]


def test_pi_configuration_redirects_provider_without_changing_models(tmp_path, model_server):
    models_file = tmp_path / "home/.pi/agent/models.json"
    models_file.parent.mkdir(parents=True)
    models_file.write_text(json.dumps({"providers": {"nemo": {"baseUrl": model_server.url, "models": ["unchanged"]}}}))
    params = {
        "directory": str(tmp_path),
        "history_transport": {"base_url": model_server.url, "timeout": 5},
        "command": ["node", "pi.js"],
    }

    with ExitStack() as stack:
        configure_transport(stack, params, harness="pi")
        provider = json.loads(models_file.read_text())["providers"]["nemo"]

        assert provider["baseUrl"] != model_server.url
        assert provider["baseUrl"].startswith("http://127.0.0.1:")
        assert provider["models"] == ["unchanged"]
        assert params["command"] == ["node", "pi.js"]


def test_codex_configuration_redirects_launcher_proxy_without_changing_cli_arguments(tmp_path, model_server):
    launcher = {
        "command": ["node", "codex.js", "exec", "--", "-"],
        "model_base_url": model_server.url,
        "invocation_id": "rollout-1",
        "timeout": 5,
        "env": {"OPENAI_API_KEY": "gym"},  # pragma: allowlist secret
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(launcher))
    params = {
        "directory": str(tmp_path),
        "history_transport": {"base_url": model_server.url, "timeout": 5},
        "command": ["node", "sandbox_launcher.mjs", str(path)],
    }

    with ExitStack() as stack:
        configure_transport(stack, params, harness="codex")
        rewritten = json.loads(path.read_text())
        local_url = rewritten["model_base_url"]
        assert local_url != model_server.url
        assert local_url.startswith("http://127.0.0.1:")
        assert rewritten == {**launcher, "model_base_url": local_url}
        assert params["command"] == ["node", "sandbox_launcher.mjs", str(path)]
