# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loopback HTTP transport, history reconstruction, SSE and native configuration tests."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import requests

from nemo_gym.agents.utils.model_history_transport import (
    DeltaState,
    HistoryTransport,
    _serialize_opencode_replacement_config,
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


@pytest.mark.parametrize("identity_header", ["thread-id", "x-session-affinity", "x-opencode-session", "x-session-id"])
def test_subagent_histories_remain_independent_on_one_connection(post_model, model_server, identity_header):
    parent = {"role": "user", "content": "parent task"}
    child = {"role": "user", "content": "child task"}
    parent_answer = {"role": "assistant", "content": "parent answer"}
    child_answer = {"role": "assistant", "content": "child answer"}
    # Codex/OpenCode identities must take precedence over a shared Gym invocation.
    parent_headers = {"x-session-id": "shared-invocation", identity_header: "parent"}
    child_headers = {"x-session-id": "shared-invocation", identity_header: "child"}

    first_parent = post_model(endpoint="responses", json={"input": [parent]}, headers=parent_headers)
    first_child = post_model(endpoint="responses", json={"input": [child]}, headers=child_headers)
    next_parent = post_model(endpoint="responses", json={"input": [parent, parent_answer]}, headers=parent_headers)
    next_child = post_model(endpoint="responses", json={"input": [child, child_answer]}, headers=child_headers)

    assert [first_parent.status_code, first_child.status_code, next_parent.status_code, next_child.status_code] == [
        200,
        200,
        200,
        200,
    ]
    parent_full, child_full, parent_delta, child_delta = [body for body, _ in model_server.wire]
    assert parent_full["_gym_history"]["conversation_id"] != child_full["_gym_history"]["conversation_id"]
    assert parent_delta["_gym_history"]["conversation_id"] == parent_full["_gym_history"]["conversation_id"]
    assert child_delta["_gym_history"]["conversation_id"] == child_full["_gym_history"]["conversation_id"]
    assert parent_delta["_gym_history"]["mode"] == child_delta["_gym_history"]["mode"] == "append"
    assert parent_delta["input"] == [parent_answer]
    assert child_delta["input"] == [child_answer]
    assert model_server.restored == [
        {"input": [parent]},
        {"input": [child]},
        {"input": [parent, parent_answer]},
        {"input": [child, child_answer]},
    ]
    assert len(model_server.upstream_clients) == 1


def test_evicted_subagent_state_sends_full_history_without_a_failed_request(post_model, model_server, monkeypatch):
    monkeypatch.setattr("nemo_gym.agents.utils.model_history_transport._MAX_HISTORY_STATES", 1)
    parent = {"role": "user", "content": "parent task"}
    child = {"role": "user", "content": "child task"}
    answer = {"role": "assistant", "content": "parent answer"}

    first_parent = post_model(endpoint="responses", json={"input": [parent]}, headers={"thread-id": "parent"})
    first_child = post_model(endpoint="responses", json={"input": [child]}, headers={"thread-id": "child"})
    next_parent = post_model(endpoint="responses", json={"input": [parent, answer]}, headers={"thread-id": "parent"})

    assert first_parent.status_code == first_child.status_code == next_parent.status_code == 200
    parent_full, child_full, resumed_parent = [body for body, _ in model_server.wire]
    assert [body["_gym_history"]["mode"] for body in [parent_full, child_full, resumed_parent]] == [
        "full",
        "full",
        "full",
    ]
    assert resumed_parent["_gym_history"]["conversation_id"] != parent_full["_gym_history"]["conversation_id"]
    assert resumed_parent["_gym_history"]["base_version"] is None
    assert resumed_parent["input"] == [parent, answer]
    assert model_server.restored == [{"input": [parent]}, {"input": [child]}, {"input": [parent, answer]}]


def test_history_state_cache_evicts_the_least_recently_used_id(monkeypatch):
    monkeypatch.setattr("nemo_gym.agents.utils.model_history_transport._MAX_HISTORY_STATES", 2)
    transport = HistoryTransport("http://127.0.0.1/v1", timeout=5)
    with transport:
        parent = transport._history_state("/v1/responses", {"thread-id": "parent"})
        child = transport._history_state("/v1/responses", {"thread-id": "child"})
        assert transport._history_state("/v1/responses", {"thread-id": "parent"}) is parent

        transport._history_state("/v1/responses", {"thread-id": "another-child"})

        assert len(transport._states) == 2
        assert transport._history_state("/v1/responses", {"thread-id": "parent"}) is parent
        assert transport._history_state("/v1/responses", {"thread-id": "child"}) is not child
        assert len(transport._states) == 2


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


@pytest.mark.parametrize("literal_replacement_prompt", [False, True], ids=["configured-prompt", "replacement-prompt"])
def test_opencode_url_rewrite_preserves_selective_prompt_escaping(tmp_path, literal_replacement_prompt):
    prompt = 'λ\n"quoted" \\path\\end\n{env:TEMPLATE_VALUE}\n{file:missing-prompt.txt}\nliteral \\u007b'
    config = {
        "model": "nemo_gym/unchanged",
        "provider": {"nemo_gym": {"options": {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "unchanged"}}},
        "agent": {
            "build": {"prompt": prompt, "description": "{env:TEMPLATE_VALUE}"},
            "plan": {"prompt": "{file:existing-prompt.txt}"},
        },
        "username": "{env:TEMPLATE_VALUE}",
    }
    original = deepcopy(config)
    params = {
        "directory": str(tmp_path),
        "history_transport": {"base_url": "http://127.0.0.1:8000/v1", "timeout": 5},
        "literal_replacement_prompt": literal_replacement_prompt,
        "env": {
            "OPENCODE_CONFIG_CONTENT": (
                _serialize_opencode_replacement_config(config) if literal_replacement_prompt else json.dumps(config)
            )
        },
        "command": ["opencode", "run"],
    }

    with ExitStack() as stack:
        configure_transport(stack, params, harness="opencode")
        serialized = params["env"]["OPENCODE_CONFIG_CONTENT"]
        updated = json.loads(serialized)

    assert config == original
    assert updated["provider"]["nemo_gym"]["options"]["baseURL"] != params["history_transport"]["base_url"]
    assert updated["provider"]["nemo_gym"]["options"]["baseURL"].startswith("http://127.0.0.1:")
    updated["provider"]["nemo_gym"]["options"]["baseURL"] = original["provider"]["nemo_gym"]["options"]["baseURL"]
    assert updated == original
    if literal_replacement_prompt:
        assert r"\u007benv:TEMPLATE_VALUE}" in serialized
        assert r"\u007bfile:missing-prompt.txt}" in serialized
    else:
        assert "{env:TEMPLATE_VALUE}" in serialized
        assert "{file:missing-prompt.txt}" in serialized
    # Native expansion remains available outside the generated build prompt.
    compact_keys = serialized.replace('": ', '":')
    assert '"description":"{env:TEMPLATE_VALUE}"' in compact_keys
    assert '"plan":{"prompt":"{file:existing-prompt.txt}"}' in compact_keys
    assert '"username":"{env:TEMPLATE_VALUE}"' in compact_keys


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


@pytest.fixture
def overlapping_model_server():
    """Hold the root response open while a child uses the same model endpoint."""
    model = SimpleNamespace(
        buffer=HistoryBuffer(),
        lock=threading.Lock(),
        wire=[],
        first_ready=threading.Event(),
        release_first=threading.Event(),
        disconnect_child=False,
        upstream_clients=set(),
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            session = self.headers["x-session-id"]
            with model.lock:
                model.upstream_clients.add(self.client_address)
                model.wire.append((session, deepcopy(body)))
                if session == "child" and model.disconnect_child:
                    self.disconnect_without_response()
                    return
                try:
                    # Gym reconstructs synchronously on one worker's event loop.
                    restored = model.buffer.reconstruct("rollout", body, history_field="input")
                except HistoryResync:
                    self.send_history_resync()
                    return
            data = json.dumps({"session": session, "input": restored["input"]}).encode()
            if body.get("stream", False):
                self.send_streamed_response(session, data)
            else:
                self.send_buffered_response(session, data)

        def disconnect_without_response(self):
            self.close_connection = True

        def send_history_resync(self):
            error = b'{"error":{"code":"gym_history_resync"}}'
            self.send_response(409)
            self.send_header("Content-Length", str(len(error)))
            self.end_headers()
            self.wfile.write(error)

        def wait_for_root_release(self):
            model.first_ready.set()
            if model.release_first.wait(10):
                return True
            self.close_connection = True
            return False

        def send_buffered_response(self, session, data):
            # Hold the entire root response; child/replacement replies are immediate.
            if session == "root" and not self.wait_for_root_release():
                return
            self.send_response(200)
            self.send_header("x-session-id", session)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_streamed_response(self, session, data):
            self.send_response(200)
            self.send_header("x-session-id", session)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk = b"data: " + data + b"\n\n"
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
            # Hold the root stream after its first event, before the terminal event.
            if session == "root" and not self.wait_for_root_release():
                return
            final = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(final):x}\r\n".encode() + final + b"\r\n0\r\n\r\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    model.url = f"http://127.0.0.1:{server.server_port}/ng-rollout/test/v1"
    try:
        yield model
    finally:
        model.release_first.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_overlapping_model_requests_receive_complete_separate_responses(overlapping_model_server, stream):
    model = overlapping_model_server
    root_request = {"input": [{"role": "user", "content": "root task"}], "stream": stream}
    child_request = {"input": [{"role": "user", "content": "child task"}], "stream": stream}
    with (
        HistoryTransport(model.url, timeout=5) as url,
        requests.Session() as root_client,
        requests.Session() as child_client,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        root_client.trust_env = child_client.trust_env = False
        root_result = executor.submit(
            root_client.post, f"{url}/responses", json=root_request, headers={"x-session-id": "root"}, timeout=10
        )
        try:
            assert model.first_ready.wait(5), "Root request did not reach the upstream server"
            child_response = child_client.post(
                f"{url}/responses", json=child_request, headers={"x-session-id": "child"}, timeout=5
            )
            assert child_response.status_code == 200
            assert child_response.headers["x-session-id"] == "child"
            assert not root_result.done(), "Root response must remain blocked while the child completes"
        finally:
            model.release_first.set()
        root_response = root_result.result(timeout=5)

    assert root_response.status_code == 200
    assert root_response.headers["x-session-id"] == "root"
    if stream:
        root_lines = root_response.content.splitlines()
        child_lines = child_response.content.splitlines()
        assert root_lines[0].startswith(b"data: ")
        assert child_lines[0].startswith(b"data: ")
        assert root_lines[1:] == [b"", b"data: [DONE]", b""]
        assert child_lines[1:] == [b"", b"data: [DONE]", b""]
        root_payload = json.loads(root_lines[0].removeprefix(b"data: "))
        child_payload = json.loads(child_lines[0].removeprefix(b"data: "))
    else:
        root_payload = root_response.json()
        child_payload = child_response.json()
    assert root_payload == {"session": "root", "input": root_request["input"]}
    assert child_payload == {"session": "child", "input": child_request["input"]}
    assert [session for session, _ in model.wire] == ["root", "child"]


def test_failed_exchange_releases_capacity_without_interrupting_other_responses(overlapping_model_server):
    model = overlapping_model_server
    model.disconnect_child = True
    root_request = {"input": [{"role": "user", "content": "root task"}]}
    child_request = {"input": [{"role": "user", "content": "child task"}]}
    replacement_request = {"input": [{"role": "user", "content": "replacement task"}]}
    with (
        HistoryTransport(model.url, timeout=5, max_connections=2) as url,
        requests.Session() as root_client,
        requests.Session() as child_client,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        root_client.trust_env = child_client.trust_env = False
        root_result = executor.submit(
            root_client.post, f"{url}/responses", json=root_request, headers={"x-session-id": "root"}, timeout=10
        )
        try:
            assert model.first_ready.wait(5), "Root request did not reach the upstream server"
            child_response = child_client.post(
                f"{url}/responses", json=child_request, headers={"x-session-id": "child"}, timeout=5
            )
            assert child_response.status_code == 502
            assert not root_result.done(), "Child failure must not interrupt the root response"
            replacement_response = child_client.post(
                f"{url}/responses", json=replacement_request, headers={"x-session-id": "replacement"}, timeout=5
            )
            assert replacement_response.status_code == 200
            assert replacement_response.json() == {"session": "replacement", "input": replacement_request["input"]}
        finally:
            model.release_first.set()
        root_response = root_result.result(timeout=5)

    assert root_response.status_code == 200
    assert root_response.json() == {"session": "root", "input": root_request["input"]}
    assert [session for session, _ in model.wire] == ["root", "child", "replacement"]  # Failed child was not retried.


def test_sequential_cli_connections_reuse_one_upstream_connection(model_server):
    user = {"role": "user", "content": "task"}
    assistant = {"role": "assistant", "content": "answer"}
    first_request = {"input": [user]}
    appended_request = {"input": [user, assistant]}
    with HistoryTransport(model_server.url, timeout=5) as url:
        with requests.Session() as first_client:
            first_client.trust_env = False
            first_response = first_client.post(f"{url}/responses", json=first_request, timeout=5)
        with requests.Session() as next_client:
            next_client.trust_env = False
            appended_response = next_client.post(f"{url}/responses", json=appended_request, timeout=5)

    assert first_response.status_code == 200
    assert appended_response.status_code == 200
    assert model_server.restored == [first_request, appended_request]
    assert model_server.wire[1][0]["input"] == [assistant]
    assert len(model_server.upstream_clients) == 1


@pytest.mark.parametrize("max_connections", [0, -1, 1.5, True])
def test_pool_capacity_must_be_a_positive_integer(max_connections):
    with pytest.raises(ValueError, match="max_connections must be a positive integer"):
        HistoryTransport("http://127.0.0.1/v1", timeout=5, max_connections=max_connections)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_exhausted_pool_waits_for_the_active_response(overlapping_model_server, monkeypatch, stream):
    model = overlapping_model_server
    transport = HistoryTransport(model.url, timeout=5, max_connections=1)
    waiting = threading.Event()
    original_wait = transport._connection_available.wait

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(transport._connection_available, "wait", observe_wait)
    root_request = {"input": [{"role": "user", "content": "root task"}], "stream": stream}
    child_request = {"input": [{"role": "user", "content": "child task"}], "stream": stream}
    with (
        transport as url,
        requests.Session() as root_client,
        requests.Session() as child_client,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        root_client.trust_env = child_client.trust_env = False
        root_result = executor.submit(
            root_client.post, f"{url}/responses", json=root_request, headers={"x-session-id": "root"}, timeout=10
        )
        try:
            assert model.first_ready.wait(5), "Root request did not reach the upstream server"
            child_result = executor.submit(
                child_client.post, f"{url}/responses", json=child_request, headers={"x-session-id": "child"}, timeout=10
            )
            assert waiting.wait(5), "Child request did not wait for the occupied connection"
            assert not child_result.done()
            assert [session for session, _ in model.wire] == ["root"]
        finally:
            model.release_first.set()
        root_response = root_result.result(timeout=5)
        child_response = child_result.result(timeout=5)

    assert root_response.status_code == 200
    assert child_response.status_code == 200
    assert root_response.headers["x-session-id"] == "root"
    assert child_response.headers["x-session-id"] == "child"
    assert [session for session, _ in model.wire] == ["root", "child"]
    assert len(model.upstream_clients) == 1


def test_waiting_for_an_exhausted_pool_uses_the_transport_timeout():
    transport = HistoryTransport("http://127.0.0.1/v1", timeout=0.01, max_connections=1)
    with transport, transport._upstream_connection():
        with pytest.raises(TimeoutError, match="Timed out waiting for a history transport connection"):
            with transport._upstream_connection():
                pytest.fail("An occupied pool must not allocate a second connection")


def test_parallel_bursts_reuse_all_cached_connections(overlapping_model_server):
    model = overlapping_model_server
    root_request = {"input": [{"role": "user", "content": "root task"}]}
    child_request = {"input": [{"role": "user", "content": "child task"}]}
    with (
        HistoryTransport(model.url, timeout=5, max_connections=2) as url,
        requests.Session() as root_client,
        requests.Session() as child_client,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        root_client.trust_env = child_client.trust_env = False
        first_root = executor.submit(
            root_client.post, f"{url}/responses", json=root_request, headers={"x-session-id": "root"}, timeout=10
        )
        try:
            assert model.first_ready.wait(5)
            first_child = child_client.post(
                f"{url}/responses", json=child_request, headers={"x-session-id": "child"}, timeout=5
            )
            assert first_child.status_code == 200
        finally:
            model.release_first.set()
        assert first_root.result(timeout=5).status_code == 200
        assert len(model.upstream_clients) == 2

        model.first_ready.clear()
        model.release_first.clear()
        next_root = executor.submit(
            root_client.post, f"{url}/responses", json=root_request, headers={"x-session-id": "root"}, timeout=10
        )
        try:
            assert model.first_ready.wait(5)
            next_child = child_client.post(
                f"{url}/responses", json=child_request, headers={"x-session-id": "child"}, timeout=5
            )
            assert next_child.status_code == 200
        finally:
            model.release_first.set()
        assert next_root.result(timeout=5).status_code == 200

    assert [session for session, _ in model.wire] == ["root", "child", "root", "child"]
    assert len(model.upstream_clients) == 2  # Both warm connections survived the first burst.


def test_shutdown_aborts_active_requests_and_rejects_waiting_leases(overlapping_model_server, monkeypatch):
    model = overlapping_model_server
    transport = HistoryTransport(model.url, timeout=30, max_connections=1)
    waiting = threading.Event()
    original_wait = transport._connection_available.wait

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(transport._connection_available, "wait", observe_wait)

    def acquire_connection():
        with transport._upstream_connection():
            pytest.fail("Shutdown must reject the queued lease")

    with ThreadPoolExecutor(max_workers=2) as executor, ExitStack() as stack:
        url = stack.enter_context(transport)
        client = stack.enter_context(requests.Session())
        client.trust_env = False
        active_request = executor.submit(
            client.post,
            f"{url}/responses",
            json={"input": [{"role": "user", "content": "root task"}]},
            headers={"x-session-id": "root"},
            timeout=30,
        )
        assert model.first_ready.wait(5), "Root request did not reach the upstream server"
        queued_lease = executor.submit(acquire_connection)
        assert waiting.wait(5), "The second lease did not wait for capacity"

        stack.close()

        with pytest.raises(OSError, match="History transport is closing"):
            queued_lease.result(timeout=5)
        with pytest.raises(requests.RequestException):
            active_request.result(timeout=5)
        with pytest.raises(OSError, match="History transport is closing"):
            with transport._upstream_connection():
                pytest.fail("A closed transport must reject new leases")

    assert [session for session, _ in model.wire] == ["root"]
    assert not model.release_first.is_set()  # Shutdown did not wait for the model response.


def test_connection_finishing_setup_after_shutdown_cannot_send(model_server, monkeypatch):
    connecting = threading.Event()
    finish_connecting = threading.Event()
    original_connect = HTTPConnection.connect
    transport = HistoryTransport(model_server.url, timeout=5)

    def delayed_connect(connection):
        connecting.set()
        if not finish_connecting.wait(5):
            raise TimeoutError("Test did not release connection setup")
        original_connect(connection)

    monkeypatch.setattr(HTTPConnection, "connect", delayed_connect)

    def send_request():
        with transport._upstream_connection() as connection:
            connection.request("POST", transport.target.path + "/responses", body=b"{}")

    with ThreadPoolExecutor(max_workers=1) as executor, ExitStack() as stack:
        stack.enter_context(transport)
        request = executor.submit(send_request)
        try:
            assert connecting.wait(5), "Request did not enter connection setup"
            stack.close()
        finally:
            finish_connecting.set()
        with pytest.raises(OSError, match="History transport is closing"):
            request.result(timeout=5)

    assert model_server.wire == []
    assert transport._connections == set()
