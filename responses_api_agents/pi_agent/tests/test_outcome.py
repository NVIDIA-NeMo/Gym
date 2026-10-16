# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


def test_outcome_extension_uses_pinned_pi_context_classifier(tmp_path: Path) -> None:
    node, pi = shutil.which("node"), shutil.which("pi")
    if not node or not pi:
        pytest.skip("Installed Pi and Node are required for the real extension-loader contract")
    package = Path(pi).resolve().parents[1]
    manifest = package / "package.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("version") != "0.80.2":
        pytest.skip("This integration targets pinned Pi 0.80.2")
    loader = package / "dist/core/extensions/loader.js"
    extension = Path(__file__).parents[1] / "outcome.mjs"
    script = """
import assert from 'node:assert/strict';
const { loadExtensions } = await import(process.argv[1]);
const loaded = await loadExtensions([process.argv[2]], process.argv[3]);
assert.deepEqual(loaded.errors, []);
const [handler] = loaded.extensions[0].handlers.get('message_end');
// Pi's JSON mode sends console.log and process.stdout.write to stderr before extensions run.
console.log = (...args) => process.stderr.write(args.join(' ') + '\\n');
process.stdout.write = (chunk) => process.stderr.write(chunk);
for (const errorMessage of [
  '400 maximum context length is 262144 tokens',
  '429 Too many requests: rate limit exceeded',
  '503 Service unavailable',
  'Connection error.',
  '401 Incorrect API key',
]) {
  await handler({message: {role: 'assistant', stopReason: 'error', errorMessage}}, {model: {contextWindow: 262144}});
}
await handler({message: {role: 'toolResult'}}, {});
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script, loader.as_uri(), str(extension), str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # The events reach the real stdout, which the sandbox runner records; tool results add none.
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {"type": "ng_pi_outcome", "context_overflow": overflow} for overflow in (True, False, False, False, False)
    ]


def _pinned_pi_package() -> Path:
    node, pi = shutil.which("node"), shutil.which("pi")
    if not node or not pi:
        pytest.skip("Installed Pi and Node are required for the real CLI contract")
    package = Path(pi).resolve().parents[1]
    manifest = package / "package.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("version") != "0.80.2":
        pytest.skip("This integration targets pinned Pi 0.80.2")
    return package


class _OverflowAfterLongTurn(BaseHTTPRequestHandler):
    """A long reasoning turn that calls bash, then a context-length error, the way Gym's model server replays them."""

    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).calls += 1
        if type(self).calls > 1:
            payload = json.dumps({"error": {"message": "This model's maximum context length is 262144 tokens."}})
            self.send_response(400)
            self.send_header("content-type", "application/json")
        else:
            base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
            deltas = [
                ({"role": "assistant"}, None),
                ({"reasoning_content": "r" * 100_000}, None),
                (
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": "true"})},
                            }
                        ]
                    },
                    None,
                ),
                ({}, "tool_calls"),
            ]
            payload = (
                "".join(
                    "data: "
                    + json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
                    + "\n\n"
                    for delta, finish in deltas
                )
                + "data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
        data = payload.encode()
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_outcome_survives_long_turns_through_the_sandbox_runner(tmp_path: Path) -> None:
    """After a long turn Pi is still writing when the overflow arrives; the event must not split or drop."""
    package = _pinned_pi_package()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    _OverflowAfterLongTurn.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", port), _OverflowAfterLongTurn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent_dir = tmp_path / "home/.pi/agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "nemo": {
                            "baseUrl": f"http://127.0.0.1:{port}/v1",
                            "api": "openai-completions",
                            "apiKey": "EMPTY",
                            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                            "models": [
                                {
                                    "id": "fake",
                                    "reasoning": True,
                                    "input": ["text"],
                                    "contextWindow": 262144,
                                    "maxTokens": 4096,
                                }
                            ],
                        }
                    }
                }
            )
        )
        (agent_dir / "settings.json").write_text(json.dumps({"compaction": {"enabled": False}}))
        source = Path(__file__).parents[1]
        shutil.copy(source / "sandbox_runner.py", tmp_path / "sandbox_runner.py")
        (tmp_path / "work").mkdir()
        command = [
            shutil.which("node"),
            str(package / "dist/cli.js"),
            "--print",
            "--mode",
            "json",
            "--no-session",
            "--provider",
            "nemo",
            "--model",
            "fake",
            "--no-extensions",
            "--extension",
            str(source / "outcome.mjs"),
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
        ]
        (tmp_path / "input.json").write_text(
            json.dumps(
                {
                    "directory": str(tmp_path),
                    "command": command,
                    "prompt": "Do the task.",
                    "cwd": str(tmp_path / "work"),
                    "env": {"PI_CODING_AGENT_DIR": str(agent_dir), "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"},
                }
            )
        )
        subprocess.run(
            ["python3", "-I", str(tmp_path / "sandbox_runner.py"), str(tmp_path / "input.json")],
            timeout=120,
            check=False,
        )
    finally:
        server.shutdown()
    events = [json.loads(line)[1] for line in (tmp_path / "events.jsonl").read_text().splitlines() if line.strip()]
    outcomes = [event["context_overflow"] for event in events if event.get("type") == "ng_pi_outcome"]
    assert outcomes == [False, True], (tmp_path / "stderr.log").read_text(errors="replace")[-2000:]


@pytest.mark.parametrize(
    "status,message,context_overflow",
    [(400, "This model's maximum context length is 262144 tokens.", True), (503, "Service unavailable", False)],
)
@pytest.mark.parametrize("delayed", [False, True])
def test_pinned_pi_classifies_model_server_http_error(
    monkeypatch: pytest.MonkeyPatch, status: int, message: str, context_overflow: bool, delayed: bool
) -> None:
    node, pi = shutil.which("node"), shutil.which("pi")
    if not node or not pi:
        pytest.skip("Installed Pi and Node are required for the real SDK contract")
    package = Path(pi).resolve().parents[1]
    manifest = package / "package.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("version") != "0.80.2":
        pytest.skip("This integration targets pinned Pi 0.80.2")
    sdk = package / "node_modules/@earendil-works/pi-ai/dist"
    server = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=8081,
            name="policy_model",
            entrypoint="app.py",
            openai_base_url="http://upstream.invalid/v1",
            openai_api_key="test",
            openai_model="test",
            propagate_upstream_http_status_codes=frozenset({status}),
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    error = ClientResponseError(MagicMock(), (), status=status, message=message)
    error.response_content = json.dumps({"error": {"message": message, "type": "upstream_error"}}).encode()

    async def reject(**kwargs):
        if delayed:
            await asyncio.sleep(0.05)
        raise error

    if delayed:
        monkeypatch.setattr("nemo_gym.base_responses_api_model._CHAT_KEEPALIVE_SECONDS", 0.001)
    server._client = MagicMock(create_chat_completion=AsyncMock(side_effect=reject))
    with TestClient(server.setup_webserver()) as client:
        response = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
    if delayed:
        assert response.status_code == 200
        assert response.text.startswith(": keep-alive\n\n")
    else:
        assert response.status_code == status
    script = """
import assert from 'node:assert/strict';
import fs from 'node:fs';
const wire = JSON.parse(fs.readFileSync(0, 'utf8'));
globalThis.fetch = async () => new Response(wire.body, {
  status: wire.status, headers: {'content-type': wire.content_type},
});
const {streamSimple} = await import(process.argv[1]);
const {isContextOverflow} = await import(process.argv[2]);
const model = {id: 'test', name: 'test', api: 'openai-completions', provider: 'openai',
  baseUrl: 'http://gym.invalid/v1', reasoning: false, input: ['text'],
  cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}, contextWindow: 262144, maxTokens: 32768};
const result = await streamSimple(model, {messages: [{role: 'user', content: 'hi', timestamp: Date.now()}]},
  {apiKey: 'test', maxRetries: 0}).result();
assert.equal(result.stopReason, 'error');
assert.ok(result.errorMessage.includes(wire.message), result.errorMessage);
assert.equal(isContextOverflow(result, model.contextWindow), wire.context_overflow);
"""
    result = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            script,
            (sdk / "api/openai-completions.js").as_uri(),
            (sdk / "index.js").as_uri(),
        ],
        input=json.dumps(
            {
                "status": response.status_code,
                "content_type": response.headers["content-type"],
                "body": response.text,
                "message": message,
                "context_overflow": context_overflow,
            }
        ),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
