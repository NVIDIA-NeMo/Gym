# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from responses_api_agents.pi_agent.tests.test_sandbox import _pinned_pi_package
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


def test_outcome_extension_uses_pinned_pi_context_classifier(tmp_path: Path) -> None:
    node, package = _pinned_pi_package()
    loader = package / "dist/core/extensions/loader.js"
    extension = Path(__file__).parents[1] / "outcome.mjs"
    script = """
import assert from 'node:assert/strict';
// Pi's JSON mode points process.stdout.write, which console.log uses, at stderr before extensions load.
process.stdout.write = (chunk) => process.stderr.write(chunk);
const { loadExtensions } = await import(process.argv[1]);
const loaded = await loadExtensions([process.argv[2]], process.argv[3]);
assert.deepEqual(loaded.errors, []);
const [handler] = loaded.extensions[0].handlers.get('message_end');
for (const errorMessage of [
  '400 maximum context length is 262144 tokens',
  '429 Too many requests: rate limit exceeded',
  '503 Service unavailable',
  'Connection error.',
  '401 Incorrect API key',
  '400 status code (no body)',
  '413 status code (no body)',
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
        {"type": "ng_pi_outcome", "context_overflow": overflow}
        for overflow in (True, False, False, False, False, False, False)
    ]


@pytest.mark.parametrize(
    "status,message,context_overflow",
    [
        (400, "This model's maximum context length is 262144 tokens.", True),
        (400, "Rejected request", False),
        (401, "Invalid key", False),
        (429, "Rate limit", False),
        (503, "Service unavailable", False),
    ],
)
@pytest.mark.parametrize("delayed", [False, True])
def test_pinned_pi_classifies_model_server_http_error(
    monkeypatch: pytest.MonkeyPatch, status: int, message: str, context_overflow: bool, delayed: bool
) -> None:
    node, package = _pinned_pi_package()
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
assert.deepEqual(result.content, []);
assert.equal(result.usage.input + result.usage.output, 0);
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
