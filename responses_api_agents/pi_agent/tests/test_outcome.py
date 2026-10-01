# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
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
const records = [];
console.log = line => records.push(JSON.parse(line));
for (const [errorMessage, expected] of [
  ['400 maximum context length is 262144 tokens', true],
  ['429 Too many requests: rate limit exceeded', false],
  ['503 Service unavailable', false],
  ['Connection error.', false],
  ['401 Incorrect API key', false],
]) {
  await handler({message: {role: 'assistant', stopReason: 'error', errorMessage}}, {model: {contextWindow: 262144}});
  assert.deepEqual(records.at(-1), {type: 'ng_pi_outcome', context_overflow: expected});
}
const count = records.length;
await handler({message: {role: 'toolResult'}}, {});
assert.equal(records.length, count);
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script, loader.as_uri(), str(extension), str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "status,message,context_overflow",
    [(400, "This model's maximum context length is 262144 tokens.", True), (503, "Service unavailable", False)],
)
def test_pinned_pi_classifies_model_server_http_error(status, message, context_overflow):
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
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    error = ClientResponseError(MagicMock(), (), status=status, message=message)
    error.response_content = json.dumps({"error": {"message": message, "type": "upstream_error"}}).encode()
    server._client = MagicMock(create_chat_completion=AsyncMock(side_effect=error))
    with TestClient(server.setup_webserver()) as client:
        response = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
    script = """
import assert from 'node:assert/strict';
import fs from 'node:fs';
const wire = JSON.parse(fs.readFileSync(0, 'utf8'));
globalThis.fetch = async () => new Response(wire.body, {
  status: wire.status, headers: {'content-type': 'application/json'},
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
