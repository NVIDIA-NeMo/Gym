# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the session-key and request-log hooks with Node; the HTTP wire is checked by real-Pi smoke runs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute Pi extensions")
EXTENSIONS = Path(__file__).parents[1]


def run_node(script: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script, *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        env={**os.environ, **env},
    )


def test_prompt_cache_key_is_added_to_every_gym_provider_request():
    script = """
import assert from 'node:assert/strict';
const { default: install } = await import(process.argv[1]);
let handle;
install({ on(event, callback) { assert.equal(event, 'before_provider_request'); handle = callback; } });
const model = { provider: 'nemo', api: 'openai-completions' };
for (const payload of [{ messages: [], reasoning_effort: 'max' }, { messages: [1, 2], prompt_cache_key: 'pi-own' }]) {
  const original = structuredClone(payload);
  assert.deepEqual(handle({ payload }, { model }), { ...payload, prompt_cache_key: 'pi-session-key' });
  assert.deepEqual(payload, original);
}
for (const other of [undefined, { provider: 'other', api: 'openai-completions' }, { provider: 'nemo', api: 'x' }]) {
  assert.equal(handle({ payload: {} }, { model: other }), undefined);
}
"""
    extension = (EXTENSIONS / "prompt-cache-key.mjs").as_uri()
    completed = run_node(script, extension, env={"NEMO_GYM_PI_PROMPT_CACHE_KEY": "pi-session-key"})
    assert completed.returncode == 0, completed.stderr
    missing = run_node(script, extension, env={"NEMO_GYM_PI_PROMPT_CACHE_KEY": ""})
    assert missing.returncode != 0
    assert "NEMO_GYM_PI_PROMPT_CACHE_KEY must be set" in missing.stderr


def test_request_log_records_fields_without_the_conversation(tmp_path):
    script = """
import assert from 'node:assert/strict';
const { default: install } = await import(process.argv[1]);
let handle;
install({ on(event, callback) { handle = callback; } });
const model = { provider: 'nemo', api: 'openai-completions' };
const tools = [{ type: 'function', function: { name: 'bash', parameters: {} } }];
const first = { model: 'deepseek-flash', messages: [{ role: 'system', content: 'secret prompt' }, { role: 'user', content: 'task' }],
  tools, reasoning_effort: 'max', max_tokens: 393216, prompt_cache_key: 'pi-key', stream: true };
assert.equal(handle({ payload: first }, { model }), undefined);
handle({ payload: { ...first, messages: [...first.messages, { role: 'assistant', content: 'x' }] } }, { model });
"""
    log = tmp_path / "pi-requests.jsonl"
    completed = run_node(script, (EXTENSIONS / "request-log.mjs").as_uri(), env={"NEMO_GYM_PI_REQUEST_LOG": str(log)})
    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["request"] for record in records] == [1, 2]
    first = records[0]
    assert {key: first[key] for key in ("model", "reasoning_effort", "max_tokens", "prompt_cache_key", "stream")} == {
        "model": "deepseek-flash",
        "reasoning_effort": "max",
        "max_tokens": 393216,
        "prompt_cache_key": "pi-key",
        "stream": True,
    }
    assert (first["provider"], first["api"]) == ("nemo", "openai-completions")
    assert (first["messages"], first["last_role"], first["tools"]) == (2, "user", ["bash"])
    assert (records[1]["messages"], records[1]["last_role"]) == (3, "assistant")
    assert len(first["tools_sha256"]) == 64
    assert "secret prompt" not in log.read_text()
    # The first request is kept in full beside the log.
    full = json.loads((tmp_path / "pi-requests.first.json").read_text())
    assert full["messages"][0]["content"] == "secret prompt"
