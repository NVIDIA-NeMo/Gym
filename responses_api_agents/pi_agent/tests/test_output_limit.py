# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Pi request hook with Node; the HTTP wire is checked by real-Pi smoke runs."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute Pi extensions")
@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        ({}, 128),
        ({"max_tokens": 64}, 64),
        ({"max_completion_tokens": 32}, 32),
        ({"max_tokens": 256, "max_completion_tokens": 512}, 128),
    ],
)
def test_caps_each_request_without_changing_history_or_sampling(existing, expected):
    extension = Path(__file__).parents[1] / "output-limit.mjs"
    script = """
import assert from 'node:assert/strict';
const { default: install } = await import(process.argv[1]);
let handle;
install({ on(event, callback) {
  assert.equal(event, 'before_provider_request');
  handle = callback;
} });
const model = { provider: 'nemo', api: 'openai-completions', maxTokens: 128 };
for (const turns of [1, 3, 8]) {
  const messages = Array.from({ length: turns }, () => ({ role: 'user', content: 'hello' }));
  const input = { messages, temperature: 0.7, ...JSON.parse(process.argv[2]) };
  const original = structuredClone(input);
  const result = handle({ payload: input }, { model });
  assert.deepEqual(result, { messages, temperature: 0.7, max_tokens: Number(process.argv[3]) });
  assert.deepEqual(input, original);
}
for (const other of [undefined, { provider: 'other', api: 'openai-completions' },
                     { provider: 'nemo', api: 'openai-responses' }]) {
  assert.equal(handle({ payload: {} }, { model: other }), undefined);
}
"""
    completed = subprocess.run(
        [
            shutil.which("node"),
            "--input-type=module",
            "-e",
            script,
            extension.as_uri(),
            json.dumps(existing),
            str(expected),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
