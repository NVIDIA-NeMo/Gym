# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
from pathlib import Path

import pytest


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
