# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute Pi extensions")
def test_shell_deadlines_preserve_shorter_limits_and_explain_search_scope() -> None:
    extension = Path(__file__).parents[1] / "runtime-guards.mjs"
    script = """
import assert from 'node:assert/strict';
const { default: install } = await import(process.argv[1]);
const handlers = {};
process.env.NEMO_GYM_PI_BASH_TIMEOUT = '900';
install({ on: (name, handler) => { handlers[name] = handler; } });
for (const [requested, expected] of [[undefined, 900], [30, 30], [0, 900],
                                    [-1, 900], [3600, 900], [0.5, 0.5]]) {
  const event = { toolName: 'bash', input: { command: 'echo hello', timeout: requested } };
  handlers.tool_call(event);
  assert.deepEqual(event.input, { command: 'echo hello', timeout: expected });
}
const read = { toolName: 'read', input: { path: 'README.md' } };
handlers.tool_call(read);
assert.deepEqual(read.input, { path: 'README.md' });
const prompt = handlers.before_agent_start({ systemPrompt: 'Original instructions' }).systemPrompt;
assert.ok(prompt.startsWith('Original instructions'));
assert.ok(prompt.includes('900 seconds'));
assert.ok(prompt.includes('find /'));
process.env.NEMO_GYM_PI_BASH_TIMEOUT = 'invalid';
assert.throws(() => install({}), /positive integer/);
"""
    result = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script, extension.as_uri()],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
