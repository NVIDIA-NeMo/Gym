# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute Pi extensions")
def test_plain_prompt_leaves_out_pis_documentation_and_the_machines_date() -> None:
    extension = Path(__file__).parents[1] / "plain-prompt.mjs"
    # Pi 0.80.2's default prompt, shortened: the documentation block sits between the guidelines and the appended
    # system prompt, and the date and the working directory come last.
    script = """
import assert from 'node:assert/strict';
const { default: install } = await import(process.argv[1]);
const handlers = {};
install({ on: (name, handler) => { handlers[name] = handler; } });
const docs = [
  'Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):',
  '- Main documentation: /tmp/runtime-0.80.2/pi/README.md',
  '- Additional docs: /tmp/runtime-0.80.2/pi/docs',
  '- Examples: /tmp/runtime-0.80.2/pi/examples (extensions, custom tools, SDK)',
  '- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)',
].join('\\n');
const head = 'You are an expert coding assistant.\\n\\nGuidelines:\\n- Be concise in your responses';
const tail = '\\n\\nYou act for the task.\\n- a line of the task\\nCurrent date: 2026-10-31' +
  '\\nCurrent working directory: /workspace\\n\\nSandbox execution: use focused commands.';
const edited = handlers.before_agent_start({ systemPrompt: head + '\\n\\n' + docs + tail }).systemPrompt;
assert.equal(edited, head + '\\n\\nYou act for the task.\\n- a line of the task' +
  '\\nCurrent working directory: /workspace\\n\\nSandbox execution: use focused commands.');
// a custom prompt has no documentation block: only the date goes
const custom = 'Custom.\\nCurrent date: 2026-10-31\\nCurrent working directory: /workspace';
assert.equal(handlers.before_agent_start({ systemPrompt: custom }).systemPrompt,
             'Custom.\\nCurrent working directory: /workspace');
"""
    result = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script, extension.as_uri()],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
