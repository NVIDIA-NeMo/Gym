# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_headers_keep_session_identity_without_guessing_reply_identity() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the native OpenCode plugin")
    plugin = Path(__file__).parents[1] / "assistant_message_header.js"
    script = (
        plugin.read_text()
        + """
import assert from 'node:assert/strict';
const hooks = AssistantMessageHeader();
const input = {sessionID: 'session-a', message: {id: 'user-a'}, agent: 'build',
               model: {id: 'model-a', providerID: 'provider-a'}};
const headers = () => {
  const output = {headers: {}};
  hooks['chat.headers'](input, output);
  return output.headers;
};
const reply = id => ({id, sessionID: input.sessionID, parentID: input.message.id,
  role: 'assistant', agent: input.agent, modelID: input.model.id,
  providerID: input.model.providerID, time: {}});
const update = info => hooks.event({event: {type: 'message.updated', properties: {info}}});
assert.deepEqual(headers(), {'X-Session-Id': 'session-a'});
update(reply('reply-a'));
assert.deepEqual(headers(), {'X-Session-Id': 'session-a', 'X-OpenCode-Assistant-Message-Id': 'reply-a'});
update(reply('reply-b'));
assert.deepEqual(headers(), {'X-Session-Id': 'session-a'});
update({...reply('reply-a'), time: {completed: 1}});
assert.equal(headers()['X-OpenCode-Assistant-Message-Id'], 'reply-b');
hooks.event({event: {type: 'session.idle', properties: {sessionID: 'session-a'}}});
assert.deepEqual(headers(), {'X-Session-Id': 'session-a'});
"""
    )
    subprocess.run([node, "--input-type=module"], input=script, text=True, check=True, timeout=10)
