# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("activation", [None, "seed:activation-1", "seed:activation-2"])
def test_opencode_headers_keep_session_identity_without_guessing_reply_identity(activation) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the OpenCode plugin")
    plugin = Path(__file__).parents[1] / "assistant_message_header.js"
    script = (
        plugin.read_text()
        + (
            f"\nprocess.env.NEMO_GYM_OPENCODE_ACTIVATION_ID = {activation!r};\n"
            if activation
            else "\ndelete process.env.NEMO_GYM_OPENCODE_ACTIVATION_ID;\n"
        )
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
const expected = process.env.NEMO_GYM_OPENCODE_ACTIVATION_ID ? {'X-Session-Id': process.env.NEMO_GYM_OPENCODE_ACTIVATION_ID + ':session-a'} : {};
assert.deepEqual(headers(), expected);
update(reply('reply-a'));
assert.deepEqual(headers(), {...expected, 'X-OpenCode-Assistant-Message-Id': 'reply-a'});
update(reply('reply-b'));
assert.deepEqual(headers(), expected);
update({...reply('reply-a'), time: {completed: 1}});
assert.equal(headers()['X-OpenCode-Assistant-Message-Id'], 'reply-b');
hooks.event({event: {type: 'session.idle', properties: {sessionID: 'session-a'}}});
assert.deepEqual(headers(), expected);
"""
    )
    subprocess.run([node, "--input-type=module"], input=script, text=True, check=True, timeout=10)
