# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest
from responses_api_agents.opencode_sandboxed_agent.app import _prepare_prompt_date_plugin


def test_plugin_staging_preserves_config_and_shell_quotes(tmp_path: Path) -> None:
    path = tmp_path / "plugin ' quoted.mjs"
    config = {"plugin": ["file:///existing.mjs"], "model": "nemo_gym/dummy_model"}
    updated, command = _prepare_prompt_date_plugin(config, plugin_path=str(path))
    subprocess.run(["sh", "-c", command], check=True)
    assert "experimental.chat.system.transform" in path.read_text()
    assert updated["plugin"] == ["file:///existing.mjs", f"file://{path}"]
    assert config == {"plugin": ["file:///existing.mjs"], "model": "nemo_gym/dummy_model"}
    assert updated["model"] == config["model"]


def test_plugin_rejects_invalid_existing_config() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        _prepare_prompt_date_plugin({"plugin": "not-a-list"}, plugin_path="/tmp/unused.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required to execute the OpenCode hook")
def test_midnight_hook_preserves_strict_lineage() -> None:
    plugin = Path(__file__).parents[1] / "stable_prompt_date.mjs"
    script = r"""
import assert from 'node:assert/strict';
const { default: factory } = await import(process.argv[1]);
const hook = (await factory())['experimental.chat.system.transform'];
const prompt = (date, location = '/testbed') => [
  `Keep all instructions.\n<env>\n  Working directory: ${location}\n  Today's date: ${date}\n</env>\nOutside Today's date: unchanged`,
  'Do not modify this second system part.',
];
const before = { system: prompt('Sun Oct 04 2026') };
const after = { system: prompt('Mon Oct 05 2026') };
const rawAfter = structuredClone(after);
const originalArray = after.system;
await hook({}, before);
await hook({}, after);
assert.deepEqual(before.system, after.system);
assert.equal(after.system, originalArray);
const edited = { system: prompt('Mon Oct 05 2026', '/different') };
await hook({}, edited);
assert.notDeepEqual(edited.system, before.system);
assert.match(edited.system[0], /Working directory: \/different/);
const newRollout = { system: prompt('Mon Oct 05 2026') };
await (await factory())['experimental.chat.system.transform']({}, newRollout);
assert.deepEqual(newRollout.system, rawAfter.system);
const auxiliary = { system: ['Summarize this conversation.'] };
await hook({}, auxiliary);
assert.deepEqual(auxiliary.system, ['Summarize this conversation.']);
await assert.rejects(hook({}, { system: ['<env>missing date</env>'] }), /Expected one/);
await assert.rejects(hook({}, { system: [...prompt('Mon Oct 05 2026'), ...prompt('Mon Oct 05 2026')] }), /found 2/);
console.log(JSON.stringify({ before: before.system, rawAfter: rawAfter.system, after: after.system, edited: edited.system }));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, plugin.as_uri()],
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = json.loads(result.stdout)
    contexts = {
        key: [{"role": "system", "content": text} for text in value]
        + [{"role": "user", "content": "Solve"}, {"role": "assistant", "content": "Inspecting"}]
        for key, value in outputs.items()
    }
    # The parent assistant matches across midnight.
    assert assistant_fingerprint(contexts["before"]) == assistant_fingerprint(contexts["rawAfter"])
    # The continuation digest counts instruction items by position only (743e65f06), so lineage
    # already survives the date change. The hook keeps the rendered system prompt, and with it the
    # prompt tokens the engine sees, identical across midnight.
    assert outputs["before"] != outputs["rawAfter"]
    assert outputs["before"] == outputs["after"]
    assert conversation_digest(contexts["before"]) == conversation_digest(contexts["after"])
    # Other prompt changes pass through unchanged.
    assert outputs["edited"] != outputs["before"]
