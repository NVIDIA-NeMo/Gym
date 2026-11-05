# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bundled provider plugin, loaded the way OpenClaw loads it."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from responses_api_agents.openclaw_agent.sandbox import PROVIDER_REPLAY_READY, PROVIDER_REPLAY_RECEIPT


PLUGIN = Path(__file__).parents[1] / "provider_plugin"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to load the provider plugin")
def test_provider_plugin_writes_the_receipt_the_adapter_checks(tmp_path):
    shutil.copytree(PLUGIN, tmp_path / "provider_plugin")
    script = """
const plugin = (await import(process.argv[1])).default;
const providers = [];
plugin.register({ registerProvider: (provider) => providers.push(provider) });
const [provider] = providers;
const policy = provider.buildReplayPolicy({ provider: provider.id, modelApi: "openai-completions" });
console.log(JSON.stringify({ id: plugin.id, providers: providers.map((p) => p.id), policy }));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, (tmp_path / "provider_plugin/index.mjs").as_uri()],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    manifest = json.loads((PLUGIN / "openclaw.plugin.json").read_text())
    assert loaded["id"] == manifest["id"] == "nemo-gym"
    assert loaded["providers"] == manifest["providers"] == ["nemo"]
    # OpenClaw's openai-completions policy without tool-call ID sanitization or reasoning removal.
    assert loaded["policy"] == {
        "sanitizeToolCallIds": False,
        "dropReasoningFromHistory": False,
        "applyAssistantFirstOrderingFix": True,
        "validateGeminiTurns": True,
        "validateAnthropicTurns": True,
    }
    assert (tmp_path / PROVIDER_REPLAY_RECEIPT).read_text() == PROVIDER_REPLAY_READY
