# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bundled provider plugin, loaded the way OpenClaw loads it."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_gym.sandbox import AsyncSandbox, DockerProvider, SandboxSpec
from responses_api_agents.openclaw_agent.app import _sandbox_prepare_command
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


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is required for the non-root upload regression")
async def test_non_root_docker_upload_keeps_the_provider_receipt_writable() -> None:
    """Set GYM_TEST_OPENCLAW_DOCKER_IMAGE to a local image with Bash and Node 22.19.0 or newer."""
    image = os.environ.get("GYM_TEST_OPENCLAW_DOCKER_IMAGE")
    if not image:
        pytest.skip("set GYM_TEST_OPENCLAW_DOCKER_IMAGE to run the real Docker provider regression")
    sandbox = AsyncSandbox(DockerProvider(create={"network": "none"}, exec={"exec_shell": "sh"}))
    directory = "/tmp/nemo-gym-openclaw-sessions/session"
    try:
        await sandbox.start(
            SandboxSpec(
                image=image,
                user="65534:65534",
                workdir="/tmp",
                ttl_s=120,
                provider_options={"run_args": ["--pull", "never"]},
            )
        )
        prepared = await sandbox.exec(
            "mkdir -p /tmp/task && " + _sandbox_prepare_command("/tmp/task", directory, "/tmp/openclaw-runtime"),
            timeout_s=30,
        )
        assert prepared.return_code == 0 and not prepared.error_type, prepared.stderr or prepared.stdout
        # Exercise DockerProvider.upload_file: it creates missing parents and uploads files as root.
        for name in ("index.mjs", "openclaw.plugin.json"):
            await sandbox.upload(PLUGIN / name, f"{directory}/provider_plugin/{name}")
        script = """
import { readFileSync, statSync } from "node:fs";
const pluginUrl = new URL(process.argv[1]);
const plugin = (await import(pluginUrl.href)).default;
let provider;
plugin.register({ registerProvider: (value) => { provider = value; } });
provider.buildReplayPolicy({ provider: "nemo", modelApi: "openai-completions" });
console.log(JSON.stringify({
  uid: process.getuid(),
  directoryOwner: statSync(new URL(".", pluginUrl)).uid,
  receiptOwner: statSync(new URL("policy.ready", pluginUrl)).uid,
  receipt: readFileSync(new URL("policy.ready", pluginUrl), "utf8"),
}));
"""
        loaded = await sandbox.exec(
            shlex.join(["node", "--input-type=module", "-e", script, f"file://{directory}/provider_plugin/index.mjs"]),
            timeout_s=30,
        )
        assert loaded.return_code == 0 and not loaded.error_type, loaded.stderr or loaded.stdout
        assert json.loads(loaded.stdout) == {
            "uid": 65534,
            "directoryOwner": 65534,
            "receiptOwner": 65534,
            "receipt": PROVIDER_REPLAY_READY,
        }
    finally:
        await sandbox.stop()
