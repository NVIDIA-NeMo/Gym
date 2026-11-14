# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
OVERLAY = ROOT / "benchmarks/opencode_multimodal_benchmarks/agentx/web_filter.yaml"


def test_web_filter_behavior(tmp_path):
    runtime = os.environ.get("GYM_TEST_JS_RUNTIME") or shutil.which("node") or shutil.which("bun")
    if runtime is None:
        pytest.skip("Node.js or Bun is required; GYM_TEST_JS_RUNTIME can select a runtime")
    config = yaml.safe_load(OVERLAY.read_text())["visual_agent"]["responses_api_agents"]["opencode_agent"]
    for target, content in config["sandbox_config"]["files"].items():
        (tmp_path / Path(target).name).write_text(content)
    result = subprocess.run(
        [runtime, str(Path(__file__).with_name("agentx_web_filter_checks.mjs")), str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "web filter checks passed" in result.stdout
    assert '"phase":"loaded"' in result.stderr
    assert '"phase":"blocked_input"' in result.stderr
    assert '"phase":"blocked_output"' in result.stderr


@pytest.mark.parametrize("preset", ["config", "opencode", "training"])
def test_agentx_stages_plugin_and_policy_through_stock_sandbox_config(monkeypatch, preset):
    from omegaconf import OmegaConf

    from nemo_gym.config import GlobalConfigDictParser
    from nemo_gym.sandbox.providers.base import SandboxSpec

    monkeypatch.chdir(ROOT)
    _, configs = GlobalConfigDictParser().load_extra_config_paths(
        [f"benchmarks/opencode_multimodal_benchmarks/agentx/{preset}.yaml"]
    )
    merged = OmegaConf.merge(*configs)
    GlobalConfigDictParser()._recursively_swap_keys(merged)
    agent = OmegaConf.to_container(merged, resolve=False)["visual_agent"]["responses_api_agents"]["opencode_agent"]
    spec = SandboxSpec(**(agent["sandbox_config"] | {"image": "test-image"}))
    (plugin,) = agent["plugins"]
    assert plugin in spec.files
    assert "tool.execute.before" in spec.files[plugin] and "tool.execute.after" in spec.files[plugin]
    policy = json.loads(spec.files[str(Path(plugin).with_suffix(".json"))])
    manifest = json.loads((ROOT / "benchmarks/opencode_multimodal_benchmarks/manifests/AgentX.json").read_text())
    assert manifest["source_repo"].lower() in policy["url_substrings"]
    assert agent["network_access"] == "inherit"
    assert spec.env["OPENCODE_ENABLE_EXA"] == "true"
    assert agent["opencode_config"]["permission"]["webfetch"] == "allow"
    assert agent["opencode_config"]["permission"]["websearch"] == "allow"
