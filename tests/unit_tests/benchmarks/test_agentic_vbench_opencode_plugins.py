# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode compaction plugins: their node tests, and what the training overlay loads matches what is staged."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from benchmarks.agentic_vbench.staging import (
    OPENCODE_PLUGINS,
    OPENCODE_PLUGINS_DIR,
    OPENCODE_PLUGINS_SANDBOX_DIR,
)
from responses_api_models.vllm_model.app import OPENCODE_AGENT_HEADER


ROOT = Path(__file__).resolve().parents[3]
OVERLAY = ROOT / "environments/harbor/agentic_vbench_train/compaction_plugins.yaml"
NODE_TESTS = sorted((OPENCODE_PLUGINS_DIR / "tests").glob("test_*.mjs"))


def test_every_plugin_has_a_node_test_and_a_license_header() -> None:
    assert {path.name.removeprefix("test_") for path in NODE_TESTS} == set(OPENCODE_PLUGINS)
    for name in OPENCODE_PLUGINS:
        assert (OPENCODE_PLUGINS_DIR / name).read_text().startswith("// SPDX-FileCopyrightText:")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", NODE_TESTS, ids=lambda path: path.stem)
def test_plugin_behaves_as_its_node_test_says(script: Path) -> None:
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.startswith("ok ")


def test_the_training_overlay_loads_the_staged_plugins_and_turns_on_summary_recovery() -> None:
    overlay = yaml.safe_load(OVERLAY.read_text())
    plugins = overlay["opencode_agent"]["responses_api_agents"]["opencode_agent"]["plugins"]
    assert plugins == [f"{OPENCODE_PLUGINS_SANDBOX_DIR}/{name}" for name in OPENCODE_PLUGINS]
    vllm = overlay["policy_model"]["responses_api_models"]["vllm_model"]
    assert set(vllm) == {"compaction_summary_max_tokens", "compaction_summary_max_attempts"}
    # The header plugin and the model server agree on the header name.
    header_plugin = (OPENCODE_PLUGINS_DIR / "compaction_summary_header.mjs").read_text()
    assert f'export const HEADER = "{OPENCODE_AGENT_HEADER}";' in header_plugin
