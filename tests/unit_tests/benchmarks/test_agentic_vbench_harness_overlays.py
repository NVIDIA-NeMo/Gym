# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Agentic-VBench entries admit two harnesses, and each harness's overlay differs only in the harness block."""

import tomllib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
ENTRIES = ROOT / "environments/harbor"
# (entry, OpenCode overlay, Pi overlay)
OVERLAYS = [
    ("agentic_vbench", "harbor_sandbox.yaml", "harbor_sandbox_pi.yaml"),
    ("agentic_vbench_train", "harbor_train.yaml", "harbor_train_pi.yaml"),
]
HARNESSES = {"opencode_agent", "pi_agent"}


def _load(entry: str, name: str) -> dict:
    return yaml.safe_load((ENTRIES / entry / name).read_text())


@pytest.mark.parametrize("entry", ["agentic_vbench", "agentic_vbench_train"])
def test_the_entries_admit_the_harnesses_that_send_the_model_tool_images(entry: str) -> None:
    dataset = tomllib.loads((ENTRIES / entry / "dataset.toml").read_text())
    assert set(dataset["gym"]["allowed_agents"]) == HARNESSES


@pytest.mark.parametrize(("entry", "opencode", "pi"), OVERLAYS)
def test_a_harness_overlay_names_only_its_own_harness(entry: str, opencode: str, pi: str) -> None:
    # A block of a harness the run did not select would be a server without an entrypoint, which Gym refuses.
    assert HARNESSES & set(_load(entry, opencode)) == {"opencode_agent"}
    pi_overlay = _load(entry, pi)
    assert HARNESSES & set(pi_overlay) == {"pi_agent"}
    assert set(pi_overlay["pi_agent"]) == {"responses_api_agents"}
    assert set(pi_overlay["pi_agent"]["responses_api_agents"]) == {"pi_agent"}


@pytest.mark.parametrize(("entry", "opencode", "pi"), OVERLAYS)
def test_the_pi_overlay_keeps_every_setting_outside_the_harness_block(entry: str, opencode: str, pi: str) -> None:
    opencode_overlay, pi_overlay = _load(entry, opencode), _load(entry, pi)
    del opencode_overlay["opencode_agent"], pi_overlay["pi_agent"]
    assert pi_overlay == opencode_overlay


def test_the_pi_overlays_carry_the_settings_the_opencode_overlays_set() -> None:
    assert _load("agentic_vbench", "harbor_sandbox_pi.yaml")["pi_agent"]["responses_api_agents"]["pi_agent"] == {
        "image_input": True
    }
    opencode = _load("agentic_vbench_train", "harbor_train.yaml")["opencode_agent"]["responses_api_agents"][
        "opencode_agent"
    ]
    pi = _load("agentic_vbench_train", "harbor_train_pi.yaml")["pi_agent"]["responses_api_agents"]["pi_agent"]
    for key in ("token_id_capture", "image_input", "timeout", "context_window", "max_output_tokens"):
        assert pi[key] == opencode[key], key
    # No compaction call in a Pi training episode: nothing joins it to the scored chain.
    assert pi["auto_compaction"] is False
    assert set(pi) == {
        "token_id_capture",
        "image_input",
        "timeout",
        "context_window",
        "max_output_tokens",
        "auto_compaction",
    }
