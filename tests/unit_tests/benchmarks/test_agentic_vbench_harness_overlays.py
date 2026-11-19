# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Agentic-VBench entries admit three harnesses, and each harness's overlay differs only in the harness block."""

import tomllib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
ENTRIES = ROOT / "environments/harbor"
# harness -> (benchmark overlay, training overlay)
OVERLAYS = {
    "opencode_agent": ("harbor_sandbox.yaml", "harbor_train.yaml"),
    "pi_agent": ("harbor_sandbox_pi.yaml", "harbor_train_pi.yaml"),
    "openclaw_agent": ("harbor_sandbox_openclaw.yaml", "harbor_train_openclaw.yaml"),
}
ADDED = [harness for harness in OVERLAYS if harness != "opencode_agent"]
ENTRY_NAMES = ("agentic_vbench", "agentic_vbench_train")
# What `harbor_train.yaml` sets on OpenCode and every other harness's training overlay sets on its own.
SHARED_TRAINING_KEYS = ("token_id_capture", "image_input", "timeout", "context_window", "max_output_tokens")


def _load(entry: str, name: str) -> dict:
    return yaml.safe_load((ENTRIES / entry / name).read_text())


def _block(overlay: dict, harness: str) -> dict:
    assert set(overlay[harness]) == {"responses_api_agents"}
    assert set(overlay[harness]["responses_api_agents"]) == {harness}
    return overlay[harness]["responses_api_agents"][harness]


@pytest.mark.parametrize("entry", ENTRY_NAMES)
def test_the_entries_admit_the_harnesses_that_send_the_model_tool_images(entry: str) -> None:
    dataset = tomllib.loads((ENTRIES / entry / "dataset.toml").read_text())
    assert dataset["gym"]["allowed_agents"] == list(OVERLAYS)


@pytest.mark.parametrize("harness", OVERLAYS)
@pytest.mark.parametrize("index", range(2), ids=ENTRY_NAMES)
def test_a_harness_overlay_names_only_its_own_harness(harness: str, index: int) -> None:
    # A block of a harness the run did not select would be a server without an entrypoint, which Gym refuses.
    overlay = _load(ENTRY_NAMES[index], OVERLAYS[harness][index])
    assert set(OVERLAYS) & set(overlay) == {harness}


@pytest.mark.parametrize("harness", ADDED)
@pytest.mark.parametrize("index", range(2), ids=ENTRY_NAMES)
def test_an_added_overlay_keeps_every_setting_outside_the_harness_block(harness: str, index: int) -> None:
    opencode = _load(ENTRY_NAMES[index], OVERLAYS["opencode_agent"][index])
    added = _load(ENTRY_NAMES[index], OVERLAYS[harness][index])
    del opencode["opencode_agent"], added[harness]
    assert added == opencode


@pytest.mark.parametrize("harness", ADDED)
def test_an_added_overlay_carries_the_settings_the_opencode_overlay_sets(harness: str) -> None:
    assert _block(_load("agentic_vbench", OVERLAYS[harness][0]), harness) == {"image_input": True}
    opencode = _block(_load("agentic_vbench_train", OVERLAYS["opencode_agent"][1]), "opencode_agent")
    added = _block(_load("agentic_vbench_train", OVERLAYS[harness][1]), harness)
    for key in SHARED_TRAINING_KEYS:
        assert added[key] == opencode[key], key
    # Pi alone has a switch for compaction; no compaction call in a Pi training episode.
    extra = {"auto_compaction": False} if harness == "pi_agent" else {}
    assert {key: value for key, value in added.items() if key not in SHARED_TRAINING_KEYS} == extra


@pytest.mark.parametrize("harness", ["pi_agent", "openclaw_agent", "codex_agent"])
def test_a_node_runtime_installer_does_not_take_the_archive_s_owner(harness: str) -> None:
    # Node's archive is owned by uid 1000. A root inside a user namespace (rootless Podman) cannot chown to it, and
    # tar then fails the whole installation.
    installer = ROOT / "responses_api_agents" / harness / f"install_{harness.removesuffix('_agent')}_runtime.sh"
    extraction = [line for line in installer.read_text().splitlines() if "--strip-components=1 -C node" in line]
    assert len(extraction) == 1
    assert extraction[0].strip().startswith("tar --no-same-owner -xzf ")
