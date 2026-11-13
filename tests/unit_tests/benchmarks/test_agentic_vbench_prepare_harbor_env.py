# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for writing the pinned Agentic-VBench tasks into the `harbor` environment server's dataset entry."""

import hashlib
import tomllib
from pathlib import Path

import pytest

import benchmarks.agentic_vbench.prepare_harbor_env as prepare_module
from benchmarks.agentic_vbench.prepare_harbor_env import ENTRY_DIR, RECORDS_DIR, adapt_task, write_entry
from benchmarks.agentic_vbench.staging import HUB_JUDGE, JUDGE_PROTOCOLS, MATERIALS_FETCH_GUARD
from nemo_gym.data.tasks.harbor.dataset_config import read_dataset_config
from nemo_gym.data.tasks.harbor.models import HarborTaskConfig


def _write_task(folder: Path, *, judged: bool, materials: bool = False) -> None:
    """A pinned-shape task: `[[steps]]` with one `solve` step, a build-mode Dockerfile and a step verifier."""
    env = ""
    if judged:
        env = '[environment.env]\nANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY:-}"\nGEMINI_API_KEY    = "${GEMINI_API_KEY:-}"\n'
    if materials:
        # The repair tasks name the golden-asset archive in the sandbox environment; the verifier downloads it.
        env = '[environment.env]\nMATERIALS_URL = "https://example.invalid/materials.zip"\n'
    url = "ARG MATERIALS_URL=https://example.invalid/materials.zip\n" if materials else ""
    (folder / "environment").mkdir(parents=True)
    (folder / "environment" / "Dockerfile").write_text(f"FROM python:3.12-slim\n{url}RUN mkdir -p /workspace\n")
    (folder / "task.toml").write_text(
        'version = "1.0"\n\n[task]\nname = "agentic-vbench/example"\n\n[environment]\nbuild_timeout_sec = 600.0\n'
        f"cpus = 2\nmemory_mb = 4096\nstorage_mb = 8192\nallow_internet = true\n\n{env}\n"
        '[[steps]]\nname = "solve"\n\n[steps.agent]\ntimeout_sec = 1800.0\n\n[steps.verifier]\ntimeout_sec = 1800.0\n'
    )
    step = folder / "steps" / "solve"
    (step / "tests").mkdir(parents=True)
    (step / "solution").mkdir()
    (step / "instruction.md").write_text("# Task\n\nAssemble the video.\n")
    retry = 'curl --retry 5 --retry-delay 3 -o /tmp/g.zip "$MATERIALS_URL"\n' if materials else ""
    (step / "tests" / "test.sh").write_text(f"#!/bin/bash\nset -euo pipefail\n{retry}python3 /tests/judge.py\n")
    (step / "tests" / "judge.py").write_text("print('original judge')\n")
    (step / "solution" / "solve.sh").write_text("#!/bin/bash\necho solved\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkout(tmp_path: Path) -> list[Path]:
    """Two families' tasks under the pinned checkout layout `tasks/agentic_vbench_<family>/<task>/`."""
    repurpose = tmp_path / "checkout" / "tasks" / "agentic_vbench_repurpose" / "boxing"
    sequencing = tmp_path / "checkout" / "tasks" / "agentic_vbench_sequencing" / "agentic-vbench-sequencing-task1"
    _write_task(repurpose, judged=True)
    _write_task(sequencing, judged=False)
    return [repurpose, sequencing]


def test_adapt_task_routes_the_judge_through_the_hub_and_keeps_the_key_out_of_the_sandbox(tmp_path: Path) -> None:
    repurpose, _ = _checkout(tmp_path)
    destination = tmp_path / "entry" / repurpose.name
    adapt_task(repurpose, destination, protocol="nvinference-hybrid", records=tmp_path / "records")

    config = tomllib.loads((destination / "task.toml").read_text())
    assert "env" not in config["environment"], "the native judge keys must not reach the agent's sandbox"
    verifier_env = config["steps"][0]["verifier"]["env"]
    assert verifier_env["NVINFERENCE_API_KEY"] == "${NVINFERENCE_API_KEY}"
    for key, value in JUDGE_PROTOCOLS["nvinference-hybrid"].items():
        assert verifier_env[key] == value
    tests = destination / "steps" / "solve" / "tests"
    assert _digest(tests / "judge.py") == _digest(HUB_JUDGE)
    assert (tests / "avb-original-judge.py").read_text() == "print('original judge')\n"
    # The record names what changed and the digests before and after.
    assert (tmp_path / "records" / "judge_protocol.json").is_file()
    # Gym's task schema accepts the adjusted task.toml (deprecated `version` and `allow_internet` included).
    HarborTaskConfig.model_validate(config)


def test_adapt_task_leaves_an_unjudged_task_byte_for_byte(tmp_path: Path) -> None:
    _, sequencing = _checkout(tmp_path)
    destination = tmp_path / "entry" / sequencing.name
    adapt_task(sequencing, destination, protocol="nvinference-hybrid", records=tmp_path / "records")

    for source in sorted(path for path in sequencing.rglob("*") if path.is_file()):
        assert _digest(destination / source.relative_to(sequencing)) == _digest(source), source
    assert not (destination.parent / f".{destination.name}.stages").exists()


def test_adapt_task_stretches_the_golden_asset_retries(tmp_path: Path) -> None:
    repair = tmp_path / "checkout" / "tasks" / "agentic_vbench_repair" / "curiosity-repair-1"
    _write_task(repair, judged=False, materials=True)
    destination = tmp_path / "entry" / repair.name
    adapt_task(repair, destination, protocol="nvinference-hybrid", records=tmp_path / "records")

    verifier = (destination / "steps" / "solve" / "tests" / "test.sh").read_text()
    assert "--retry 12 --retry-delay 20 --retry-all-errors" in verifier
    assert "--retry 5 --retry-delay 3" not in verifier
    # A download that still fails is reported as inconclusive, not left without a reward.
    assert verifier.count(MATERIALS_FETCH_GUARD) == 1


def test_adapt_task_refuses_an_unknown_judge_protocol(tmp_path: Path) -> None:
    repurpose, _ = _checkout(tmp_path)
    with pytest.raises(ValueError, match="judge protocol"):
        adapt_task(repurpose, tmp_path / "entry" / repurpose.name, protocol="bogus", records=tmp_path / "records")


def test_write_entry_is_flat_keeps_committed_files_and_replaces_previous_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare_module, "STAGING_DIR", tmp_path / "staging")
    entry = tmp_path / "environments" / "harbor" / "agentic_vbench"
    entry.mkdir(parents=True)
    (entry / "README.md").write_text("# entry\n")
    (entry / "dataset.toml").write_text('[gym]\ntype = "benchmark"\n')
    (entry / "harbor_sandbox.yaml").write_text("sandbox: {}\n")
    (entry / "stale-task").mkdir()
    (entry / "stale-task" / "task.toml").write_text("")
    task_dirs = _checkout(tmp_path)

    assert write_entry(task_dirs, entry) == entry
    assert sorted(path.name for path in entry.iterdir()) == [
        RECORDS_DIR,
        "README.md",
        "agentic-vbench-sequencing-task1",
        "boxing",
        "dataset.toml",
        "harbor_sandbox.yaml",
    ]
    assert (entry / "README.md").read_text() == "# entry\n"
    assert (entry / RECORDS_DIR / "boxing" / "judge_protocol.json").is_file()
    # The family folder of the checkout is gone: `harbor` discovers the task folders by name.
    assert (entry / "boxing" / "task.toml").is_file()
    assert not any(path.name.startswith(".") and path.name != RECORDS_DIR for path in entry.iterdir())


def test_write_entry_refuses_duplicate_task_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_module, "STAGING_DIR", tmp_path / "staging")
    repurpose, _ = _checkout(tmp_path)
    twin = tmp_path / "checkout" / "tasks" / "agentic_vbench_assembly" / repurpose.name
    _write_task(twin, judged=False)
    with pytest.raises(ValueError, match="unique"):
        write_entry([repurpose, twin], tmp_path / "entry")


def test_committed_entry_declares_the_benchmark_and_its_harnesses() -> None:
    config = read_dataset_config(ENTRY_DIR)
    assert config.source is None, "the tasks come from the pinned git checkout, not from a Harbor hub"
    assert config.type == "benchmark"
    assert config.domain == "agent"
    assert config.license == "Apache 2.0"
    assert config.allowed_agents == ["opencode_agent"]
    assert (ENTRY_DIR / "README.md").is_file() and (ENTRY_DIR / "harbor_sandbox.yaml").is_file()
