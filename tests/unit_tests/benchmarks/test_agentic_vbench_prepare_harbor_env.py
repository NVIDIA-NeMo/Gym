# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for writing the pinned Agentic-VBench tasks into the `harbor` environment server's dataset entry."""

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import pytest

import benchmarks.agentic_vbench.prepare_harbor_env as prepare_module
from benchmarks.agentic_vbench.prepare_harbor_env import (
    ENTRY_NAME,
    RECORDS_DIR,
    TASK_DATA_FILE,
    TRAIN_ENTRY_NAME,
    adapt_task,
    custom_task_dirs,
    prepare,
    task_family,
    write_entry,
)
from benchmarks.agentic_vbench.staging import (
    HUB_JUDGE,
    JUDGE_PROTOCOLS,
    MATERIALS_ARTIFACT,
    MATERIALS_FETCH_GUARD,
    MATERIALS_RETRY_PATCHED,
    OPENCODE_PLUGINS,
    OPENCODE_PLUGINS_DIR,
    OPENCODE_PLUGINS_WORKDIR,
    OUTPUT_ARTIFACT,
)
from nemo_gym.data.tasks.harbor.models import HarborTaskConfig
from nemo_gym.data.tasks.harbor.task_data import load_task_data_schema, read_task_fields


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
    (folder / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\n{url}RUN mkdir -p /workspace\nWORKDIR /workspace\n"
    )
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
    assert "env" not in config["environment"], "the upstream judge keys must not reach the agent's sandbox"
    verifier_env = config["steps"][0]["verifier"]["env"]
    assert verifier_env["NVINFERENCE_API_KEY"] == "${NVINFERENCE_API_KEY}"
    for key, value in JUDGE_PROTOCOLS["nvinference-hybrid"].items():
        assert verifier_env[key] == value
    tests = destination / "steps" / "solve" / "tests"
    assert _digest(tests / "judge.py") == _digest(HUB_JUDGE)
    assert (tests / "avb-original-judge.py").read_text() == "print('original judge')\n"
    # The verifier reads only the agent's output, declared as the step's artifact so it can grade separately.
    assert config["steps"][0]["artifacts"] == [OUTPUT_ARTIFACT]
    # The record names what changed and the digests before and after.
    assert (tmp_path / "records" / "judge_protocol.json").is_file()
    # Gym's task schema accepts the adjusted task.toml (deprecated `version` and `allow_internet` included).
    HarborTaskConfig.model_validate(config)


def test_adapt_task_leaves_an_unjudged_task_byte_for_byte_but_for_its_family_and_the_plugins(tmp_path: Path) -> None:
    _, sequencing = _checkout(tmp_path)
    destination = tmp_path / "entry" / sequencing.name
    adapt_task(sequencing, destination, protocol="nvinference-hybrid", records=tmp_path / "records")

    for source in sorted(path for path in sequencing.rglob("*") if path.is_file()):
        assert _digest(destination / source.relative_to(sequencing)) == _digest(source), source
    assert not (destination.parent / f".{destination.name}.stages").exists()
    # The two additions every task gets: its family as a typed task field, and the OpenCode plugins the training
    # overlay may load, under the step's workdir so they land in the sandbox's working directory.
    assert json.loads((destination / TASK_DATA_FILE).read_text()) == {"family": "sequencing"}
    plugins = destination / "steps" / "solve" / "workdir" / OPENCODE_PLUGINS_WORKDIR
    assert sorted(path.name for path in plugins.iterdir()) == sorted(OPENCODE_PLUGINS)
    for name in OPENCODE_PLUGINS:
        assert _digest(plugins / name) == _digest(OPENCODE_PLUGINS_DIR / name)
    added = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()} - {
        path.relative_to(sequencing).as_posix() for path in sequencing.rglob("*") if path.is_file()
    }
    assert added == {
        TASK_DATA_FILE,
        *(f"steps/solve/workdir/{OPENCODE_PLUGINS_WORKDIR}/{name}" for name in OPENCODE_PLUGINS),
    }
    record = json.loads((tmp_path / "records" / "opencode_plugins.json").read_text())
    assert set(record["files"]) == {
        f"steps/solve/workdir/{OPENCODE_PLUGINS_WORKDIR}/{name}" for name in OPENCODE_PLUGINS
    }
    # The entry's task_data.py schema accepts the field the prepare script wrote.
    shutil.copy(prepare_module.ENTRIES_DIR / ENTRY_NAME / "task_data.py", destination.parent / "task_data.py")
    schema = load_task_data_schema(destination.parent)
    assert schema is not None
    schema.validate(read_task_fields(destination), task_id=destination.name)
    with pytest.raises(ValueError):
        schema.validate({"family": "understanding"}, task_id=destination.name)


def test_adapt_task_refuses_a_task_outside_the_family_layout_or_with_its_own_task_data(tmp_path: Path) -> None:
    stray = tmp_path / "checkout" / "tasks" / "boxing"
    _write_task(stray, judged=False)
    with pytest.raises(ValueError, match="sits under tasks/agentic_vbench_<family>/"):
        task_family(stray)
    _, sequencing = _checkout(tmp_path)
    (sequencing / TASK_DATA_FILE).write_text('{"family": "repair"}\n')
    with pytest.raises(ValueError, match="ships its own task_data.json"):
        adapt_task(
            sequencing, tmp_path / "entry" / sequencing.name, protocol="nvinference-hybrid", records=tmp_path / "r"
        )


def test_adapt_task_refuses_a_task_that_does_not_work_in_workspace(tmp_path: Path) -> None:
    _, sequencing = _checkout(tmp_path)
    dockerfile = sequencing / "environment" / "Dockerfile"
    dockerfile.write_text(dockerfile.read_text().replace("WORKDIR /workspace\n", "WORKDIR /app\n"))
    with pytest.raises(ValueError, match="works in /workspace"):
        adapt_task(
            sequencing, tmp_path / "entry" / sequencing.name, protocol="nvinference-hybrid", records=tmp_path / "r"
        )


def test_adapt_task_stretches_the_golden_asset_retries(tmp_path: Path) -> None:
    repair = tmp_path / "checkout" / "tasks" / "agentic_vbench_repair" / "curiosity-repair-1"
    _write_task(repair, judged=False, materials=True)
    destination = tmp_path / "entry" / repair.name
    adapt_task(repair, destination, protocol="nvinference-hybrid", records=tmp_path / "records")

    verifier = (destination / "steps" / "solve" / "tests" / "test.sh").read_text()
    assert MATERIALS_RETRY_PATCHED in verifier
    assert "--retry 5 --retry-delay 3" not in verifier
    # A download that still fails exits 75 (verifier unavailable), not without a reward.
    assert verifier.count(MATERIALS_FETCH_GUARD) == 1
    config = tomllib.loads((destination / "task.toml").read_text())
    assert config["steps"][0]["artifacts"] == [OUTPUT_ARTIFACT, MATERIALS_ARTIFACT]
    HarborTaskConfig.model_validate(config)
    record = json.loads((tmp_path / "records" / "materials.json").read_text())
    assert record["retry"]["staged"] == MATERIALS_RETRY_PATCHED
    assert set(record["verifiers"]) == {"steps/solve/tests/test.sh"}


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
    (entry / "task_data.py").write_text("class TaskData: ...\n")
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
        "task_data.py",
    ]
    assert (entry / "README.md").read_text() == "# entry\n"
    assert (entry / "task_data.py").read_text() == "class TaskData: ...\n"
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


def _training_root(tmp_path: Path) -> Path:
    """Generated tasks in the pinned layout, each family folder holding links to the tasks' real folders."""
    real = tmp_path / "generated"
    root = tmp_path / "root"
    for family, name in (
        ("assembly", "curiosity-assembly-1"),
        ("assembly", "curiosity-assembly-2"),
        ("repair", "curiosity-repair-1"),
    ):
        _write_task(real / name, judged=False, materials=family == "repair")
        (root / "tasks" / f"agentic_vbench_{family}").mkdir(parents=True, exist_ok=True)
        (root / "tasks" / f"agentic_vbench_{family}" / name).symlink_to(real / name, target_is_directory=True)
    return root


def test_custom_task_dirs_discovers_the_selected_families_through_links_and_checks_names(tmp_path: Path) -> None:
    root = _training_root(tmp_path)
    assert [task.name for task in custom_task_dirs(root, ["assembly", "repair", "repurpose"])] == [
        "curiosity-assembly-1",
        "curiosity-assembly-2",
        "curiosity-repair-1",
    ]
    assert [task.name for task in custom_task_dirs(root, ["repair"])] == ["curiosity-repair-1"]
    with pytest.raises(ValueError, match="No tasks of"):
        custom_task_dirs(root, ["repurpose"])
    with pytest.raises(ValueError, match="Unknown families"):
        custom_task_dirs(root, ["understanding"])
    twin = root / "tasks" / "agentic_vbench_sequencing" / "curiosity-assembly-1"
    _write_task(twin, judged=False)
    with pytest.raises(ValueError, match="unique across families"):
        custom_task_dirs(root, ["assembly", "sequencing"])


def test_prepare_writes_a_task_root_to_the_training_entry_with_the_same_adjustments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare_module, "STAGING_DIR", tmp_path / "staging")
    entries = tmp_path / "environments" / "harbor"
    real_entries = prepare_module.ENTRIES_DIR
    monkeypatch.setattr(prepare_module, "ENTRIES_DIR", entries)
    train = entries / TRAIN_ENTRY_NAME
    train.mkdir(parents=True)
    (train / "dataset.toml").write_text('[gym]\ntype = "train"\nlicense = "Apache 2.0"\n')
    shutil.copy(real_entries / TRAIN_ENTRY_NAME / "task_data.py", train / "task_data.py")
    root = _training_root(tmp_path)

    assert prepare(tasks_root=root, protocol="nvinference-hybrid") == train

    assert sorted(path.name for path in train.iterdir()) == [
        RECORDS_DIR,
        "curiosity-assembly-1",
        "curiosity-assembly-2",
        "curiosity-repair-1",
        "dataset.toml",
        "task_data.py",
    ]
    # Links are resolved: the entry holds the tasks' files, adjusted as the benchmark entry's are.
    assert not (train / "curiosity-repair-1").is_symlink()
    repair = (train / "curiosity-repair-1" / "steps" / "solve" / "tests" / "test.sh").read_text()
    assert MATERIALS_RETRY_PATCHED in repair and MATERIALS_FETCH_GUARD in repair
    assert json.loads((train / "curiosity-repair-1" / TASK_DATA_FILE).read_text()) == {"family": "repair"}
    assert json.loads((train / "curiosity-assembly-2" / TASK_DATA_FILE).read_text()) == {"family": "assembly"}
    schema = load_task_data_schema(train)
    assert schema is not None
    for task in ("curiosity-assembly-1", "curiosity-assembly-2", "curiosity-repair-1"):
        schema.validate(read_task_fields(train / task), task_id=task)
    # The pinned benchmark entry is never the target of a task root.
    with pytest.raises(ValueError, match="pinned benchmark"):
        prepare(tasks_root=root, entry_name=ENTRY_NAME, protocol="nvinference-hybrid")
