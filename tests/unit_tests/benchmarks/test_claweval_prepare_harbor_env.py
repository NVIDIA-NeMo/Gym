# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for writing ClawEval task packages into the `harbor` environment server's training entry."""

import json
import shutil
import tomllib
from pathlib import Path

import pytest
import yaml

from benchmarks.agentic_vbench.staging import OPENCODE_PLUGINS, OPENCODE_PLUGINS_WORKDIR
from benchmarks.claweval.prepare_harbor_env import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    ENTRY_DIR,
    JUDGE_KEY_VARIABLE,
    PREPARED_RECORD,
    TASK_DATA_FILE,
    main,
    package_digest,
    prepare,
    read_package,
    select_packages,
    write_task,
)
from nemo_gym.data.tasks.harbor.models import HarborTaskConfig
from nemo_gym.data.tasks.harbor.task_data import load_task_data_schema, read_task_fields


PROMPT = "Watch /workspace/fixtures/video.mp4 and write the speaker's name to /workspace/output.txt.\n"


def _write_package(root: Path, task_id: str, *, category: str = "video_qa", prompt: str = PROMPT) -> Path:
    folder = root / task_id
    (folder / "fixtures").mkdir(parents=True)
    (folder / "grader_reference").mkdir()
    (folder / "fixtures" / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(64))
    (folder / "grader_reference" / "00.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(16))
    (folder / ".grade_artifacts.py").write_text("print('{\"valid\": true}')\n")
    (folder / "grader.py").write_text("class Grader:\n    pass\n")
    (folder / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": task_id,
                "category": category,
                "prompt": {"text": prompt, "language": "en"},
                "sandbox_files": ["fixtures/video.mp4"],
                "sandbox_grader_files": [".grade_artifacts.py"],
                "local_grader_files": ["grader_reference/00.png"],
                "env_snapshot_files": ["/workspace/output.txt"],
                "env_snapshot_commands": ["python /workspace/.grade_artifacts.py"],
                "environment": {"timeout_seconds": 900, "max_turns": 40, "env_snapshot_timeout": 120},
                "judge_rubric": '{"family": "video_qa", "outputs": ["output.txt"], "reference_answer": "secret"}',
                "reference_solution": "secret",
            },
            sort_keys=False,
        )
    )
    return folder


def _write_source(root: Path) -> Path:
    package = root / "claw_eval_checkout" / "src" / "claw_eval"
    (package / "graders").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "graders" / "__init__.py").write_text("")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "x.pyc").write_bytes(b"")
    return root / "claw_eval_checkout"


def _entry(tmp_path: Path) -> Path:
    entry = tmp_path / "entry"
    entry.mkdir()
    for name in ("dataset.toml", "task_data.py", "harbor_train.yaml", "README.md"):
        shutil.copyfile(ENTRY_DIR / name, entry / name)
    return entry


def test_write_task_splits_agent_and_grader_material(tmp_path: Path) -> None:
    folder = _write_package(tmp_path / "root", "B_video_qa_0001")
    source = _write_source(tmp_path)
    destination = tmp_path / "entry" / "B_video_qa_0001"
    destination.parent.mkdir()
    shutil.copyfile(ENTRY_DIR / "task_data.py", destination.parent / "task_data.py")

    task_data = write_task(read_package(folder), destination, claw_eval_source=source)

    config = tomllib.loads((destination / "task.toml").read_text())
    assert HarborTaskConfig.unknown_keys(config) == []
    task = HarborTaskConfig.model_validate(config)
    assert config["task"]["name"] == "claweval/B_video_qa_0001"
    assert config["metadata"]["category"] == "video_qa"
    assert [step.name for step in task.steps] == ["solve"]
    assert task.steps[0].agent.timeout_sec == 900.0
    assert task.steps[0].verifier.timeout_sec == 1800.0  # the floor, above snapshot timeout + 600
    assert task.steps[0].verifier.env == {
        "CLAW_EVAL_JUDGE_API_KEY": "${" + JUDGE_KEY_VARIABLE + "}",
        "CLAW_EVAL_JUDGE_BASE_URL": DEFAULT_JUDGE_BASE_URL,
        "CLAW_EVAL_JUDGE_MODEL": DEFAULT_JUDGE_MODEL,
    }
    assert "docker_image" not in config["environment"]
    assert "WORKDIR /workspace" in (destination / "environment" / "Dockerfile").read_text().splitlines()

    step = destination / "steps" / "solve"
    assert (step / "instruction.md").read_text() == PROMPT
    workdir_files = sorted(str(p.relative_to(step / "workdir")) for p in (step / "workdir").rglob("*") if p.is_file())
    assert workdir_files == sorted(
        ["AGENTS.md", "fixtures/video.mp4"] + [f"{OPENCODE_PLUGINS_WORKDIR}/{name}" for name in OPENCODE_PLUGINS]
    )
    tests = step / "tests"
    assert (tests / "test.sh").read_text().startswith("#!/bin/bash")
    for name in ("judge_claweval.py", "judge_transport.py", "grader.py", "task.yaml", ".grade_artifacts.py"):
        assert (tests / name).is_file(), name
    assert (tests / "grader_reference" / "00.png").is_file()
    assert (tests / "claw_eval" / "graders" / "__init__.py").is_file()
    assert not (tests / "claw_eval" / "__pycache__").exists()
    staged = yaml.safe_load((tests / "task.yaml").read_text())
    assert staged["reference_solution"] == "secret" and staged["prompt"]["text"] == PROMPT
    assert (step / "solution" / "solve.sh").read_text().startswith("#!/bin/bash")
    # Nothing that grades sits where the agent works.
    for name in ("grader.py", "task.yaml", ".grade_artifacts.py", "grader_reference"):
        assert not (step / "workdir" / name).exists(), name

    assert json.loads((destination / TASK_DATA_FILE).read_text()) == task_data
    assert task_data["family"] == "video_qa"
    assert task_data["task_package_sha256"] == package_digest(
        folder, yaml.safe_load((folder / "task.yaml").read_text())
    )
    schema = load_task_data_schema(destination.parent)
    assert schema is not None
    schema.validate(read_task_fields(destination), task_id=destination.name)


def test_write_task_names_a_prebuilt_image_and_judge(tmp_path: Path) -> None:
    folder = _write_package(tmp_path / "root", "B_video_ocr_0002", category="video_ocr")
    source = _write_source(tmp_path)
    destination = tmp_path / "out" / "B_video_ocr_0002"
    destination.parent.mkdir()
    write_task(
        read_package(folder),
        destination,
        claw_eval_source=source,
        docker_image="localhost/claweval-env:1",
        judge_model="gcp/google/other-judge",
        copy_fixtures=True,
    )
    config = tomllib.loads((destination / "task.toml").read_text())
    assert config["environment"]["docker_image"] == "localhost/claweval-env:1"
    assert config["steps"][0]["verifier"]["env"]["CLAW_EVAL_JUDGE_MODEL"] == "gcp/google/other-judge"
    HarborTaskConfig.model_validate(config)
    fixture = destination / "steps" / "solve" / "workdir" / "fixtures" / "video.mp4"
    assert fixture.stat().st_nlink == 1  # copied, not linked
    with pytest.raises(FileExistsError):
        write_task(read_package(folder), destination, claw_eval_source=source)


def test_package_digest_covers_scoring_inputs_not_media(tmp_path: Path) -> None:
    folder = _write_package(tmp_path / "root", "B_video_qa_0003")
    definition = yaml.safe_load((folder / "task.yaml").read_text())
    before = package_digest(folder, definition)
    (folder / "fixtures" / "video.mp4").write_bytes(b"other media of any length")
    assert package_digest(folder, definition) == before
    (folder / "grader.py").write_text("class Grader:\n    changed = True\n")
    assert package_digest(folder, definition) != before


def test_read_package_refuses_malformed_packages(tmp_path: Path) -> None:
    root = tmp_path / "root"
    folder = _write_package(root, "B_video_qa_0004")
    (folder / "task.yaml").write_text((folder / "task.yaml").read_text().replace("B_video_qa_0004", "other-id"))
    with pytest.raises(ValueError, match="task_id must be the package folder's name"):
        read_package(folder)
    folder = _write_package(root, "B_video_qa_0005")
    definition = yaml.safe_load((folder / "task.yaml").read_text())
    definition["sandbox_files"] = ["../escape.mp4"]
    (folder / "task.yaml").write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="leaves the package"):
        read_package(folder)
    folder = _write_package(root, "B_video_qa_0006")
    (folder / "grader.py").unlink()
    with pytest.raises(ValueError, match="no grader.py"):
        read_package(folder)


def test_select_packages_by_id(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for task_id in ("B_video_qa_0007", "B_video_qa_0008", "B_video_ocr_0009"):
        _write_package(root, task_id)
    assert [p["task_id"] for p in select_packages(root)] == ["B_video_ocr_0009", "B_video_qa_0007", "B_video_qa_0008"]
    assert [p["task_id"] for p in select_packages(root, ["B_video_qa_0008", "B_video_qa_0008"])] == ["B_video_qa_0008"]
    with pytest.raises(ValueError, match="no package for task ids"):
        select_packages(root, ["B_video_qa_9999"])
    with pytest.raises(ValueError, match="no <task_id>/task.yaml"):
        select_packages(tmp_path / "root" / "B_video_qa_0007" / "fixtures")


def test_prepare_replaces_previous_tasks_and_keeps_entry_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    for task_id in ("B_video_qa_0010", "B_video_qa_0011"):
        _write_package(root, task_id)
    source = _write_source(tmp_path)
    entry = _entry(tmp_path)
    (entry / "stale-task").mkdir()
    (entry / "stale-task" / "task.toml").write_text("")

    written = prepare(root, claw_eval_source=source, entry_dir=entry, task_ids=["B_video_qa_0011"])

    assert [task["task_id"] for task in written] == ["B_video_qa_0011"]
    assert not (entry / "stale-task").exists()
    assert not (entry / "B_video_qa_0010").exists()
    for name in ("dataset.toml", "task_data.py", "harbor_train.yaml", "README.md"):
        assert (entry / name).is_file(), name
    record = json.loads((entry / PREPARED_RECORD).read_text())
    assert record["tasks"] == written and record["judge_model"] == DEFAULT_JUDGE_MODEL
    schema = load_task_data_schema(entry)
    schema.validate(read_task_fields(entry / "B_video_qa_0011"), task_id="B_video_qa_0011")

    # The command line: environment defaults for the root and the checkout, a task-ids file, another judge model.
    ids = tmp_path / "ids.txt"
    ids.write_text("# chosen\nB_video_qa_0010\n\n")
    monkeypatch.setenv("CLAWEVAL_TASKS_ROOT", str(root))
    monkeypatch.setenv("CLAWEVAL_SOURCE", str(source))
    monkeypatch.setenv("CLAWEVAL_JUDGE_MODEL", "gcp/google/env-judge")
    main(["--entry", str(entry), "--task-ids", str(ids)])
    assert sorted(p.name for p in entry.iterdir() if p.is_dir()) == ["B_video_qa_0010"]
    config = tomllib.loads((entry / "B_video_qa_0010" / "task.toml").read_text())
    assert config["steps"][0]["verifier"]["env"]["CLAW_EVAL_JUDGE_MODEL"] == "gcp/google/env-judge"
