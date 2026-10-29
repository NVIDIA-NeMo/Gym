# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser
from nemo_gym.config.config_types import maybe_get_server_instance_config
from nemo_gym.data.task_data import TaskDataValidator, load_task_data_schema
from nemo_gym.environment.benchmarks import BenchmarkConfig, _is_benchmark_config
from nemo_gym.environment.registry import discover_environment_catalog
from resources_servers.swe_atlas_qa import prepare


def test_benchmark_catalog_discovery_and_single_dataset_composition() -> None:
    root = prepare.PACKAGE_DIR.parents[1]
    config_path = root / "benchmarks/swe_atlas_qa/config.yaml"
    assert _is_benchmark_config(config_path)
    entries = [entry for entry in discover_environment_catalog() if entry.name == "swe_atlas_qa"]
    assert [(entry.kind, entry.config_path.resolve()) for entry in entries] == [("benchmark", config_path)]
    benchmark = BenchmarkConfig.from_config_path(config_path, strict=False)
    assert benchmark.name == "swe_atlas_qa"
    _, configs = GlobalConfigDictParser().load_extra_config_paths([str(config_path)])
    resolved = OmegaConf.merge(*configs)
    agent = resolved.swe_atlas_qa_agent.responses_api_agents.harness_agent
    standalone = OmegaConf.load(prepare.PACKAGE_DIR / "configs/swe_atlas_qa.yaml")
    assert len(agent.datasets) == 1
    standalone_datasets = standalone.swe_atlas_qa_agent.responses_api_agents.harness_agent.datasets
    assert agent.datasets == [dataset for dataset in standalone_datasets if dataset.type == "benchmark"]
    assert [dataset.type for dataset in standalone_datasets] == ["example", "benchmark"]
    assert agent.resources_server.name == "swe_atlas_qa_resources_server"
    assert agent.agent == "opencode"
    model, error = maybe_get_server_instance_config(name="policy_model", server_type_config_dict=resolved.policy_model)
    assert error is None
    assert model.get_inner_run_server_config().port == 8000


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is required for the local source fixture")
    source_dir = tmp_path / "source"
    task_count = getattr(request, "param", 2)
    for index in range(task_count):
        task_dir = source_dir / "data/qa" / f"task-{index:024x}"
        (task_dir / "tests").mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            f'[environment]\ndocker_image = "synthetic-image-{index}"\n[verifier]\nprivate = "PRIVATE_CONFIG"\n',
            encoding="utf-8",
        )
        (task_dir / "instruction.md").write_text(f"  Synthetic question {index}: café\n\n", encoding="utf-8")
        (task_dir / "tests/rubrics.json").write_text('{"private": "PRIVATE_RUBRIC"}', encoding="utf-8")
        (task_dir / "tests/prompt.txt").write_text("PRIVATE_PROMPT", encoding="utf-8")
        (task_dir / "solution").mkdir()
        (task_dir / "solution/answer.txt").write_text("PRIVATE_ANSWER", encoding="utf-8")
    subprocess.run(["git", "init", str(source_dir)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=source_dir, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--no-gpg-sign",
            "-m",
            "Synthetic fixture",
        ],
        cwd=source_dir,
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_dir, text=True).strip()
    monkeypatch.setattr(prepare, "SOURCE_REVISION", revision)
    monkeypatch.setattr(prepare, "EXPECTED_TASK_COUNT", task_count)
    return source_dir


def test_prepare_rows_preserve_instruction_and_exclude_private_assets(source: Path, tmp_path: Path):
    output = tmp_path / "output/benchmark.jsonl"
    assert prepare.prepare(source_dir=source, output_path=output) == output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    validator = TaskDataValidator(
        server_name="swe_atlas_qa",
        adapter=load_task_data_schema(prepare.PACKAGE_DIR),
        dataset_fpath=str(output),
    )
    for index, row in enumerate(rows):
        task_id = f"task-{index:024x}"
        assert row == {
            "task_id": task_id,
            "responses_create_params": {
                "input": [{"role": "user", "content": f"  Synthetic question {index}: café\n\n"}],
            },
            "verifier_metadata": {"task_id": task_id},
        }
        validator.validate_row(index, row)
    assert validator.report.clean
    assert "PRIVATE_" not in output.read_text(encoding="utf-8")
    assert not output.with_name("example.jsonl").exists()


def test_prepare_preserves_example_fixtures(source: Path, tmp_path: Path):
    example = tmp_path / "example.jsonl"
    example.write_text("existing smoke fixture\n", encoding="utf-8")
    prepare.prepare(source_dir=source, output_path=tmp_path / "benchmark.jsonl")
    assert example.read_text(encoding="utf-8") == "existing smoke fixture\n"


def test_load_task_returns_toml_config(source: Path):
    tasks_dir = source / "data/qa"
    task_id = f"task-{0:024x}"
    expected = tomllib.loads((tasks_dir / task_id / "task.toml").read_text(encoding="utf-8"))
    assert prepare.load_task(tasks_dir, task_id) == expected


@pytest.mark.parametrize("task_id", ["", ".", "..", "../outside", "/tmp/outside", "task-invalid", "task-" + "a" * 25])
def test_load_task_rejects_invalid_ids(tmp_path: Path, task_id: str):
    with pytest.raises(ValueError, match="Invalid SWE-Atlas task ID"):
        prepare.load_task(tmp_path, task_id)


def test_load_task_rejects_symlink_escape(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tasks_dir / f"task-{0:024x}").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Invalid SWE-Atlas task ID"):
        prepare.load_task(tasks_dir, f"task-{0:024x}")


def test_load_task_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        prepare.load_task(tmp_path, f"task-{0:024x}")


def test_prepare_rejects_wrong_revision(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prepare, "SOURCE_REVISION", "0" * 40)
    output = tmp_path / "benchmark.jsonl"
    with pytest.raises(ValueError, match="source revision"):
        prepare.prepare(source_dir=source, output_path=output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["modified", "staged", "untracked", "deleted"])
def test_prepare_rejects_dirty_qa(source: Path, tmp_path: Path, change: str):
    instruction = source / "data/qa" / f"task-{0:024x}" / "instruction.md"
    if change == "deleted":
        instruction.unlink()
    elif change == "untracked":
        (instruction.parent / "extra.txt").write_text("synthetic extra", encoding="utf-8")
    else:
        instruction.write_text("synthetic modification", encoding="utf-8")
        if change == "staged":
            subprocess.run(["git", "add", str(instruction)], cwd=source, check=True, capture_output=True)
    output = tmp_path / "benchmark.jsonl"
    output.write_text("existing output", encoding="utf-8")
    with pytest.raises(ValueError, match="changes under data/qa"):
        prepare.prepare(source_dir=source, output_path=output)
    assert output.read_text(encoding="utf-8") == "existing output"


def test_prepare_checks_task_count(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prepare, "EXPECTED_TASK_COUNT", 124)
    with pytest.raises(ValueError, match="Expected 124 SWE-Atlas QA tasks, found 2"):
        prepare.prepare(source_dir=source, output_path=tmp_path / "benchmark.jsonl")


def test_prepare_clones_and_checks_out_pin(source: Path, tmp_path: Path):
    destination = tmp_path / "download/source"
    real_run = subprocess.run

    def mock_clone(command, **kwargs):
        if command[1] == "clone":
            assert command == ["git", "clone", prepare.SOURCE_REPOSITORY, str(destination)]
            shutil.copytree(source, destination)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return real_run(command, **kwargs)

    with patch.object(prepare.subprocess, "run", side_effect=mock_clone) as run:
        output = prepare.prepare(source_dir=destination, output_path=tmp_path / "benchmark.jsonl")
    commands = [call.args[0] for call in run.call_args_list]
    assert commands[1] == ["git", "checkout", "--detach", prepare.SOURCE_REVISION]
    assert commands[2] == ["git", "rev-parse", "HEAD"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_cli_passes_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    source = tmp_path / "source"
    output = tmp_path / "benchmark.jsonl"
    monkeypatch.setattr(sys, "argv", ["prepare", "--source-dir", str(source), "--output", str(output)])
    with patch.object(prepare, "prepare", return_value=output) as mocked:
        prepare.main()
    mocked.assert_called_once_with(source_dir=source, output_path=output)
    assert capsys.readouterr().out.strip() == str(output)
