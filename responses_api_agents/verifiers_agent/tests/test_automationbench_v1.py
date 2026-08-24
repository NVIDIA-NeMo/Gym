# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import json
import sys
from pathlib import Path

import pytest
import verifiers.v1 as vf


AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_DIR.parents[1]
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.automationbench import prepare as prepare_module
from benchmarks.automationbench import taskset as taskset_module
from benchmarks.automationbench.prepare import stage_tasks
from benchmarks.automationbench.taskset import (
    AutomationBenchTaskset,
    AutomationBenchTasksetConfig,
    decode_public_info,
)


def _write_task(path: Path, *, name: str, source_id: int, domain: str = "hr") -> None:
    path.write_text(
        json.dumps(
            {
                "task_name": name,
                "task_id": source_id,
                "domain": domain,
                "user_prompt": f"Run {name}",
                "initial_state": {},
                "assertions": [],
                "zapier_tools": [],
            }
        ),
        encoding="utf-8",
    )


def test_external_tasks_load_in_sorted_path_order(tmp_path: Path) -> None:
    _write_task(tmp_path / "task_02.json", name="second", source_id=2)
    _write_task(tmp_path / "task_01.json", name="first", source_id=1)

    config = AutomationBenchTasksetConfig(id="automationbench_v1", tasks_dir=str(tmp_path))
    tasks = list(AutomationBenchTaskset(config))

    assert [task.data.idx for task in tasks] == [0, 1]
    assert [task.data.name for task in tasks] == ["first", "second"]
    assert [task.data.source_id for task in tasks] == [1, 2]
    assert [task.data.prompt_text for task in tasks] == ["Run first", "Run second"]


def test_public_info_accepts_serialized_and_decoded_mappings() -> None:
    expected = {"initial_state": {"meta": {"schema_version": "0.1.0"}}, "assertions": []}

    assert decode_public_info(json.dumps(expected)) == expected
    assert decode_public_info(expected) == expected


def test_external_tasks_honor_domain_and_size_filters(tmp_path: Path) -> None:
    _write_task(tmp_path / "task_01.json", name="hr-one", source_id=1)
    _write_task(tmp_path / "task_02.json", name="sales-one", source_id=2, domain="sales")
    _write_task(tmp_path / "task_03.json", name="hr-two", source_id=3)

    config = AutomationBenchTasksetConfig(
        id="automationbench_v1",
        tasks_dir=str(tmp_path),
        domains="hr",
        num_examples=1,
    )

    assert [task.data.name for task in AutomationBenchTaskset(config)] == ["hr-one"]


def test_external_tasks_reject_missing_directory(tmp_path: Path) -> None:
    config = AutomationBenchTasksetConfig(
        id="automationbench_v1",
        tasks_dir=str(tmp_path / "missing"),
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        list(AutomationBenchTaskset(config))


def test_external_tasks_resolve_repository_relative_path(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    tasks_dir = repo_root / "benchmarks" / "automationbench" / "data" / "tasks"
    component_dir = tmp_path / "component"
    tasks_dir.mkdir(parents=True)
    component_dir.mkdir()
    _write_task(tasks_dir / "task_01.json", name="relative", source_id=1)
    config = AutomationBenchTasksetConfig(
        id="automationbench_v1",
        tasks_dir="benchmarks/automationbench/data/tasks",
    )
    monkeypatch.setattr(taskset_module, "REPO_ROOT", repo_root)
    monkeypatch.chdir(component_dir)

    assert [task.data.name for task in AutomationBenchTaskset(config)] == ["relative"]


def test_stage_tasks_is_stable_and_refuses_a_different_source(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    staged = data_dir / "tasks"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_task(first / "task_01.json", name="first", source_id=1)
    _write_task(second / "task_01.json", name="second", source_id=2)
    monkeypatch.setattr(prepare_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(prepare_module, "STAGED_TASKS_DIR", staged)

    assert stage_tasks(first) == staged
    assert staged.is_symlink()
    assert staged.resolve() == first.resolve()
    assert stage_tasks(first) == staged

    with pytest.raises(FileExistsError, match="pass --force"):
        stage_tasks(second)


def test_automationbench_plugin_resolves_stable_v1_environment(tmp_path: Path) -> None:
    _write_task(tmp_path / "task_01.json", name="smoke", source_id=1)

    config = vf.resolve_env_config(
        {
            "taskset": {
                "id": "automationbench_v1",
                "tasks_dir": str(tmp_path),
            },
            "agent": {
                "harness": {"id": "null"},
                "max_turns": 50,
            },
        }
    )
    env = vf.load_environment(config)
    task = next(iter(env.taskset))

    assert type(env).__name__ == "SingleAgentEnv"
    assert env.config.agent.harness.id == "null"
    assert env.config.agent.max_turns == 50
    assert task.data.name == "smoke"


def test_openai_override_provides_verifiers_context_management_type() -> None:
    from openai.types.responses.response_create_params import ContextManagement

    assert importlib.metadata.version("openai") == "2.54.0"
    assert ContextManagement is not None
