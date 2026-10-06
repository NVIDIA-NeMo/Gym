# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from benchmarks.terminal_bench_2_1 import prepare_nooa
from nemo_gym.episode_types import MaterializedTask
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput
from resources_servers.terminal_bench_2_1.task_metadata import read_image_startup


ROOT = Path(__file__).resolve().parents[3]


def _task(directory: Path, *, name: str = "regex-log", agent_timeout: str = "900.0") -> Path:
    task = directory / "tasks" / name
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("private verifier\n")
    (task / "instruction.md").write_text("Canonical instruction.\nPreserve these bytes.\n")
    (task / "task.toml").write_text(
        f'[task]\nname = "terminal-bench/{name}"\n'
        f'[environment]\ndocker_image = "alexgshaw/{name}:20251031"\n'
        f"[agent]\ntimeout_sec = {agent_timeout}\n[verifier]\ntimeout_sec = 450.0\n"
    )
    return task


def test_canonical_instruction_image_and_separate_budgets_are_preserved(tmp_path: Path) -> None:
    task = _task(tmp_path)
    row = prepare_nooa.task_row(task)
    assert row["task_id"] == {"taskset": "terminal-bench-2.1-nooa", "task_id": "terminal-bench/regex-log"}
    body = row["task_input"]
    assert body["responses_create_params"]["input"] == [
        {"role": "user", "content": (task / "instruction.md").read_text()}
    ]
    assert body["agent_timeout_seconds"] == 900
    validated = MaterializedTask[SingleAgentTurnTaskInput].model_validate(row)
    assert validated.task_input.responses_create_params.max_output_tokens == 32768
    assert validated.task_input.responses_create_params.temperature == 1.0
    assert validated.task_input.responses_create_params.top_p == 1.0
    assert body["task_data"]["verifier_timeout_seconds"] == 450
    assert body["task_data"]["docker_image"] == "alexgshaw/regex-log:20251031"
    assert body["task_data"]["task_revision"] == prepare_nooa.TASK_REVISION
    assert "private verifier" not in json.dumps(row)


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "true", '"900"'])
def test_invalid_canonical_timeout_is_rejected(tmp_path: Path, timeout: str) -> None:
    with pytest.raises(ValueError, match="Invalid agent timeout"):
        prepare_nooa.task_row(_task(tmp_path, agent_timeout=timeout))


def test_preparer_materializes_explicit_startup_bound_to_image_and_source(tmp_path: Path) -> None:
    task = _task(tmp_path, name="install-windows-3.11")
    (task / "environment").mkdir()
    dockerfile = task / "environment/Dockerfile"
    dockerfile.write_text('FROM ubuntu:24.04\nCMD ["supervisord","-c","/etc/supervisor/supervisord.conf"]\n')
    row = prepare_nooa.task_row(task)
    assert row["task_input"]["task_data"]["image_startup"] == {
        "docker_image": "alexgshaw/install-windows-3.11:20251031",
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "command": ["supervisord", "-c", "/etc/supervisor/supervisord.conf"],
    }


def test_startup_reader_uses_only_final_stage_and_preserves_exec_form(tmp_path: Path) -> None:
    task = _task(tmp_path)
    (task / "environment").mkdir()
    dockerfile = task / "environment/Dockerfile"
    dockerfile.write_text('FROM ubuntu AS build\nCMD ["build-only"]\nFROM ubuntu\n')
    assert read_image_startup(task) is None
    dockerfile.write_text('FROM ubuntu\nENTRYPOINT ["server"]\nCMD ["--foreground"]\n')
    assert read_image_startup(task).command == ["server", "--foreground"]


@pytest.mark.parametrize("directive", ["CMD shell command", "CMD []", 'CMD ["ok", 5]'])
def test_startup_reader_rejects_unsupported_forms(tmp_path: Path, directive: str) -> None:
    task = _task(tmp_path)
    (task / "environment").mkdir()
    (task / "environment/Dockerfile").write_text(f"FROM ubuntu\n{directive}\n")
    with pytest.raises(ValueError, match="Canonical CMD"):
        prepare_nooa.task_row(task)


def test_preparation_checks_revision_and_count_before_writing(tmp_path: Path, monkeypatch) -> None:
    _task(tmp_path)
    output = tmp_path / "native.jsonl"
    check = Mock(side_effect=[prepare_nooa.TASK_REVISION + "\n", ""])
    monkeypatch.setattr(prepare_nooa.subprocess, "check_output", check)
    with pytest.raises(ValueError, match="Expected 89 unique"):
        prepare_nooa.prepare_native(repository_path=tmp_path, output=output)
    assert not output.exists()
    check.side_effect = ["wrong-revision\n"]
    with pytest.raises(ValueError, match="Expected Terminal-Bench task revision"):
        prepare_nooa.prepare_native(repository_path=tmp_path, output=output)
    check.side_effect = [prepare_nooa.TASK_REVISION + "\n", " M tasks/regex-log/instruction.md\n"]
    with pytest.raises(ValueError, match="modified or untracked"):
        prepare_nooa.prepare_native(repository_path=tmp_path, output=output)
    assert not output.exists()


def test_all_89_rows_are_written_once_with_unique_identity(tmp_path: Path, monkeypatch) -> None:
    for index in range(89):
        _task(tmp_path, name=f"task-{index:02d}")
    monkeypatch.setattr(
        prepare_nooa.subprocess, "check_output", Mock(side_effect=[prepare_nooa.TASK_REVISION + "\n", ""])
    )
    output = prepare_nooa.prepare_native(repository_path=tmp_path, output=tmp_path / "native.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 89
    assert [row["task_id"]["task_id"] for row in rows] == [f"terminal-bench/task-{index:02d}" for index in range(89)]
    assert all(row["task_input"]["agent_timeout_seconds"] == 900 for row in rows)


def test_deployed_paths_preserve_relative_directories_and_all_task_semantics(tmp_path: Path, monkeypatch) -> None:
    for index in range(89):
        _task(tmp_path, name=f"task-{index:02d}")
    check = Mock(side_effect=[prepare_nooa.TASK_REVISION + "\n", ""] * 2)
    monkeypatch.setattr(prepare_nooa.subprocess, "check_output", check)
    original = prepare_nooa.prepare_native(repository_path=tmp_path, output=tmp_path / "original.jsonl")
    deployed_root = Path("/owned/deployment/terminal-bench-2-1")
    deployed = prepare_nooa.prepare_native(
        repository_path=tmp_path,
        output=tmp_path / "deployed.jsonl",
        deployed_repository_path=deployed_root,
    )
    original_rows = [json.loads(line) for line in original.read_text().splitlines()]
    deployed_rows = [json.loads(line) for line in deployed.read_text().splitlines()]
    assert len(original_rows) == len(deployed_rows) == 89
    for index, (before, after) in enumerate(zip(original_rows, deployed_rows, strict=True)):
        expected = str(deployed_root / "tasks" / f"task-{index:02d}")
        assert after["task_input"]["task_data"]["task_folder"] == expected
        before["task_input"]["task_data"]["task_folder"] = expected
        assert before == after  # Identity, prompts, images, startup and both deadlines stay identical.


def test_relative_deployment_root_is_rejected_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "native.jsonl"
    with pytest.raises(ValueError, match="Deployed repository path must be absolute"):
        prepare_nooa.prepare_native(
            repository_path=tmp_path, output=output, deployed_repository_path=Path("relative-checkout")
        )
    assert not output.exists()


def test_nooa_recipe_uses_native_borrowing_one_repeat_and_configurable_provider() -> None:
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from resources_servers.terminal_bench_2_1.app import TerminalBench21ResourcesServerConfig

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(ROOT / "benchmarks/terminal_bench_2_1/nooa.yaml")])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="terminal_bench_2_1_nooa",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config.terminal_bench_2_1_nooa.environment_servers.single_agent_turn, resolve=True),
    )
    agent = config[environment.agent_server.name].responses_api_agents.nooa_agent
    resources = TerminalBench21ResourcesServerConfig(
        name=environment.resources_server.name,
        host="localhost",
        port=8002,
        **OmegaConf.to_container(
            config[environment.resources_server.name].resources_servers.terminal_bench_2_1, resolve=True
        ),
    )
    assert agent.nooa.execution_mode == "sandboxed"
    assert agent.max_policy_calls == 100
    assert agent.context_window == 262144
    assert agent.num_workers == resources.num_workers == 1
    assert resources.sandbox_provider == "sandbox"
    assert resources.sandbox_config["ttl_s"] == 30000
    assert environment.default_episode_timeout_seconds == 30000
    assert not environment.resources_tool_transports
    resource_config = config[environment.resources_server.name].resources_servers.terminal_bench_2_1
    assert resource_config.allowed_agents == ["nooa_agent"]
    assert resource_config.datasets[0].num_repeats == 1
