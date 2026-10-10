# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.harbor.prepare_utils import build_images, provisioning
from nemo_gym import PARENT_DIR
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


HELLO_WORLD_CONFIG = PARENT_DIR / "benchmarks" / "harbor" / "hello_world" / "config.yaml"
SWE_BENCH_CONFIG = PARENT_DIR / "benchmarks" / "harbor" / "swe_bench_verified" / "config.yaml"
OPENCODE_CONFIG = PARENT_DIR / "responses_api_agents" / "harbor_harness_agent" / "configs" / "opencode.yaml"


def record(name: str, *, image: str | None = "env:aaa", builds: bool = True, unsupported=(), **extra) -> dict:
    return {
        "harbor_dataset": "hello_world",
        "task_name": name,
        "instruction": f"Solve {name}.\n",
        "agent_timeout_sec": 120.0,
        "agent_user": None,
        "image": image,
        "builds_image": builds,
        "environment_dir": f"/tasks/{name}/environment",
        "unsupported": list(unsupported),
    } | extra


@pytest.mark.parametrize(
    ("config", "alias", "name"),
    [
        (HELLO_WORLD_CONFIG, "hello_world", "harbor/hello-world"),
        (SWE_BENCH_CONFIG, "swe_bench_verified", "swe-bench/swe-bench-verified"),
    ],
)
def test_settings_come_from_the_benchmark_config(config: Path, alias: str, name: str) -> None:
    settings = provisioning.load_harbor_tasks_settings(config)
    assert list(settings.datasets) == [alias]
    assert settings.datasets[alias]["name"] == name
    # Pinned to one published version, so prepare and the server resolve the same tasks.
    assert settings.datasets[alias]["ref"].startswith("sha256:")
    assert settings.image_template == "nemo-gym-harbor-env:{environment_hash}"


def test_local_dataset_paths_resolve_against_the_repository() -> None:
    settings = provisioning.load_harbor_tasks_settings(
        PARENT_DIR / "resources_servers" / "harbor_tasks" / "configs" / "harbor_tasks.yaml"
    )
    assert settings.datasets["example"]["path"] == str(PARENT_DIR / "resources_servers/harbor_tasks/data/tasks")


def test_task_row_carries_the_instruction_and_agent_settings() -> None:
    row = provisioning.task_row(record("org/task", agent_user="agent"))
    assert row == {
        "task_id": "org/task",
        "harbor_dataset": "hello_world",
        "task_name": "org/task",
        "responses_create_params": {
            "input": [{"role": "user", "content": "Solve org/task.\n"}],
            "metadata": {"harbor_agent_timeout_sec": "120.0", "harbor_agent_user": "agent"},
        },
    }


def test_prepare_rows_writes_supported_tasks_and_reports_images_to_build(monkeypatch, tmp_path, capsys) -> None:
    records = [record("a"), record("b", image="env:aaa"), record("c", unsupported=["docker-compose environments"])]
    monkeypatch.setattr(provisioning, "describe_tasks", lambda settings: records)
    with pytest.raises(ValueError, match="'c' uses unsupported features: docker-compose environments"):
        provisioning.prepare_rows(HELLO_WORLD_CONFIG, tmp_path / "rows.jsonl")

    output = provisioning.prepare_rows(HELLO_WORLD_CONFIG, tmp_path / "rows.jsonl", skip_unsupported=True)
    assert [json.loads(line)["task_name"] for line in output.read_text().splitlines()] == ["a", "b"]
    # Two tasks share one environment, so one image is reported, and nothing is built here.
    assert "1 environment image(s) are built from environment/Dockerfile" in capsys.readouterr().out


def test_build_plan_has_one_entry_per_built_image() -> None:
    plan = build_images.plan_builds(
        [
            record("a", image="env:aaa"),
            record("b", image="env:aaa"),
            record("c", image="env:ccc"),
            record("prebuilt", image="python:3.12", builds=False),
            record("compose", image=None, unsupported=["docker-compose environments"]),
        ]
    )
    assert plan == {"env:aaa": "/tasks/a/environment", "env:ccc": "/tasks/c/environment"}


@pytest.mark.parametrize("dry_run", [False, True])
def test_build_images_builds_only_missing_images(monkeypatch, dry_run: bool) -> None:
    monkeypatch.setattr(
        build_images,
        "describe_tasks",
        lambda settings, **kwargs: [record("a", image="env:a"), record("b", image="env:b")],
    )
    monkeypatch.setattr(build_images, "image_exists", lambda image, push: image == "env:a")
    built = []
    monkeypatch.setattr(build_images, "build_image", lambda image, path, push: built.append((image, push)))
    argv = ["build_images", "--config", str(HELLO_WORLD_CONFIG), "--load", *(["--dry-run"] if dry_run else [])]
    monkeypatch.setattr(sys, "argv", argv)
    build_images.main()
    assert built == ([] if dry_run else [("env:b", False)])


def test_build_images_fails_when_a_build_fails(monkeypatch) -> None:
    monkeypatch.setattr(build_images, "describe_tasks", lambda settings, **kwargs: [record("a", image="env:a")])
    monkeypatch.setattr(build_images, "image_exists", lambda image, push: False)
    monkeypatch.setattr(build_images, "build_image", lambda image, path, push: f"{image}: boom")
    monkeypatch.setattr(sys, "argv", ["build_images", "--config", str(HELLO_WORLD_CONFIG), "--push"])
    with pytest.raises(SystemExit, match="env:a: boom"):
        build_images.main()


def test_benchmark_runs_the_default_harbor_agent() -> None:
    benchmark = BenchmarkConfig.from_config_path(HELLO_WORLD_CONFIG, strict=False)
    assert benchmark.agent_name == "harbor_hello_world_harbor_harness_agent"
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/harbor/hello_world/data/benchmark.jsonl")


def test_agent_type_swaps_another_harbor_agent_onto_the_benchmark() -> None:
    """`--agent-type harbor_harness_agent/opencode` adds the flavor config; it replaces the benchmark's agent."""
    initial = OmegaConf.merge(
        OmegaConf.create({"config_paths": [str(HELLO_WORLD_CONFIG), str(OPENCODE_CONFIG)]}),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    agent = resolved.harbor_hello_world_harbor_harness_agent.responses_api_agents.harbor_harness_agent
    environment = resolved.harbor_hello_world.environment_servers.single_agent_turn_legacy
    assert agent.harbor_agent.name == "opencode"
    assert agent.resources_server.name == "harbor_hello_world_resources_server"
    assert environment.agent_server.name == "harbor_hello_world_harbor_harness_agent"
    assert "harbor_harness_agent_opencode" not in resolved
