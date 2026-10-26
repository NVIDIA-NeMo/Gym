# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from benchmarks.openthoughts_tblite import prepare as preparation
from nemo_gym.environment.manifest import load_manifest, resolve_manifest_config_path
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


BENCHMARK_DIR = Path(__file__).resolve().parents[1]

TASK_TOMLS = {
    # [environment] between other tables, with a comment on its header
    "alpha": (
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 600.0\n\n[environment]  # sandbox\nbuild_timeout_sec = 600.0\n'
        "cpus = 2\nmemory_mb = 4096\n\n[agent]\ntimeout_sec = 900.0\n"
    ),
    # no [environment] at all, as 40 of the tasks
    "beta": 'version = "1.0"\n\n[verifier]\ntimeout_sec = 60.0\n\n[agent]\ntimeout_sec = 3600.0\n',
    # [environment] last
    "gamma": "[agent]\ntimeout_sec = 300\n\n[environment]\nbuild_timeout_sec = 300\n",
}


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def pinned_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A task repository pinned like the real one; returns its checkout."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--quiet")
    git(origin, "config", "user.name", "Preparation Test")
    git(origin, "config", "user.email", "test@example.invalid")
    for name, task_toml in TASK_TOMLS.items():
        (origin / name / "tests").mkdir(parents=True)
        (origin / name / "environment").mkdir()
        (origin / name / "task.toml").write_text(task_toml)
        (origin / name / "instruction.md").write_text(f"Solve {name}.\nKeep the café heading.\n\n", encoding="utf-8")
        (origin / name / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        (origin / name / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (origin / "README.md").write_text("Not a task.\n")
    git(origin, "add", ".")
    git(origin, "commit", "--quiet", "-m", "Fixture tasks")
    gym_root = tmp_path / "gym"
    monkeypatch.setattr(preparation, "GYM_ROOT", gym_root)
    monkeypatch.setattr(preparation, "TASKS_DIR", gym_root / "benchmarks/openthoughts_tblite/data/tasks")
    monkeypatch.setattr(preparation, "OUTPUT_PATH", gym_root / "benchmarks/openthoughts_tblite/data/benchmark.jsonl")
    monkeypatch.setattr(preparation, "TASK_REPOSITORY", str(origin))
    monkeypatch.setattr(preparation, "TASK_REVISION", git(origin, "rev-parse", "HEAD"))
    monkeypatch.setattr(preparation, "DOCKER_IMAGE", "example.invalid/tblite:{task}")
    monkeypatch.setattr(preparation, "NUM_TASKS", len(TASK_TOMLS))
    return origin


def test_prepare_copies_pinned_tasks_and_adds_only_their_images(pinned_tasks: Path) -> None:
    output = preparation.prepare()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": f"Solve {name}.\nKeep the café heading.\n\n"}]
            },
            "task_id": name,
            "task_name": name,
            "docker_image": f"example.invalid/tblite:{name}",
            "task_folder": f"benchmarks/openthoughts_tblite/data/tasks/{name}",
        }
        for name in TASK_TOMLS
    ]
    for name, task_toml in TASK_TOMLS.items():
        copied = preparation.TASKS_DIR / name
        expected = tomllib.loads(task_toml)
        expected.setdefault("environment", {})["docker_image"] = f"example.invalid/tblite:{name}"
        # The task keeps its own agent and verifier limits; it only gains its image.
        assert tomllib.loads((copied / "task.toml").read_text()) == expected
        for relative in ("instruction.md", "tests/test.sh", "environment/Dockerfile"):
            assert (copied / relative).read_bytes() == (pinned_tasks / name / relative).read_bytes()
    assert sorted(path.name for path in preparation.TASKS_DIR.iterdir()) == [".source", *TASK_TOMLS]
    assert preparation.is_prepared_data_current(output)


def test_prepare_reuses_current_tasks_until_the_pin_changes(
    pinned_tasks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = preparation.prepare()
    original = output.read_bytes()
    output.write_text("stale cached inputs\n")
    fetches = []
    fetch = preparation._fetch_tasks
    monkeypatch.setattr(preparation, "_fetch_tasks", lambda: fetches.append(1) or fetch())

    preparation.prepare()
    assert fetches == []
    assert output.read_bytes() == original

    monkeypatch.setattr(preparation, "DOCKER_IMAGE", "example.invalid/other:{task}")
    assert not preparation.is_prepared_data_current(output)
    preparation.prepare()
    assert fetches == [1]
    assert json.loads(output.read_text().splitlines()[0])["docker_image"] == "example.invalid/other:alpha"
    assert preparation.is_prepared_data_current(output)


def test_wrong_task_count_keeps_the_previous_tasks(pinned_tasks: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = preparation.prepare()
    original = output.read_bytes()
    monkeypatch.setattr(preparation, "DOCKER_IMAGE", "example.invalid/other:{task}")
    monkeypatch.setattr(preparation, "NUM_TASKS", len(TASK_TOMLS) + 1)

    with pytest.raises(ValueError, match="Expected 4 tasks"):
        preparation.prepare()
    assert output.read_bytes() == original
    assert "example.invalid/tblite:alpha" in (preparation.TASKS_DIR / "alpha/task.toml").read_text()
    assert [path.name for path in preparation.TASKS_DIR.parent.iterdir() if path.name.startswith("tblite-")] == []


def test_a_task_that_already_names_an_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="already names an image"):
        preparation._with_docker_image('[environment]\ndocker_image = "x"\n', "y")


def _resolve(*config_paths: str, **values: object) -> dict:
    initial = {
        "config_paths": [f"benchmarks/openthoughts_tblite/{config_paths[0]}", *config_paths[1:]],
        "policy_base_url": "http://model.invalid:8000/v1",
        "policy_api_key": "unused",
        "policy_model_name": "offline-model",
        **values,
    }
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(initial),
        )
    )
    return OmegaConf.to_container(config, resolve=True)


DATASETS = [
    {
        "name": "openthoughts_tblite",
        "type": "benchmark",
        "jsonl_fpath": "benchmarks/openthoughts_tblite/data/benchmark.jsonl",
        "prepare_script": "benchmarks/openthoughts_tblite/prepare.py",
        "num_repeats": 3,
    }
]


def _harbor_agent(config: dict) -> dict:
    environment = config["openthoughts_tblite_environment_server"]["environment_servers"]["legacy_agent"]
    assert environment["agent_server"]["name"] == "openthoughts_tblite"
    return config["openthoughts_tblite"]["responses_api_agents"]["harbor_agent_general"]


def test_harbor_profile_runs_each_task_with_its_own_limits() -> None:
    agent = _harbor_agent(_resolve("config.yaml", output_jsonl_fpath="/results/run/rollouts.jsonl"))
    assert agent["datasets"] == DATASETS
    assert agent["harbor_resolve_paths_from_gym_root"] is True
    assert agent["harbor_unique_job_names"] is True
    assert agent["harbor_dataset"] == {"path": "benchmarks/openthoughts_tblite/data/tasks"}
    assert agent["harbor_jobs_dir"] == "/results/run/rollouts.jsonl"
    assert agent["harbor_max_retries"] == 0
    assert agent["harbor_environment"] == {"type": "docker"}
    harbor_agent = agent["harbor_agent"]
    assert harbor_agent["name"] == "terminus-2"
    assert harbor_agent["model_name"] == "hosted_vllm/offline-model"
    assert harbor_agent["kwargs"]["api_base"] == "http://model.invalid:8000/v1"
    # The key reaches LiteLLM through the worker's environment, not Harbor's job config.
    assert agent["harbor_worker_env"] == {"HOSTED_VLLM_API_KEY": "unused"}
    assert "api_key" not in json.dumps(harbor_agent)
    # No time override anywhere: Harbor applies each task.toml's own limits.
    assert "override_timeout_sec" not in harbor_agent
    assert "harbor_verifier" not in agent
    assert set(harbor_agent["kwargs"]["model_info"]) == {
        "max_input_tokens",
        "max_output_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
    }


def test_harbor_profile_builds_a_harbor_job_for_one_task(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    from responses_api_agents.harbor_agent_general.app import GYM_ROOT, HarborAgentConfig

    agent = _harbor_agent(_resolve("config.yaml", output_jsonl_fpath=str(tmp_path / "rollouts.jsonl")))
    config = HarborAgentConfig.model_validate({"name": "openthoughts_tblite", "host": "127.0.0.1", "port": 1, **agent})
    job = config.build_job_config("amuse-install", "t0-r0")
    assert job.jobs_dir == tmp_path / "harbor"
    assert job.datasets[0].path == GYM_ROOT / "benchmarks/openthoughts_tblite/data/tasks"
    assert job.datasets[0].task_names == ["amuse-install"]
    assert job.environment.type == "docker"
    assert job.agents[0].name == "terminus-2"
    assert job.agents[0].override_timeout_sec is None
    assert job.timeout_multiplier == 1.0
    assert "unused" not in job.model_dump_json()


def test_harbor_profile_gives_every_attempt_its_own_job_folder(tmp_path: Path) -> None:
    """Two runs writing next to each other (model-a.jsonl, model-b.jsonl) share one harbor/ folder; neither may
    pick up the other's trial for the same task and attempt."""
    pytest.importorskip("harbor")
    from responses_api_agents.harbor_agent_general.app import HarborAgentConfig

    jobs = []
    for output in ("model-a.jsonl", "model-b.jsonl"):
        agent = _harbor_agent(_resolve("config.yaml", output_jsonl_fpath=str(tmp_path / output)))
        config = HarborAgentConfig.model_validate(
            {"name": "openthoughts_tblite", "host": "127.0.0.1", "port": 1, **agent}
        )
        jobs.append(config.build_job_config("amuse-install", "t0-r0"))
    first, second = jobs
    assert first.jobs_dir == second.jobs_dir == tmp_path / "harbor"
    assert first.job_name != second.job_name
    assert first.job_name.startswith("t0-r0-") and second.job_name.startswith("t0-r0-")


def test_terminus_2_profile_runs_the_same_rows_on_the_terminal_bench_server() -> None:
    # `--model-type vllm_model` adds the model server that Gym's Terminus-2 calls.
    config = _resolve("terminus_2.yaml", "responses_api_models/vllm_model/configs/vllm_model.yaml")
    environment = config["openthoughts_tblite_environment_server"]["environment_servers"]["legacy_agent"]
    assert environment["agent_server"]["name"] == "openthoughts_tblite_terminus_2_sandboxed_agent"
    agent = config["openthoughts_tblite_terminus_2_sandboxed_agent"]["responses_api_agents"][
        "terminus_2_sandboxed_agent"
    ]
    assert agent["datasets"] == DATASETS
    assert agent["resources_server"]["name"] == "openthoughts_tblite_terminus_2_sandboxed_resources_server"
    resources = config[agent["resources_server"]["name"]]["resources_servers"]["terminal_bench_2_1"]
    assert "terminus_2_sandboxed_agent" in resources["allowed_agents"]
    assert resources["sandbox_config"]["metadata"]["benchmark"] == "openthoughts-tblite"
    # One limit for every task: TB-Lite's longest agent and verifier limits.
    assert agent["sandbox_timeout"] == resources["evaluation_timeout"] == 3600
    assert resources["is_verifying_golden_patch"] is False


@pytest.mark.parametrize("provider", ["docker", "opensandbox"])
def test_terminus_2_profile_uses_the_one_provider_the_run_names(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README's command adds a provider config; agent and resources server both resolve to it alone."""
    from nemo_gym.sandbox.config import resolve_provider_config

    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")
    config = _resolve(
        "terminus_2.yaml",
        "responses_api_models/vllm_model/configs/vllm_model.yaml",
        f"nemo_gym/sandbox/providers/{provider}/configs/{provider}.yaml",
    )
    agent = config["openthoughts_tblite_terminus_2_sandboxed_agent"]["responses_api_agents"][
        "terminus_2_sandboxed_agent"
    ]
    resources = config[agent["resources_server"]["name"]]["resources_servers"]["terminal_bench_2_1"]
    for server in (agent, resources):
        assert list(resolve_provider_config(server["sandbox_provider"], config)) == [provider]


@pytest.mark.parametrize(
    ("manifest_path", "config_name", "agent_server"),
    [
        ("manifest.yaml", "config.yaml", "harbor_agent_general"),
        ("terminus_2/manifest.yaml", "terminus_2.yaml", "terminus_2_sandboxed_agent"),
    ],
)
def test_manifests_match_their_configs(manifest_path: str, config_name: str, agent_server: str) -> None:
    manifest = load_manifest(BENCHMARK_DIR / manifest_path)
    assert resolve_manifest_config_path(BENCHMARK_DIR / manifest_path, manifest) == BENCHMARK_DIR / config_name
    assert manifest.agent_server == agent_server
    assert [dataset.model_dump(exclude_none=True) for dataset in manifest.datasets] == DATASETS
    assert yaml.safe_load((BENCHMARK_DIR / ".gitignore").read_text()) == "data/"
