# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import DatasetConfig
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import VLLMConverter
from responses_api_agents.harbor_agent.task_data import TaskData


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "responses_api_agents/harbor_agent/example/onboarding"
TASK = RECIPE / "tasks/write-answer"
HARBOR_REVISION = "9dddd797b57ab8a0f9d6352a20fce73abbb29573"
UBUNTU_DIGEST = "sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254"


def _run_script(script: Path, *args: Path) -> None:
    assert script.is_file(), f"Missing executable task fixture: {script}"
    subprocess.run(["bash", str(script), *map(str, args)], check=True, capture_output=True, text=True, timeout=5)


def test_oracle_produces_verifier_reward_one(tmp_path: Path) -> None:
    answer = tmp_path / "answer.txt"
    rewards = tmp_path / "verifier"
    _run_script(TASK / "solution/solve.sh", answer)
    _run_script(TASK / "tests/test.sh", answer, rewards)
    assert answer.read_bytes() == b"42\n"
    assert (rewards / "reward.txt").read_text() == "1.0\n"


@pytest.mark.parametrize("answer_text", ["41\n", "42", "42\nextra\n", ""])
def test_wrong_answer_produces_verifier_reward_zero(tmp_path: Path, answer_text: str) -> None:
    answer = tmp_path / "answer.txt"
    answer.write_text(answer_text)
    rewards = tmp_path / "verifier"
    _run_script(TASK / "tests/test.sh", answer, rewards)
    assert (rewards / "reward.txt").read_text() == "0.0\n"


def test_missing_answer_produces_verifier_reward_zero(tmp_path: Path) -> None:
    _run_script(TASK / "tests/test.sh", tmp_path / "missing.txt", tmp_path / "verifier")
    assert (tmp_path / "verifier/reward.txt").read_text() == "0.0\n"


def test_recipe_routes_one_tracked_task_and_bounds_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GYM_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("GYM_HARBOR_RUN_DIR", str(tmp_path))
    config = OmegaConf.to_container(OmegaConf.load(RECIPE / "harbor_onboarding.yaml"), resolve=True)
    agent = config["harbor_onboarding"]["responses_api_agents"]["harbor_agent"]
    datasets = [DatasetConfig.model_validate(data) for data in agent["datasets"]]
    assert len(datasets) == 1
    dataset = datasets[0]
    assert dataset.type == "validation"
    assert dataset.num_repeats == 1
    rows = [json.loads(line) for line in Path(dataset.jsonl_fpath).read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    TaskData.model_validate(row)
    assert row["instance_id"] == "onboarding::write-answer"
    assert "agent_ref" not in row  # Dataset collation injects task_source; dispatch resolves the owning agent.
    assert row["responses_create_params"]["max_output_tokens"] == 512
    assert Path(agent["harbor_datasets"]["onboarding"]["local_dataset_path"]) / "write-answer" == TASK
    assert agent["concurrency"] == 1
    assert agent["harbor_environment_type"] == "docker"
    assert agent["harbor_no_delete"] is False
    assert agent["harbor_jobs_dir"] == str(tmp_path / "native-jobs")
    assert agent["harbor_agent_kwargs"]["max_turns"] == 3
    assert agent["harbor_agent_kwargs"]["enable_summarize"] is False
    assert agent["harbor_agent_kwargs"]["nemo_model_server_timeout_sec"] <= 60
    assert agent["harbor_agent_max_timeout"] <= 180
    assert agent["harbor_verifier_max_timeout"] <= 10
    assert agent["harbor_timeout_multiplier"] == 1.0


def test_task_has_standard_harbor_layout_and_no_runtime_network() -> None:
    task = tomllib.loads((TASK / "task.toml").read_text())
    for relative in ["instruction.md", "environment/Dockerfile", "solution/solve.sh", "tests/test.sh"]:
        assert (TASK / relative).is_file()
    assert task["version"] == "1.0"
    assert task["agent"]["timeout_sec"] <= 180
    assert task["verifier"]["timeout_sec"] <= 10
    assert task["environment"]["build_timeout_sec"] <= 300
    assert task["environment"]["allow_internet"] is False
    assert task["environment"]["cpus"] == 1
    assert task["environment"]["gpus"] == 0
    dockerfile = (TASK / "environment/Dockerfile").read_text()
    assert f"FROM ubuntu:24.04@{UBUNTU_DIGEST}" in dockerfile
    assert "tmux" in dockerfile
    assert "asciinema" in dockerfile
    assert "COPY" not in dockerfile  # Neither the oracle nor verifier belongs in the agent's image.
    requirements = (REPO_ROOT / "responses_api_agents/harbor_agent/requirements.txt").read_text()
    assert f"harbor @ git+https://github.com/laude-institute/harbor.git@{HARBOR_REVISION}" in requirements


def test_recipe_token_limit_survives_chat_conversion() -> None:
    row = json.loads((RECIPE / "input.jsonl").read_text())
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    chat = VLLMConverter(return_token_id_information=True).responses_to_chat_completion_create_params(params)
    assert chat.max_tokens == 512
    assert chat.temperature == 0


def test_task_parses_with_optional_pinned_harbor() -> None:
    pytest.importorskip("harbor")
    from harbor.models.task.task import Task

    assert version("harbor") == "0.1.42", "Run this integration check with harbor_agent/requirements.txt."
    task = Task(TASK)
    assert task.config.environment.allow_internet is False
    assert task.config.agent.timeout_sec == 180
    assert task.config.verifier.timeout_sec == 10
    assert task.paths.is_valid()
    assert task.name == "write-answer"
    assert len(task.checksum) == 64


def test_recipe_builds_optional_pinned_harbor_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    from harbor.models.job.config import JobConfig

    from responses_api_agents.harbor_agent.app import HarborAgent, HarborAgentConfig

    monkeypatch.setenv("GYM_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("GYM_HARBOR_RUN_DIR", str(tmp_path))
    config = OmegaConf.to_container(OmegaConf.load(RECIPE / "harbor_onboarding.yaml"), resolve=True)
    agent_config = config["harbor_onboarding"]["responses_api_agents"]["harbor_agent"]
    server = HarborAgent.model_construct(
        config=HarborAgentConfig(name="harbor_onboarding", host="127.0.0.1", port=8080, **agent_config)
    )
    row = json.loads((RECIPE / "input.jsonl").read_text())
    job = JobConfig.model_validate(
        server._build_job_config(
            dataset_alias="onboarding",
            task_name="write-answer",
            model_name="approved-test-model",
            api_base="http://127.0.0.1:9000/v1",
            job_name="offline-config-check",
            jobs_dir=tmp_path / "native-jobs",
            responses_create_params=row["responses_create_params"],
        )
    )
    assert job.environment.delete is True
    assert job.environment.kwargs == {"workdir": "/app"}
    assert job.orchestrator.n_concurrent_trials == 1
    assert job.orchestrator.retry.max_retries == 0
    assert job.n_attempts == 1
    assert job.datasets[0].path == TASK.parent
    assert job.datasets[0].task_names == ["write-answer"]
    assert job.agents[0].max_timeout_sec == 180
    assert job.agents[0].kwargs["responses_create_params"]["max_output_tokens"] == 512
    assert job.verifier.max_timeout_sec == 10


def test_recipe_collates_validation_split_without_servers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nemo_gym.train_data_utils import TrainDataProcessor, TrainDataProcessorConfig

    monkeypatch.setenv("GYM_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("GYM_HARBOR_RUN_DIR", str(tmp_path))
    config = OmegaConf.load(RECIPE / "harbor_onboarding.yaml")
    # Collation writes per-dataset sidecars: use a temporary copy, never mutate tracked fixture data.
    input_path = tmp_path / "input.jsonl"
    input_path.write_bytes((RECIPE / "input.jsonl").read_bytes())
    config.harbor_onboarding.responses_api_agents.harbor_agent.datasets[0].jsonl_fpath = str(input_path)
    config.output_dirpath = str(tmp_path / "collated")
    config.mode = "train_preparation"
    config.should_download = False
    config.task_data_validation = "error"
    processor = TrainDataProcessor()
    processor_config = TrainDataProcessorConfig.model_validate(config)
    instances = processor.load_and_validate_server_instance_configs(processor_config, config)
    processor.load_datasets(processor_config, instances)
    metrics = processor.validate_samples_and_aggregate_metrics(instances, overwrite_metrics_conflicts=False)
    processor.collate_samples(processor_config, instances, metrics)
    rows = [json.loads(line) for line in (tmp_path / "collated/validation.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["task_source"] == "harbor_onboarding"
    assert rows[0]["instance_id"] == "onboarding::write-answer"
    assert rows[0]["responses_create_params"]["max_output_tokens"] == 512
    assert "agent_ref" not in rows[0]


def test_prefetch_preserves_installed_harbor_but_provisions_missing_model(tmp_path: Path) -> None:
    from nemo_gym.cli.setup_command import get_venv_path, setup_env_command

    config = OmegaConf.create(
        {
            "uv_venv_dir": str(tmp_path / "venvs"),
            "python_version": "3.13.14",
            "skip_venv_if_present": True,
            "head_server_deps": ["ray[default]==2.56.1", "openai==2.44.0"],
            "pip_install_verbose": False,
        }
    )
    harbor_dir = REPO_ROOT / "responses_api_agents/harbor_agent"
    harbor_venv = get_venv_path(harbor_dir, config)
    (harbor_venv / "bin").mkdir(parents=True)
    (harbor_venv / "bin/python").touch()
    (harbor_venv / "bin/activate").touch()
    harbor_command = setup_env_command(harbor_dir, config, "harbor_onboarding")
    assert str(harbor_venv / "bin/activate") in harbor_command
    assert "uv pip install" not in harbor_command
    assert "uv venv" not in harbor_command

    model_dir = REPO_ROOT / "responses_api_models/vllm_model"
    model_command = setup_env_command(model_dir, config, "policy_model")
    assert str(get_venv_path(model_dir, config)) in model_command
    assert "uv pip install" in model_command
    assert "ray[default]==2.56.1" in model_command
    assert "openai==2.44.0" in model_command
