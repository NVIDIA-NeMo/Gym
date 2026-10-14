# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from responses_api_agents.harbor_agent_general.app import GYM_ROOT, HarborAgentConfig, temporary_worker_env


def _make_config(tmp_path: Path, harbor_jobs_dir: Path, **config_overrides: object) -> HarborAgentConfig:
    config = {
        "name": "harbor_agent_general",
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "harbor_jobs_dir": harbor_jobs_dir,
        "harbor_dataset": {"path": str(tmp_path / "dataset")},
        "harbor_environment": {"type": "docker"},
        "harbor_agent": {"name": "opencode", "model_name": "test-model"},
    }
    config.update(config_overrides)
    return HarborAgentConfig.model_validate(config)


def test_normalize_jobs_dir_keeps_directory_path(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"

    config = _make_config(tmp_path, jobs_dir)

    assert config.harbor_jobs_dir == jobs_dir.resolve()


def test_normalize_jobs_dir_maps_jsonl_path_to_harbor_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "logs" / "rollouts.jsonl"

    config = _make_config(tmp_path, input_path)

    assert config.harbor_jobs_dir == (tmp_path / "logs" / "harbor").resolve()


def test_relative_paths_keep_existing_cwd_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, Path("relative/jobs"), harbor_dataset={"path": "relative/tasks"})

    assert config.harbor_jobs_dir == tmp_path / "relative/jobs"
    assert config.harbor_dataset.path == Path("relative/tasks")


@pytest.mark.parametrize("profile", ["docker_opencode.yaml", "singularity_opencode.yaml"])
def test_generic_profiles_keep_path_and_resume_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SINGULARITY_CACHEDIR", str(tmp_path / "cache"))
    config_path = GYM_ROOT / "responses_api_agents/harbor_agent_general/configs" / profile
    source = OmegaConf.merge(
        OmegaConf.load(config_path),
        {"harbor_jobs_dir": "relative/jobs", "harbor_dataset_path": "relative/tasks", "policy_model_name": "test"},
    )
    values = OmegaConf.to_container(
        source.harbor_agent_general.responses_api_agents.harbor_agent_general, resolve=True
    )
    assert isinstance(values, dict)
    config = _make_config(tmp_path, Path(values.pop("harbor_jobs_dir")), **values)

    assert config.harbor_jobs_dir == tmp_path / "relative/jobs"
    assert config.harbor_dataset.path == Path("relative/tasks")
    assert config.harbor_resolve_paths_from_gym_root is False
    assert config.harbor_unique_job_names is False
    assert config.build_job_config(task_name="task", job_name="t0-r0").job_name == "t0-r0"


def test_relative_paths_opt_in_to_gym_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        tmp_path,
        Path("relative/jobs"),
        harbor_dataset={"path": "relative/tasks"},
        harbor_resolve_paths_from_gym_root=True,
    )

    assert config.harbor_jobs_dir == GYM_ROOT / "relative/jobs"
    assert config.harbor_dataset.path == GYM_ROOT / "relative/tasks"


@pytest.mark.parametrize("from_gym_root", [False, True])
def test_absolute_paths_are_unchanged(tmp_path: Path, from_gym_root: bool) -> None:
    config = _make_config(tmp_path, tmp_path / "jobs", harbor_resolve_paths_from_gym_root=from_gym_root)
    assert config.harbor_jobs_dir == tmp_path / "jobs"
    assert config.harbor_dataset.path == tmp_path / "dataset"


def test_unique_jobs_preserve_prior_rollout_directories(tmp_path: Path) -> None:
    config = _make_config(tmp_path, tmp_path / "jobs", harbor_unique_job_names=True)
    first = config.build_job_config(task_name="task-001", job_name="t0-r0")
    first_dir = first.jobs_dir / first.job_name
    first_dir.mkdir(parents=True)
    evidence = first_dir / "result.json"
    evidence.write_text("original evidence")

    second = config.build_job_config(task_name="task-001", job_name="t0-r0")
    second_dir = second.jobs_dir / second.job_name
    second_dir.mkdir()  # A repeated rollout must not collide with the earlier job.

    assert first_dir != second_dir
    assert first.job_name.startswith("t0-r0-")
    assert second.job_name.startswith("t0-r0-")
    assert evidence.read_text() == "original evidence"


def test_reward_key_defaults_to_reward_and_accepts_override(tmp_path: Path) -> None:
    assert _make_config(tmp_path, tmp_path / "default").harbor_reward_key == "reward"
    assert _make_config(tmp_path, tmp_path / "custom", harbor_reward_key="score").harbor_reward_key == "score"


def test_worker_environment_is_empty_by_default_and_accepts_values(tmp_path: Path) -> None:
    default_config = _make_config(tmp_path, tmp_path / "default")
    configured = _make_config(
        tmp_path,
        tmp_path / "configured",
        harbor_worker_env={"OPENAI_API_KEY": "test-key", "JUDGE_MODEL": "test-judge"},
    )

    assert default_config.harbor_worker_env == {}
    assert configured.harbor_worker_env == {
        "OPENAI_API_KEY": "test-key",
        "JUDGE_MODEL": "test-judge",
    }


def test_temporary_worker_environment_restores_process_state(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING_WORKER_VALUE", "before")
    monkeypatch.delenv("NEW_WORKER_VALUE", raising=False)

    with temporary_worker_env({"EXISTING_WORKER_VALUE": "during", "NEW_WORKER_VALUE": "temporary"}):
        assert os.environ["EXISTING_WORKER_VALUE"] == "during"
        assert os.environ["NEW_WORKER_VALUE"] == "temporary"

    assert os.environ["EXISTING_WORKER_VALUE"] == "before"
    assert "NEW_WORKER_VALUE" not in os.environ


def test_build_job_config_applies_single_trial_defaults(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    config = _make_config(tmp_path, jobs_dir)

    job_config = config.build_job_config(task_name="task-001", job_name="t3-r2")

    assert job_config.job_name == "t3-r2"
    assert job_config.jobs_dir == jobs_dir.resolve()
    assert job_config.n_attempts == 1
    assert job_config.n_concurrent_trials == 1
    assert job_config.retry.max_retries == 0
    assert job_config.datasets[0].task_names == ["task-001"]
    assert job_config.environment.delete is True
    assert job_config.agents == [config.harbor_agent]
    assert job_config.verifier == config.harbor_verifier


def test_build_job_config_preserves_harbor_config_overrides(tmp_path: Path) -> None:
    config = _make_config(tmp_path, tmp_path / "jobs")
    config.harbor_dataset = config.harbor_dataset.model_copy(update={"task_names": ["original"]})
    config.harbor_environment = config.harbor_environment.model_copy(
        update={"type": "singularity", "kwargs": {"singularity_force_pull": True}}
    )
    config.harbor_agent = config.harbor_agent.model_copy(update={"model_name": "custom-model"})

    job_config = config.build_job_config(task_name="task-002", job_name="job-002")

    assert job_config.datasets[0].task_names == ["task-002"]
    assert job_config.environment.type == "singularity"
    assert job_config.environment.kwargs == {"singularity_force_pull": True}
    assert job_config.agents[0].model_name == "custom-model"
