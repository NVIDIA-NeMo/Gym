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
from harbor.models.job.config import JobConfig
from omegaconf import OmegaConf
from pydantic import ValidationError

from responses_api_agents.harbor_agent_general.app import GYM_ROOT, HarborAgentConfig, temporary_worker_env


def _config_dict(tmp_path: Path, **job_overrides: object) -> dict:
    return {
        "name": "harbor_agent_general",
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "job": {
            "jobs_dir": tmp_path / "jobs",
            "datasets": [{"path": str(tmp_path / "dataset")}],
            "agents": [{"name": "opencode", "model_name": "test-model"}],
            **job_overrides,
        },
    }


@pytest.mark.parametrize("suffix, expected", [("jobs", "jobs"), ("logs/rollouts.jsonl", "logs/harbor")])
def test_normalize_jobs_dir(tmp_path: Path, suffix: str, expected: str) -> None:
    config = HarborAgentConfig.model_validate(_config_dict(tmp_path, jobs_dir=tmp_path / suffix))
    assert config.job.jobs_dir == tmp_path / expected


def test_gym_defaults_and_native_timeout_defaults(tmp_path: Path) -> None:
    config = HarborAgentConfig.model_validate(_config_dict(tmp_path))
    job = config.build_job_config(task_name="task-001", job_name="t3-r2")

    assert job.job_name == "t3-r2"
    assert job.jobs_dir == tmp_path / "jobs"
    assert job.n_attempts == job.n_concurrent_trials == 1
    assert job.quiet is True
    assert job.retry.max_retries == 0
    assert job.datasets[0].task_names == ["task-001"]
    assert job.environment.delete is True
    assert job.timeout_multiplier == 1.0
    assert job.agent_timeout_multiplier is None
    assert job.verifier_timeout_multiplier is None
    assert job.agent_setup_timeout_multiplier is None
    assert job.environment_build_timeout_multiplier is None
    assert config.harbor_reward_key == "reward"
    assert config.harbor_ray_task_num_cpus == 0.25


def test_native_options_survive_worker_serialization_and_trials_are_isolated(tmp_path: Path) -> None:
    config = HarborAgentConfig.model_validate(
        _config_dict(
            tmp_path,
            timeout_multiplier=1.5,
            agent_timeout_multiplier=2.0,
            verifier_timeout_multiplier=3.0,
            agent_setup_timeout_multiplier=4.0,
            environment_build_timeout_multiplier=5.0,
            quiet=False,
            retry={"max_retries": 3, "max_wait_sec": 17, "exclude_exceptions": ["AgentTimeoutError"]},
            extra_instructions=["Use the provided workspace."],
            artifacts=["/workspace/report.txt"],
            environment={"type": "singularity", "kwargs": {"singularity_force_pull": True}},
            verifier={"override_timeout_sec": 123},
        )
    )
    first = config.build_job_config(task_name="first-task", job_name="t0-r0")
    second = config.build_job_config(task_name="second-task", job_name="t1-r0")
    # Exercise the same serialization boundary as the Ray worker.
    worker_job = JobConfig.model_validate(first.model_dump(mode="json"))
    expected = config.job.model_dump(mode="json")
    expected["job_name"] = "t0-r0"
    expected["datasets"][0]["task_names"] = ["first-task"]
    assert worker_job.model_dump(mode="json") == expected
    assert worker_job.retry.max_wait_sec == 17
    assert worker_job.extra_instructions == ["Use the provided workspace."]
    assert worker_job.artifacts == ["/workspace/report.txt"]
    assert worker_job.environment.kwargs == {"singularity_force_pull": True}
    assert worker_job.verifier.override_timeout_sec == 123
    assert worker_job.timeout_multiplier == 1.5
    assert worker_job.agent_timeout_multiplier == 2.0
    assert worker_job.verifier_timeout_multiplier == 3.0
    assert worker_job.agent_setup_timeout_multiplier == 4.0
    assert worker_job.environment_build_timeout_multiplier == 5.0
    assert worker_job.quiet is False

    first.environment.kwargs["singularity_force_pull"] = False
    first.agents[0].model_name = "changed"
    assert second.environment.kwargs["singularity_force_pull"] is True
    assert second.agents[0].model_name == "test-model"
    assert second.datasets[0].task_names == ["second-task"]
    assert config.job.datasets[0].task_names is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"job_name": "fixed"}, "job.job_name"),
        ({"n_attempts": 2}, "must be 1"),
        ({"n_concurrent_trials": 2}, "must be 1"),
        ({"datasets": []}, "exactly one dataset"),
        ({"datasets": [{"path": "a"}, {"path": "b"}]}, "exactly one dataset"),
        ({"agents": []}, "exactly one dataset and one agent"),
        ({"agents": [{"name": "oracle"}, {"name": "oracle"}]}, "exactly one dataset and one agent"),
        ({"tasks": [{"path": "task"}]}, "per-row execution"),
        ({"source_jobs": [{"action": "regrade", "type": "local", "path": "previous-job"}]}, "per-row execution"),
        ({"install_only": True}, "per-row execution"),
        ({"datasets": [{"path": "tasks", "task_names": ["fixed"]}]}, "controlled by Gym input rows"),
        ({"datasets": [{"path": "tasks", "exclude_task_names": ["fixed"]}]}, "controlled by Gym input rows"),
        ({"datasets": [{"path": "tasks", "n_tasks": 1}]}, "controlled by Gym input rows"),
        ({"environment": {"delete": False}}, "cleans up its environment"),
        ({"retry": {"max_retries": -1}}, "greater than or equal to 0"),
    ],
)
def test_reject_conflicting_or_invalid_native_settings(tmp_path: Path, overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        HarborAgentConfig.model_validate(_config_dict(tmp_path, **overrides))


def test_native_model_input_is_not_mutated(tmp_path: Path) -> None:
    raw = _config_dict(tmp_path)
    native = JobConfig.model_validate(raw["job"])
    original = native.model_dump()
    config = HarborAgentConfig.model_validate(raw | {"job": native})
    assert config.job.n_concurrent_trials == 1
    assert native.model_dump() == original


def test_legacy_config_migrates_to_equivalent_native_job(tmp_path: Path) -> None:
    raw = _config_dict(tmp_path)
    native = raw.pop("job")
    legacy = raw | {
        "harbor_jobs_dir": native["jobs_dir"],
        "harbor_dataset": native["datasets"][0],
        "harbor_agent": native["agents"][0],
        "harbor_environment": {"type": "singularity"},
        "harbor_verifier": {"override_timeout_sec": 123},
        "harbor_debug": True,
        "harbor_max_retries": 3,
        "harbor_reward_key": "score",
    }
    config = HarborAgentConfig.model_validate(legacy)
    expected = HarborAgentConfig.model_validate(
        _config_dict(
            tmp_path,
            environment={"type": "singularity"},
            verifier={"override_timeout_sec": 123},
            debug=True,
            retry={"max_retries": 3},
        )
    )
    assert config.build_job_config("task", "job") == expected.build_job_config("task", "job")
    assert config.job.debug is True
    assert config.harbor_reward_key == "score"
    assert "harbor_dataset" in legacy  # Validation must not consume the caller's input.
    assert "job" not in legacy


def test_reject_mixed_native_and_legacy_settings(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not both"):
        HarborAgentConfig.model_validate(_config_dict(tmp_path) | {"harbor_debug": True})


@pytest.mark.parametrize(
    "filename, server_name, environment_type",
    [
        ("responses_api_agents/harbor_agent_general/configs/docker_opencode.yaml", "harbor_agent_general", "docker"),
        (
            "responses_api_agents/harbor_agent_general/configs/singularity_opencode.yaml",
            "harbor_agent_general",
            "singularity",
        ),
        ("benchmarks/biomnibench_da/config.yaml", "biomnibench_da", "singularity"),
        ("benchmarks/genebench_pro/config.yaml", "genebench_pro", "singularity"),
        ("environments/labbench2_pdf/config.yaml", "harbor_agent_general", "docker"),
        ("environments/labbench2_pdf/config_singularity.yaml", "harbor_agent_general", "singularity"),
    ],
)
def test_shipped_configs_resolve_native_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, server_name: str, environment_type: str
) -> None:
    monkeypatch.setenv("SINGULARITY_CACHEDIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SINGULARITY_FORCE_PULL", raising=False)
    repo = Path(__file__).resolve().parents[3]
    merged = OmegaConf.merge(
        OmegaConf.load(repo / filename),
        {
            "harbor_jobs_dir": str(tmp_path / "jobs"),
            "harbor_dataset_path": str(tmp_path / "dataset"),
            "policy_model_name": "test-model",
            "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl"),
        },
    )
    settings = OmegaConf.to_container(merged[server_name].responses_api_agents.harbor_agent_general, resolve=True)
    config = HarborAgentConfig.model_validate(settings | {"name": server_name, "host": "0.0.0.0", "port": 8080})
    job = config.build_job_config("selected-task", "t0-r0")
    assert job.datasets[0].task_names == ["selected-task"]
    assert job.environment.type == environment_type
    assert "job" in settings
    if filename.startswith("environments/labbench2_pdf/"):
        assert job.agents[0].model_name == "openai/test-model"
        assert job.jobs_dir == repo / "results/labbench2_pdf/harbor_jobs"
        dataset_name = "tasks_docker" if environment_type == "docker" else "tasks_singularity"
        assert job.datasets[0].path == repo / "environments/labbench2_pdf/data" / dataset_name
        assert job.job_name.startswith("t0-r0-")
        assert job.agents[0].override_setup_timeout_sec == 900
        assert job.agents[0].env["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"
        assert config.harbor_worker_env["JUDGE_MODEL"] == ""
    else:
        assert job.agents[0].model_name == "test-model"
    if server_name in {"biomnibench_da", "genebench_pro"}:
        assert job.retry.max_retries == 3
        assert job.jobs_dir == tmp_path / "harbor"
    if server_name == "biomnibench_da":
        assert config.harbor_reward_key == "score"


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


def test_relative_paths_keep_existing_cwd_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, Path("relative/jobs"), harbor_dataset={"path": "relative/tasks"})

    assert config.job.jobs_dir == tmp_path / "relative/jobs"
    assert config.job.datasets[0].path == Path("relative/tasks")


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
    config = HarborAgentConfig.model_validate(
        values | {"name": "harbor_agent_general", "host": "0.0.0.0", "port": 8080}
    )

    assert config.job.jobs_dir == tmp_path / "relative/jobs"
    assert config.job.datasets[0].path == Path("relative/tasks")
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

    assert config.job.jobs_dir == GYM_ROOT / "relative/jobs"
    assert config.job.datasets[0].path == GYM_ROOT / "relative/tasks"


@pytest.mark.parametrize("from_gym_root", [False, True])
def test_absolute_paths_are_unchanged(tmp_path: Path, from_gym_root: bool) -> None:
    config = _make_config(tmp_path, tmp_path / "jobs", harbor_resolve_paths_from_gym_root=from_gym_root)
    assert config.job.jobs_dir == tmp_path / "jobs"
    assert config.job.datasets[0].path == tmp_path / "dataset"


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
