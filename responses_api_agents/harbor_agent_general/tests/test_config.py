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

from pathlib import Path

from responses_api_agents.harbor_agent_general.app import HarborAgentConfig


def _make_config(tmp_path: Path, harbor_jobs_dir: Path) -> HarborAgentConfig:
    return HarborAgentConfig(
        name="harbor_agent_general",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        harbor_jobs_dir=harbor_jobs_dir,
        harbor_dataset={"path": str(tmp_path / "dataset")},
        harbor_environment={"type": "docker"},
        harbor_agent={"name": "opencode", "model_name": "test-model"},
    )


def test_normalize_jobs_dir_keeps_directory_path(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"

    config = _make_config(tmp_path, jobs_dir)

    assert config.harbor_jobs_dir == jobs_dir.resolve()


def test_normalize_jobs_dir_maps_jsonl_path_to_harbor_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "logs" / "rollouts.jsonl"

    config = _make_config(tmp_path, input_path)

    assert config.harbor_jobs_dir == (tmp_path / "logs" / "harbor").resolve()


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
