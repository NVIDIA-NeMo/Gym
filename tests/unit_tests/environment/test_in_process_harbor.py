# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`plan_benchmark` and `serve_benchmark` for a benchmark whose config sets `harbor_target`."""

import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig, ConfigError
from nemo_gym.data.tasks.harbor.hub import DATASETS_DIR_ENV
from nemo_gym.environment.in_process import PlannedBenchmark, TaskRunner, plan_benchmark, serve_benchmark
from tests.unit_tests.data.tasks.harbor.harbor_build_helpers import PULL_DOCKERFILE, write_base_record, write_task


AGENT_CONFIG = "responses_api_agents/harbor_oracle_agent/configs/harbor_oracle_agent.yaml"
PINNED = "ubuntu@sha256:" + "a" * 64
REPINNED = "ubuntu@sha256:" + "b" * 64
HEAD = BaseServerConfig(host="127.0.0.1", port=4242)


@pytest.fixture
def harbor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A two-task Harbor dataset and a benchmark `fixture_harbor` that runs it with eight repeats."""
    dataset = tmp_path / "datasets" / "fixture_ds"
    for name in ("alpha", "beta"):
        write_task(dataset / name, dockerfile=PULL_DOCKERFILE, extra={})
    write_base_record(dataset)
    benchmark = tmp_path / "extra" / "benchmarks" / "fixture_harbor" / "config.yaml"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(
        yaml.safe_dump(
            {
                "harbor_target": str(dataset),
                "config_paths": [AGENT_CONFIG],
                "builds": {"enabled": False},
                "num_repeats": 8,
            }
        )
    )
    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path / "extra"))
    monkeypatch.setenv(DATASETS_DIR_ENV, str(tmp_path / "datasets"))
    return dataset


def _plan(work_dir: Path, task_ids: list[str] | None = None) -> PlannedBenchmark:
    return plan_benchmark("fixture_harbor", task_ids=task_ids, work_dir=work_dir)


class TestPlanHarborBenchmark:
    def test_plans_the_selected_task_as_a_materialized_row(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])

        assert planned.task_ids() == ["alpha"]
        assert planned.plan.selected_task_ids == ["alpha"]
        assert planned.plan.benchmark.name == "fixture_harbor"
        (row,) = planned.rows
        assert row["task_id"] == {"taskset": "fixture_ds", "task_id": "alpha"}
        assert row["task_input"]["responses_create_params"]["input"][0]["content"] == "Do it.\n"

    def test_plans_every_task_once_whatever_the_benchmarks_repeats(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work")

        assert planned.task_ids() == ["alpha", "beta"]
        assert len(planned.rows) == 2

    def test_a_task_id_is_matched_exactly_not_as_a_pattern(self, harbor: Path, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="No runnable task left"):
            _plan(tmp_path / "work", ["*"])

    def test_a_task_id_the_target_does_not_have_is_refused(self, harbor: Path, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"no tasks with ids \['gamma'\]"):
            _plan(tmp_path / "work", ["alpha", "gamma"])

    def test_a_work_dir_is_required_because_the_servers_start_from_it(self, harbor: Path) -> None:
        with pytest.raises(ConfigError, match="pass work_dir"):
            plan_benchmark("fixture_harbor")

    def test_records_the_dataset_the_images_and_the_configuration_files(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])

        run = planned.harbor
        assert run.target == str(harbor)
        assert run.agent == "harbor_oracle_agent"
        assert run.sandbox is None
        assert len(run.dataset_digest) == 64
        assert run.images == ("ubuntu:24.04",)
        names = [Path(path).name for path in run.config_paths]
        assert names[0] == "run_config_harbor_oracle_agent.yaml"
        assert "harbor.yaml" in names
        assert all(Path(path).is_file() for path in run.config_paths)

    def test_the_run_config_serves_only_the_selected_task(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])

        run_config = yaml.safe_load(Path(planned.harbor.config_paths[0]).read_text())
        server = run_config["harbor_fixture_5fds_environment"]["environment_servers"]["harbor"]
        assert list(server["tasksets"]["fixture_ds"]["tasks"]) == ["alpha"]

    def test_the_sandbox_provider_config_joins_the_configuration_files(self, harbor: Path, tmp_path: Path) -> None:
        planned = plan_benchmark("fixture_harbor", work_dir=tmp_path / "work", sandbox="docker")

        assert planned.harbor.sandbox == "docker"
        assert any(path.endswith("providers/docker/configs/docker.yaml") for path in planned.harbor.config_paths)

    def test_overrides_reach_the_settings_preparation_reads(self, harbor: Path, tmp_path: Path) -> None:
        planned = plan_benchmark(
            "fixture_harbor", work_dir=tmp_path / "work", overrides={"builds": {"enabled": False, "tool": "docker"}}
        )

        assert planned.task_ids() == ["alpha", "beta"]
        assert yaml.safe_load((tmp_path / "work" / "overrides.yaml").read_text())["builds"]["tool"] == "docker"

    def test_a_sandbox_is_refused_for_a_benchmark_that_composes_its_own_servers(self) -> None:
        with pytest.raises(ConfigError, match="not one"):
            plan_benchmark("aime24", sandbox="docker")

    def test_the_planned_benchmark_survives_json(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])

        restored = PlannedBenchmark.from_json(planned.to_json())

        assert restored.harbor == planned.harbor
        assert restored.plan == planned.plan
        assert restored.rows == planned.rows

    def test_a_plan_saved_without_a_harbor_record_still_loads(self, harbor: Path, tmp_path: Path) -> None:
        saved = json.loads(_plan(tmp_path / "work", ["alpha"]).to_json())
        del saved["harbor"]

        assert PlannedBenchmark.from_json(json.dumps(saved)).harbor is None


class TestChangedInputs:
    def test_the_same_inputs_plan_the_same_digests_in_another_work_dir(self, harbor: Path, tmp_path: Path) -> None:
        first = _plan(tmp_path / "one", ["alpha"])
        second = _plan(tmp_path / "two", ["alpha"])

        assert first.task_digests() == second.task_digests()
        assert first.harbor.dataset_digest == second.harbor.dataset_digest
        assert first.harbor.images == second.harbor.images

    def test_a_changed_verifier_changes_the_task_digest(self, harbor: Path, tmp_path: Path) -> None:
        before = _plan(tmp_path / "one", ["alpha"])
        (harbor / "alpha" / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")

        after = _plan(tmp_path / "two", ["alpha"])

        assert after.task_digests()["alpha"] != before.task_digests()["alpha"]

    def test_a_change_to_a_task_that_is_not_selected_still_changes_every_task_digest(
        self, harbor: Path, tmp_path: Path
    ) -> None:
        """Each row carries its dataset's digest, which covers every task folder, selected or not."""
        before = _plan(tmp_path / "one", ["alpha"])
        (harbor / "beta" / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")

        after = _plan(tmp_path / "two", ["alpha"])

        assert after.harbor.dataset_digest != before.harbor.dataset_digest
        assert after.task_digests() != before.task_digests()

    def test_a_changed_task_image_changes_the_recorded_images(self, harbor: Path, tmp_path: Path) -> None:
        records = harbor / "compose-images.json"
        record = json.loads(records.read_text())["ubuntu:24.04"]
        records.write_text(json.dumps({"ubuntu:24.04": {**record, "image": PINNED}}))
        before = _plan(tmp_path / "one", ["alpha"])
        records.write_text(json.dumps({"ubuntu:24.04": {**record, "image": REPINNED}}))

        after = _plan(tmp_path / "two", ["alpha"])

        assert before.harbor.images == (PINNED,)
        assert after.harbor.images == (REPINNED,)
        assert after.task_digests() == before.task_digests()


class _FakeServerSupervisor:
    started_with = None

    def start(self, parser_config) -> None:
        type(self).started_with = parser_config

    def shutdown(self) -> None:
        pass


class TestServeHarborBenchmark:
    @pytest.fixture(autouse=True)
    def fakes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeServerSupervisor.started_with = None
        monkeypatch.setattr("nemo_gym.environment.in_process.serving.ServerSupervisor", _FakeServerSupervisor)
        monkeypatch.setattr(ServerClient, "load_head_server_config", classmethod(lambda cls: HEAD))

    def test_starts_the_servers_from_the_planned_configuration(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])

        with serve_benchmark("fixture_harbor", planned=planned, overrides={"policy_base_url": "http://model/v1"}):
            pass

        initial = _FakeServerSupervisor.started_with.initial_global_config_dict
        paths = list(initial["config_paths"])
        assert paths[:-1] == list(planned.harbor.config_paths)
        assert paths[-1].endswith("responses_api_models/openai_model/configs/openai_model.yaml")
        assert OmegaConf.to_container(initial)["policy_base_url"] == "http://model/v1"

    def test_the_planned_benchmark_is_required(self, harbor: Path) -> None:
        with pytest.raises(ConfigError, match="PlannedBenchmark that plan_benchmark returned"):
            with serve_benchmark("fixture_harbor"):
                pass
        assert _FakeServerSupervisor.started_with is None

    def test_a_plan_of_a_benchmark_without_a_harbor_record_is_refused(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])
        plain = PlannedBenchmark(plan=planned.plan, rows=planned.rows)

        with pytest.raises(ConfigError, match="PlannedBenchmark that plan_benchmark returned"):
            with serve_benchmark("fixture_harbor", planned=plain):
                pass

    def test_a_plan_for_another_target_is_refused(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])
        other = PlannedBenchmark(
            plan=planned.plan,
            rows=planned.rows,
            harbor=type(planned.harbor)(**{**vars(planned.harbor), "target": "x"}),
        )

        with pytest.raises(ConfigError, match="plan is for Harbor target 'x'"):
            with serve_benchmark("fixture_harbor", planned=other):
                pass

    def test_a_plan_whose_files_are_on_another_machine_is_refused(self, harbor: Path, tmp_path: Path) -> None:
        planned = _plan(tmp_path / "work", ["alpha"])
        (tmp_path / "work" / "fixture_ds" / "run_config_harbor_oracle_agent.yaml").unlink()

        with pytest.raises(ConfigError, match="plan it again here"):
            with serve_benchmark("fixture_harbor", planned=planned):
                pass


def test_task_runner_takes_the_materialized_rows_of_a_harbor_plan(harbor: Path, tmp_path: Path) -> None:
    planned = _plan(tmp_path / "work", ["alpha"])

    runner = TaskRunner(planned.rows, registry_server=HEAD)

    assert runner.task_ids() == ["alpha"]
