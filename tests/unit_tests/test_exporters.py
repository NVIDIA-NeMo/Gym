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
import math
from pathlib import Path
from typing import Any, ClassVar, Optional
from unittest.mock import ANY, MagicMock

import orjson
import pytest
from mlflow.utils.validation import (
    MAX_ENTITY_KEY_LENGTH,
    MAX_PARAM_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
    _validate_metric_name,
)
from omegaconf import DictConfig
from pytest import MonkeyPatch

import nemo_gym.exporters as exporters_module
import nemo_gym.exporters.mlflow as mlflow_module
import nemo_gym.exporters.wandb as wandb_module
from nemo_gym.config_types import ExporterConfig, LangSmithConfig, MLFlowConfig, WANDBConfig
from nemo_gym.exporters import (
    export_metrics,
    export_rollouts,
    get_exporters,
    setup_exporters,
    teardown_exporters,
)
from nemo_gym.exporters.base import BaseExporter
from nemo_gym.exporters.mlflow import MLflowExporter, _flatten_config, _sanitize_key
from nemo_gym.exporters.wandb import WandbExporter


class RecordingConfig(ExporterConfig):
    recording_enabled: bool = False

    @property
    def is_available(self) -> bool:
        return self.recording_enabled


class RecordingExporter(BaseExporter):
    """In-memory backend that records every call, so fan-out and masking are observable."""

    name: ClassVar[str] = "recording"
    fail_on_setup: ClassVar[bool] = False

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.configs: list[DictConfig] = []
        self.metrics: list[tuple[dict[str, Any], Optional[int]]] = []
        self.rollouts: list[list[dict[str, Any]]] = []
        self.torn_down = False

    def setup(self) -> None:
        if self.fail_on_setup:
            raise RuntimeError("backend unreachable")

    def teardown(self) -> None:
        self.torn_down = True

    def _log_config(self, config_dict: DictConfig) -> None:
        self.configs.append(config_dict)

    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self.metrics.append((metrics, step))

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        self.rollouts.append(rollouts)


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    teardown_exporters()
    yield
    teardown_exporters()


@pytest.fixture
def wandb_config() -> DictConfig:
    return DictConfig(
        {
            "recording_enabled": True,
            "results_dir": "/tmp/results",
            "wandb_project": "proj",
            "wandb_name": "run",
            "wandb_api_key": "secret-key",  # pragma: allowlist secret
        }
    )


@pytest.fixture
def mlflow_config() -> DictConfig:
    return DictConfig(
        {
            "mlflow_tracking_uri": "https://tracking.example",
            "mlflow_tracking_token": "secret-token",  # pragma: allowlist secret
            "mlflow_experiment_name": "gym",
            "mlflow_run_name": "run",
        }
    )


@pytest.fixture
def langsmith_config() -> DictConfig:
    return DictConfig(
        {
            "langsmith_api_key": "secret-key",  # pragma: allowlist secret
            "langsmith_endpoint": "https://langsmith.example",
            "langsmith_workspace_id": "workspace-1",
            "langsmith_dataset_name": "gym-dataset",
            "langsmith_experiment_name": "gym-experiment",
        }
    )


def _register_recording(monkeypatch: MonkeyPatch) -> MagicMock:
    """Point the registry at RecordingExporter and hand back the (spied) lazy loader."""
    loader = MagicMock(return_value=RecordingExporter)
    monkeypatch.setattr(exporters_module, "EXPORTER_REGISTRY", {"recording": (RecordingConfig, "recording")})
    monkeypatch.setattr(exporters_module, "_load_exporter_class", loader)
    return loader


def _open_mlflow_exporter(monkeypatch: MonkeyPatch, config: DictConfig) -> tuple[MLflowExporter, MagicMock]:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
    client.create_run.return_value = MagicMock(info=MagicMock(run_id="run-1"))
    monkeypatch.setattr(mlflow_module, "MlflowClient", MagicMock(return_value=client))
    exporter = MLflowExporter(config)
    exporter.setup()
    return exporter, client


class TestRegistry:
    def test_setup_loads_langsmith_when_configured(self, monkeypatch: MonkeyPatch) -> None:
        """Tests to make sure Langsmith exporter is loaded via lazy loader when configured.
        Tests the registry functionality of the exporters and doesn't actually run any
        LangSmith code. test_setup_creates_client_from_config tests the actual LangSmith code."""

        loader = MagicMock(return_value=RecordingExporter)  # create fake lazy loader

        # Mock the lazy loader to avoid importing the real LangSmith implementation (test isolation)
        monkeypatch.setattr(exporters_module, "_load_exporter_class", loader)
        config = DictConfig(
            {
                "langsmith_api_key": "secret-key",  # pragma: allowlist secret
                "langsmith_dataset_name": "gym-dataset",
                "langsmith_experiment_name": "gym-experiment",
            }
        )

        opened = setup_exporters(config)

        loader.assert_called_once_with("nemo_gym.exporters.langsmith:LangSmithExporter")
        assert len(opened) == 1  # only 1 exporter should be opened bc other exporters were not configured

    def test_setup_skips_backends_that_are_not_configured(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        loader = _register_recording(monkeypatch)
        wandb_config["recording_enabled"] = False

        assert setup_exporters(wandb_config) == []
        assert get_exporters() == []
        # The point of keying availability off the config model: an unconfigured backend's module
        # (and its tracking SDK) is never imported.
        loader.assert_not_called()

    def test_setup_opens_configured_backend_and_logs_config(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        _register_recording(monkeypatch)

        opened = setup_exporters(wandb_config)

        assert len(opened) == 1
        assert get_exporters() == opened
        assert len(opened[0].configs) == 1

    def test_exported_config_masks_secrets_but_exporter_keeps_the_live_key(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        _register_recording(monkeypatch)

        exporter = setup_exporters(wandb_config)[0]

        assert exporter.configs[0]["wandb_api_key"] == "****"
        # The masking must not have mutated the config the exporter authenticates with.
        assert wandb_config["wandb_api_key"] == "secret-key"  # pragma: allowlist secret

    def test_setup_failure_is_skipped_rather_than_raised(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        _register_recording(monkeypatch)
        monkeypatch.setattr(RecordingExporter, "fail_on_setup", True)

        assert setup_exporters(wandb_config) == []
        assert get_exporters() == []

    def test_setup_replaces_previously_opened_exporters(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        _register_recording(monkeypatch)
        first = setup_exporters(wandb_config)[0]

        second = setup_exporters(wandb_config)[0]

        assert first is not second
        assert first.torn_down
        assert get_exporters() == [second]

    def test_module_level_helpers_fan_out_to_exporter(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        _register_recording(monkeypatch)
        exporter = setup_exporters(wandb_config)[0]

        export_metrics({"reward": 1.0}, step=7)
        export_rollouts([{"reward": 1.0}])

        assert exporter.metrics == [({"reward": 1.0}, 7)]
        assert exporter.rollouts == [[{"reward": 1.0}]]

    def test_helpers_are_noops_when_no_backend_is_configured(self) -> None:
        export_metrics({"reward": 1.0})
        export_rollouts([{"reward": 1.0}])

        assert get_exporters() == []

    def test_teardown_closes_exporters_and_empties_the_registry(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        _register_recording(monkeypatch)
        exporter = setup_exporters(wandb_config)[0]

        teardown_exporters()

        assert exporter.torn_down
        assert get_exporters() == []


class TestGuard:
    def test_backend_exception_does_not_propagate(self, monkeypatch: MonkeyPatch, wandb_config: DictConfig) -> None:
        _register_recording(monkeypatch)
        exporter = setup_exporters(wandb_config)[0]
        monkeypatch.setattr(
            exporter, "_log_metrics", MagicMock(side_effect=RuntimeError("tracking server is down")), raising=False
        )

        export_metrics({"reward": 1.0})

        assert exporter.metrics == []
        assert get_exporters() == [exporter]

    def test_a_failing_backend_does_not_block_the_others(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        _register_recording(monkeypatch)
        failing = RecordingExporter(wandb_config)
        monkeypatch.setattr(failing, "_log_metrics", MagicMock(side_effect=RuntimeError("down")), raising=False)
        healthy = RecordingExporter(wandb_config)
        monkeypatch.setattr(exporters_module, "_EXPORTERS", [failing, healthy])

        export_metrics({"reward": 1.0})

        assert healthy.metrics == [({"reward": 1.0}, None)]


class TestWandbConfigAvailability:
    def test_requires_project_name_and_key(self) -> None:
        assert not WANDBConfig(wandb_project="proj").is_available
        assert not WANDBConfig(wandb_project="proj", wandb_name="run").is_available
        assert WANDBConfig(
            wandb_project="proj",
            wandb_name="run",
            wandb_api_key="k",  # pragma: allowlist secret
        ).is_available

    def test_a_masked_key_does_not_count_as_configured(self) -> None:
        assert not WANDBConfig(wandb_project="proj", wandb_name="run", wandb_api_key="****").is_available


class TestWandbExporter:
    def test_setup_initializes_a_run_under_the_results_dir(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        init = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(wandb_module.wandb, "init", init)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        exporter = WandbExporter(wandb_config)
        exporter.setup()

        init.assert_called_once_with(project="proj", name="run", dir=str(Path("/tmp/results") / "wandb"))
        assert wandb_module.environ["WANDB_API_KEY"] == "secret-key"  # pragma: allowlist secret

    def test_logging_before_setup_raises(self, wandb_config: DictConfig) -> None:
        with pytest.raises(RuntimeError, match="not open"):
            WandbExporter(wandb_config)._log_metrics({"reward": 1.0})

    def test_rollouts_are_logged_as_one_json_blob_per_row(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        run = MagicMock()
        monkeypatch.setattr(wandb_module.wandb, "init", MagicMock(return_value=run))
        exporter = WandbExporter(wandb_config)
        exporter.setup()

        exporter._log_rollouts([{"reward": 1.0}, {"reward": 0.0}])

        (payload,) = run.log.call_args.args
        table = payload["Rollouts"]
        assert table.columns == ["Rollout"]
        assert [orjson.loads(row) for (row,) in table.data] == [{"reward": 1.0}, {"reward": 0.0}]

    def test_metrics_pass_the_step_through(self, monkeypatch: MonkeyPatch, wandb_config: DictConfig) -> None:
        run = MagicMock()
        monkeypatch.setattr(wandb_module.wandb, "init", MagicMock(return_value=run))
        exporter = WandbExporter(wandb_config)
        exporter.setup()

        exporter._log_metrics({"reward": 1.0}, step=3)

        run.log.assert_called_once_with({"reward": 1.0}, step=3, commit=True)

    def test_teardown_finishes_the_run_and_is_idempotent(
        self, monkeypatch: MonkeyPatch, wandb_config: DictConfig
    ) -> None:
        run = MagicMock()
        monkeypatch.setattr(wandb_module.wandb, "init", MagicMock(return_value=run))
        exporter = WandbExporter(wandb_config)
        exporter.setup()

        exporter.teardown()
        exporter.teardown()

        run.finish.assert_called_once_with()
        assert exporter.run is None

    def test_teardown_without_setup_is_a_noop(self, monkeypatch: MonkeyPatch, wandb_config: DictConfig) -> None:
        init = MagicMock()
        monkeypatch.setattr(wandb_module.wandb, "init", init)
        exporter = WandbExporter(wandb_config)

        exporter.teardown()

        init.assert_not_called()
        assert exporter.run is None


class TestFlattenConfig:
    def test_nested_dicts_become_dotted_keys(self) -> None:
        assert list(_flatten_config({"a": {"b": {"c": 1}}})) == [("a.b.c", 1)]

    def test_lists_are_indexed(self) -> None:
        assert list(_flatten_config({"a": [1, {"b": 2}]})) == [("a[0]", 1), ("a[1].b", 2)]

    def test_empty_containers_yield_nothing(self) -> None:
        assert list(_flatten_config({"a": {}, "b": []})) == []


class TestSanitizeKey:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("simple_agent/mean/reward", "simple_agent/mean/reward"),
            ("a b:c-d.e/f", "a b:c-d.e/f"),
            ("gpqa_simple_agent/pass@1/accuracy", "gpqa_simple_agent/pass_at_1/accuracy"),
            ("weird[0]", "weird_0_"),
            ("/leading", "leading"),
            ("trailing/", "trailing"),
            ("a//b", "a/b"),
            ("a/../b", "a/__/b"),
            ("...x", "___x"),
        ],
    )
    def test_rewrites_only_what_mlflow_rejects(self, raw: str, expected: str) -> None:
        assert _sanitize_key(raw) == expected

    @pytest.mark.parametrize("raw", ["", "/", "//"])
    def test_names_with_nothing_left_are_dropped(self, raw: str) -> None:
        assert _sanitize_key(raw) is None

    def test_over_long_names_keep_both_ends(self) -> None:
        raw = "gpqa/" + "z" * (MAX_ENTITY_KEY_LENGTH + 10) + "/mean/reward"

        sanitized = _sanitize_key(raw)

        assert len(sanitized) == MAX_ENTITY_KEY_LENGTH
        assert sanitized.startswith("gpqa/")
        assert sanitized.endswith("/mean/reward")

    def test_a_name_lengthened_past_the_limit_is_elided_not_dropped(self) -> None:
        # `@` becomes `_at_`, so a name at the limit overflows it.
        sanitized = _sanitize_key("a" * (MAX_ENTITY_KEY_LENGTH - 2) + "@1")

        assert len(sanitized) == MAX_ENTITY_KEY_LENGTH
        assert sanitized.endswith("_at_1")

    @pytest.mark.parametrize("raw", ["pass@1", "a//b", "/x/", "...", "a/../b", "x" * 400, "%^&*"])
    def test_every_sanitized_name_is_accepted_by_mlflow(self, raw: str) -> None:
        _validate_metric_name(_sanitize_key(raw))


class TestMLflowConfigAvailability:
    def test_requires_uri_experiment_and_run_name(self, mlflow_config: DictConfig) -> None:
        assert MLFlowConfig.model_validate(mlflow_config).is_available

    def test_the_token_is_optional(self, mlflow_config: DictConfig) -> None:
        without_token = MLFlowConfig.model_validate({**mlflow_config, "mlflow_tracking_token": None})

        assert without_token.is_available

    def test_registry_only_credentials_are_not_enough(self) -> None:
        # The GitLab model registry shares this model but needs only the URI and token, so those
        # two alone must not switch the exporter on.
        wandb_config = MLFlowConfig(
            mlflow_tracking_uri="https://gitlab",
            mlflow_tracking_token="t",  # pragma: allowlist secret
        )

        assert not wandb_config.is_available

    def test_a_masked_token_does_not_count_as_configured(self, mlflow_config: DictConfig) -> None:
        masked = MLFlowConfig.model_validate({**mlflow_config, "mlflow_tracking_token": "****"})

        assert not masked.is_available


class TestMLflowExporter:
    def test_setup_creates_a_run_in_the_named_experiment(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)

        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        mlflow_module.MlflowClient.assert_called_once_with(tracking_uri="https://tracking.example")
        client.create_experiment.assert_not_called()
        client.create_run.assert_called_once_with("exp-1", run_name="run")
        assert exporter.run_id == "run-1"
        assert mlflow_module.environ["MLFLOW_TRACKING_TOKEN"] == "secret-token"  # pragma: allowlist secret

    def test_setup_without_a_token_leaves_the_env_var_unset(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        mlflow_config["mlflow_tracking_token"] = None

        _open_mlflow_exporter(monkeypatch, mlflow_config)

        assert "MLFLOW_TRACKING_TOKEN" not in mlflow_module.environ

    def test_setup_creates_the_experiment_when_it_does_not_exist(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        client.create_experiment.return_value = "exp-new"
        client.create_run.return_value = MagicMock(info=MagicMock(run_id="run-1"))
        monkeypatch.setattr(mlflow_module, "MlflowClient", MagicMock(return_value=client))

        MLflowExporter(mlflow_config).setup()

        client.create_experiment.assert_called_once_with("gym")
        client.create_run.assert_called_once_with("exp-new", run_name="run")

    def test_logging_before_setup_raises(self, mlflow_config: DictConfig) -> None:
        with pytest.raises(RuntimeError, match="not open"):
            MLflowExporter(mlflow_config)._log_metrics({"reward": 1.0})

    def test_config_is_logged_as_an_artifact_and_flattened_params(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        exporter._log_config(DictConfig({"policy": {"model_name": "qwen"}, "seed": 7}))

        client.log_dict.assert_called_once_with(
            "run-1", {"policy": {"model_name": "qwen"}, "seed": 7}, "global_config.json"
        )
        params = client.log_batch.call_args.kwargs["params"]
        assert {(param.key, param.value) for param in params} == {("policy.model_name", "qwen"), ("seed", "7")}

    def test_over_long_param_values_are_truncated(self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        exporter._log_config(DictConfig({"prompt": "x" * (MAX_PARAM_VAL_LENGTH + 100)}))

        (param,) = client.log_batch.call_args.kwargs["params"]
        assert param.value == "x" * MAX_PARAM_VAL_LENGTH

    def test_params_are_batched_within_the_mlflow_limit(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)
        wide = {f"k{index}": index for index in range(MAX_PARAMS_TAGS_PER_BATCH + 5)}

        exporter._log_config(DictConfig(wide))

        batch_sizes = [len(call.kwargs["params"]) for call in client.log_batch.call_args_list]
        assert batch_sizes == [MAX_PARAMS_TAGS_PER_BATCH, 5]

    def test_metrics_are_split_into_numeric_metrics_and_string_tags(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        exporter._log_metrics({"reward": 0.5, "passed": True, "agent": "openhands", "missing": None}, step=4)

        metrics = client.log_batch.call_args_list[0].kwargs["metrics"]
        assert {(metric.key, metric.value, metric.step) for metric in metrics} == {
            ("reward", 0.5, 4),
            ("passed", 1.0, 4),
        }
        tags = client.log_batch.call_args_list[1].kwargs["tags"]
        assert [(tag.key, tag.value) for tag in tags] == [("agent", "openhands")]
        assert "missing" not in {metric.key for metric in metrics} | {tag.key for tag in tags}

    def test_non_finite_metrics_are_logged(self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        exporter._log_metrics({"reward": 1.0, "std": float("nan"), "drift": float("inf")})

        metrics = {metric.key: metric.value for metric in client.log_batch.call_args_list[0].kwargs["metrics"]}
        assert metrics["reward"] == 1.0
        assert math.isnan(metrics["std"])
        assert math.isinf(metrics["drift"])

    def test_step_defaults_to_zero(self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        exporter._log_metrics({"reward": 1.0})

        (metric,) = client.log_batch.call_args.kwargs["metrics"]
        assert metric.step == 0

    def test_rollouts_are_logged_as_one_json_blob_per_row(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        exporter._log_rollouts([{"reward": 1.0}, {"reward": 0.0}])

        kwargs = client.log_table.call_args.kwargs
        assert [orjson.loads(row) for row in kwargs["data"]["Rollout"]] == [{"reward": 1.0}, {"reward": 0.0}]
        # log_table rejects any other extension.
        assert kwargs["artifact_file"].endswith(".json")

    def test_teardown_terminates_the_run_and_is_idempotent(
        self, monkeypatch: MonkeyPatch, mlflow_config: DictConfig
    ) -> None:
        exporter, client = _open_mlflow_exporter(monkeypatch, mlflow_config)

        exporter.teardown()
        exporter.teardown()

        client.set_terminated.assert_called_once_with("run-1", status="FINISHED")
        assert exporter.run_id is None

    def test_teardown_without_setup_is_a_noop(self, mlflow_config: DictConfig) -> None:
        MLflowExporter(mlflow_config).teardown()


class TestLangSmithConfigAvailability:
    def test_requires_api_key_dataset_and_experiment(self) -> None:
        configured = {
            "langsmith_api_key": "secret-key",  # pragma: allowlist secret
            "langsmith_dataset_name": "gym-dataset",
            "langsmith_experiment_name": "gym-experiment",
        }

        assert LangSmithConfig.model_validate(configured).is_available

        for required_field in configured:
            incomplete = {**configured, required_field: None}
            assert not LangSmithConfig.model_validate(incomplete).is_available

    def test_workspace_is_optional_and_endpoint_has_a_default(self) -> None:
        config = LangSmithConfig(
            langsmith_api_key="secret-key",  # pragma: allowlist secret
            langsmith_dataset_name="gym-dataset",
            langsmith_experiment_name="gym-experiment",
        )

        assert config.langsmith_endpoint == "https://api.smith.langchain.com"
        assert config.langsmith_workspace_id is None

    def test_a_masked_key_does_not_count_as_configured(self) -> None:
        config = LangSmithConfig(
            langsmith_api_key="****",
            langsmith_dataset_name="gym-dataset",
            langsmith_experiment_name="gym-experiment",
        )

        assert not config.is_available


class TestLangSmithRolloutMapping:
    def test_maps_responses_io_metadata_and_reward(self) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        rollout = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": "What is 2 + 2?",
                    }
                ],
                "temperature": 0.2,
            },
            "response": {
                "id": "resp-1",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "4",
                            }
                        ],
                    }
                ],
            },
            "reward": 0.75,
            "score": 0.75,
            "extracted_answer": "4",
            "_ng_task_index": 12,
            "_ng_rollout_index": 3,
            "agent_ref": {"name": "deep-agent"},
            "task_source": "math",
        }

        mapped = langsmith_module._map_rollout(rollout)

        assert mapped.inputs == rollout["responses_create_params"]
        assert mapped.outputs == rollout["response"]
        assert mapped.reward == 0.75
        assert mapped.metadata == {
            "score": 0.75,
            "extracted_answer": "4",
            "_ng_task_index": 12,
            "_ng_rollout_index": 3,
            "agent_ref": {"name": "deep-agent"},
            "task_source": "math",
            "ls_runner": "nemo-gym",
            "source": "nemo-gym",
        }


class TestLangSmithExporter:
    def test_setup_creates_client_from_config(self, monkeypatch: MonkeyPatch, langsmith_config: DictConfig) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client_constructor = MagicMock()
        monkeypatch.setattr(langsmith_module, "Client", client_constructor)

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()

        client_constructor.assert_called_once_with(
            api_url="https://langsmith.example",
            api_key="secret-key",  # pragma: allowlist secret
            workspace_id="workspace-1",
        )
        assert exporter.client is client_constructor.return_value

    def test_setup_creates_dataset_and_linked_experiment_when_missing(
        self,
        monkeypatch: MonkeyPatch,
        langsmith_config: DictConfig,
    ) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = False
        client.create_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"
        monkeypatch.setattr(
            langsmith_module,
            "Client",
            MagicMock(return_value=client),
        )

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()

        client.has_dataset.assert_called_once_with(
            dataset_name="gym-dataset",
        )
        client.create_dataset.assert_called_once_with(
            "gym-dataset",
            description="Evaluation dataset exported by NeMo Gym.",
            metadata={
                "ls_runner": "nemo-gym",
                "source": "nemo-gym",
            },
        )
        client.create_project.assert_called_once_with(
            "gym-experiment",
            upsert=True,
            reference_dataset_id="dataset-1",
            metadata={
                "ls_runner": "nemo-gym",
                "source": "nemo-gym",
            },
        )
        assert exporter.dataset_id == "dataset-1"
        assert exporter.experiment_id == "experiment-1"

    def test_setup_reuses_existing_dataset(
        self,
        monkeypatch: MonkeyPatch,
        langsmith_config: DictConfig,
    ) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"
        monkeypatch.setattr(
            langsmith_module,
            "Client",
            MagicMock(return_value=client),
        )

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()

        client.read_dataset.assert_called_once_with(
            dataset_name="gym-dataset",
        )
        client.create_dataset.assert_not_called()
        client.create_project.assert_called_once_with(
            "gym-experiment",
            upsert=True,
            reference_dataset_id="dataset-1",
            metadata={
                "ls_runner": "nemo-gym",
                "source": "nemo-gym",
            },
        )
        assert exporter.dataset_id == "dataset-1"
        assert exporter.experiment_id == "experiment-1"

    def test_log_rollouts_creates_one_dataset_example_per_task(
        self,
        monkeypatch: MonkeyPatch,
        langsmith_config: DictConfig,
    ) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"
        monkeypatch.setattr(
            langsmith_module,
            "Client",
            MagicMock(return_value=client),
        )

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()

        rollouts = [
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 2 + 2?"}],
                },
                "response": {"output_text": "4"},
                "reward": 1.0,
                "score": 1.0,
                "_ng_task_index": 7,
                "_ng_rollout_index": 0,
                "agent_ref": {"name": "deep-agent"},
                "task_source": "math",
            },
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 2 + 2?"}],
                },
                "response": {"output_text": "Four"},
                "reward": 1.0,
                "score": 1.0,
                "_ng_task_index": 7,
                "_ng_rollout_index": 1,
                "agent_ref": {"name": "deep-agent"},
                "task_source": "math",
            },
        ]

        exporter._log_rollouts(rollouts)

        client.create_examples.assert_called_once()
        kwargs = client.create_examples.call_args.kwargs

        assert kwargs["dataset_id"] == "dataset-1"
        assert len(kwargs["examples"]) == 1

        example = kwargs["examples"][0]
        assert example["inputs"] == rollouts[0]["responses_create_params"]
        assert example["metadata"] == {
            "_ng_task_index": 7,
            "task_source": "math",
            "ls_runner": "nemo-gym",
            "source": "nemo-gym",
        }
        assert "outputs" not in example
        assert "id" in example

    def test_log_rollouts_creates_new_and_updates_existing_dataset_examples(
        self,
        monkeypatch: MonkeyPatch,
        langsmith_config: DictConfig,
    ) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"

        existing_example_id = langsmith_module._stable_uuid(
            "dataset-1",
            "example",
            "math",
            7,
        )
        existing_example = MagicMock()
        existing_example.id = existing_example_id
        client.list_examples.return_value = [existing_example]

        monkeypatch.setattr(
            langsmith_module,
            "Client",
            MagicMock(return_value=client),
        )

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()

        rollouts = [
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 2 + 2?"}],
                },
                "response": {"output_text": "4"},
                "reward": 1.0,
                "_ng_task_index": 7,
                "_ng_rollout_index": 0,
                "agent_ref": {"name": "deep-agent"},
                "task_source": "math",
            },
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 3 + 3?"}],
                },
                "response": {"output_text": "6"},
                "reward": 1.0,
                "_ng_task_index": 8,
                "_ng_rollout_index": 0,
                "agent_ref": {"name": "deep-agent"},
                "task_source": "math",
            },
        ]

        exporter._log_rollouts(rollouts)

        new_example_id = langsmith_module._stable_uuid(
            "dataset-1",
            "example",
            "math",
            8,
        )
        client.list_examples.assert_called_once_with(
            dataset_id="dataset-1",
            example_ids=[existing_example_id, new_example_id],
        )

        client.create_examples.assert_called_once()
        create_kwargs = client.create_examples.call_args.kwargs
        assert create_kwargs["dataset_id"] == "dataset-1"
        assert [example["id"] for example in create_kwargs["examples"]] == [new_example_id]

        client.update_examples.assert_called_once()
        update_kwargs = client.update_examples.call_args.kwargs
        assert update_kwargs["dataset_id"] == "dataset-1"
        assert [example["id"] for example in update_kwargs["updates"]] == [existing_example_id]

    def test_log_rollouts_creates_completed_run_for_each_rollout(
        self,
        monkeypatch: MonkeyPatch,
        langsmith_config: DictConfig,
    ) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"
        monkeypatch.setattr(
            langsmith_module,
            "Client",
            MagicMock(return_value=client),
        )

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()

        rollouts = [
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 2 + 2?"}],
                },
                "response": {"output_text": "4"},
                "reward": 1.0,
                "_ng_task_index": 7,
                "_ng_rollout_index": 0,
                "agent_ref": {"name": "deep-agent"},
                "task_source": "math",
            },
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 2 + 2?"}],
                },
                "response": {"output_text": "Four"},
                "reward": 1.0,
                "_ng_task_index": 7,
                "_ng_rollout_index": 1,
                "agent_ref": {"name": "deep-agent"},
                "task_source": "math",
            },
        ]

        exporter._log_rollouts(rollouts)

        assert client.create_run.call_count == 2

        example_id = client.create_examples.call_args.kwargs["examples"][0]["id"]

        for rollout, call in zip(
            rollouts,
            client.create_run.call_args_list,
            strict=True,
        ):
            kwargs = call.kwargs
            rollout_index = rollout["_ng_rollout_index"]

            assert kwargs["name"] == f"task-7-rollout-{rollout_index}"
            assert kwargs["run_type"] == "chain"
            assert kwargs["inputs"] == rollout["responses_create_params"]
            assert kwargs["outputs"] == rollout["response"]
            assert kwargs["project_name"] == "gym-experiment"
            assert kwargs["reference_example_id"] == example_id
            assert "trace_id" not in kwargs
            assert "dotted_order" not in kwargs
            assert kwargs["tags"] == ["nemo-gym"]
            assert kwargs["extra"]["metadata"]["_ng_rollout_index"] == rollout_index
            assert kwargs["start_time"] <= kwargs["end_time"]

        assert client.create_run.call_args_list[0].kwargs["id"] != client.create_run.call_args_list[1].kwargs["id"]

    def test_log_rollouts_attaches_reward_feedback_to_the_run(
        self,
        monkeypatch: MonkeyPatch,
        langsmith_config: DictConfig,
    ) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"
        monkeypatch.setattr(
            langsmith_module,
            "Client",
            MagicMock(return_value=client),
        )

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()
        exporter._log_rollouts(
            [
                {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "What is 2 + 2?"}],
                    },
                    "response": {"output_text": "4"},
                    "reward": 0.75,
                    "_ng_task_index": 7,
                    "_ng_rollout_index": 0,
                    "agent_ref": {"name": "deep-agent"},
                    "task_source": "math",
                }
            ]
        )

        client.create_feedback.assert_called_once()
        kwargs = client.create_feedback.call_args.kwargs
        run_id = client.create_run.call_args.kwargs["id"]

        assert kwargs["key"] == "reward"
        assert kwargs["score"] == 0.75
        assert kwargs["run_id"] == run_id
        assert kwargs["trace_id"] == run_id
        assert kwargs["session_id"] == "experiment-1"
        assert kwargs["feedback_id"] == langsmith_module._stable_uuid(
            run_id,
            "feedback",
            "reward",
        )
        assert kwargs["source_info"] == {
            "ls_runner": "nemo-gym",
            "source": "nemo-gym",
        }

    def test_log_rollouts_updates_existing_reward_feedback(
        self,
        monkeypatch: MonkeyPatch,
        langsmith_config: DictConfig,
    ) -> None:
        pytest.importorskip("langsmith")
        from langsmith.utils import LangSmithConflictError

        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"
        client.create_feedback.side_effect = LangSmithConflictError("feedback already exists")
        monkeypatch.setattr(
            langsmith_module,
            "Client",
            MagicMock(return_value=client),
        )

        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()
        exporter._log_rollouts(
            [
                {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "What is 2 + 2?"}],
                    },
                    "response": {"output_text": "4"},
                    "reward": 0.75,
                    "_ng_task_index": 7,
                    "_ng_rollout_index": 0,
                    "agent_ref": {"name": "deep-agent"},
                    "task_source": "math",
                }
            ]
        )

        run_id = client.create_run.call_args.kwargs["id"]
        feedback_id = langsmith_module._stable_uuid(
            run_id,
            "feedback",
            "reward",
        )

        client.update_feedback.assert_called_once_with(
            feedback_id,
            score=0.75,
        )

    def test_teardown_ends_experiment_closes_client_and_is_idempotent(
        self, monkeypatch: MonkeyPatch, langsmith_config: DictConfig
    ) -> None:
        pytest.importorskip("langsmith")
        import nemo_gym.exporters.langsmith as langsmith_module

        client = MagicMock()
        client.has_dataset.return_value = False
        client.create_dataset.return_value.id = "dataset-1"
        client.create_project.return_value.id = "experiment-1"
        monkeypatch.setattr(langsmith_module, "Client", MagicMock(return_value=client))
        exporter = langsmith_module.LangSmithExporter(langsmith_config)
        exporter.setup()

        exporter.teardown()
        exporter.teardown()

        client.update_project.assert_called_once_with(
            "experiment-1",
            end_time=ANY,
        )
        client.close.assert_called_once_with()
        assert exporter.client is None
        assert exporter.dataset_id is None
        assert exporter.experiment_id is None
