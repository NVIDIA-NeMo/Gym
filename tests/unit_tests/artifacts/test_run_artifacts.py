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

from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.utils.path_utils import exporter_run_id_path_for


def test_exporter_run_id_path_for() -> None:
    assert exporter_run_id_path_for(Path("out/rollouts.jsonl"), "mlflow") == Path("out/rollouts_mlflow_run_id")


def test_run_artifacts_derives_every_sibling_of_the_rollouts_file() -> None:
    artifacts = RunArtifacts("out/weather_rollouts.jsonl")

    assert artifacts.rollouts_jsonl_fpath == Path("out/weather_rollouts.jsonl")
    assert artifacts.failures_jsonl_fpath == Path("out/weather_rollouts_failures.jsonl")
    assert artifacts.materialized_inputs_jsonl_fpath == Path("out/weather_rollouts_materialized_inputs.jsonl")
    assert artifacts.aggregate_metrics_json_fpath == Path("out/weather_rollouts_aggregate_metrics.json")
    assert artifacts.resolved_config_yaml_fpath == Path("out/weather_rollouts_resolved_config.yaml")
    assert artifacts.exporter_run_id_fpath("wandb") == Path("out/weather_rollouts_wandb_run_id")


def test_run_artifacts_derives_paths_without_requiring_files_to_exist(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path / "rollouts.jsonl")

    assert not artifacts.aggregate_metrics_json_fpath.exists()
    assert artifacts.aggregate_metrics_json_fpath.parent == tmp_path
