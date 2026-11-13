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
"""A run that selected its tasks by id is tagged as a subset on its MLflow run, so its score is never read as the
benchmark's."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import DictConfig
from pytest import MonkeyPatch

import nemo_gym.artifacts.exporters.mlflow as mlflow_module
from nemo_gym.artifacts.exporters.mlflow import MLflowExporter
from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.utils.task_ids import task_ids_digest


MLFLOW = DictConfig(
    {"mlflow_tracking_uri": "https://tracking.example", "mlflow_experiment_name": "gym", "mlflow_run_name": "run"}
)


def _open_mlflow_exporter(monkeypatch: MonkeyPatch, config: DictConfig) -> tuple[MLflowExporter, MagicMock]:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
    client.create_run.return_value = MagicMock(info=MagicMock(run_id="run-1"))
    monkeypatch.setattr(mlflow_module, "MlflowClient", MagicMock(return_value=client))
    exporter = MLflowExporter(config)
    exporter.setup()
    return exporter, client


def tags(client) -> dict[str, str]:
    return {tag.key: tag.value for batch in client.log_batch.call_args_list for tag in batch.kwargs.get("tags", ())}


def test_a_live_subset_run_is_tagged_with_its_digest(monkeypatch: MonkeyPatch) -> None:
    exporter, client = _open_mlflow_exporter(monkeypatch, MLFLOW)

    exporter._log_config(DictConfig({"task_ids": ["b", "a", 7]}))

    assert tags(client) == {
        "gym_task_selection": "subset",
        "gym_task_selection_source": "task_ids",
        "gym_task_ids_sha256": task_ids_digest(["a", "b", "7"]),
    }


def test_a_live_run_from_an_id_file_is_tagged_with_the_file_and_its_digest(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    ids = tmp_path / "subset_v1.txt"
    ids.write_text("# v1\nb\na\n")
    exporter, client = _open_mlflow_exporter(monkeypatch, MLFLOW)

    exporter._log_config(DictConfig({"task_ids_fpath": str(ids)}))

    assert tags(client)["gym_task_selection_source"] == str(ids)
    assert tags(client)["gym_task_ids_sha256"] == task_ids_digest(["a", "b"])


def test_an_unreadable_id_file_still_tags_a_subset(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    exporter, client = _open_mlflow_exporter(monkeypatch, MLFLOW)

    exporter._log_config(DictConfig({"task_ids_fpath": str(tmp_path / "missing.txt")}))

    assert tags(client)["gym_task_selection"] == "subset"
    assert tags(client)["gym_task_ids_sha256"] == "unknown"


def test_a_post_run_export_takes_the_digest_from_the_saved_record(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # The id file may have changed since the run: the record beside the rollouts says what ran.
    (tmp_path / "rollouts_task_selection.json").write_text(
        json.dumps({"source": "subset_v1.txt", "task_ids_sha256": "ab" * 32, "task_ids": ["a"]})
    )
    exporter, client = _open_mlflow_exporter(monkeypatch, MLFLOW)
    exporter._exported_artifacts = RunArtifacts(tmp_path / "rollouts.jsonl")

    exporter._log_config(DictConfig({"task_ids_fpath": "subset_v1.txt"}))

    assert tags(client)["gym_task_ids_sha256"] == "ab" * 32


def test_a_run_of_every_task_gets_no_subset_tag(monkeypatch: MonkeyPatch) -> None:
    exporter, client = _open_mlflow_exporter(monkeypatch, MLFLOW)

    exporter._log_config(DictConfig({"seed": 7}))

    assert tags(client) == {}
