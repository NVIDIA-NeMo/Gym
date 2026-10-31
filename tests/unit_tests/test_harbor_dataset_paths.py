# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig

from nemo_gym.config.config_types import ResponsesAPIAgentServerInstanceConfig
from nemo_gym.tasks.harbor.cli import prepare_target
from nemo_gym.tasks.harbor.materialize import dataset_entry
from nemo_gym.train_data_utils import TrainDataProcessor, TrainDataProcessorConfig
from tests.unit_tests.test_harbor_tasks import write_task


@pytest.mark.parametrize("absolute", [False, True])
def test_rows_inside_checkout_use_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    rows = Path("results/ds/tasks.jsonl")
    entry = dataset_entry("ds", tmp_path / rows if absolute else rows)
    assert entry["jsonl_fpath"] == str(rows)


@pytest.mark.parametrize("relative", [False, True])
def test_external_rows_keep_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    external = tmp_path / "shared" / "tasks.jsonl"
    entry = dataset_entry("ds", Path("../shared/tasks.jsonl") if relative else external)
    assert entry["jsonl_fpath"] == str(external)


def test_symlink_to_external_rows_keeps_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    external = tmp_path / "shared"
    external.mkdir()
    try:
        (checkout / "data").symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    monkeypatch.chdir(checkout)
    assert dataset_entry("ds", Path("data/tasks.jsonl"))["jsonl_fpath"] == str(external / "tasks.jsonl")


@pytest.mark.parametrize("changed_tasks", [False, True])
def test_collation_across_checkouts_preserves_metrics_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_tasks: bool
) -> None:
    output = tmp_path / "shared-output"
    config = TrainDataProcessorConfig(output_dirpath=str(output), mode="train_preparation")
    first_rows = None
    first_metrics = None
    for index in range(2):
        checkout = tmp_path / f"checkout-{index}"
        write_task(checkout / "ds" / "a")
        if index and changed_tasks:
            write_task(checkout / "ds" / "b")
        monkeypatch.chdir(checkout)
        prepared = prepare_target(str(checkout / "ds"))
        entry = dataset_entry("ds", prepared.rows_path)
        block = {"simple_agent": {"entrypoint": "app.py", "datasets": [entry]}}
        server = ResponsesAPIAgentServerInstanceConfig(
            name="harbor_ds_agent",
            responses_api_agents=block,
            server_type_config_dict=DictConfig({"responses_api_agents": block}),
        )
        processor = TrainDataProcessor()
        metrics = processor.validate_samples_and_aggregate_metrics([server], False)
        if index and changed_tasks:
            with pytest.raises(ValueError, match="Found conflicting aggregate metrics"):
                processor.collate_samples(config, [server], metrics)
            assert (output / "benchmark_metrics.json").read_bytes() == first_metrics
            assert (output / "benchmark.jsonl").read_bytes() == first_rows
        else:
            processor.collate_samples(config, [server], metrics)
            rows = (output / "benchmark.jsonl").read_bytes()
            saved_metrics = (output / "benchmark_metrics.json").read_bytes()
            assert json.loads(saved_metrics)["Number of examples"] == 1
            if index:
                assert rows == first_rows
                assert saved_metrics == first_metrics
                assert not (output / "benchmark_metrics_conflict.json").exists()
            first_rows, first_metrics = rows, saved_metrics
