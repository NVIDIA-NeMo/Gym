# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared acceptance checks collected by each schema-owning training component."""

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.data import TrainDataProcessor, materialize_task
from nemo_gym.data.dataset_preparation import TrainDataProcessorConfig
from resources_servers.opencode_multimodal_benchmarks.tests.test_rewards import tool_response, trajectory


class AdapterChecks:
    component: str
    parent: str
    invalid_fields: dict

    def component_dir(self):
        return Path(__file__).resolve().parents[2] / self.component

    def row(self):
        return json.loads((self.component_dir() / "data/example.jsonl").read_text().splitlines()[0])

    def test_committed_examples_pass_ci_data_validation(self):
        from nemo_gym.cli.env import TestConfig, _validate_data_single

        _validate_data_single(TestConfig(entrypoint=f"resources_servers/{self.component}", should_validate_data=True))
        directory = self.component_dir() / "data"
        rows = [json.loads(line) for line in (directory / "example.jsonl").read_text().splitlines()]
        results = [json.loads(line) for line in (directory / "example_rollouts.jsonl").read_text().splitlines()]
        schema = importlib.import_module(f"resources_servers.{self.parent}.task_data").TaskData
        for row, result in zip(rows, results, strict=True):
            schema.model_validate(row)
            assert NeMoGymResponseCreateParamsNonStreaming.model_validate(
                result["responses_create_params"]
            ) == NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
            assert result["expected_answer"] == row["expected_answer"]
            assert result["response"]["output"]

    def collate(self, tmp_path, monkeypatch, row):
        config = OmegaConf.load(self.component_dir() / "configs" / f"{self.component}.yaml")
        if self.parent == "math_with_judge":
            config.policy_model = {"responses_api_models": {"vllm_model": {"entrypoint": "app.py"}}}
        rows = tmp_path / "rows.jsonl"
        rows.write_text(json.dumps(row) + "\n")
        config[self.component].resources_servers[self.component].datasets = [
            {"name": "native", "type": "validation", "license": "Apache 2.0", "jsonl_fpath": str(rows)}
        ]
        monkeypatch.chdir(tmp_path)
        parsed = GlobalConfigDictParser().parse_no_environment(config)
        processor = TrainDataProcessor()
        settings = TrainDataProcessorConfig(
            output_dirpath=str(tmp_path / "out"), mode="train_preparation", task_data_validation="error"
        )
        instances = processor.load_and_validate_server_instance_configs(settings, parsed)
        paths = processor._collate_samples_single_type(
            "validation", instances, task_data_validation=settings.effective_task_data_validation
        )
        return [json.loads(line) for path in paths for line in path.read_text().splitlines()]

    def test_strict_collation_uses_native_schema(self, tmp_path, monkeypatch):
        rows = self.collate(tmp_path, monkeypatch, self.row())
        assert len(rows) == 1
        # Collation materializes the task, preserving native grader fields.
        task_data = materialize_task(rows[0], taskset="native", task_index=0)["task_input"]
        for key, value in self.row().items():
            if key != "responses_create_params":
                assert task_data[key] == value
        assert "verifier_metadata" not in task_data
        assert "benchmark" not in task_data

    @pytest.mark.parametrize("bad_field_index", [0, 1])
    def test_strict_collation_rejects_parent_schema_errors(self, tmp_path, monkeypatch, bad_field_index):
        field, invalid = list(self.invalid_fields.items())[bad_field_index]
        row = self.row() | {field: invalid}
        schema = importlib.import_module(f"resources_servers.{self.parent}.task_data").TaskData
        with pytest.raises(ValidationError):
            schema.model_validate(row)
        with pytest.raises(ValueError, match=field):
            self.collate(tmp_path, monkeypatch, row)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("correct", [True, False])
    @pytest.mark.parametrize("status", ["completed", "failed"])
    async def test_verifier_preserves_response_and_default_penalties(self, correct, status):
        module = importlib.import_module(f"resources_servers.{self.component}.app")
        config = OmegaConf.to_container(
            OmegaConf.load(self.component_dir() / "configs" / f"{self.component}.yaml"), resolve=True
        )[self.component]["resources_servers"][self.component]
        server = module.ToolRewardServer(
            config=module.ToolRewardConfig.model_validate(config | {"name": "grader", "host": "127.0.0.1", "port": 1}),
            server_client=MagicMock(spec=ServerClient),
        )
        row = self.row()
        answer = row["expected_answer"] if correct else "wrong"
        if self.parent != "string_match":
            answer = "\\boxed{" + answer + "}"
        original = tool_response(answer)
        request = module.ToolRewardVerifyRequest.model_validate(
            row | {"response": original, "ng_trajectory": trajectory(status)}
        )
        result = await server.verify(request)
        expected = (0.95 if status == "failed" else 1.0) if correct else 0.0
        assert result.reward == pytest.approx(expected)
        assert result.response.model_dump() == original.model_dump()
        assert result.reward_components["verifier"] == float(correct)
        assert result.opencode_tool_reward.base_reward == float(correct)
