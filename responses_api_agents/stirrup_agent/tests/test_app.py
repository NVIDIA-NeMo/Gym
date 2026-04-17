# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.stirrup_agent.app import (
    StirrupAgentWrapper,
    StirrupAgentWrapperConfig,
    _load_task_registry,
    get_task_strategy,
)
from responses_api_agents.stirrup_agent.task_strategy import TaskStrategy


STIRRUP_AGENT_DIR = Path(__file__).resolve().parent.parent


class TestTaskRegistry:
    def test_registry_includes_gdpval(self) -> None:
        registry = _load_task_registry()
        assert "gdpval" in registry

    def test_get_task_strategy_returns_instance(self) -> None:
        strategy = get_task_strategy("gdpval")
        assert isinstance(strategy, TaskStrategy)

    def test_get_task_strategy_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_strategy("this_task_does_not_exist")


class TestApp:
    def test_sanity(self) -> None:
        """Config instantiation + wrapper construction should not raise."""
        config = StirrupAgentWrapperConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="stirrup_agent",
            task="gdpval",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
        )
        StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))


class TestExampleDataset:
    def test_example_jsonl_is_valid(self) -> None:
        """The shipped example dataset should parse and contain the GDPVal schema."""
        example_path = STIRRUP_AGENT_DIR / "data" / "example.jsonl"
        assert example_path.is_file(), f"missing {example_path}"

        lines = example_path.read_text().strip().splitlines()
        assert len(lines) >= 1

        for line in lines:
            record = json.loads(line)
            params = record["responses_create_params"]
            metadata = params["metadata"]
            # Schema contract required by GDPValTask.extract_task_info.
            assert "task_id" in metadata
            assert "prompt" in metadata
            # Metadata must be all strings (OpenAI Metadata type constraint).
            for key, value in metadata.items():
                assert isinstance(value, str), f"metadata['{key}'] is {type(value).__name__}, not str"
            # reference_files / rubric_json are JSON-encoded strings.
            json.loads(metadata["reference_files"])
            json.loads(metadata["rubric_json"])
