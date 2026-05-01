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
from stirrup import Agent
from stirrup.clients.utils import to_openai_messages
from stirrup.core.models import AssistantMessage, TokenUsage, ToolCall

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.stirrup_agent.app import (
    StirrupAgentWrapper,
    StirrupAgentWrapperConfig,
    _load_task_registry,
    get_task_strategy,
)
from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent, NeMoUserMessage
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
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="gdpval_resources_server",
            ),
        )
        StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_summarization_history_restores_tool_messages_for_openai(self, monkeypatch) -> None:
        """NeMo user-role tool results must become tool messages for model calls."""
        messages = [
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(tool_call_id="call_1", name="code_exec", arguments='{"cmd":"true"}')],
                token_usage=TokenUsage(input=1, answer=1, reasoning=0),
            ),
            NeMoUserMessage(content="ok", name="code_exec", success=True, tool_call_id="call_1"),
        ]

        captured_messages = None

        async def capture_summarization_messages(_self, messages):
            nonlocal captured_messages
            captured_messages = messages
            return messages

        monkeypatch.setattr(Agent, "summarize_messages", capture_summarization_messages)
        agent = NeMoAgent(client=MagicMock(), name="stirrup_agent", tools=[], tool_response_as_user=True)

        await agent.summarize_messages(messages)

        assert captured_messages is not None
        openai_messages = to_openai_messages(captured_messages)

        assert openai_messages[0]["role"] == "assistant"
        assert openai_messages[0]["tool_calls"][0]["id"] == "call_1"
        assert openai_messages[1]["role"] == "tool"
        assert openai_messages[1]["tool_call_id"] == "call_1"


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
