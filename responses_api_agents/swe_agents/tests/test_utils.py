# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.swe_agents.utils import (
    _extract_text_from_message,
    _get_workspace_root,
    _resolve_setup_directory,
    _run_setup_shell_script,
    _setup_directory_lock,
    convert_tools_to_function_format,
    convert_trajectory_to_output_items,
    extract_data_from_trajectory,
    extract_input_messages_from_trajectory,
    extract_messages,
    extract_problem_info,
    get_model_endpoint,
    get_openhands_trajectory_from_completions,
    get_trajectory_and_tools,
)


class TestExtractProblemInfo:
    def test_extract_problem_info(self) -> None:
        """Test extracting problem information from request."""
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[],
            metadata={
                "problem_statement": "Fix the bug",
                "instance_id": "test-123",
                "base_commit": "abc123",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": "{}",
            },
        )

        problem_info = extract_problem_info(body, "docker://container/{instance_id}")

        assert problem_info["problem_statement"] == "Fix the bug"
        assert problem_info["instance_id"] == "test-123"
        assert problem_info["base_commit"] == "abc123"
        assert problem_info["dataset_name"] == "SWE-bench"
        assert problem_info["split"] == "test"
        assert problem_info["container_formatter"] == "docker://container/{instance_id}"

    def test_extract_problem_info_with_list_container_formatter(self) -> None:
        """Test extracting problem information with list container_formatter."""
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[],
            metadata={
                "problem_statement": "Fix the bug",
                "instance_id": "test-123",
                "base_commit": "abc123",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": "{}",
            },
        )

        container_formatters = [
            "docker://container1/{instance_id}",
            "docker://container2/{instance_id}",
        ]
        problem_info = extract_problem_info(body, container_formatters)

        assert problem_info["container_formatter"] == container_formatters


class TestExtractTextFromMessage:
    def test_extract_text_from_message(self) -> None:
        message = SimpleNamespace(content=[{"type": "input_text", "text": "hello"}])
        assert _extract_text_from_message(message) == "hello"

    def test_extract_text_from_message_missing_content(self) -> None:
        assert _extract_text_from_message(SimpleNamespace()) is None
        message = SimpleNamespace(content=[{"type": "text", "text": "ignored"}])
        assert _extract_text_from_message(message) is None


class TestExtractInputMessagesFromTrajectory:
    def test_stops_on_assistant(self) -> None:
        system_msg = SimpleNamespace(role="system", content=[{"type": "input_text", "text": "sys"}])
        dev_msg = SimpleNamespace(role="developer", content=[{"type": "input_text", "text": "dev"}])
        assistant_msg = SimpleNamespace(role="assistant", content="hi")

        input_messages, filtered_output = extract_input_messages_from_trajectory([system_msg, dev_msg, assistant_msg])

        assert [m.role for m in input_messages] == ["system", "developer"]
        assert all(isinstance(m, NeMoGymEasyInputMessage) for m in input_messages)
        assert filtered_output == [assistant_msg]

    def test_empty_response_output(self) -> None:
        input_messages, filtered_output = extract_input_messages_from_trajectory([])
        assert input_messages == []
        assert filtered_output == []

    def test_none_response_output(self) -> None:
        input_messages, filtered_output = extract_input_messages_from_trajectory(None)
        assert input_messages == []
        assert filtered_output == []

    def test_stops_on_function_call(self) -> None:
        system_msg = SimpleNamespace(role="system", content=[{"type": "input_text", "text": "sys"}])
        func_call = SimpleNamespace(type="function_call", call_id="call1")

        input_messages, filtered_output = extract_input_messages_from_trajectory([system_msg, func_call])

        assert len(input_messages) == 1
        assert filtered_output == [func_call]

    def test_item_without_role_goes_to_filtered(self) -> None:
        no_role_item = SimpleNamespace(data="some data")
        assistant_msg = SimpleNamespace(role="assistant", content="hi")

        input_messages, filtered_output = extract_input_messages_from_trajectory([no_role_item, assistant_msg])

        assert input_messages == []
        assert filtered_output == [no_role_item, assistant_msg]


class TestConvertTrajectoryToOutputItems:
    def test_openhands(self) -> None:
        trajectory = [
            {"role": "system", "content": [{"type": "text", "text": "sys prompt"}]},
            {
                "role": "assistant",
                "content": "First reply",
                "tool_calls": [{"id": "call1", "function": {"name": "tool_fn", "arguments": '{"k": "v"}'}}],
            },
            {"role": "tool", "content": "tool output", "tool_call_id": "call1"},
        ]

        output_items = convert_trajectory_to_output_items(trajectory, "openhands")

        assert output_items[0].role == "system"
        assert output_items[0].content[0]["text"] == "sys prompt"
        assert output_items[1].role == "assistant"
        assert output_items[1].content[0].text == "First reply"
        assert output_items[2].name == "tool_fn"
        assert output_items[3].call_id == "call1"
        assert output_items[3].output == "tool output"

    def test_swe_agent(self) -> None:
        trajectory = [
            {"role": "system", "content": "sys prompt"},
            {
                "role": "assistant",
                "content": "assistant reply",
                "tool_calls": [{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}],
                "provider_specific_fields": {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [0.1, 0.2],
                },
            },
            {"role": "tool", "content": "ok", "tool_call_ids": ["tc1"]},
        ]

        output_items = convert_trajectory_to_output_items(trajectory, "swe_agent")

        assert output_items[0].role == "system"
        assert output_items[1].role == "assistant"
        assert output_items[2].name == "bash"
        assert output_items[3].call_id == "tc1"

    def test_openhands_string_content(self) -> None:
        trajectory = [{"role": "user", "content": "Simple string content"}]
        output_items = convert_trajectory_to_output_items(trajectory, "openhands")
        assert len(output_items) == 1
        assert output_items[0].content[0]["text"] == "Simple string content"

    def test_openhands_tool_with_tool_call_ids_list(self) -> None:
        trajectory = [{"role": "tool", "content": "tool result", "tool_call_ids": ["call_abc123"]}]
        output_items = convert_trajectory_to_output_items(trajectory, "openhands")
        assert len(output_items) == 1
        assert output_items[0].call_id == "call_abc123"

    def test_swe_agent_function_as_string(self) -> None:
        trajectory = [
            {
                "role": "assistant",
                "content": "response",
                "tool_calls": [{"id": "tc1", "function": '{"name": "bash", "arguments": "{}"}'}],
            },
        ]
        output_items = convert_trajectory_to_output_items(trajectory, "swe_agent")
        assert len(output_items) == 2
        assert output_items[1].name == "bash"

    def test_swe_agent_function_as_invalid_string(self) -> None:
        trajectory = [
            {
                "role": "assistant",
                "content": "response",
                "tool_calls": [{"id": "tc1", "function": "invalid_function_name"}],
            },
        ]
        output_items = convert_trajectory_to_output_items(trajectory, "swe_agent")
        assert len(output_items) == 2
        assert output_items[1].name == "invalid_function_name"


class TestExtractMessages:
    def test_with_tool_calls(self) -> None:
        trajectory_item = {
            "response": "assistant msg",
            "observation": "tool output",
            "tool_calls": [{"id": "call-123", "function": {"name": "bash"}}],
            "tool_call_ids": ["call-123"],
        }
        messages = extract_messages(trajectory_item)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "tool"

    def test_non_dict_trajectory_item(self) -> None:
        assert extract_messages("not a dict") == []
        assert extract_messages(None) == []
        assert extract_messages([1, 2, 3]) == []

    def test_extra_info_not_dict(self) -> None:
        trajectory_item = {"response": "msg", "extra_info": "not a dict", "tool_calls": None}
        messages = extract_messages(trajectory_item)
        assert len(messages) == 1
        assert messages[0]["provider_specific_fields"] == {}


class TestExtractDataFromTrajectory:
    def test_standard_path(self) -> None:
        trajectory_data = [
            {
                "query": [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}],
                "response": "resp",
                "observation": "obs",
                "tool_calls": [{"id": "c1"}],
                "tool_call_ids": ["c1"],
            }
        ]
        result = extract_data_from_trajectory(trajectory_data, history=[])
        assert [msg["role"] for msg in result] == ["system", "user", "assistant", "tool"]

    def test_empty_trajectories(self) -> None:
        assert extract_data_from_trajectory([], []) == []

    def test_last_item_not_dict(self) -> None:
        assert extract_data_from_trajectory(["not a dict"], []) == []

    def test_query_key_missing(self) -> None:
        assert extract_data_from_trajectory([{"other_key": "value"}], []) == []


class TestGetModelEndpoint:
    def test_uses_server_config(self) -> None:
        with (
            patch("responses_api_agents.swe_agents.utils.ServerClient.load_from_global_config") as mock_load,
            patch("responses_api_agents.swe_agents.utils.get_first_server_config_dict") as mock_get_first,
        ):
            mock_load.return_value = SimpleNamespace(global_config_dict={"servers": []})
            mock_get_first.return_value = {"host": "localhost", "port": 9999}
            endpoint = get_model_endpoint("my-model")
            assert endpoint == "http://localhost:9999/v1"


class TestResolveSetupDirectory:
    def test_uses_workspace_root(self) -> None:
        with patch("responses_api_agents.swe_agents.utils._get_workspace_root", return_value=Path("/tmp/workspace")):
            resolved = _resolve_setup_directory(None, "subdir")
            assert resolved == Path("/tmp/workspace/subdir").resolve()


class TestSetupDirectoryLock:
    def test_creates_lock_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir) / "lock-target"
            setup_dir.mkdir()
            lock_file = setup_dir.parent / f".{setup_dir.name}.lock"
            with _setup_directory_lock(setup_dir, "test-lock"):
                assert lock_file.exists()


class TestRunSetupShellScript:
    def test_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir)
            script_content = f'#!/bin/bash\necho "hello" > "{setup_dir}/out.txt"\n'
            _run_setup_shell_script(
                setup_dir=setup_dir,
                script_name="setup_test.sh",
                script_content=script_content,
                timeout_seconds=5,
                label="TestSetup",
            )
            assert (setup_dir / "out.txt").read_text().strip() == "hello"

    def test_script_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir)
            script_content = "#!/bin/bash\nexit 1\n"
            with pytest.raises(RuntimeError, match="setup failed with return code 1"):
                _run_setup_shell_script(
                    setup_dir=setup_dir,
                    script_name="failing_script.sh",
                    script_content=script_content,
                    timeout_seconds=5,
                    label="FailingSetup",
                )


class TestGetOpenhandsTrajectoryFromCompletions:
    def test_get_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectories_dir = Path(tmpdir)
            instance_id = "astropy__astropy-7671"

            completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)

            completion_data = {
                "messages": [
                    {"content": [{"type": "text", "text": "OpenHands agent"}], "role": "system"},
                    {"content": [{"type": "text", "text": "Fix the issue"}], "role": "user"},
                ],
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": [],
                                "role": "assistant",
                                "tool_calls": [
                                    {"id": "call_1", "type": "function", "function": {"name": "execute_bash"}}
                                ],
                            }
                        }
                    ]
                },
                "kwargs": {"tools": [{"type": "function", "function": {"name": "execute_bash"}}]},
            }

            with open(completions_dir / "001_completion.json", "w") as f:
                json.dump(completion_data, f)

            messages, tools = get_openhands_trajectory_from_completions(trajectories_dir, instance_id)

            assert len(messages) == 3
            assert messages[0]["role"] == "system"
            assert len(tools) == 1

    def test_missing_completions_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            messages, tools = get_openhands_trajectory_from_completions(Path(tmpdir), "nonexistent")
            assert messages == []
            assert tools == []

    def test_invalid_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectories_dir = Path(tmpdir)
            instance_id = "test-instance"
            completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)
            with open(completions_dir / "001_completion.json", "w") as f:
                f.write("not valid json")
            messages, tools = get_openhands_trajectory_from_completions(trajectories_dir, instance_id)
            assert messages == []
            assert tools == []


class TestGetSweAgentTrajectory:
    def test_get_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectories_dir = Path(tmpdir)
            instance_id = "django__django-12345"
            traj_file = trajectories_dir / f"{instance_id}.traj"
            trajectory_data = {
                "history": [],
                "trajectory": [
                    {
                        "query": [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}],
                        "response": "resp",
                        "observation": "obs",
                        "tool_calls": [{"id": "c1"}],
                        "tool_call_ids": ["c1"],
                    }
                ],
            }
            with open(traj_file, "w") as f:
                json.dump(trajectory_data, f)

            trajectory, tools = get_trajectory_and_tools(
                trajectories_dir, instance_id, "swe_agent", "configs/swe_agent_tools_openai_format.json"
            )
            assert trajectory is not None

    def test_unsupported_agent_framework(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory, tools = get_trajectory_and_tools(Path(tmpdir), "instance", "unknown", None)
            assert trajectory is None
            assert tools == []


class TestConvertToolsToFunctionFormat:
    def test_convert_tools(self) -> None:
        chat_tools = [
            {"type": "function", "function": {"name": "bash", "description": "Run bash", "parameters": {}}},
            {"type": "not_function", "other": "data"},
        ]
        tools = convert_tools_to_function_format(chat_tools)
        assert len(tools) == 1
        assert tools[0].name == "bash"


class TestGetWorkspaceRoot:
    def test_returns_path(self) -> None:
        result = _get_workspace_root()
        assert isinstance(result, Path)
        assert result.exists()


class TestSetupSwebenchEnvironment:
    def test_already_exists(self) -> None:
        from responses_api_agents.swe_agents.utils import setup_swebench_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir)
            swebench_dir = setup_dir / "SWE-bench"
            swebench_dir.mkdir()
            (swebench_dir / "venv" / "bin").mkdir(parents=True)
            (swebench_dir / "venv" / "bin" / "python").touch()
            (setup_dir / "uv").mkdir()
            (setup_dir / "python").mkdir()
            result = setup_swebench_environment(setup_dir=setup_dir)
            assert result == setup_dir


class TestSetupR2eGymEnvironment:
    def test_already_exists(self) -> None:
        from responses_api_agents.swe_agents.utils import setup_r2e_gym_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir)
            r2e_dir = setup_dir / "R2E-Gym"
            r2e_dir.mkdir()
            (r2e_dir / "venv" / "bin").mkdir(parents=True)
            (r2e_dir / "venv" / "bin" / "python").touch()
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = SimpleNamespace(returncode=0)
                result = setup_r2e_gym_environment(setup_dir=setup_dir)
                assert result == setup_dir


class TestSetupOpenhandsEnvironment:
    def test_already_exists(self) -> None:
        from responses_api_agents.swe_agents.utils import setup_openhands_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir)
            oh_dir = setup_dir / "OpenHands"
            oh_dir.mkdir()
            (oh_dir / ".venv" / "bin").mkdir(parents=True)
            (oh_dir / ".venv" / "bin" / "python").touch()
            (setup_dir / "miniforge3").mkdir()
            result = setup_openhands_environment(setup_dir=setup_dir)
            assert result == setup_dir


class TestRunSwebenchEvaluation:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        from responses_api_agents.swe_agents.utils import run_swebench_evaluation

        with tempfile.TemporaryDirectory() as tmpdir:
            problem_info = {
                "instance_id": "test-123",
                "container_formatter": "/mock/{instance_id}.sif",
            }
            body = NeMoGymResponseCreateParamsNonStreaming(
                input=[],
                model="test-model",
                metadata={
                    "problem_statement": "Fix",
                    "instance_id": "test-123",
                    "base_commit": "abc",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": "{}",
                },
            )
            mock_result = {"trajectory": [{"role": "assistant", "content": "Done"}], "tools": []}

            with patch("responses_api_agents.swe_agents.utils.RunOpenHandsAgent") as mock_agent_class:
                mock_agent = mock_agent_class.return_value
                mock_agent.process_single_datapoint = AsyncMock(return_value=mock_result)

                result = await run_swebench_evaluation(
                    problem_info=problem_info,
                    model_endpoint="http://localhost:8000/v1",
                    body=body,
                    run_session_id="test-session",
                    agent_framework="openhands",
                    agent_config=None,
                    agent_tools_file=None,
                    agent_max_turns=100,
                    swebench_agent_timeout=2700,
                    swebench_tests_timeout=1800,
                    openhands_setup_dir=Path(tmpdir),
                    swebench_setup_dir=Path(tmpdir),
                    r2e_gym_setup_dir=Path(tmpdir),
                    dataset_path="/mock/dataset.jsonl",
                    instance_dir="test_instance",
                )
                assert "tools" in result
                assert "trajectory" in result
