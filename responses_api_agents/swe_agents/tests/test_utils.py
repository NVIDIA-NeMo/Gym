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

# Add parent directory to path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    build_nemo_skills_command,
    convert_tools_to_function_format,
    ensure_nemo_run_symlink,
    extract_problem_info,
    get_openhands_trajectory_from_completions,
    get_trajectory_and_tools,
)

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def test_extract_problem_info():
    """Test extracting problem information from request."""
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[],
        metadata={
            "problem_statement": "Fix the bug",
            "instance_id": "test-123",
            "base_commit": "abc123",
            "dataset_name": "SWE-bench",
            "split": "test",
        },
    )

    problem_info = extract_problem_info(body, "docker://container/{instance_id}")

    assert problem_info["problem_statement"] == "Fix the bug"
    assert problem_info["instance_id"] == "test-123"
    assert problem_info["base_commit"] == "abc123"
    assert problem_info["dataset_name"] == "SWE-bench"
    assert problem_info["split"] == "test"
    assert problem_info["container_formatter"] == "docker://container/{instance_id}"
    print("✓ test_extract_problem_info")


def test_build_nemo_skills_command():
    """Test building NeMo-Skills command."""
    input_file = Path("/tmp/input.jsonl")
    output_file = Path("/tmp/output.jsonl")
    model_endpoint = "http://localhost:8000/v1"

    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[],
        model="gpt-4",
        temperature=0.7,
        max_output_tokens=2048,
    )

    cmd = build_nemo_skills_command(input_file, output_file, model_endpoint, body, "swe_agent", None, 100, 1800, {})

    cmd_str = " ".join(cmd)
    assert "python" in cmd[0]
    assert "nemo_skills.inference.eval.swebench" in cmd_str
    assert "++input_file=/tmp/input.jsonl" in cmd_str
    assert "++output_file=/tmp/output.jsonl" in cmd_str
    assert "++agent_framework=swe_agent" in cmd_str
    assert "++server.model=gpt-4" in cmd_str
    assert "++server.base_url=http://localhost:8000/v1" in cmd_str
    assert "++inference.temperature=0.7" in cmd_str
    assert "++inference.tokens_to_generate=2048" in cmd_str
    print("✓ test_build_nemo_skills_command")

    # Test with agent framework repo and commit
    cmd_with_repo = build_nemo_skills_command(
        input_file,
        output_file,
        model_endpoint,
        body,
        "swe_agent",
        None,
        100,
        1800,
        {},
        agent_framework_repo="https://github.com/custom/repo.git",
        agent_framework_commit="main",
    )
    cmd_with_repo_str = " ".join(cmd_with_repo)
    assert "++agent_framework_repo=https://github.com/custom/repo.git" in cmd_with_repo_str
    assert "++agent_framework_commit=main" in cmd_with_repo_str
    print("✓ test_build_nemo_skills_command (with repo/commit)")


def test_get_openhands_trajectory_from_completions():
    """Test getting OpenHands trajectory from completion files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectories_dir = Path(tmpdir)
        instance_id = "astropy__astropy-7671"

        # Create mock completion file with realistic OpenHands format
        completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
        completions_dir.mkdir(parents=True)

        # Realistic OpenHands completion data
        completion_data = {
            "messages": [
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.",
                        }
                    ],
                    "role": "system",
                },
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "Fix the issue from an open-source repository.\n\nYour task is to fix an issue from an open-source repository.",
                        }
                    ],
                    "role": "user",
                },
            ],
            "response": {
                "choices": [
                    {
                        "message": {
                            "content": [],  # Empty content when only tool calls
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_vng0nRsLdlsS5RdiAq5TKKR5",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_bash",
                                        "arguments": '{"command":"ls -l /workspace/astropy__astropy__1.3","security_risk":"LOW"}',
                                    },
                                }
                            ],
                            "function_call": None,
                            "annotations": [],
                        },
                        "provider_specific_fields": {},
                    }
                ]
            },
            "kwargs": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "description": "Execute a bash command in the terminal.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string", "description": "The bash command to execute."},
                                    "security_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                                },
                                "required": ["command", "security_risk"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "str_replace_editor",
                            "description": "Custom editing tool for viewing, creating and editing files",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                                    },
                                    "path": {"type": "string"},
                                    "security_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                                },
                                "required": ["command", "path", "security_risk"],
                            },
                        },
                    },
                ]
            },
        }

        with open(completions_dir / "001_completion.json", "w") as f:
            json.dump(completion_data, f)

        messages, tools = get_openhands_trajectory_from_completions(trajectories_dir, instance_id)

        # Verify messages were extracted correctly
        assert len(messages) == 3  # system, user, assistant
        assert messages[0]["role"] == "system"
        assert "OpenHands agent" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "Fix the issue" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == ""  # Empty content when only tool calls
        assert "tool_calls" in messages[2]
        assert len(messages[2]["tool_calls"]) == 1
        assert messages[2]["tool_calls"][0]["function"]["name"] == "execute_bash"

        # Verify tools were extracted correctly
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "execute_bash"
        assert tools[1]["function"]["name"] == "str_replace_editor"
        print("✓ test_get_openhands_trajectory_from_completions")


def test_get_swe_agent_trajectory():
    """Test getting SWE-agent trajectory from .traj files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectories_dir = Path(tmpdir)
        instance_id = "django__django-12345"

        # Create mock .traj file with SWE-agent format
        traj_file = trajectories_dir / f"{instance_id}.traj"
        trajectory_data = {
            "history": [
                {"role": "system", "content": "You are a software development assistant.", "message_type": "system"},
                {"role": "user", "content": "Fix the bug in the Django ORM query optimizer.", "message_type": "user"},
                {
                    "role": "assistant",
                    "content": "I'll help you fix the bug. Let me first explore the codebase.",
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "function": {
                                "name": "str_replace",
                                "arguments": '{"file": "django/db/models/query.py", "old": "def optimize()", "new": "def optimize_query()"}',
                            },
                        }
                    ],
                    "message_type": "action",
                },
                {
                    "role": "tool",
                    "content": "File updated successfully",
                    "tool_call_ids": ["call_001"],
                    "message_type": "observation",
                },
            ],
            "info": {"instance_id": instance_id, "model": "gpt-4", "total_cost": 0.15},
        }

        with open(traj_file, "w") as f:
            json.dump(trajectory_data, f)

        # Use the actual SWE-agent tools configuration file
        agent_tools_file = "configs/swe_agent_tools_openai_format.json"

        # Test getting trajectory and tools
        trajectory, tools = get_trajectory_and_tools(trajectories_dir, instance_id, "swe_agent", agent_tools_file)

        # Verify trajectory was extracted correctly
        assert trajectory is not None
        assert len(trajectory) == 4  # system, user, assistant, tool messages
        assert trajectory[0]["role"] == "system"
        assert trajectory[1]["role"] == "user"
        assert trajectory[2]["role"] == "assistant"
        assert trajectory[3]["role"] == "tool"

        # Verify tools were loaded from the actual config file
        assert len(tools) > 0  # Should have tools from the actual config
        # Check that we have the expected SWE-agent tools
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "bash" in tool_names
        assert "str_replace_editor" in tool_names
        assert "submit" in tool_names

        print("✓ test_get_swe_agent_trajectory")


def test_convert_tools_to_function_format():
    """Test converting tools from ChatCompletion to FunctionTool format."""
    # Test with various tool configurations
    chat_tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace",
                "description": "Replace text in file",
                "parameters": {"type": "object", "properties": {"file": {"type": "string"}}},
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                # No strict field
            },
        },
        {
            "type": "not_function",  # Should be skipped
            "other": "data",
        },
    ]

    tools = convert_tools_to_function_format(chat_tools)

    assert len(tools) == 2  # Only the two function tools
    assert tools[0].name == "str_replace"
    assert tools[0].description == "Replace text in file"
    assert tools[0].strict is True
    assert tools[1].name == "bash"
    assert tools[1].description == "Execute bash command"
    assert tools[1].strict is None
    print("✓ test_convert_tools_to_function_format")


def test_ensure_nemo_run_symlink():
    """Test symlink creation for NeMo-Skills."""
    with patch("os.path.exists") as mock_exists:
        with patch("subprocess.run") as mock_run:
            with patch("os.listdir", return_value=["python3.10"]):
                # Simulate: venv exists, nemo_skills exists, symlink doesn't exist
                mock_exists.side_effect = [True, True, False]
                mock_run.return_value.returncode = 0

                ensure_nemo_run_symlink()

                # Should create the symlink
                assert mock_run.call_count >= 1
                print("✓ test_ensure_nemo_run_symlink")


def run_all_tests():
    """Run all tests manually."""
    print("Running utility tests...\n")

    try:
        test_extract_problem_info()
    except Exception as e:
        print(f"✗ test_extract_problem_info: {e}")

    try:
        test_build_nemo_skills_command()
    except Exception as e:
        print(f"✗ test_build_nemo_skills_command: {e}")

    try:
        test_get_openhands_trajectory_from_completions()
    except Exception as e:
        print(f"✗ test_get_openhands_trajectory_from_completions: {e}")

    try:
        test_get_swe_agent_trajectory()
    except Exception as e:
        print(f"✗ test_get_swe_agent_trajectory: {e}")

    try:
        test_convert_tools_to_function_format()
    except Exception as e:
        print(f"✗ test_convert_tools_to_function_format: {e}")

    try:
        test_ensure_nemo_run_symlink()
    except Exception as e:
        print(f"✗ test_ensure_nemo_run_symlink: {e}")

    print("\nAll utility tests completed!")


if __name__ == "__main__":
    # Try to import pytest
    try:
        import pytest

        pytest.main([__file__, "-v"])
    except ImportError:
        # Run tests manually
        run_all_tests()
