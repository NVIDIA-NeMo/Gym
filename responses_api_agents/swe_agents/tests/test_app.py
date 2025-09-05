"""Tests for SWE agents."""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock, mock_open

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
    SWEBenchRunRequest,
    SWEBenchVerifyResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputMessage,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return SWEBenchWrapperConfig(
        host="localhost",
        port=9003,
        entrypoint="responses_api_agents/swe_agents",
        agent_framework="swe_agent",
        agent_config="eval/swe-bench/swe-agent/default_one_tool",
        agent_tools_file="configs/swe_agent_tools_openai_format.json",
        agent_max_turns=100,
        container_formatter="docker://swebench/sweb.eval.x86_64.{instance_id}",
        swebench_tests_timeout=1800,
        nemo_skills_config={
            "max_samples": -1,
            "skip_filled": False,
            "dry_run": False,
        }
    )


@pytest.fixture
def openhands_config():
    """Create OpenHands test configuration."""
    return SWEBenchWrapperConfig(
        host="localhost",
        port=9003,
        entrypoint="responses_api_agents/swe_agents",
        agent_framework="openhands",
        agent_config="eval/swe-bench/openhands/default",
        agent_max_turns=100,
        container_formatter="docker://swebench/sweb.eval.x86_64.{instance_id}",
        swebench_tests_timeout=1800,
    )


@pytest.fixture
def agent(config):
    """Create test agent instance."""
    with patch('app.NEMO_SKILLS_AVAILABLE', True):
        with patch.object(SWEBenchWrapper, '_ensure_nemo_run_symlink'):
            agent = SWEBenchWrapper(config=config)
            agent.server_client = AsyncMock()
            return agent


@pytest.fixture
def openhands_agent(openhands_config):
    """Create OpenHands test agent instance."""
    with patch('app.NEMO_SKILLS_AVAILABLE', True):
        with patch.object(SWEBenchWrapper, '_ensure_nemo_run_symlink'):
            agent = SWEBenchWrapper(config=openhands_config)
            agent.server_client = AsyncMock()
            return agent


@pytest.mark.asyncio
async def test_extract_problem_info(agent):
    """Test problem info extraction from request."""
    # Test with string input
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Fix the bug in DataFrame.groupby()",
        metadata={
            "instance_id": "pandas__pandas-12345",
            "base_commit": "abc123",
            "dataset_name": "custom/dataset",
            "split": "dev",
        }
    )
    
    problem_info = agent._extract_problem_info(body)
    
    assert problem_info["problem_statement"] == "Fix the bug in DataFrame.groupby()"
    assert problem_info["instance_id"] == "pandas__pandas-12345"
    assert problem_info["base_commit"] == "abc123"
    assert problem_info["dataset_name"] == "custom/dataset"
    assert problem_info["split"] == "dev"
    assert problem_info["container_formatter"] == "docker://swebench/sweb.eval.x86_64.{instance_id}"
    
    # Test with message list input
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "I'll help"},
            {"role": "user", "content": "The bug is in groupby"},
        ],
        metadata={"instance_id": "test-123"}
    )
    
    problem_info = agent._extract_problem_info(body)
    assert problem_info["problem_statement"] == "The bug is in groupby"
    assert problem_info["instance_id"] == "test-123"
    assert problem_info["dataset_name"] == "princeton-nlp/SWE-bench_Verified"  # default
    assert problem_info["split"] == "test"  # default


def test_get_model_endpoint_with_server_config(agent):
    """Test model endpoint determination with server configuration."""
    body = NeMoGymResponseCreateParamsNonStreaming(input="test")
    
    # Mock the server configuration
    with patch('app.ServerClient') as mock_client:
        with patch('app.get_first_server_config_dict') as mock_get_config:
            mock_client.load_from_global_config.return_value.global_config_dict = {
                "servers": {"openai_model": {"host": "test-host", "port": 8000}}
            }
            mock_get_config.return_value = {"host": "test-host", "port": 8000}
            
            endpoint = agent._get_model_endpoint(body)
            assert endpoint == "http://test-host:8000/v1"


def test_get_model_endpoint_error(agent):
    """Test model endpoint error handling."""
    body = NeMoGymResponseCreateParamsNonStreaming(input="test")
    
    # Mock server configuration to raise error
    with patch('app.ServerClient') as mock_client:
        mock_client.load_from_global_config.side_effect = Exception("Config error")
        
        with pytest.raises(RuntimeError, match="Could not determine model endpoint"):
            agent._get_model_endpoint(body)


def test_build_nemo_skills_command(agent):
    """Test command building for NeMo-Skills."""
    input_file = Path("/tmp/input.jsonl")
    output_file = Path("/tmp/output.jsonl")
    model_endpoint = "http://test:8000/v1"
    
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="test",
        model="gpt-4",
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=2048,
        inference={
            "top_k": 50,
            "random_seed": 42,
        }
    )
    
    cmd = agent._build_nemo_skills_command(
        input_file, output_file, model_endpoint, body
    )
    
    # Check basic command structure
    assert cmd[0].endswith("python") or cmd[0] == "python"
    assert cmd[1:3] == ["-m", "nemo_skills.inference.eval.swebench"]
    
    # Check parameters are included
    cmd_str = " ".join(cmd)
    assert "++input_file=/tmp/input.jsonl" in cmd_str
    assert "++output_file=/tmp/output.jsonl" in cmd_str
    assert "++agent_framework=swe_agent" in cmd_str
    assert "++server.model=gpt-4" in cmd_str
    assert "++server.base_url=http://test:8000/v1" in cmd_str
    assert "++inference.temperature=0.7" in cmd_str
    assert "++inference.top_p=0.9" in cmd_str
    assert "++inference.tokens_to_generate=2048" in cmd_str
    assert "++inference.top_k=50" in cmd_str
    assert "++inference.random_seed=42" in cmd_str
    assert "++agent_max_turns=100" in cmd_str
    assert "++swebench_tests_timeout=1800" in cmd_str


def test_build_nemo_skills_command_with_config(agent):
    """Test command building with agent config."""
    input_file = Path("/tmp/input.jsonl")
    output_file = Path("/tmp/output.jsonl")
    model_endpoint = "http://test:8000/v1"
    
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="test",
        model="gpt-4",
    )
    
    cmd = agent._build_nemo_skills_command(
        input_file, output_file, model_endpoint, body
    )
    
    cmd_str = " ".join(cmd)
    assert "++agent_config=eval/swe-bench/swe-agent/default_one_tool" in cmd_str


@pytest.mark.asyncio
async def test_get_openhands_trajectory_from_completions(openhands_agent):
    """Test OpenHands trajectory extraction from completion files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectories_dir = Path(tmpdir)
        instance_id = "test-123"
        
        # Create mock completion file structure
        completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
        completions_dir.mkdir(parents=True)
        
        # Create mock completion file
        completion_data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Fix the bug"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "I'll help"}]},
            ],
            "response": {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Fixed!"}],
                        "tool_calls": []
                    }
                }]
            },
            "kwargs": {
                "tools": [
                    {"type": "function", "function": {"name": "test_tool"}}
                ]
            }
        }
        
        completion_file = completions_dir / "001_completion.json"
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f)
        
        messages, tools = openhands_agent._get_openhands_trajectory_from_completions(
            trajectories_dir, instance_id
        )
        
        assert len(messages) == 3
        assert messages[0]["content"] == "Fix the bug"
        assert messages[1]["content"] == "I'll help"
        assert messages[2]["content"] == "Fixed!"
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_get_openhands_trajectory_empty_content(openhands_agent):
    """Test OpenHands trajectory with empty content lists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectories_dir = Path(tmpdir)
        instance_id = "test-123"
        
        completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
        completions_dir.mkdir(parents=True)
        
        # Create completion with empty content (tool-only message)
        completion_data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Fix bug"}]},
            ],
            "response": {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": [],  # Empty content list
                        "tool_calls": [{"id": "call_1", "function": {"name": "bash"}}]
                    }
                }]
            },
            "kwargs": {"tools": []}
        }
        
        completion_file = completions_dir / "001_completion.json"
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f)
        
        messages, tools = openhands_agent._get_openhands_trajectory_from_completions(
            trajectories_dir, instance_id
        )
        
        assert len(messages) == 2
        assert messages[1]["content"] == ""  # Empty content converted to empty string
        assert "tool_calls" in messages[1]


@pytest.mark.asyncio
async def test_responses_success(agent):
    """Test successful SWE-bench evaluation."""
    mock_result = {
        "instance_id": "test-123",
        "swe-bench-metrics": {
            "resolved": True,
            "patch_exists": True,
            "patch_successfully_applied": True,
        },
        "swe-bench-outputs": {
            "model_patch": "diff --git a/file.py...",
        },
        "trajectory": [{"action": "test"}]
    }
    
    with patch('app.ServerClient'):
        with patch('app.get_first_server_config_dict') as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 8000}
            
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # Setup mock process
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.stdout.readline = AsyncMock(side_effect=[
                    b"Processing...\n",
                    b"",  # EOF
                ])
                mock_process.wait = AsyncMock(return_value=0)
                mock_subprocess.return_value = mock_process
                
                # Mock file operations
                with patch("builtins.open", mock_open(read_data=json.dumps(mock_result))):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("pathlib.Path.mkdir"):
                            body = NeMoGymResponseCreateParamsNonStreaming(
                                input="Fix the bug",
                                metadata={"instance_id": "test-123"},
                                model="gpt-4",
                            )
                            
                            response = await agent.responses(body)
                            
                            assert response.id == "swebench-test-123"
                            assert len(response.output) == 1
                            assert response.output[0].type == "message"
                            assert isinstance(response.output[0], NeMoGymResponseOutputMessage)
                            assert response.metadata["agent_framework"] == "swe_agent"


@pytest.mark.asyncio
async def test_responses_error(agent):
    """Test error handling in SWE-bench evaluation."""
    with patch('app.ServerClient'):
        with patch('app.get_first_server_config_dict') as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 8000}
            
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # Setup mock process that fails
                mock_process = AsyncMock()
                mock_process.returncode = 1
                mock_process.stdout.readline = AsyncMock(side_effect=[
                    b"Error: Container not found\n",
                    b"",  # EOF
                ])
                mock_process.wait = AsyncMock(return_value=1)
                mock_subprocess.return_value = mock_process
                
                with patch("pathlib.Path.mkdir"):
                    with patch("builtins.open", mock_open()):
                        body = NeMoGymResponseCreateParamsNonStreaming(
                            input="Fix the bug",
                            metadata={"instance_id": "test-123"},
                        )
                        
                        response = await agent.responses(body)
                        
                        assert "error" in response.id
                        assert "Error:" in response.output[0].content[0].text
                        assert "error" in response.metadata


@pytest.mark.asyncio
async def test_run_method(agent):
    """Test the run method for verification."""
    mock_result = {
        "instance_id": "test-123",
        "swe-bench-metrics": {
            "resolved": True,
            "patch_exists": True,
            "patch_successfully_applied": True,
        },
        "swe-bench-outputs": {
            "model_patch": "diff...",
        }
    }
    
    # Create a mock response with proper structure
    mock_response_output = NeMoGymResponseOutputMessage(
        id="msg-test-123",
        content=[
            NeMoGymResponseOutputText(
                type="output_text",
                text=json.dumps(mock_result),
                annotations=[]
            )
        ],
        role="assistant",
        status="completed",
        type="message"
    )
    
    mock_response = NeMoGymResponse(
        id="swebench-test-123",
        created_at=1234567890,
        model="gpt-4",
        object="response",
        output=[mock_response_output],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        metadata={"swebench_result": mock_result}
    )
    
    # Mock the responses method
    agent.responses = AsyncMock(return_value=mock_response)
    
    request = SWEBenchRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input="Fix bug",
            metadata={"instance_id": "test-123"},
        )
    )
    
    result = await agent.run(request)
    
    assert isinstance(result, SWEBenchVerifyResponse)
    assert result.reward == 1.0
    assert result.swebench_metrics["resolved"] is True
    assert result.metadata["instance_id"] == "test-123"
    assert result.metadata["patch_exists"] is True


@pytest.mark.asyncio
async def test_run_method_failure(agent):
    """Test the run method with failed resolution."""
    mock_result = {
        "instance_id": "test-123",
        "swe-bench-metrics": {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
        }
    }
    
    mock_response_output = NeMoGymResponseOutputMessage(
        id="msg-test-123",
        content=[
            NeMoGymResponseOutputText(
                type="output_text",
                text=json.dumps(mock_result),
                annotations=[]
            )
        ],
        role="assistant",
        status="completed",
        type="message"
    )
    
    mock_response = NeMoGymResponse(
        id="swebench-test-123",
        created_at=1234567890,
        model="gpt-4",
        object="response",
        output=[mock_response_output],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        metadata={"swebench_result": mock_result}
    )
    
    agent.responses = AsyncMock(return_value=mock_response)
    
    request = SWEBenchRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input="Fix bug",
            metadata={"instance_id": "test-123"},
        )
    )
    
    result = await agent.run(request)
    
    assert result.reward == 0.0
    assert result.swebench_metrics["resolved"] is False
    assert result.metadata["patch_successfully_applied"] is False


def test_ensure_nemo_run_symlink():
    """Test symlink creation for NeMo-Skills."""
    with patch('app.NEMO_SKILLS_AVAILABLE', True):
        with patch('os.path.exists') as mock_exists:
            with patch('subprocess.run') as mock_run:
                mock_exists.side_effect = [
                    True,   # venv_lib exists
                    True,   # nemo_skills path exists
                    False,  # /nemo_run/code doesn't exist
                ]
                mock_run.return_value.returncode = 0
                
                with patch('os.listdir', return_value=['python3.10']):
                    agent = SWEBenchWrapper(config=SWEBenchWrapperConfig())
                    
                    # Check that mkdir and ln commands were called
                    assert mock_run.call_count >= 1


def test_nemo_skills_not_available():
    """Test that agent raises error when NeMo-Skills is not available."""
    with patch('app.NEMO_SKILLS_AVAILABLE', False):
        with pytest.raises(ImportError, match="NeMo-Skills is required"):
            SWEBenchWrapper(config=SWEBenchWrapperConfig())


@pytest.mark.asyncio
async def test_trajectory_loading_swe_agent(agent):
    """Test SWE-agent trajectory loading."""
    mock_result = {
        "instance_id": "test-123",
        "swe-bench-metrics": {"resolved": True},
    }
    
    mock_trajectory = {"history": [{"action": "edit", "file": "test.py"}]}
    mock_tools = {"tools": [{"type": "function", "function": {"name": "str_replace"}}]}
    
    with patch('app.ServerClient'):
        with patch('app.get_first_server_config_dict') as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 8000}
            
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.stdout.readline = AsyncMock(side_effect=[b"", ])
                mock_process.wait = AsyncMock(return_value=0)
                mock_subprocess.return_value = mock_process
                
                with patch("builtins.open", mock_open(read_data=json.dumps(mock_result))) as m:
                    # Add side effect for trajectory and tools files
                    def open_side_effect(file, *args, **kwargs):
                        file_str = str(file)
                        if ".traj" in file_str:
                            return mock_open(read_data=json.dumps(mock_trajectory))()
                        elif "swe_agent_tools" in file_str:
                            return mock_open(read_data=json.dumps(mock_tools))()
                        else:
                            return mock_open(read_data=json.dumps(mock_result))()
                    
                    m.side_effect = open_side_effect
                    
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("pathlib.Path.mkdir"):
                            with patch("pathlib.Path.glob") as mock_glob:
                                mock_glob.return_value = [Path("/tmp/test.traj")]
                                
                                body = NeMoGymResponseCreateParamsNonStreaming(
                                    input="Fix bug",
                                    metadata={"instance_id": "test-123"},
                                )
                                
                                response = await agent.responses(body)
                                
                                # Check that trajectory was processed
                                assert response.id == "swebench-test-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])