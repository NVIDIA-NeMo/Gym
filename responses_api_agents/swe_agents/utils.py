"""Utility functions for SWE-bench agent wrapper."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymFunctionCallOutput,
    NeMoGymMessage,
    NeMoGymEasyInputMessage,
)

LOG = logging.getLogger(__name__)


def extract_problem_info(body: NeMoGymResponseCreateParamsNonStreaming, container_formatter: str) -> Dict:
    """Extract SWE-bench problem information from request.
    
    Args:
        body: Request body with parameters
        container_formatter: Container path template
        
    Returns:
        Dictionary with problem information
    """

    # Get metadata
    metadata = getattr(body, "metadata", {})
    
    # Build problem info
    problem_info = {
        "problem_statement": metadata["problem_statement"],
        "instance_id": metadata["instance_id"],
        "base_commit": metadata["base_commit"],
        "dataset_name": metadata["dataset_name"],
        "split": metadata["split"],
        "container_formatter": container_formatter,
    }
    
    
    return problem_info


def extract_input_messages_from_trajectory(response_output: List) -> Tuple[List[NeMoGymEasyInputMessage], List]:
    """Extract initial input messages from response output and return filtered output.
    
    These are the system/user messages that were actually sent to the agent,
    which should be populated in the input field of responses_create_params.
    
    Args:
        response_output: List of NeMoGymResponseOutputItem objects from the response
        
    Returns:
        Tuple of (input_messages, filtered_output):
        - input_messages: List of NeMoGymEasyInputMessage objects
        - filtered_output: List with system/user/developer messages removed
    """
    input_messages = []
    filtered_output = []
    
    if not response_output:
        return input_messages, []
    
    # Find where the assistant/function calls start
    for i, item in enumerate(response_output):
        # Check if this is an assistant message or function call
        is_assistant = hasattr(item, 'role') and item.role == 'assistant'
        is_function = hasattr(item, 'type') and item.type in ['function_call', 'function_call_output']
        
        if is_assistant or is_function:
            # Add all remaining items starting from this one
            filtered_output.extend(response_output[i:])
            break
            
        # Process system/user/developer messages
        if hasattr(item, 'role') and item.role in ['system', 'user', 'developer']:
            # Try to extract text content
            text_content = _extract_text_from_message(item)
            if text_content:
                input_messages.append(NeMoGymEasyInputMessage(
                    role=item.role,
                    content=text_content,
                    type="message"
                ))
                # Skip adding to filtered output (we're removing these)
                continue
        
        # Add any other items to filtered output (edge case handling)
        filtered_output.append(item)
    
    return input_messages, filtered_output


def _extract_text_from_message(item) -> Optional[str]:
    """Helper to extract text content from a message item."""
    if not (hasattr(item, 'content') and item.content):
        return None
    
    for content_item in item.content:
        if isinstance(content_item, dict) and content_item.get('type') == 'input_text':
            return content_item.get('text', '')
    
    return None


def convert_trajectory_to_output_items(
    trajectory: List[Any], 
    problem_info: Dict,
    agent_framework: str
) -> List[NeMoGymResponseOutputItem]:
    """Convert trajectory data to NeMoGym output items.
    
    Args:
        trajectory: Raw trajectory data
        problem_info: Problem information
        agent_framework: Agent framework (swe_agent or openhands)
        
    Returns:
        List of NeMoGym output items
    """
    output_items = []
    
    # For OpenHands, trajectory is already in OpenAI format
    if agent_framework == "openhands" and isinstance(trajectory, list):
        for item in trajectory:
            if isinstance(item, dict):
                role = item.get("role", "")
                
                if role in ["user", "system", "developer"]:
                    # Create input message
                    content_data = item.get("content", "")
                    text_content = ""
                    
                    if isinstance(content_data, str):
                        text_content = content_data
                    elif isinstance(content_data, list):
                        # Handle list of content items
                        for c in content_data:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text_content = c.get("text", "")
                                break  # Take first text content
                    
                    if text_content:
                        output_items.append(NeMoGymMessage(
                            content=[{"type": "input_text", "text": text_content}],
                            role=role,
                            status="completed",
                            type="message"
                        ))
                
                elif role == "assistant":
                    # Handle assistant messages with potential tool calls
                    tool_calls = item.get("tool_calls", [])
                    content_data = item.get("content", "")
                    
                    # Add assistant message if there's content (even if there are also tool calls)
                    if content_data:
                        output_items.append(NeMoGymResponseOutputMessage(
                            id=f"msg-{len(output_items)}",
                            content=[
                                NeMoGymResponseOutputText(
                                    type="output_text",
                                    text=content_data if isinstance(content_data, str) else str(content_data),
                                    annotations=[]
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message"
                        ))
                    
                    # Also add tool calls if present  
                    if tool_calls:
                        # Create function call items
                        for tc in tool_calls:
                            if "function" in tc:
                                output_items.append(NeMoGymResponseFunctionToolCall(
                                    arguments=tc["function"].get("arguments", ""),
                                    call_id=tc.get("id", ""),
                                    name=tc["function"].get("name", ""),
                                    type="function_call",
                                    id=tc.get("id"),
                                    status="completed"
                                ))
                
                elif role == "tool":
                    # Tool response
                    content = item.get("content", "")
                    tool_call_id = item.get("tool_call_id")
                    if not tool_call_id and "tool_call_ids" in item:
                        tool_call_ids = item.get("tool_call_ids", [])
                        tool_call_id = tool_call_ids[0] if tool_call_ids else None
                    
                    if tool_call_id:
                        output_items.append(NeMoGymFunctionCallOutput(
                            call_id=tool_call_id,
                            output=content if isinstance(content, str) else json.dumps(content),
                            type="function_call_output",
                            status="completed"
                        ))
    
    # For SWE-agent, trajectory format is similar to OpenAI but with additional fields
    elif agent_framework == "swe_agent" and isinstance(trajectory, list):
        for item in trajectory:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                message_type = item.get("message_type", "")
                
                if role in ["system", "user"]:
                    # Create input message
                    if content:
                        output_items.append(NeMoGymMessage(
                            content=[{"type": "input_text", "text": content}],
                            role="system" if role == "system" else "user",
                            status="completed",
                            type="message"
                        ))
                
                elif role == "assistant":
                    # Handle assistant messages which may have tool calls
                    tool_calls = item.get("tool_calls", [])
                    
                    # Add assistant message if there's content (even if there are also tool calls)
                    if content:
                        output_items.append(NeMoGymResponseOutputMessage(
                            id=f"msg-{len(output_items)}",
                            content=[
                                NeMoGymResponseOutputText(
                                    type="output_text",
                                    text=content,
                                    annotations=[]
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message"
                        ))
                    
                    # Also add tool calls if present
                    if tool_calls:
                        for tc in tool_calls:
                            if "function" in tc:
                                output_items.append(NeMoGymResponseFunctionToolCall(
                                    arguments=tc["function"].get("arguments", ""),
                                    call_id=tc.get("id", ""),
                                    name=tc["function"].get("name", ""),
                                    type="function_call",
                                    id=tc.get("id"),
                                    status="completed"
                                ))
                
                elif role == "tool":
                    # Tool response
                    tool_call_ids = item.get("tool_call_ids", [])
                    if tool_call_ids and content:
                        output_items.append(NeMoGymFunctionCallOutput(
                            call_id=tool_call_ids[0],  # Use first ID
                            output=content if isinstance(content, str) else json.dumps(content),
                            type="function_call_output",
                            status="completed"
                        ))
    
    return output_items


def get_model_endpoint(model_server_name: Optional[str]) -> str:
    """Determine the model API endpoint.
    
    Args:
        model_server_name: Name of the model server
        
    Returns:
        Model API endpoint URL
        
    Raises:
        RuntimeError: If endpoint cannot be determined
    """
    try:
        from nemo_gym.server_utils import ServerClient, get_first_server_config_dict
        
        global_config_dict = (
            ServerClient.load_from_global_config().global_config_dict
        )
        
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            model_server_name or "openai_model",
        )
        
        base_url = (
            f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
        )
        return base_url
        
    except Exception as e:
        LOG.error(f"Failed to get server config for {model_server_name}: {e}")
        raise RuntimeError(f"Could not determine model endpoint for server '{model_server_name}': {e}")


async def run_swebench_evaluation(
    problem_info: Dict, 
    model_endpoint: str, 
    body: NeMoGymResponseCreateParamsNonStreaming,
    agent_framework: str,
    agent_config: Optional[str],
    agent_tools_file: Optional[str],
    agent_max_turns: int,
    swebench_tests_timeout: int,
    nemo_skills_config: Dict[str, Any]
) -> Dict:
    """Run SWE-bench evaluation using NeMo-Skills.
    
    Args:
        problem_info: Problem information
        model_endpoint: Model API endpoint
        body: Request body
        agent_framework: Agent framework (swe_agent or openhands)
        agent_config: Path to agent configuration file
        agent_tools_file: Path to tools JSON file (for SWE-agent)
        agent_max_turns: Maximum iterations for the agent
        swebench_tests_timeout: Timeout for running tests
        nemo_skills_config: Additional NeMo-Skills configuration
        
    Returns:
        Evaluation results dictionary
        
    Raises:
        RuntimeError: If evaluation fails
    """
    # Create persistent directory for I/O and logs in local workspace
    workspace_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent  # Go up to nemo-gym root
    instance_id = problem_info.get('instance_id', 'unknown')
    timestamp = int(time.time() * 1000)  # Millisecond timestamp for uniqueness
    persistent_dir = workspace_root / "temp_swebench" / f"{instance_id}_{timestamp}"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = persistent_dir / "input.jsonl"
    output_file = persistent_dir / "output.jsonl"
    
    # Write input file
    with open(input_file, 'w') as f:
        json.dump(problem_info, f)
        f.write('\n')
    
    # Build command to run NeMo-Skills
    cmd = build_nemo_skills_command(
        input_file, output_file, model_endpoint, body,
        agent_framework, agent_config, agent_max_turns,
        swebench_tests_timeout, nemo_skills_config
    )
    
    LOG.info(f"Running NeMo-Skills command: {' '.join(cmd)}")
    
    # Run in subprocess to avoid event loop conflicts
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT  # Merge stderr into stdout
    )
        
    # Stream output in real-time
    stdout_lines = []
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        line_str = line.decode().strip()
        if line_str:
            LOG.info(f"NeMo-Skills: {line_str}")
            stdout_lines.append(line_str)
    
    await process.wait()
    
    if process.returncode != 0:
        error_msg = f"NeMo-Skills failed with code {process.returncode}"
        if stdout_lines:
            error_msg += f": {' '.join(stdout_lines[-5:])}"  # Show last 5 lines
        LOG.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Read results
    if not output_file.exists():
        raise RuntimeError("No output file generated")
    
    with open(output_file, 'r') as f:
        result = json.loads(f.read().strip())
    
    # Try to find and include trajectory file
    trajectories_dir = persistent_dir / "trajectories"
    trajectory_data, tools = get_trajectory_and_tools(
        trajectories_dir, instance_id, agent_framework, 
        agent_tools_file if agent_framework == "swe_agent" else None
    )
    
    # Add trajectory and tools to result if found
    if trajectory_data:
        result["trajectory"] = trajectory_data
        LOG.info("Added trajectory data to result")
    if tools:
        result["tools"] = tools
        LOG.info(f"Added {len(tools)} tools to result")
    
    return result


def build_nemo_skills_command(
    input_file: Path, 
    output_file: Path, 
    model_endpoint: str, 
    body: NeMoGymResponseCreateParamsNonStreaming,
    agent_framework: str,
    agent_config: Optional[str],
    agent_max_turns: int,
    swebench_tests_timeout: int,
    nemo_skills_config: Dict[str, Any]
) -> list:
    """Build command to run NeMo-Skills SWE-bench evaluation.
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        model_endpoint: Model API endpoint
        body: Request body
        agent_framework: Agent framework
        agent_config: Path to agent configuration file
        agent_max_turns: Maximum iterations
        swebench_tests_timeout: Test timeout
        nemo_skills_config: Additional configuration
        
    Returns:
        Command as list of strings
    """
    # Extract model name from endpoint or body
    model_name = getattr(body, "model", "gpt-4.1-2025-04-14")
    
    # Build base command
    cmd = [
        sys.executable, "-m", "nemo_skills.inference.eval.swebench",
        f"++input_file={input_file}",
        f"++output_file={output_file}",
        f"++agent_framework={agent_framework}",
        f"++server.model={model_name}",
        f"++server.base_url={model_endpoint}",
        f"++agent_max_turns={agent_max_turns}",
        f"++swebench_tests_timeout={swebench_tests_timeout}",
    ]
    
    # Add agent config if specified
    if agent_config:
        cmd.append(f"++agent_config={agent_config}")
    
    # Add inference parameters
    inference_params = getattr(body, "inference", {})
    if hasattr(body, "temperature") and body.temperature is not None:
        inference_params["temperature"] = body.temperature
    if hasattr(body, "top_p") and body.top_p is not None:
        inference_params["top_p"] = body.top_p
    if hasattr(body, "max_output_tokens") and body.max_output_tokens is not None:
        inference_params["tokens_to_generate"] = body.max_output_tokens
    
    for key, value in inference_params.items():
        cmd.append(f"++inference.{key}={value}")
    
    # Add any additional NeMo-Skills config
    for key, value in nemo_skills_config.items():
        cmd.append(f"++{key}={value}")
    
    return cmd


def get_trajectory_and_tools(
    trajectories_dir: Path, 
    instance_id: str, 
    agent_framework: str,
    agent_tools_file: Optional[str] = None
) -> tuple:
    """Get trajectory and tools from evaluation results.
    
    Args:
        trajectories_dir: Directory containing trajectories
        instance_id: Instance ID
        agent_framework: Agent framework
        agent_tools_file: Path to tools JSON file (for SWE-agent)
        
    Returns:
        Tuple of (trajectory_data, tools)
    """
    trajectory_data = None
    tools = []
    
    # Handle different agent frameworks' trajectory storage
    if agent_framework == "openhands":
        # Get trajectory from completion files (complete and in OpenAI format)
        trajectory_data, tools = get_openhands_trajectory_from_completions(trajectories_dir, instance_id)
        if trajectory_data:
            LOG.info(f"Loaded OpenHands trajectory from llm_completions ({len(trajectory_data)} messages)")
        else:
            LOG.warning(f"No trajectory files found in {trajectories_dir}")
            
    elif agent_framework == "swe_agent":
        # For SWE-agent, look for .traj files
        if trajectories_dir.exists():
            traj_files = [f for f in trajectories_dir.glob("**/*.traj") if "demonstrations" not in str(f)]
            
            if traj_files:
                # Read the first trajectory file found
                try:
                    with open(traj_files[0], 'r') as f:
                        trajectory_data = json.load(f)["history"]
                    LOG.info(f"Found and loaded SWE-agent trajectory file: {traj_files[0]}")
                except Exception as e:
                    LOG.warning(f"Failed to read trajectory file {traj_files[0]}: {e}")
                
                # Load SWE-agent tools from the configured JSON file
                if agent_tools_file:
                    tools_file = Path(__file__).parent / agent_tools_file
                    if tools_file.exists():
                        with open(tools_file, 'r') as f:
                            tools_data = json.load(f)
                            tools = tools_data.get("tools", [])
                            LOG.info(f"Loaded SWE-agent tools from {tools_file}")
                    else:
                        LOG.warning(f"SWE-agent tools file not found: {tools_file}")
                else:
                    LOG.warning("No agent_tools_file configured for SWE-agent")
        else:
            LOG.warning(f"No trajectory files found in {trajectories_dir}")
    else:
        LOG.warning(f"Unsupported agent framework: {agent_framework}")
    
    return trajectory_data, tools


def get_openhands_trajectory_from_completions(trajectories_dir: Path, instance_id: str) -> tuple:
    """Get trajectory from llm_completions directory for OpenHands.
    
    Args:
        trajectories_dir: Trajectories directory
        instance_id: Instance ID
        
    Returns:
        Tuple of (messages, tools)
    """
    messages = []
    tools = []
    completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
    
    if not completions_dir.exists():
        LOG.warning(f"No llm_completions directory found: {completions_dir}")
        return messages, tools
    
    # Get all completion files sorted by timestamp
    completion_files = sorted(completions_dir.glob("*.json"))
    
    if not completion_files:
        LOG.warning(f"No completion files found in: {completions_dir}")
        return messages, tools
    
    # Use the last file for messages since it contains the cumulative conversation
    last_file = completion_files[-1]
    
    try:
        with open(last_file, 'r') as f:
            data = json.load(f)
        
        # Get all messages from the last file
        messages = data["messages"]
        
        # Add the final assistant response
        messages.append(data["response"]["choices"][0]["message"])
        
        # Get tools (they should be the same across all turns)
        tools = data.get("kwargs", {}).get("tools", [])
            
        LOG.info(f"Loaded {len(messages)} messages from last completion file: {last_file.name}")
            
    except Exception as e:
        LOG.error(f"Failed to read completion file {last_file}: {e}")
        return [], []
    
    # Convert content format if needed (OpenHands uses list of dicts for content)
    for msg in messages:
        if "content" in msg:
            if msg["content"] is None:
                msg["content"] = ""
            elif isinstance(msg["content"], list):
                # Handle empty content lists (e.g., assistant messages with only tool calls)
                if len(msg["content"]) == 0:
                    msg["content"] = ""
                elif len(msg["content"]) == 1:
                    # Extract the single text item
                    item = msg["content"][0]
                    if not isinstance(item, dict) or item.get("type") != "text" or "text" not in item:
                        raise ValueError(f"Expected content item to be {{type: 'text', text: '...'}}, got {item}")
                    msg["content"] = item["text"]
                else:
                    raise ValueError(f"Expected 0 or 1 content items, got {len(msg['content'])}")
        else:
            raise ValueError(f"Expected content in message, got {msg}")
    
    return messages, tools


def ensure_nemo_run_symlink():
    """Ensure /nemo_run/code symlink exists pointing to nemo_skills package.
    
    Raises:
        RuntimeError: If symlink cannot be created
    """
    # Find nemo_skills in the .venv directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_lib = os.path.join(current_dir, ".venv/lib")
    
    nemo_skills_path = None
    if os.path.exists(venv_lib):
        # Look for python* directories
        for python_dir in os.listdir(venv_lib):
            if python_dir.startswith("python"):
                potential_path = os.path.join(venv_lib, python_dir, "site-packages/nemo_skills")
                if os.path.exists(potential_path):
                    nemo_skills_path = potential_path
                    break
    
    if not nemo_skills_path:
        raise RuntimeError(f"Could not find nemo_skills package in {venv_lib}")
    
    # Create symlink if it doesn't exist
    if not os.path.exists("/nemo_run/code"):
        # Create directory
        result = subprocess.run(["mkdir", "-p", "/nemo_run"], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create /nemo_run directory: {result.stderr.decode()}")
        
        # Create symlink
        result = subprocess.run(["ln", "-sf", nemo_skills_path, "/nemo_run/code"], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create symlink: {result.stderr.decode()}")
        
        LOG.info(f"Created symlink: /nemo_run/code -> {nemo_skills_path}")
