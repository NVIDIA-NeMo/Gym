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
import copy
import fcntl
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai.types.responses.function_tool import FunctionTool

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymMessage,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient, get_first_server_config_dict
from responses_api_agents.swe_agents.run_openhands import (
    RunOpenHandsAgent,
    SupportedAgentFrameworks,
    SweBenchGenerationConfig,
    SweBenchInferenceConfig,
)


### Trajectory Conversion Utils ###


def _extract_text_from_message(item) -> Optional[str]:
    """Helper to extract text content from a message item."""
    if not (hasattr(item, "content") and item.content):
        return None

    for content_item in item.content:
        if isinstance(content_item, dict) and content_item.get("type") == "input_text":
            return content_item.get("text", "")

    return None


def extract_input_messages_from_trajectory(
    response_output: List,
) -> Tuple[List[NeMoGymEasyInputMessage], List]:
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
        return [], []

    # Find where the assistant/function calls start
    # TODO (sugam): check if we need the function call check.
    for i, item in enumerate(response_output):
        # Check if this is an assistant message or function call
        is_assistant = hasattr(item, "role") and item.role == "assistant"
        is_function = hasattr(item, "type") and item.type in [
            "function_call",
            "function_call_output",
        ]

        if is_assistant or is_function:
            filtered_output.extend(response_output[i:])
            break

        # Process system/user/developer messages
        if hasattr(item, "role") and item.role in ["system", "user", "developer"]:
            # Try to extract text content
            text_content = _extract_text_from_message(item)
            if text_content:
                input_messages.append(
                    NeMoGymEasyInputMessage(
                        role=item.role,
                        content=text_content,
                        type="message",
                    )
                )
                continue

        filtered_output.append(item)

    return input_messages, filtered_output


def convert_trajectory_to_output_items(
    trajectory: List[Any],
    agent_framework: str,
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
                role = item["role"]

                # Extract text content from content data
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

                if role in ["user", "system", "developer"]:
                    if text_content:
                        output_items.append(
                            NeMoGymMessage(
                                content=[{"type": "input_text", "text": text_content}],
                                role=role,
                                status="completed",
                                type="message",
                            )
                        )

                elif role == "assistant":
                    # Handle assistant messages with potential tool calls
                    tool_calls = item.get("tool_calls", [])

                    # Add assistant message if there's content (even if there are also tool calls)
                    prompt_token_ids = item.get("prompt_token_ids", [])
                    generation_token_ids = item.get("generation_token_ids", [])
                    generation_log_probs = item.get("generation_log_probs", [])

                    output_items.append(
                        NeMoGymResponseOutputMessageForTraining(
                            id=f"msg-{len(output_items)}",
                            content=[
                                NeMoGymResponseOutputText(
                                    type="output_text",
                                    text=text_content,
                                    annotations=[],
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message",
                            prompt_token_ids=prompt_token_ids,
                            generation_token_ids=generation_token_ids,
                            generation_log_probs=generation_log_probs,
                        )
                    )

                    # Also add tool calls if present
                    if tool_calls:
                        # Create function call items
                        for tc in tool_calls:
                            if "function" in tc:
                                output_items.append(
                                    NeMoGymResponseFunctionToolCall(
                                        arguments=tc["function"].get("arguments", ""),
                                        call_id=tc.get("id", ""),
                                        name=tc["function"].get("name", ""),
                                        type="function_call",
                                        id=tc.get("id"),
                                        status="completed",
                                    )
                                )

                elif role == "tool":
                    # Tool response
                    content = item.get("content", "")
                    tool_call_id = item.get("tool_call_id")
                    if not tool_call_id and "tool_call_ids" in item:
                        tool_call_ids = item.get("tool_call_ids", [])
                        tool_call_id = tool_call_ids[0] if tool_call_ids else None
                    if tool_call_id:
                        output_items.append(
                            NeMoGymFunctionCallOutput(
                                call_id=tool_call_id,
                                output=text_content,
                                type="function_call_output",
                                status="completed",
                            )
                        )

    # For SWE-agent, trajectory format is similar to OpenAI but with additional fields
    elif agent_framework == "swe_agent" and isinstance(trajectory, list):
        for item in trajectory:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")

                if role in ["system", "user"]:
                    # Create input message
                    if content:
                        output_items.append(
                            NeMoGymMessage(
                                content=[{"type": "input_text", "text": content}],
                                role="system" if role == "system" else "user",
                                status="completed",
                                type="message",
                            )
                        )

                elif role == "assistant":
                    # Handle assistant messages which may have tool calls
                    tool_calls = item.get("tool_calls", [])

                    prompt_token_ids = item.get("provider_specific_fields", {}).get("prompt_token_ids", [])
                    generation_token_ids = item.get("provider_specific_fields", {}).get("generation_token_ids", [])
                    generation_log_probs = item.get("provider_specific_fields", {}).get("generation_log_probs", [])
                    # Add assistant message if there's content (even if there are also tool calls)
                    if content:
                        output_items.append(
                            NeMoGymResponseOutputMessageForTraining(
                                id=f"msg-{len(output_items)}",
                                content=[
                                    NeMoGymResponseOutputText(
                                        type="output_text",
                                        text=content,
                                        annotations=[],
                                        logprobs=None,
                                    )
                                ],
                                role="assistant",
                                status="completed",
                                type="message",
                                prompt_token_ids=prompt_token_ids,
                                generation_token_ids=generation_token_ids,
                                generation_log_probs=generation_log_probs,
                            )
                        )

                    # Also add tool calls if present
                    if tool_calls:
                        for tc in tool_calls:
                            if "function" in tc:
                                # Handle both dict and string formats for tc["function"]
                                func = tc["function"]
                                if isinstance(func, str):
                                    # If it's a string, try to parse as JSON or use as name
                                    try:
                                        func = json.loads(func)
                                    except (json.JSONDecodeError, TypeError):
                                        # If not valid JSON, treat the string as the function name
                                        func = {"name": func, "arguments": ""}

                                output_items.append(
                                    NeMoGymResponseFunctionToolCall(
                                        arguments=func.get("arguments", ""),
                                        call_id=tc.get("id", ""),
                                        name=func.get("name", ""),
                                        type="function_call",
                                        id=tc.get("id"),
                                        status="completed",
                                    )
                                )

                elif role == "tool":
                    # Tool response
                    tool_call_ids = item.get("tool_call_ids", [])
                    if tool_call_ids and content:
                        output_items.append(
                            NeMoGymFunctionCallOutput(
                                call_id=tool_call_ids[0],  # Use first ID
                                output=content if isinstance(content, str) else json.dumps(content),
                                type="function_call_output",
                                status="completed",
                            )
                        )

    return output_items


def get_trajectory_and_tools(
    trajectories_dir: Path,
    instance_id: str,
    agent_framework: str,
    agent_tools_file: Optional[str] = None,
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

    if agent_framework == "openhands":
        trajectory_data, tools = get_openhands_trajectory_from_completions(trajectories_dir, instance_id)
        # if trajectory_data:
        #     print(
        #         f"Loaded OpenHands trajectory from llm_completions ({len(trajectory_data)} messages)",
        #         flush=True,
        #     )
        # else:
        #     print(f"No trajectory files found in {trajectories_dir}", flush=True)

    elif agent_framework == "swe_agent":
        # For SWE-agent, look for .traj files
        if trajectories_dir.exists():
            traj_files = [f for f in trajectories_dir.glob("**/*.traj") if "demonstrations" not in str(f)]

            if traj_files:
                # Read the first trajectory file found
                try:
                    with open(traj_files[0], "r") as f:
                        traj_content = json.load(f)
                        history = traj_content["history"]
                        trajectory_steps = traj_content["trajectory"]
                        trajectory_data = extract_data_from_trajectory(trajectory_steps, history)
                    print(f"Found and loaded SWE-agent trajectory file: {traj_files[0]}", flush=True)
                except Exception as e:
                    print(f"Failed to read trajectory file {traj_files[0]}: {e}", flush=True)

                # Load SWE-agent tools from the configured JSON file
                if agent_tools_file:
                    tools_file = Path(__file__).parent / agent_tools_file
                    if tools_file.exists():
                        with open(tools_file, "r") as f:
                            tools_data = json.load(f)
                            tools = tools_data.get("tools", [])
                            print(f"Loaded SWE-agent tools from {tools_file}", flush=True)
                    else:
                        print(f"SWE-agent tools file not found: {tools_file}", flush=True)
                else:
                    print("No agent_tools_file configured for SWE-agent", flush=True)
        else:
            print(f"No trajectory files found in {trajectories_dir}", flush=True)
    else:
        print(f"Unsupported agent framework: {agent_framework}", flush=True)

    return trajectory_data, tools


def convert_tools_to_function_format(raw_tools: List[Dict]) -> List:
    """Convert tools from ChatCompletion format to Response FunctionTool format.

    Args:
        raw_tools: List of tools in ChatCompletion format

    Returns:
        List of FunctionTool objects
    """

    tools = []
    for tool in raw_tools:
        # Tools from SWE-agent are in ChatCompletion format with nested structure
        # Convert to Response FunctionTool format which is flat
        if tool.get("type") == "function" and "function" in tool:
            func_def = tool["function"]
            # Create FunctionTool object with flat structure
            function_tool = FunctionTool(
                type="function",
                name=func_def.get("name", ""),
                description=func_def.get("description"),
                parameters=func_def.get("parameters"),
                strict=func_def.get("strict"),  # May be None
            )
            tools.append(function_tool)
    return tools


### SWE Agent Harness Utils ###


def extract_messages(trajectory_item) -> List[Dict]:
    """
    Trajectory might have failed assistant messages, hence we take trajectory as ground truth instead of history.
    Convert a trajectory item into assistant and tool messages.
    Returns a list of messages.
    """
    # Defensive check: if trajectory_item is not a dict, return empty list
    if not isinstance(trajectory_item, dict):
        print(f"trajectory_item is not a dict (type: {type(trajectory_item)}). Skipping.", flush=True)
        return []

    tool_calls = trajectory_item.get("tool_calls")
    final_message = []

    # Get extra_info safely
    extra_info = trajectory_item.get("extra_info", {})
    if isinstance(extra_info, dict):
        provider_specific_fields = extra_info.get("provider_specific_fields", {})
    else:
        provider_specific_fields = {}

    # Create assistant message
    assistant_msg = {
        "role": "assistant",
        "content": trajectory_item.get("response", ""),
        "thought": trajectory_item.get("thought", ""),
        "action": trajectory_item.get("action", ""),
        "agent": "main",
        "tool_calls": tool_calls,
        "message_type": "action",
        "thinking_blocks": [],
        "provider_specific_fields": provider_specific_fields,
    }
    final_message.append(assistant_msg)
    if tool_calls is not None:
        # Create tool message
        tool_msg = {
            "role": "tool",
            "content": trajectory_item.get("observation", ""),
            "agent": "main",
            "message_type": "observation",
            "tool_call_ids": trajectory_item.get("tool_call_ids", [""]),
        }
        final_message.append(tool_msg)

    return final_message


def extract_data_from_trajectory(
    trajectory_data: List[Dict], history: List[Dict]
) -> Tuple[List[Dict], Dict[int, Dict]]:
    """
    Extract final trajectory from trajectory and history.
    """
    final_trajectory = []
    history_copy = copy.deepcopy(history)
    trajectories_copy = copy.deepcopy(trajectory_data)

    # Defensive checks for trajectory_data structure
    if not trajectories_copy or len(trajectories_copy) == 0:
        print("Empty trajectories_copy, returning empty trajectory", flush=True)
        return []

    # Check if last trajectory item is a dict
    if not isinstance(trajectories_copy[-1], dict):
        print(
            f"Last trajectory item is not a dict (type: {type(trajectories_copy[-1])}), returning empty trajectory",
            flush=True,
        )
        return []

    # Check if "query" key exists and is a list
    if "query" not in trajectories_copy[-1] or not isinstance(trajectories_copy[-1]["query"], list):
        print("'query' key missing or not a list in last trajectory item, returning empty trajectory", flush=True)
        return []

    if len(trajectories_copy[-1]["query"]) > 0 and len(trajectories_copy[-1]["query"][0]) == 0:  # error case
        if len(trajectories_copy) < 2:
            print("Not enough trajectory items for error case, returning empty trajectory", flush=True)
            return []
        if not isinstance(trajectories_copy[-2], dict) or "query" not in trajectories_copy[-2]:
            print("Second-to-last trajectory item is malformed, returning empty trajectory", flush=True)
            return []
        final_trajectory = trajectories_copy[-2]["query"].copy()
        final_trajectory.extend(extract_messages(trajectories_copy[-2]))
        if len(history_copy) >= 2:
            user_message = history_copy.pop()
            assistant_message = history_copy.pop()
            if isinstance(user_message, dict) and isinstance(assistant_message, dict):
                user_message["content"] = user_message.get("content", "") + "." + assistant_message.get("content", "")
                final_trajectory.append(user_message)
    else:
        final_trajectory = trajectories_copy[-1]["query"].copy()
        final_trajectory.extend(extract_messages(trajectories_copy[-1]))

    # Filter out any non-dict items that might have been added
    final_trajectory = [item for item in final_trajectory if isinstance(item, dict)]

    return final_trajectory


### OpenHands Harness Utils ###


def get_openhands_trajectory_from_completions(
    trajectories_dir: Path,
    instance_id: str,
) -> tuple:
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
        print(f"No llm_completions directory found: {completions_dir}", flush=True)
        return messages, tools

    completion_files = sorted(completions_dir.glob("*.json"))

    if not completion_files:
        print(f"No completion files found in: {completions_dir}", flush=True)
        return messages, tools

    last_file = completion_files[-1]

    try:
        with open(last_file, "r") as f:
            data = json.load(f)

        messages = data["messages"]
        provider_specific_fields = data.get("provider_specific_fields", {})
        final_assistant_message = data["response"]["choices"][0]["message"]

        for key in ["prompt_token_ids", "generation_token_ids", "generation_log_probs"]:
            if key in provider_specific_fields:
                final_assistant_message[key] = provider_specific_fields[key]

        if final_assistant_message.get("content") or final_assistant_message.get("tool_calls"):
            messages.append(final_assistant_message)

        tools = data.get("kwargs", {}).get("tools", [])

        # print(
        #     f"Loaded {len(messages)} messages from last completion file: {last_file}",
        #     flush=True,
        # )

    except Exception as e:
        print(f"Failed to read completion file {last_file}: {e}", flush=True)
        return [], []

    for msg in messages:
        if "content" in msg:
            msg["content"] = msg["content"] or ""
            if isinstance(msg["content"], list):
                # Handle empty content lists (e.g., assistant messages with only tool calls)
                if len(msg["content"]) == 0:
                    msg["content"] = ""
                elif len(msg["content"]) == 1:
                    item = msg["content"][0]
                    if not isinstance(item, dict) or item.get("type") != "text" or "text" not in item:
                        raise ValueError(f"Expected content item to be {{type: 'text', text: '...'}}, got {item}")
                    msg["content"] = item["text"]
                else:
                    raise ValueError(f"Expected 0 or 1 content items, got {len(msg['content'])}")
        else:
            raise ValueError(f"Expected content in message, got {msg}")

    return messages, tools


### Run SWE Harness Utils ###


def extract_problem_info(
    body: NeMoGymResponseCreateParamsNonStreaming,
    container_formatter: str | list[str],
) -> Dict:
    # Get metadata
    metadata = body.metadata

    # Build problem info
    problem_info = {
        "problem_statement": metadata["problem_statement"],
        "instance_id": metadata["instance_id"],
        "base_commit": metadata["base_commit"],
        "dataset_name": metadata["dataset_name"],
        "split": metadata["split"],
        # TODO (sugam): refactor this to a cleaner approach
        "instance_dict": metadata["instance_dict"],
        "container_formatter": container_formatter,
    }

    return problem_info


def get_model_endpoint(model_server_name: str) -> str:
    global_config_dict = ServerClient.load_from_global_config().global_config_dict

    model_server_config = get_first_server_config_dict(
        global_config_dict,
        model_server_name,
    )

    base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
    return base_url


async def run_swebench_evaluation(
    problem_info: Dict,
    model_endpoint: str,
    body: NeMoGymResponseCreateParamsNonStreaming,
    run_session_id: str,
    agent_framework: str,
    agent_config: Optional[str],
    agent_tools_file: Optional[str],
    agent_max_turns: int,
    swebench_tests_timeout: int,
    swebench_agent_timeout: int,
    agent_framework_repo: Optional[str] = None,
    agent_framework_commit: str = "HEAD",
    openhands_setup_dir: Optional[Path] = None,
    swebench_setup_dir: Optional[Path] = None,
    r2e_gym_setup_dir: Optional[Path] = None,
    dataset_path: Optional[str] = None,
    instance_dir: Optional[str] = None,
) -> Dict:
    # Create persistent directory for I/O and logs in local workspace
    workspace_root = Path(os.path.dirname(os.path.abspath(__file__)))
    instance_id = problem_info.get("instance_id", "unknown")
    persistent_dir = workspace_root / f"swebench_results_{run_session_id}" / instance_dir
    persistent_dir.mkdir(parents=True, exist_ok=True)
    output_file = persistent_dir / "output.jsonl"

    inference_params = {}

    for param, key in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "tokens_to_generate"),
    ]:
        value = getattr(body, param, None)
        if value is not None:
            inference_params[key] = value

    inference_config = SweBenchInferenceConfig(**inference_params)
    server = {
        "model": body.model,
        "base_url": model_endpoint,
    }

    cfg = SweBenchGenerationConfig(
        output_file=output_file,
        agent_framework=SupportedAgentFrameworks.openhands,
        agent_framework_repo=agent_framework_repo,
        agent_framework_commit=agent_framework_commit,
        agent_config=agent_config,
        agent_max_turns=agent_max_turns,
        swebench_tests_timeout=swebench_tests_timeout,
        swebench_agent_timeout=swebench_agent_timeout,
        inference=inference_config,
        server=server,
    )

    run_oh = RunOpenHandsAgent(
        cfg=cfg,
        openhands_setup_dir=openhands_setup_dir,
        swebench_setup_dir=swebench_setup_dir,
        r2e_gym_setup_dir=r2e_gym_setup_dir,
        dataset_path=dataset_path,
    )
    result = await run_oh.process_single_datapoint(problem_info)
    print(f"Process completed for {instance_id}", flush=True)

    try:
        with open(output_file, "w") as f:
            json.dump(result, f)
    except Exception as e:
        print(f"Failed to write result to {output_file}: {e}", flush=True)
        raise e

    # Read results
    if not output_file.exists():
        raise RuntimeError(f"No output file generated: {output_file}")

    # Try to find and include trajectory file
    trajectories_dir = persistent_dir / "trajectories"
    trajectory_data, tools = get_trajectory_and_tools(
        trajectories_dir,
        instance_id,
        agent_framework,
        agent_tools_file if agent_framework == "swe_agent" else None,
    )

    # tools = convert_tools_to_function_format(tools) if tools else []

    result["tools"] = tools
    result["trajectory"] = trajectory_data

    return result


### Harness and Evaluation Setup Utils ###


def _get_workspace_root() -> Path:
    return Path(os.path.dirname(os.path.abspath(__file__)))


def _resolve_setup_directory(provided_dir: Optional[Path], default_subdir: str) -> Path:
    base_dir = provided_dir or (_get_workspace_root() / default_subdir)
    return base_dir.resolve()


@contextmanager
def _setup_directory_lock(setup_dir: Path, label: str):
    """File-based lock to ensure only one process performs the setup."""
    lock_dir = setup_dir.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{setup_dir.name}.lock"

    with open(lock_path, "w") as lock_file:
        print(f"Acquiring {label} setup lock at {lock_path}", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _run_setup_shell_script(
    setup_dir: Path,
    script_name: str,
    script_content: str,
    timeout_seconds: int,
    label: str,
    timeout_error_message: Optional[str] = None,
) -> None:
    script_path = setup_dir / script_name

    with open(script_path, "w") as f:
        f.write(script_content)
    script_path.chmod(0o755)

    print(f"Running {label} setup script...", flush=True)
    print(f"Setup script: {script_path}", flush=True)

    process = None
    try:
        process = subprocess.Popen(
            [str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines: List[str] = []
        if process.stdout is None:
            raise RuntimeError("Failed to capture script output")

        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)

        process.wait(timeout=timeout_seconds)

        if process.returncode != 0:
            full_output = "".join(output_lines)
            raise RuntimeError(f"{label} setup failed with return code {process.returncode}:\n{full_output}")

        print(f"{label} setup completed successfully!", flush=True)
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        message = timeout_error_message or f"{label} setup timed out after {timeout_seconds} seconds"
        raise RuntimeError(message)
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"{label} setup failed: {exc}") from exc
    finally:
        if process and process.stdout:
            process.stdout.close()


def setup_swebench_environment(
    swebench_repo: Optional[str] = "https://github.com/HeyyyyyyG/SWE-bench.git",
    swebench_commit: str = "HEAD",
    setup_dir: Optional[Path] = None,
) -> Path:
    setup_dir = _resolve_setup_directory(setup_dir, "swe_swebench_setup")

    with _setup_directory_lock(setup_dir, "SWE-bench"):
        swebench_dir = setup_dir / "SWE-bench"
        uv_dir = setup_dir / "uv"
        python_dir = setup_dir / "python"

        if swebench_dir.exists():
            print(f"SWE-bench already set up at {setup_dir}", flush=True)
            print(f"  - SWE-bench: {swebench_dir}", flush=True)
            print(f"  - venv: {swebench_dir / 'venv'}", flush=True)
            print(f"  - uv: {uv_dir}", flush=True)
            print(f"  - Python: {python_dir}", flush=True)
            return setup_dir

        print(f"Setting up SWE-bench environment at {setup_dir}...", flush=True)
        setup_dir.mkdir(parents=True, exist_ok=True)

        script_name = "setup_swebench.sh"
        script_content = f"""#!/bin/bash
set -e
set -x

cd {setup_dir}

export UV_INSTALL_DIR="{uv_dir}"
export UV_PYTHON_INSTALL_DIR="{python_dir}"
if [ ! -f "{uv_dir}/bin/uv" ]; then
    echo "Installing uv to {uv_dir}..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed at {uv_dir}"
fi

export PATH="{uv_dir}/bin:$PATH"
echo "Verifying uv installation..."
which uv
uv --version

# Clone SWE-bench
if [ ! -d "{swebench_dir}/.git" ]; then
    echo "Cloning SWE-bench..."
    # Clean up any partial clone
    rm -rf "{swebench_dir}"
    git clone {swebench_repo} {swebench_dir}
else
    echo "SWE-bench already cloned at {swebench_dir}"
fi

cd {swebench_dir}
echo "Checking out {swebench_commit}..."
git checkout {swebench_commit}

echo "Installing Python 3.12 to portable location..."
uv python install 3.12

echo "Python installations:"
uv python list

echo "Creating virtual environment with uv..."
rm -rf venv
uv venv --python 3.12 venv

echo "Installing SWE-bench..."
uv pip install -p {swebench_dir}/venv/bin/python -e .

if [ -d venv ] && [ -f venv/bin/python ]; then
    echo "✓ venv created at $(pwd)/venv"
    echo "✓ Python version: $(venv/bin/python --version)"
else
    echo "✗ ERROR: venv was not created properly!"
    exit 1
fi

echo "SWE-bench setup complete!"
"""

        _run_setup_shell_script(
            setup_dir=setup_dir,
            script_name=script_name,
            script_content=script_content,
            timeout_seconds=600,
            label="SWE-bench",
            timeout_error_message="SWE-bench setup timed out after 10 minutes",
        )

        print(f"Setup directory: {setup_dir}", flush=True)
        print(f"  - SWE-bench: {swebench_dir}", flush=True)
        print(f"  - venv: {swebench_dir / 'venv'}", flush=True)
        print(f"  - uv: {uv_dir}", flush=True)
        print(f"  - Python: {python_dir}", flush=True)

        return setup_dir


def setup_r2e_gym_environment(
    eval_harness_repo: Optional[str] = None,
    eval_harness_commit: str = "local-eval",
    setup_dir: Optional[Path] = None,
) -> Path:
    """Set up R2E-Gym environment once during initialization.

    This function builds R2E-Gym in a persistent location that can be mounted
    into Apptainer containers, avoiding repeated setup for each request.

    Args:
        eval_harness_repo: URL of the R2E-Gym repo (default: official repo)
        eval_harness_commit: Commit/branch to use (default: local-eval)
        setup_dir: Directory to set up R2E-Gym (default: workspace_root/swe_r2e_gym_setup)

    Returns:
        Path to the built R2E-Gym directory

    Raises:
        RuntimeError: If setup fails
    """
    if eval_harness_repo is None:
        eval_harness_repo = "https://github.com/ludwig-n/R2E-Gym.git"

    setup_dir = _resolve_setup_directory(setup_dir, "swe_r2e_gym_setup")

    with _setup_directory_lock(setup_dir, "R2E-Gym"):
        r2e_gym_dir = setup_dir / "R2E-Gym"
        uv_dir = setup_dir / "uv"
        python_dir = setup_dir / "python"

        # Check if setup is complete by verifying venv and installed module
        venv_dir = r2e_gym_dir / "venv"
        if r2e_gym_dir.exists() and venv_dir.exists():
            # Verify r2egym module is actually installed
            python_bin = venv_dir / "bin" / "python"
            if python_bin.exists():
                import subprocess

                try:
                    result = subprocess.run([str(python_bin), "-c", "import r2egym"], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        print(f"R2E-Gym already set up at {setup_dir}", flush=True)
                        print(f"  - R2E-Gym: {r2e_gym_dir}", flush=True)
                        print(f"  - venv: {venv_dir}", flush=True)
                        print(f"  - uv: {uv_dir}", flush=True)
                        print(f"  - Python: {python_dir}", flush=True)
                        return setup_dir
                    else:
                        print("R2E-Gym directory exists but module not properly installed, rebuilding...", flush=True)
                except (subprocess.TimeoutExpired, Exception) as e:
                    print(f"R2E-Gym verification failed: {e}, rebuilding...", flush=True)

        print(f"Setting up R2E-Gym environment at {setup_dir}...", flush=True)
        setup_dir.mkdir(parents=True, exist_ok=True)

        script_name = "setup_r2e_gym.sh"
        script_content = f"""#!/bin/bash
set -e
set -x

cd {setup_dir}

export UV_INSTALL_DIR="{uv_dir}"
export UV_PYTHON_INSTALL_DIR="{python_dir}"
if [ ! -f "{uv_dir}/bin/uv" ]; then
    echo "Installing uv to {uv_dir}..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed at {uv_dir}"
fi

export PATH="{uv_dir}/bin:$PATH"
echo "Verifying uv installation..."
which uv
uv --version

# Clone R2E-Gym
if [ ! -d "{r2e_gym_dir}/.git" ]; then
    echo "Cloning R2E-Gym..."
    # Clean up any partial clone
    rm -rf "{r2e_gym_dir}"
    git clone {eval_harness_repo} {r2e_gym_dir}
else
    echo "R2E-Gym already cloned at {r2e_gym_dir}"
fi

cd {r2e_gym_dir}
echo "Checking out {eval_harness_commit}..."
git checkout {eval_harness_commit}

echo "Installing Python 3.12 to portable location..."
uv python install 3.12

echo "Python installations:"
uv python list

echo "Creating virtual environment with uv..."
rm -rf venv
uv venv --python 3.12 venv

echo "Installing R2E-Gym in editable mode..."
uv pip install -p {r2e_gym_dir}/venv/bin/python -e . --no-cache

echo "Verifying installation..."
{r2e_gym_dir}/venv/bin/python -c "import r2egym; print('✓ r2egym installed successfully')"

if [ -d venv ] && [ -f venv/bin/python ]; then
    echo "✓ venv created at $(pwd)/venv"
    echo "✓ Python version: $(venv/bin/python --version)"
else
    echo "✗ ERROR: venv was not created properly!"
    exit 1
fi

echo "R2E-Gym setup complete!"
"""

        _run_setup_shell_script(
            setup_dir=setup_dir,
            script_name=script_name,
            script_content=script_content,
            timeout_seconds=1200,
            label="R2E-Gym",
            timeout_error_message="R2E-Gym setup timed out after 20 minutes",
        )

        print(f"Setup directory: {setup_dir}", flush=True)
        print(f"  - R2E-Gym: {r2e_gym_dir}", flush=True)
        print(f"  - venv: {r2e_gym_dir / '.venv'}", flush=True)
        print(f"  - uv: {uv_dir}", flush=True)
        print(f"  - Python: {python_dir}", flush=True)

        return setup_dir


def setup_openhands_environment(
    agent_framework_repo: Optional[str] = "https://github.com/sdevare-nv/nv-OpenHands.git",
    agent_framework_commit: str = "gym",
    setup_dir: Optional[Path] = None,
) -> Path:
    setup_dir = _resolve_setup_directory(setup_dir, "swe_openhands_setup")

    with _setup_directory_lock(setup_dir, "OpenHands"):
        openhands_dir = setup_dir / "OpenHands"
        miniforge_dir = setup_dir / "miniforge3"

        if openhands_dir.exists() and Path(openhands_dir / ".venv" / "bin" / "python").exists():
            print(f"OpenHands already set up at {setup_dir}", flush=True)
            print(f"  - Miniforge: {miniforge_dir}", flush=True)
            print(f"  - OpenHands: {openhands_dir}", flush=True)
            return setup_dir

        print(f"Setting up OpenHands environment at {setup_dir}...", flush=True)
        shutil.rmtree(setup_dir, ignore_errors=True)
        setup_dir.mkdir(parents=True, exist_ok=True)

        script_name = "setup_openhands.sh"
        script_content = f"""#!/bin/bash
set -e
set -x  # Enable debug output

cd {setup_dir}

# Install miniforge if not properly installed
if [ ! -f "{miniforge_dir}/bin/conda" ] || [ ! -f "{miniforge_dir}/bin/mamba" ]; then
    echo "Installing miniforge..."
    # Clean up any partial installation
    rm -rf "{miniforge_dir}"
    rm -f Miniforge3-*.sh

    echo "Downloading miniforge..."
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

    echo "Running miniforge installer..."
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p {miniforge_dir}

    echo "Cleaning up installer..."
    rm Miniforge3-$(uname)-$(uname -m).sh
else
    echo "Miniforge already installed at {miniforge_dir}"
fi

# Add conda to PATH and source conda setup
echo "Setting up conda environment..."
export PATH="{miniforge_dir}/bin:$PATH"
source {miniforge_dir}/etc/profile.d/conda.sh
conda activate base

# Verify conda and mamba are available
echo "Verifying conda installation..."
which conda
which mamba
conda --version
mamba --version

# Install required packages
echo "Installing conda packages (this may take 5-10 minutes)..."
mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs conda-forge::poetry conda-forge::tmux

# Verify installations
echo "Verifying package installations..."
which python
which node
which poetry

# Clone OpenHands
if [ ! -d "{openhands_dir}/.git" ]; then
    echo "Cloning OpenHands..."
    # Clean up any partial clone
    rm -rf "{openhands_dir}"
    git clone {agent_framework_repo} {openhands_dir}
else
    echo "OpenHands already cloned at {openhands_dir}"
fi

cd {openhands_dir}
echo "Checking out {agent_framework_commit}..."
git checkout {agent_framework_commit}

# Build OpenHands
echo "Building OpenHands (this may take 5-10 minutes)..."
export INSTALL_DOCKER=0


# Remove any cached virtualenvs from previous runs
echo "Removing any cached poetry virtualenvs..."
rm -rf ~/.cache/pypoetry/virtualenvs/openhands-* || true

# CRITICAL: Unset any active virtualenv from the host .venv
# This prevents poetry from getting confused about which venv to use
echo "Unsetting host virtualenv to avoid poetry confusion..."
unset VIRTUAL_ENV
unset PYTHONHOME
# Remove any venv paths from PATH to ensure clean environment
export PATH=$(echo "$PATH" | tr ':' '\\n' | grep -v '\\.venv' | tr '\\n' ':' | sed 's/:$//')

# Configure poetry to create virtualenv in the project directory (so it's mounted in container)
export POETRY_VIRTUALENVS_IN_PROJECT=true

# Retry `make build` with a timeout guard on the first attempt
MAX_MAKE_BUILD_ATTEMPTS=2
MAKE_BUILD_TIMEOUT_SECONDS=$((2 * 60))
MAKE_BUILD_TIMEOUT_MINUTES=$((MAKE_BUILD_TIMEOUT_SECONDS / 60))

attempt=1
while [ "$attempt" -le "$MAX_MAKE_BUILD_ATTEMPTS" ]; do
    echo "Running make build (attempt $attempt/$MAX_MAKE_BUILD_ATTEMPTS)..."

    if [ "$attempt" -lt "$MAX_MAKE_BUILD_ATTEMPTS" ]; then
        if timeout "$MAKE_BUILD_TIMEOUT_SECONDS" make build; then
            echo "make build completed successfully."
            break
        fi

        exit_code=$?
        if [ "$exit_code" -eq 124 ]; then
            echo "make build timed out after $MAKE_BUILD_TIMEOUT_MINUTES minutes."
        else
            echo "make build failed with exit code $exit_code."
        fi

        echo "Retrying make build after cleanup..."
        make clean || true
        attempt=$((attempt + 1))
        continue
    fi

    if make build; then
        echo "make build completed successfully."
        break
    fi

    exit_code=$?
    echo "make build failed on the final attempt with exit code $exit_code."
done


# Install Python dependencies with poetry
echo "Installing Python dependencies (creating .venv in OpenHands directory)..."
poetry install --no-interaction --no-root

# Install datasets package
echo "Installing datasets package..."
poetry run python -m pip install datasets

mkdir -p evaluation/oh
mkdir -p logs
mkdir -p .eval_sessions

echo "Verifying .venv was created..."
if [ -d .venv ]; then
    echo "✓ .venv created at $(pwd)/.venv"
else
    echo "✗ ERROR: .venv was not created!"
    exit 1
fi

echo "OpenHands setup complete!"
"""

        _run_setup_shell_script(
            setup_dir=setup_dir,
            script_name=script_name,
            script_content=script_content,
            timeout_seconds=1800,
            label="OpenHands",
            timeout_error_message="OpenHands setup timed out after 30 minutes",
        )

        print(f"Setup directory: {setup_dir}", flush=True)
        print(f"  - Miniforge: {miniforge_dir}", flush=True)
        print(f"  - OpenHands: {openhands_dir}", flush=True)

        return setup_dir
