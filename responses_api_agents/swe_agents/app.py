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
import logging
import time
from typing import Any, Dict, Optional

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)

# Import utility functions from local utils module
from responses_api_agents.swe_agents.utils import (
    convert_tools_to_function_format,
    convert_trajectory_to_output_items,
    ensure_nemo_run_symlink,
    extract_input_messages_from_trajectory,
    extract_problem_info,
    get_model_endpoint,
    run_swebench_evaluation,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Check if NeMo-Skills is available
try:
    import importlib.util

    nemo_skills_spec = importlib.util.find_spec("nemo_skills")
    NEMO_SKILLS_AVAILABLE = nemo_skills_spec is not None
except ImportError:
    NEMO_SKILLS_AVAILABLE = False

if not NEMO_SKILLS_AVAILABLE:
    LOG.warning("NeMo-Skills is not installed. Please install with: uv sync --extra nemo-skills")


class SWEBenchWrapperConfig(BaseResponsesAPIAgentConfig):
    """Configuration for SWE-bench wrapper agent."""

    # Agent framework configuration
    agent_framework: str = Field(default="swe_agent", description="Agent framework to use: swe_agent or openhands")
    agent_config: Optional[str] = Field(default=None, description="Path to agent configuration file")
    agent_tools_file: Optional[str] = Field(
        default=None, description="Path to JSON file containing tool definitions in OpenAI format (for SWE-agent)"
    )
    agent_max_turns: int = Field(default=100, description="Maximum iterations for the agent")
    agent_framework_repo: Optional[str] = Field(
        default=None,
        description="URL of the SWE-agent/OpenHands repo to pass to git clone. If None, will use the official repo",
    )

    agent_framework_commit: str = Field(
        default="HEAD", description="Which commit to use when cloning the SWE-agent/OpenHands repo"
    )
    # Container configuration
    container_formatter: str = Field(
        default="docker://swebench/sweb.eval.x86_64.{instance_id}", description="Container path template"
    )
    swebench_tests_timeout: int = Field(default=1800, description="Timeout for running tests (seconds)")

    # Model server reference (optional - can also be passed in request)
    model_server: Optional[ModelServerRef] = None

    # Additional NeMo-Skills config options
    nemo_skills_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration to pass to NeMo-Skills"
    )


class SWEBenchRunRequest(BaseRunRequest):
    """Request format for SWE-bench runs."""

    # Allow extra fields for flexibility
    model_config = {"extra": "allow"}


class SWEBenchVerifyRequest(BaseVerifyRequest):
    """Request format for SWE-bench verification."""

    model_config = {"extra": "allow"}


class SWEBenchVerifyResponse(BaseVerifyResponse):
    """Response format for SWE-bench verification."""

    model_config = {"extra": "allow"}

    # Additional SWE-bench specific fields
    swebench_metrics: Optional[Dict[str, Any]] = None

    # Additional numeric fields for rollout statistics
    resolved: Optional[float] = None  # 1.0 if resolved, 0.0 otherwise
    patch_exists: Optional[float] = None  # 1.0 if patch exists, 0.0 otherwise
    patch_successfully_applied: Optional[float] = None  # 1.0 if patch applied, 0.0 otherwise


class SWEBenchWrapper(SimpleResponsesAPIAgent):
    """Wrapper for NeMo-Skills SWE-bench evaluation in NeMo-Gym."""

    config: SWEBenchWrapperConfig

    def __init__(self, *args, **kwargs):
        """Initialize the wrapper and check dependencies."""
        super().__init__(*args, **kwargs)

        if not NEMO_SKILLS_AVAILABLE:
            raise ImportError(
                "NeMo-Skills is required for SWE-bench wrapper. Please install it with: uv sync --extra nemo-skills"
            )

        # Ensure symlink exists for /nemo_run/code
        ensure_nemo_run_symlink()

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        """Run NeMo-Skills SWE-bench evaluation."""

        LOG.info(f"Starting SWE-bench evaluation with framework: {self.config.agent_framework}")

        # Extract problem information from request
        problem_info = extract_problem_info(body, self.config.container_formatter)

        # Determine model endpoint
        model_endpoint = get_model_endpoint(self.config.model_server.name if self.config.model_server else None)

        # Run SWE-bench evaluation
        try:
            result = await run_swebench_evaluation(
                problem_info,
                model_endpoint,
                body,
                self.config.agent_framework,
                self.config.agent_config,
                self.config.agent_tools_file,
                self.config.agent_max_turns,
                self.config.swebench_tests_timeout,
                self.config.nemo_skills_config,
                self.config.agent_framework_repo,
                self.config.agent_framework_commit,
            )

            # Extract trajectory and convert to proper NeMoGym format
            output_items = []
            trajectory = result.get("trajectory", [])

            # Convert tools from ChatCompletion format to Response FunctionTool format
            raw_tools = result.get("tools", [])
            tools = convert_tools_to_function_format(raw_tools) if raw_tools else []

            # Convert trajectory to NeMoGym output items
            if trajectory:
                output_items = convert_trajectory_to_output_items(
                    trajectory, problem_info, self.config.agent_framework
                )

            # If no trajectory or empty output, create a summary message
            if not output_items:
                output_items = [
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{problem_info.get('instance_id', 'unknown')}",
                        content=[
                            NeMoGymResponseOutputText(
                                type="output_text",
                                text=json.dumps(
                                    {k: v for k, v in result.items() if k not in ["trajectory", "tools"]}, indent=2
                                ),
                                annotations=[],
                            )
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ]

            # Store the full result in metadata for the verify step
            # Note: metadata values must be strings for NeMoGymResponse
            metadata = {
                "agent_framework": self.config.agent_framework,
                "has_trajectory": str(bool(trajectory)),
                "instance_id": result.get("instance_id", problem_info.get("instance_id", "unknown")),
            }

            # Add evaluation results to metadata (convert to strings)
            for key in ["resolved", "patch_exists", "patch_successfully_applied"]:
                if key in result:
                    metadata[key] = str(result[key])

            # For complex metrics, store as JSON string
            if "swe-bench-metrics" in result:
                metadata["swe-bench-metrics"] = json.dumps(result["swe-bench-metrics"])

            return NeMoGymResponse(
                id=f"swebench-{problem_info.get('instance_id', 'unknown')}",
                created_at=int(time.time()),
                model=getattr(body, "model", "gpt-4.1-2025-04-14"),
                object="response",
                output=output_items,
                parallel_tool_calls=(False if self.config.agent_framework == "swe_agent" else True),
                tool_choice="auto",
                tools=tools,
                metadata=metadata,
            )

        except Exception as e:
            LOG.error(f"SWE-bench evaluation failed: {str(e)}")
            # Return error response
            error_message = NeMoGymResponseOutputMessage(
                id=f"msg-{problem_info.get('instance_id', 'unknown')}-error",
                content=[NeMoGymResponseOutputText(type="output_text", text=f"Error: {str(e)}", annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )

            return NeMoGymResponse(
                id=f"swebench-{problem_info.get('instance_id', 'unknown')}-error",
                created_at=int(time.time()),
                model=getattr(body, "model", "gpt-4.1-2025-04-14"),
                object="response",
                output=[error_message],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
                metadata={"error": str(e)},
            )

    async def run(self, body: SWEBenchRunRequest) -> SWEBenchVerifyResponse:
        """Run and verify SWE-bench solution."""
        # Fix None values in responses_create_params to use defaults
        # This is needed because the pydantic model has non-Optional fields with defaults
        update_dict = {}
        # SWE-agent processes tool calls sequentially, OpenHands can do parallel
        update_dict["parallel_tool_calls"] = False if self.config.agent_framework == "swe_agent" else True
        if body.responses_create_params.tool_choice is None:
            update_dict["tool_choice"] = "auto"  # OpenAI default

        # Create a copy with the fixed values if needed
        fixed_params = (
            body.responses_create_params.model_copy(update=update_dict)
            if update_dict
            else body.responses_create_params
        )

        # Run the evaluation
        response = await self.responses(fixed_params)

        # Extract initial input messages from the response output and get filtered output
        # These are the system/user messages that were actually sent to the agent
        input_messages, filtered_output = extract_input_messages_from_trajectory(response.output)

        # Update response with filtered output (system/user messages removed)
        response = response.model_copy(update={"output": filtered_output})

        # Add the extracted input messages and tools to the params
        # Note: tools should already be in the correct format from the response
        params_with_input = fixed_params.model_copy(
            update={"input": input_messages, "tools": response.tools if response.tools else []}
        )

        # Extract metrics from response metadata
        metadata = response.metadata or {}
        # Remove metadata from response after extracting metrics
        response = response.model_copy(update={"metadata": None})

        # Parse metrics from JSON string if present
        metrics = json.loads(metadata.get("swe-bench-metrics", "{}")) if "swe-bench-metrics" in metadata else {}

        # Extract individual metrics with proper type conversion
        resolved = metrics.get("resolved") or (metadata.get("resolved") == "True")
        patch_exists = metrics.get("patch_exists") or (metadata.get("patch_exists") == "True")
        patch_applied = metrics.get("patch_successfully_applied") or (
            metadata.get("patch_successfully_applied") == "True"
        )

        reward = 1.0 if resolved else 0.0

        # Build verification response with top-level numeric fields for statistics
        return SWEBenchVerifyResponse(
            responses_create_params=params_with_input,  # Include the input messages
            response=response,
            reward=reward,
            resolved=1.0 if resolved else 0.0,  # Top-level numeric field
            patch_exists=1.0 if patch_exists else 0.0,  # Top-level numeric field
            patch_successfully_applied=1.0 if patch_applied else 0.0,  # Top-level numeric field
            swebench_metrics=metrics,
            metadata={
                "instance_id": metadata.get("instance_id", "unknown"),
                "agent_framework": self.config.agent_framework,
                "patch_exists": patch_exists,
                "patch_successfully_applied": patch_applied,
                "resolved": resolved,
            },
        )


if __name__ == "__main__":
    SWEBenchWrapper.run_webserver()
