import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    SimpleResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    Body,
)
from nemo_gym.config_types import ResourcesServerRef, ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputMessage,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Check if NeMo-Skills is available
try:
    import nemo_skills
    NEMO_SKILLS_AVAILABLE = True
except ImportError:
    NEMO_SKILLS_AVAILABLE = False
    LOG.warning(
        "NeMo-Skills is not installed. Please install with: uv sync --extra nemo-skills"
    )


class SWEBenchWrapperConfig(BaseResponsesAPIAgentConfig):
    """Configuration for SWE-bench wrapper agent."""
    
    # Agent framework configuration
    agent_framework: str = Field(
        default="swe_agent",
        description="Agent framework to use: swe_agent or openhands"
    )
    agent_config: Optional[str] = Field(
        default=None,
        description="Path to agent configuration file"
    )
    agent_tools_file: Optional[str] = Field(
        default=None,
        description="Path to JSON file containing tool definitions in OpenAI format (for SWE-agent)"
    )
    agent_max_turns: int = Field(
        default=100,
        description="Maximum iterations for the agent"
    )
    
    # Container configuration
    container_formatter: str = Field(
        default="docker://swebench/sweb.eval.x86_64.{instance_id}",
        description="Container path template"
    )
    swebench_tests_timeout: int = Field(
        default=1800,
        description="Timeout for running tests (seconds)"
    )
    
    # Model server reference (optional - can also be passed in request)
    model_server: Optional[ModelServerRef] = None
    
    # Additional NeMo-Skills config options
    nemo_skills_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration to pass to NeMo-Skills"
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
    trajectory: Optional[List[Any]] = None
    
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
                "NeMo-Skills is required for SWE-bench wrapper. "
                "Please install it with: uv sync --extra nemo-skills"
            )
        
        # Ensure symlink exists for /nemo_run/code
        self._ensure_nemo_run_symlink()
    
    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Run NeMo-Skills SWE-bench evaluation."""
        
        LOG.info(f"Starting SWE-bench evaluation with framework: {self.config.agent_framework}")
        
        # Extract problem information from request
        problem_info = self._extract_problem_info(body)
        
        # Determine model endpoint
        model_endpoint = self._get_model_endpoint(body)
        
        # Run SWE-bench evaluation
        try:
            result = await self._run_swebench_evaluation(
                problem_info, model_endpoint, body
            )
            
            # Format successful result
            output_message = NeMoGymResponseOutputMessage(
                id=f"msg-{problem_info.get('instance_id', 'unknown')}",
                content=[
                    NeMoGymResponseOutputText(
                        type="output_text",
                        text=json.dumps(result, indent=2),
                        annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message"
            )
            
            return NeMoGymResponse(
                id=f"swebench-{problem_info.get('instance_id', 'unknown')}",
                created_at=int(time.time()),
                model=getattr(body, "model", "gpt-4.1-2025-04-14"),
                object="response",
                output=[output_message],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
                metadata={
                    "agent_framework": self.config.agent_framework
                }
            )
            
        except Exception as e:
            LOG.error(f"SWE-bench evaluation failed: {str(e)}")
            # Return error response
            error_message = NeMoGymResponseOutputMessage(
                id=f"msg-{problem_info.get('instance_id', 'unknown')}-error",
                content=[
                    NeMoGymResponseOutputText(
                        type="output_text",
                        text=f"Error: {str(e)}",
                        annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message"
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
                metadata={"error": str(e)}
            )
    
    def _extract_problem_info(self, body: NeMoGymResponseCreateParamsNonStreaming) -> Dict:
        """Extract SWE-bench problem information from request."""
        # Handle different input formats
        input_data = getattr(body, "input", "")
        
        if isinstance(input_data, str):
            problem_statement = input_data
        elif isinstance(input_data, list) and len(input_data) > 0:
            # Extract from last message
            last_message = input_data[-1]
            problem_statement = last_message.get("content", "")
        else:
            problem_statement = ""
        
        # Get metadata
        metadata = getattr(body, "metadata", {})
        
        # Build problem info
        problem_info = {
            "problem_statement": problem_statement,
            "instance_id": metadata.get("instance_id", "unknown"),
            "base_commit": metadata.get("base_commit", ""),
            "dataset_name": metadata.get("dataset_name", "princeton-nlp/SWE-bench_Verified"),
            "split": metadata.get("split", "test"),
            "container_formatter": self.config.container_formatter,
        }
        
        # Add any additional metadata
        problem_info.update(metadata)
        
        return problem_info
    
    def _get_model_endpoint(self, body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        """Determine the model API endpoint."""
        # Try to get server config from global config
        server_name = (
            self.config.model_server.name 
            if self.config.model_server 
            else "openai_model"
        )
        
        try:
            from nemo_gym.server_utils import ServerClient, get_first_server_config_dict
            
            global_config_dict = (
                ServerClient.load_from_global_config().global_config_dict
            )
            
            model_server_config = get_first_server_config_dict(
                global_config_dict,
                server_name,
            )
            
            base_url = (
                f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            )
            return base_url
            
        except Exception as e:
            LOG.error(f"Failed to get server config for {server_name}: {e}")
            raise RuntimeError(f"Could not determine model endpoint for server '{server_name}': {e}")
    
    async def _run_swebench_evaluation(
        self, problem_info: Dict, model_endpoint: str, body: Dict
    ) -> Dict:
        """Run SWE-bench evaluation using NeMo-Skills."""
        
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
        cmd = self._build_nemo_skills_command(
            input_file, output_file, model_endpoint, body
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
        trajectory_data = None
        trajectories_dir = persistent_dir / "trajectories"
        
        # Handle different agent frameworks' trajectory storage
        if self.config.agent_framework == "openhands":
            instance_id = problem_info.get('instance_id', 'unknown')
            
            # Get trajectory from completion files (complete and in OpenAI format)
            trajectory_data, tools = self._get_openhands_trajectory_from_completions(trajectories_dir, instance_id)
            if trajectory_data:
                LOG.info(f"Loaded OpenHands trajectory from llm_completions ({len(trajectory_data)} messages)")
                # Add tools to result
                if tools:
                    result["tools"] = tools
                else:
                    LOG.warning(f"No tools found in {trajectories_dir}")
            else:
                LOG.warning(f"No trajectory files found in {trajectories_dir}")
        elif self.config.agent_framework == "swe_agent":
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
                    if hasattr(self.config, 'agent_tools_file') and self.config.agent_tools_file:
                        tools_file = Path(__file__).parent / self.config.agent_tools_file
                        if tools_file.exists():
                            with open(tools_file, 'r') as f:
                                tools_data = json.load(f)
                                result["tools"] = tools_data.get("tools", [])
                                LOG.info(f"Loaded SWE-agent tools from {tools_file}")
                        else:
                            raise FileNotFoundError(f"SWE-agent tools file not found: {tools_file}")
                    else:
                        raise ValueError("No agent_tools_file configured for SWE-agent")
            else:
                LOG.warning(f"No trajectory files found in {trajectories_dir}")
        else:
            raise ValueError(f"Unsupported agent framework: {self.config.agent_framework}")
        
        # Add trajectory to result if found
        if trajectory_data:
            result["trajectory"] = trajectory_data
            LOG.info("Added trajectory data to result")
        else:
            # Log warning but don't fail - trajectory might not always be available
            LOG.warning(f"No trajectory files found in {trajectories_dir}")
            # Don't raise an error, just continue without trajectory data
        
        return result
    
    def _get_openhands_trajectory_from_completions(self, trajectories_dir: Path, instance_id: str) -> tuple:
        """Get trajectory from llm_completions directory for OpenHands.
        
        Returns:
            tuple: (messages, tools)
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
    

    def _build_nemo_skills_command(
        self, input_file: Path, output_file: Path, model_endpoint: str, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> list:
        """Build command to run NeMo-Skills SWE-bench evaluation."""
        
        # Extract model name from endpoint or body
        model_name = getattr(body, "model", "gpt-4.1-2025-04-14")
        
        # Build base command
        cmd = [
            sys.executable, "-m", "nemo_skills.inference.eval.swebench",
            f"++input_file={input_file}",
            f"++output_file={output_file}",
            f"++agent_framework={self.config.agent_framework}",
            f"++server.model={model_name}",
            f"++server.base_url={model_endpoint}",
            f"++agent_max_turns={self.config.agent_max_turns}",
            f"++swebench_tests_timeout={self.config.swebench_tests_timeout}",
        ]
        
        # Add agent config if specified
        if self.config.agent_config:
            cmd.append(f"++agent_config={self.config.agent_config}")
        
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
        for key, value in self.config.nemo_skills_config.items():
            cmd.append(f"++{key}={value}")
        
        return cmd
    
    def _ensure_nemo_run_symlink(self):
        """Ensure /nemo_run/code symlink exists pointing to nemo_skills package."""
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
    
    async def run(self, body: SWEBenchRunRequest) -> SWEBenchVerifyResponse:
        """Run and verify SWE-bench solution."""
        # Run the evaluation
        response = await self.responses(body.responses_create_params)
        
        # Extract results from response - the result is in the output message content
        result = {}
        metrics = {}
        
        # Parse the result from the response output
        if response.output and len(response.output) > 0:
            message = response.output[0]
            if message.content and len(message.content) > 0:
                text_content = message.content[0].text
                try:
                    result = json.loads(text_content)
                    # NeMo-Skills might output metrics under different keys
                    # Try "swe-bench-metrics" first, then check direct fields in result
                    metrics = result.get("swe-bench-metrics", {})
                    if not metrics:
                        # Check if metrics are directly in the result
                        if any(key in result for key in ["resolved", "patch_exists", "patch_successfully_applied"]):
                            metrics = {
                                "resolved": result.get("resolved", False),
                                "patch_exists": result.get("patch_exists", False),
                                "patch_successfully_applied": result.get("patch_successfully_applied", False),
                            }
                except json.JSONDecodeError:
                    LOG.warning(f"Failed to parse result JSON from response: {text_content[:200]}")
        
        # Calculate reward and other metrics
        resolved = metrics.get("resolved", False)
        patch_exists = metrics.get("patch_exists", False)
        patch_applied = metrics.get("patch_successfully_applied", False)
        
        reward = 1.0 if resolved else 0.0
        
        # Build verification response with top-level numeric fields for statistics
        return SWEBenchVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=response,
            reward=reward,
            resolved=1.0 if resolved else 0.0,  # Top-level numeric field
            patch_exists=1.0 if patch_exists else 0.0,  # Top-level numeric field
            patch_successfully_applied=1.0 if patch_applied else 0.0,  # Top-level numeric field
            swebench_metrics=metrics,
            trajectory=result.get("trajectory", []),
            metadata={
                "instance_id": result.get("instance_id", "unknown"),
                "agent_framework": self.config.agent_framework,
                "patch_exists": patch_exists,
                "patch_successfully_applied": patch_applied,
                "resolved": resolved,
            }
        )


if __name__ == "__main__":
    SWEBenchWrapper.run_webserver()
