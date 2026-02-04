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
from typing import Dict, Optional

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.swe_agents.run_openhands import (
    RunOpenHandsAgent,
    SweBenchGenerationConfig,
    SweBenchInferenceConfig,
)


def get_openhands_trajectory_from_completions(trajectories_dir: Path, instance_id: str) -> tuple:
    """
    This reads the trajectories directly dumped by OpenHands.
    """
    messages, tools = [], []

    completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
    if not completions_dir.exists():
        print(f"No llm_completions directory found: {completions_dir}", flush=True)
        return messages, tools

    completion_files = sorted(completions_dir.glob("*.json"))
    if not completion_files:
        print(f"No completion files found in: {completions_dir}", flush=True)
        return messages, tools

    last_file = completion_files[-1]

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

    return messages, tools


async def run_swebench_evaluation(
    problem_info: Dict,
    model_endpoint: str,
    body: NeMoGymResponseCreateParamsNonStreaming,
    agent_framework: str,
    agent_config: Optional[str],
    agent_tools_file: Optional[str],
    agent_max_turns: int,
    swebench_tests_timeout: int,
    swebench_agent_timeout: int,
    persistent_dir: Path,
    metrics_fpath: Path,
    ng_global_config_dict_str: str,
    model_server_name: str,
    agent_framework_repo: Optional[str] = None,
    agent_framework_commit: str = "HEAD",
    openhands_setup_dir: Optional[Path] = None,
    swebench_setup_dir: Optional[Path] = None,
    r2e_gym_setup_dir: Optional[Path] = None,
    dataset_path: Optional[str] = None,
    ray_queue_time: Optional[float] = None,
    ray_submit_time: Optional[float] = None,
    openhands_should_log: bool = False,
    debug: bool = False,
    apptainer_memory_limit_mb: Optional[int] = None,
    command_exec_timeout: Optional[int] = None,
) -> Dict:
    instance_id = problem_info.get("instance_id", "unknown")
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
        agent_framework_repo=agent_framework_repo,
        agent_framework_commit=agent_framework_commit,
        agent_config=agent_config,
        agent_max_turns=agent_max_turns,
        swebench_tests_timeout=swebench_tests_timeout,
        swebench_agent_timeout=swebench_agent_timeout,
        apptainer_memory_limit_mb=apptainer_memory_limit_mb,
        command_exec_timeout=command_exec_timeout,
        inference=inference_config,
        server=server,
    )

    run_oh = RunOpenHandsAgent(
        cfg=cfg,
        openhands_setup_dir=openhands_setup_dir,
        swebench_setup_dir=swebench_setup_dir,
        r2e_gym_setup_dir=r2e_gym_setup_dir,
        dataset_path=dataset_path,
        ng_global_config_dict_str=ng_global_config_dict_str,
        openhands_should_log=openhands_should_log,
        debug=debug,
        model_server_name=model_server_name,
        metrics_fpath=metrics_fpath,
    )

    result = await run_oh.process_single_datapoint(problem_info, persistent_dir)
    print(f"Process completed for {instance_id}", flush=True)

    result["oh_time_metrics"]["ray_time_in_queue"] = ray_submit_time - ray_queue_time

    with open(output_file, "w") as f:
        json.dump(result, f)

    # Try to find and include trajectory file
    trajectories_dir = persistent_dir / "trajectories"
    trajectory_data, tools = get_openhands_trajectory_from_completions(trajectories_dir, instance_id)

    result["tools"] = tools
    result["trajectory"] = trajectory_data

    return result
