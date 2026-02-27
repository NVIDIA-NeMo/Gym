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

"""
Example client for the GDPVal agent with a bash sandbox resources server.

Prerequisites:
    - The head server, model server, resources server, and agent server must all
      be running. See the configs/ directory for example YAML configurations.

Usage:
    python client.py
"""
import asyncio
import json

from nemo_gym.server_utils import ServerClient, get_response_json


# Tool definitions exposed to the model.
# NOTE: session_id and output_dir are injected by the agent — they should NOT
# appear in these definitions since the model doesn't manage sessions.
TOOLS = [
    {
        "type": "function",
        "name": "run_command",
        "description": "Execute a bash command in the sandbox environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum seconds to wait for the command (default 30).",
                },
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "finish",
        "description": (
            "Mark the task as complete and end the session. "
            "Optionally provide a list of file paths (relative to the sandbox working directory) "
            "to save as permanent output files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths in the sandbox to save as output.",
                },
            },
            "required": [],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a capable coding assistant with access to a bash sandbox. "
    "Use the run_command tool to execute bash commands. "
    "When you have completed the task, call the finish tool. "
    "If you created output files you want to keep, pass their paths to finish."
)

TASK_PROMPT = (
    "Write a Python script called hello.py that prints 'Hello, World!'. "
    "Run it to verify it works, then finish the task and save hello.py as output."
)


async def main():
    # Connect to the running NeMo Gym infrastructure
    server_client = ServerClient.load_from_global_config()

    # Build the run request body matching GDPValAgentRunRequest
    run_request = {
        # NeMoGymResponseCreateParamsNonStreaming — the model server params
        "responses_create_params": {
            "input": "",  # Overridden by the agent with system + task prompts
            "tools": TOOLS,
        },
        # GDPVal agent-specific fields
        "task_prompt": TASK_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "output_dir": "/tmp/gdpval_output",
        "task_id": "hello_world_task",
    }

    print("Submitting task to GDPVal agent...")
    print(f"  Task: {TASK_PROMPT}")
    print(f"  Output dir: {run_request['output_dir']}")
    print()

    # POST to the agent's /run endpoint
    response = await server_client.post(
        server_name="bash_sandbox_agent",
        url_path="/run",
        json=run_request,
    )

    result = await get_response_json(response)

    print("=" * 60)
    print("Task Result:")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    # Print a summary of saved files
    saved_files = result.get("output_files", [])
    if saved_files:
        print()
        print(f"Saved {len(saved_files)} output file(s):")
        for f in saved_files:
            print(f"  -> {f.get('output_path', 'unknown')} ({f.get('size', '?')} bytes)")
    else:
        print("\nNo output files were saved.")


if __name__ == "__main__":
    asyncio.run(main())
