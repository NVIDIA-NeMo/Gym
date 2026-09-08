#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free kernel coding agent for the task sandbox."""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from uuid import uuid4


def run(command: str) -> str:
    try:
        result = subprocess.run(command, cwd="/workspace", shell=True, text=True, capture_output=True, timeout=300)
        output = f"exit_code={result.returncode}\n{result.stdout}{result.stderr}"
    except subprocess.TimeoutExpired as exc:
        output = f"exit_code=124\n{exc.stdout or ''}{exc.stderr or ''}"
    return output[-50_000:]


def complete(url: str, key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> None:
    config = json.loads(os.environ["NGKB_AGENT_KWARGS"])
    model = config["model"].removeprefix("openai/")
    instruction = Path("/trajectories_mount/instruction.txt").read_text()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a CUDA kernel optimization agent. Work autonomously in /workspace. "
                "Inspect reference.py and solution.py, use shell commands to develop and test, "
                "and leave the final implementation in solution.py. Do not only explain the answer."
            ),
        },
        {"role": "user", "content": instruction},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command in /workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    final_text = ""
    max_turns = config.get("fabric_config", {}).get("runtime", {}).get("max_turns", 90)
    for _ in range(int(max_turns)):
        data = complete(
            os.environ["NGKB_MODEL_URL"],
            config["model_api_key"],
            {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto"},
        )
        message = data["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            final_text = message.get("content") or ""
            break
        for call in calls:
            arguments = json.loads(call["function"]["arguments"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": run(arguments["command"])})

    response = {
        "id": f"kernel-gym-runner-{uuid4().hex}",
        "created_at": int(time.time()),
        "model": model,
        "object": "response",
        "output": [
            {
                "id": f"msg_{uuid4().hex}",
                "content": [{"type": "output_text", "text": final_text, "annotations": []}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
    }
    Path("/trajectories_mount/response.json").write_text(json.dumps(response))
    print(f"agent finished after {len(messages)} messages", flush=True)


if __name__ == "__main__":
    main()
