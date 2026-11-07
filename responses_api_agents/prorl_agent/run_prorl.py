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
"""
ProRL-Agent-Server integration for prorl_agent.

Handles input/output conversion between Gym's SWEBenchRunRequest and
ProRL's /process endpoint.

ProRL server lifecycle (caller's responsibility before using this module):
    POST {prorl_url}/start
    POST {prorl_url}/add_llm_server  {"address": "http://..."}

ProRL /process input:
    {
        "instance": dict,          # full SWE-bench instance dict
        "sampling_params": dict,   # temperature, max_output_tokens, top_p, ...
        "job_id": str | None
    }

ProRL /process output:
    {
        "instance_id": str,
        "trajectory_id": str,
        "git_patch": str | None,
        "success": bool,
        "finish": bool,
        "messages": list[dict],    # see convert_prorl_messages_to_output_items
        "tools": list[dict],       # ChatCompletion tool format
        "end_properly": bool,
        "resolved": bool,
        "critical_error": str | None,   # "init" | "run" | "eval" | "timeout" | None
        "filter": bool,
        "error": str | None,
    }

ProRL message format (each item in "messages"):
    {
        "role": "system" | "user" | "assistant" | "tool",
        "content": str,
        # assistant only:
        "tool_calls": [{"name": str, "arguments": str | dict}],  # already flattened
        "token_ids": list[int] | None,    # generation token IDs
        "input_ids": list[int] | None,    # prompt token IDs
        "logprobs": list[float] | None,
        "repetition_penalty": list[float] | None,
        # tool only (may be absent):
        "tool_call_id": str | None,
    }
"""

import json
from typing import Any, Optional

import aiohttp

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


def convert_prorl_messages_to_output_items(
    messages: list[dict[str, Any]],
) -> list[NeMoGymResponseOutputItem]:
    """Convert ProRL messages list to NeMoGym output items.

    Field mapping:
        ProRL token_ids  -> generation_token_ids
        ProRL input_ids  -> prompt_token_ids
        ProRL logprobs   -> generation_log_probs
        ProRL tool_calls (flattened) -> NeMoGymResponseFunctionToolCall
        ProRL tool role  -> NeMoGymFunctionCallOutput
    """
    output_items: list[NeMoGymResponseOutputItem] = []
    # Maps sequential index to call_id for matching tool responses.
    pending_call_ids: list[str] = []
    call_counter = 0

    for item in messages:
        role = item.get("role", "")
        content = item.get("content", "") or ""

        if role in ("system", "user", "developer"):
            if content:
                output_items.append(
                    NeMoGymMessage(
                        content=[{"type": "input_text", "text": content}],
                        role=role,
                        status="completed",
                        type="message",
                    )
                )

        elif role == "assistant":
            tool_calls = item.get("tool_calls") or []
            generation_token_ids = item.get("token_ids") or []
            prompt_token_ids = item.get("input_ids") or []
            generation_log_probs = item.get("logprobs") or []

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

            # ProRL tool_calls are already flattened: {"name": str, "arguments": str|dict}
            for tc in tool_calls:
                call_counter += 1
                call_id = f"call_{call_counter}"
                name = tc.get("name", "")
                arguments = tc.get("arguments", "")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        name=name,
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )
                pending_call_ids.append(call_id)

        elif role == "tool":
            # Match with the oldest un-consumed tool call.
            tool_call_id = item.get("tool_call_id")
            if tool_call_id is None and pending_call_ids:
                tool_call_id = pending_call_ids.pop(0)
            elif tool_call_id is not None and tool_call_id in pending_call_ids:
                pending_call_ids.remove(tool_call_id)

            if tool_call_id:
                output_str = content if isinstance(content, str) else json.dumps(content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=tool_call_id,
                        output=output_str,
                        type="function_call_output",
                        status="completed",
                    )
                )

    return output_items


async def call_prorl_process(
    prorl_url: str,
    instance: dict[str, Any],
    sampling_params: dict[str, Any],
    job_id: Optional[str] = None,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """POST to ProRL /process and return the result dict."""
    request_body: dict[str, Any] = {
        "instance": instance,
        "sampling_params": sampling_params,
    }
    if job_id is not None:
        request_body["job_id"] = job_id

    client_timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(f"{prorl_url}/process", json=request_body) as resp:
            resp.raise_for_status()
            return await resp.json()


def initialize_prorl_server(prorl_url: str, model_endpoint: str) -> None:
    """Start ProRL server and register the LLM endpoint (synchronous, called from model_post_init).

    Calls /start (idempotent — ignores 400 if already running) then /add_llm_server.
    """
    import requests

    # /start — idempotent
    try:
        resp = requests.post(f"{prorl_url}/start", timeout=30)
        if resp.status_code == 400:
            print("ProRL server already running.", flush=True)
        else:
            resp.raise_for_status()
            print("ProRL server started.", flush=True)
    except Exception as e:
        print(f"Warning: ProRL /start call failed: {e}", flush=True)

    # /add_llm_server
    try:
        resp = requests.post(
            f"{prorl_url}/add_llm_server",
            json={"address": model_endpoint},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"ProRL LLM server registered: {model_endpoint}", flush=True)
    except Exception as e:
        print(f"Warning: ProRL /add_llm_server call failed: {e}", flush=True)
