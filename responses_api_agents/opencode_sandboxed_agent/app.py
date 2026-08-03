# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import sys
from pathlib import Path
from shlex import quote
from time import time
from typing import Any, Dict
from uuid import uuid4

from fastapi import Request
from pydantic import Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, get_server_url, raise_for_status


class OpenCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    opencode_version: str
    opencode_config: Dict[str, Any] = Field(default_factory=dict)

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]


class OpenCodeSandboxedAgentRunRequest(BaseRunRequest):
    pass


class OpenCodeSandboxedAgentVerifyResponse(BaseVerifyResponse):
    turns_used: int = 0
    finished_naturally: bool = False


class OpenCodeSandboxedAgent(SimpleResponsesAPIAgent):
    config: OpenCodeSandboxedAgentConfig

    async def _start_sandbox(self) -> AsyncSandbox:
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        sandbox_spec = SandboxSpec(
            image="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907",  # This is just the first SWE Bench Verified image for now
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=dict(),
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
            },
            resources=SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {})),
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(sandbox_spec)

        return sandbox

    def _create_opencode_config(self) -> Dict[str, Any]:
        return {
            "model": "nemo_gym/dummy_model",
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "nemo_gym": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": f"{get_server_url(self.config.model_server.name)}/v1",
                        "apiKey": "dummy_key",
                    },
                    "models": {
                        "dummy_model": {},  # TODO @bxyu-nvidia: Propogate sampling params here.
                    },
                }
            },
        }

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox = await self._start_sandbox()

        query = None
        # This can be modified to handle system/developer prompts too.
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]

        assert query, body.input

        export_fname = "export.json"
        command = f"""
        echo "Shell: $SHELL" \
        && curl -fsSL https://opencode.ai/install | VERSION={self.config.opencode_version} bash \
        && export PATH=$HOME/.opencode/bin:$PATH \
        && opencode run {quote(query)} \
        && session_id=$(opencode session list --format json | jq -r '.[0].id') \
        && opencode export $session_id > {export_fname}
        """

        result = await sandbox.exec(
            command=command,
            env={"OPENCODE_CONFIG_CONTENT": json.dumps(self._create_opencode_config())},
        )

        print("STDOUT: ", result.stdout, file=sys.stderr)
        print("STDERR: ", result.stderr, file=sys.stderr)

        pwd_result = await sandbox.exec(command="pwd")
        results_remote_fpath = Path(pwd_result.stdout) / export_fname

        results_dir: Path = Path(__file__).parent / "results" / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        results_local_fpath = results_dir / export_fname
        print(f"Downloading results from {results_remote_fpath} to {results_local_fpath}", file=sys.stderr)
        await sandbox.download(str(results_remote_fpath), results_local_fpath)

        await sandbox.stop()

        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = None, None
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        output_items, usage, model_name = await self._run_opencode(user_message, system_prompt)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def run(
        self, request: Request, body: OpenCodeSandboxedAgentRunRequest
    ) -> OpenCodeSandboxedAgentVerifyResponse:
        cookies = request.cookies

        seed_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_resp)
        cookies = seed_resp.cookies

        agent_resp = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_resp)
        cookies = agent_resp.cookies
        agent_resp_json = await get_response_json(agent_resp)

        verify_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": agent_resp_json},
            cookies=cookies,
        )
        await raise_for_status(verify_resp)
        verify_json = await get_response_json(verify_resp)

        gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
        turns = sum(
            1
            for item in gym_resp.output
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
        )
        last = gym_resp.output[-1] if gym_resp.output else None
        naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

        return OpenCodeSandboxedAgentVerifyResponse.model_validate(
            verify_json | {"turns_used": turns, "finished_naturally": naturally}
        )


if __name__ == "__main__":
    OpenCodeSandboxedAgent.run_webserver()
