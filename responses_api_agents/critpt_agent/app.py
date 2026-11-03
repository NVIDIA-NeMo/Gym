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
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import yaml
from fastapi import HTTPException, Request, Response
from pydantic import ConfigDict

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ModelCallRef


_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.DOTALL)

# Repo root: responses_api_agents/critpt_agent/app.py -> nemo-gym/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class CritPtAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    turn2_prompt_fpath: str


class CritPtAgentRunRequest(BaseRunRequest):
    # extra="allow" so extra row fields reach the resources server's seed and verify payloads unchanged.
    model_config = ConfigDict(extra="allow")

    problem_id: str
    code_template: str


class CritPtAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@dataclass
class CritPtSessionState(AgentSessionState):
    """Per-episode evidence retained until the environment closes the session."""

    observations: AgentObservationBundle | None = None


class CritPtAgent(SimpleResponsesAPIAgent):
    config: CritPtAgentConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        path = Path(self.config.turn2_prompt_fpath)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        prompt_yaml = yaml.safe_load(path.read_text())
        self._turn2_user_template: str = prompt_yaml["user"]

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> CritPtSessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("CritPt Agent does not support sandbox access")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError("CritPt Agent does not support required tool access")
        return CritPtSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, CritPtSessionState):
            raise TypeError("Expected CritPt session state")
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=state.observations,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Solve, then format using the deferred final user message in prepared input.

        Input ends with two user messages: the problem and the prepared formatting prompt.
        The formatting prompt is withheld from the solve call. Only the formatting output
        is returned for verification; usage covers both model calls.
        """
        messages = body.input
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or any(getattr(message, "role", None) != "user" for message in messages[-2:])
        ):
            raise HTTPException(
                422,
                "CritPt input must end with the problem and formatting user messages; "
                "run gym eval prepare --benchmark critpt to prepare both turns.",
            )
        session_id = self._agent_session_id_from_request(request)
        state = self._require_agent_session(session_id) if session_id is not None else None
        invocation = AgentInvocation(invocation_id="root", conversation=messages[:-1])
        if state is not None:
            if not isinstance(state, CritPtSessionState):
                raise TypeError("Expected CritPt session state")
            state.observations = AgentObservationBundle(source=self.config.name, records=[invocation])
        cookies = request.cookies
        started = monotonic()

        async def call_model(params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            nonlocal cookies
            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/responses", request),
                json=params,
                cookies=cookies,
            )
            await raise_for_status(model_response)
            cookies = cookies | {key: value.value for key, value in model_response.cookies.items()}
            for key, value in model_response.cookies.items():
                response.set_cookie(key, value.value)
            result = NeMoGymResponse.model_validate(await get_response_json(model_response))
            invocation.model_calls.append(ModelCallRef(model_ref=self.config.model_server, response_id=result.id))
            invocation.conversation = list(params.input) + result.output
            return result

        try:
            turn1 = await call_model(body.model_copy(update={"input": messages[:-1]}))
            if turn1.status == "failed" or turn1.error is not None:
                invocation.status = "failed"
                return turn1
            turn1_text = _strip_thinking_blocks(_extract_output_text(turn1.model_dump()))
            turn2_input = messages[:-1] + [
                NeMoGymEasyInputMessage(role="assistant", content=turn1_text),
                messages[-1],
            ]
            turn2 = await call_model(body.model_copy(update={"input": turn2_input}))
            invocation.status = (
                "failed"
                if turn2.status == "failed" or turn2.error is not None
                else "incomplete"
                if turn2.status == "incomplete"
                else "completed"
            )
            usage = None
            if turn1.usage is not None and turn2.usage is not None:
                usage = accumulate_response_usage(turn1.usage, turn2.usage)
            return turn2.model_copy(update={"usage": usage})
        except asyncio.CancelledError:
            invocation.status = "incomplete"
            raise
        except Exception as error:
            invocation.status = "failed"
            invocation.error_type = type(error).__name__
            raise
        finally:
            invocation.duration_ms = (monotonic() - started) * 1000

    async def run(self, request: Request, body: CritPtAgentRunRequest) -> CritPtAgentVerifyResponse:
        cookies = request.cookies

        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        self_responses_url_path = self.url_path_for_run("/v1/responses", body)

        # Older flat rows carry only the solve input. Adapt them to the same two-turn
        # request used by sessions, keeping verifier-only row fields out of model input.
        params = body.responses_create_params
        messages = params.input
        if isinstance(messages, str):
            messages = [NeMoGymEasyInputMessage(role="user", content=messages)]
        if len(messages) < 2 or any(getattr(message, "role", None) != "user" for message in messages[-2:]):
            messages = list(messages) + [
                NeMoGymEasyInputMessage(
                    role="user", content=self._turn2_user_template.format(code_template=body.code_template)
                )
            ]
        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self_responses_url_path,
            json=params.model_copy(update={"input": messages}),
            cookies=cookies,
        )
        await raise_for_status(agent_response)
        cookies = agent_response.cookies
        turn2_json = await get_response_json(agent_response)

        # Verify Turn 2 output against the Artificial Analysis API
        verify_request_data = body.model_dump() | {"response": turn2_json}
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request_data,
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return CritPtAgentVerifyResponse.model_validate(await get_response_json(verify_response))


def _strip_thinking_blocks(text: str) -> str:
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


def _extract_output_text(response_json: dict) -> str:
    parts = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


if __name__ == "__main__":
    CritPtAgent.run_webserver()
