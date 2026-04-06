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
"""
Orchestrator agent: decompose > dispatch sub-agents > synthesize.

Asks the model to decompose a problem into sub-tasks, solves each
sub-task with an independent LLM call, then synthesizes a final answer.

Graph: decompose -> dispatch (loop per subtask) -> synthesize -> END
"""

import re
from typing import Annotated, List, TypedDict

from app import LangGraphAgentAdapter, LangGraphAgentConfig
from fastapi import Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


DECOMPOSE_PROMPT = """Break the following problem into independent sub-tasks that can each be solved separately. \
For each sub-task, write it as a self-contained question that can be answered without context from the others.
You may use up to 5 subtasks.

Format your response like so:
SUBTASK 1: <question>
SUBTASK 2: <question>
SUBTASK 3: <question>

If the problem is simple enough to solve directly, just write:
SUBTASK 1: <the original problem>

Problem: {task}"""

SYNTHESIZE_PROMPT = """You decomposed a problem into sub-tasks and solved each one. \
Now combine the sub-task results into a final answer to the original problem.

Original problem: {task}

{subtask_results}

Synthesize these results into a single final answer. Show your reasoning, then wrap your final answer \
in <answer></answer> tags."""

SUBTASK_REGEX = r"SUBTASK\s+\d+:\s*(.+)"


class OrchestratorAgentConfig(LangGraphAgentConfig):
    max_subtasks: int = 5


class OrchestratorRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OrchestratorVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class OrchestratorVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    policy_outputs: list
    cookies: dict
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_policy_response: NeMoGymResponse
    task: str
    subtasks: List[str]
    subtask_results: dict
    current_subtask: int


def _extract_text(outputs):
    return "".join(c.text for o in outputs if o.type == "message" for c in o.content if c.type == "output_text")


# TODO: Use LangGraph's Send() API for the parallel worker dispatch, see langgraphs workflows.md Orchestrator-Worker pattern.
class OrchestratorAgent(LangGraphAgentAdapter):
    config: OrchestratorAgentConfig

    async def _call_model(self, state, prompt):
        input_messages = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        request_body = state["request_body"].model_copy(update={"input": input_messages + state["policy_outputs"]})
        resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=request_body,
            cookies=state["cookies"],
        )
        await raise_for_status(resp)
        return NeMoGymResponse.model_validate(await resp.json()), resp.cookies

    def build_graph(self):
        graph = StateGraph(OrchestratorState)

        async def decompose(state):
            task = state["task"]
            prompt = DECOMPOSE_PROMPT.format(task=task)
            prompt_msg = NeMoGymEasyInputMessage(role="user", content=prompt)

            policy_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(policy_response.output)

            matches = re.findall(SUBTASK_REGEX, text)
            subtasks = [m.strip() for m in matches[: self.config.max_subtasks]]

            # If no subtasks parsed, use the original task
            if not subtasks:
                subtasks = [task]

            return {
                "messages": [HumanMessage(content=prompt), AIMessage(content=text)],
                "policy_outputs": state["policy_outputs"] + [prompt_msg] + policy_response.output,
                "cookies": cookies,
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
                "subtasks": subtasks,
                "subtask_results": {},
                "current_subtask": 0,
            }

        async def dispatch(state):
            idx = state["current_subtask"]
            subtask = state["subtasks"][idx]
            prompt = f"Solve the following sub-task completely. Show your work.\n\nSub-task: {subtask}"
            prompt_msg = NeMoGymEasyInputMessage(role="user", content=prompt)

            policy_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(policy_response.output)

            new_results = {**state["subtask_results"], f"subtask_{idx + 1}": text}

            return {
                "messages": [HumanMessage(content=prompt), AIMessage(content=text)],
                "policy_outputs": state["policy_outputs"] + [prompt_msg] + policy_response.output,
                "cookies": cookies,
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
                "subtask_results": new_results,
                "current_subtask": idx + 1,
            }

        async def synthesize(state):
            task = state["task"]
            results_text = "\n\n".join(
                f"--- Sub-task {i + 1}: {state['subtasks'][i]} ---\nResult: {state['subtask_results'].get(f'subtask_{i + 1}', 'N/A')}"
                for i in range(len(state["subtasks"]))
            )
            prompt = SYNTHESIZE_PROMPT.format(task=task, subtask_results=results_text)
            prompt_msg = NeMoGymEasyInputMessage(role="user", content=prompt)

            policy_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(policy_response.output)

            return {
                "messages": [HumanMessage(content=prompt), AIMessage(content=text)],
                "policy_outputs": state["policy_outputs"] + [prompt_msg] + policy_response.output,
                "cookies": cookies,
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
            }

        def route_dispatch(state):
            if state["current_subtask"] >= len(state["subtasks"]):
                return "synthesize"
            return "dispatch"

        graph.add_node("decompose", decompose)
        graph.add_node("dispatch", dispatch)
        graph.add_node("synthesize", synthesize)
        graph.set_entry_point("decompose")
        graph.add_conditional_edges("decompose", route_dispatch, {"dispatch": "dispatch", "synthesize": "synthesize"})
        graph.add_conditional_edges("dispatch", route_dispatch, {"dispatch": "dispatch", "synthesize": "synthesize"})
        graph.add_edge("synthesize", END)

        return graph.compile()

    async def get_initial_state(self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict) -> dict:
        if isinstance(body.input, str):
            task = body.input
        else:
            task = ""
            for msg in body.input:
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "user")
                if role in ["user", "human"] and isinstance(content, str):
                    task = content

        return {
            "messages": [HumanMessage(content=task)],
            "policy_outputs": [],
            "cookies": cookies,
            "request_body": body,
            "last_policy_response": None,
            "task": task,
            "subtasks": [],
            "subtask_results": {},
            "current_subtask": 0,
        }

    def extract_outputs(self, final_state: dict) -> list:
        return final_state["policy_outputs"]

    async def run(self, request: Request, body: OrchestratorRunRequest) -> OrchestratorVerifyResponse:
        cookies = request.cookies

        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        resp = await self.server_client.post(
            server_name=self.config.name, url_path="/v1/responses", json=body.responses_create_params, cookies=cookies
        )
        await raise_for_status(resp)

        verify_request = OrchestratorVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(resp)}
        )

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resp.cookies,
        )
        await raise_for_status(verify)
        return OrchestratorVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    OrchestratorAgent.run_webserver()
