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
See https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/rewoo/rewoo.ipynb
ReWOO (Reasoning Without Observation) agent.

Generates a full plan with variable substitution in a single LLM call,
then executes steps sequentially, substituting prior results. Last,
a solver synthesizes all results into a final answer.

Graph: plan -> worker -> (loop for each step) -> solve -> END
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


ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}

PLAN_PROMPT = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) LLM[input]: A pretrained LLM. Useful when you need to act with general world knowledge, \
reasoning, and common sense. Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x \
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours \
less than Toby. How many hours did Rebecca work?
Plan: Translate the problem into algebraic expressions and solve. #E1 = LLM[Solve x + (2x - 10) + ((2x - 10) - 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = LLM[Calculate (2 * #E2 - 10) - 8]

Begin!
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}"""

SOLVE_PROMPT = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer \
directly. Wrap your final answer in <answer></answer> tags.

Task: {task}
Response:"""

# Regex to match: Plan: <reasoning> #E1 = Tool[argument]
STEP_REGEX = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"


class ReWOOAgentConfig(LangGraphAgentConfig):
    pass


class ReWOORunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ReWOOVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ReWOOVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ReWOOState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    policy_outputs: list
    cookies: dict
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_policy_response: NeMoGymResponse
    task: str
    plan_string: str
    steps: List
    results: dict
    current_step: int


def _extract_text(outputs):
    return "".join(c.text for o in outputs if o.type == "message" for c in o.content if c.type == "output_text")


class ReWOOAgent(LangGraphAgentAdapter):
    config: ReWOOAgentConfig

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
        graph = StateGraph(ReWOOState)

        async def plan(state):
            task = state["task"]
            prompt = PLAN_PROMPT.format(task=task)
            prompt_msg = NeMoGymEasyInputMessage(role="user", content=prompt)

            policy_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(policy_response.output)

            matches = re.findall(STEP_REGEX, text)

            return {
                "messages": [HumanMessage(content=prompt), AIMessage(content=text)],
                "policy_outputs": state["policy_outputs"] + [prompt_msg] + policy_response.output,
                "cookies": cookies,
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
                "plan_string": text,
                "steps": matches,
                "results": {},
                "current_step": 0,
            }

        async def worker(state):
            step_idx = state["current_step"]
            _, step_name, tool, tool_input = state["steps"][step_idx]

            # Variable substitution: replace #E1, #E2, etc. with prior results
            for k, v in state["results"].items():
                tool_input = tool_input.replace(k, v)

            prompt = tool_input
            prompt_msg = NeMoGymEasyInputMessage(role="user", content=f"Step {step_name}: {prompt}")

            policy_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(policy_response.output)

            new_results = {**state["results"], step_name: text}

            return {
                "messages": [
                    HumanMessage(content=f"Step {step_name}: {prompt}"),
                    AIMessage(content=text),
                ],
                "policy_outputs": state["policy_outputs"] + [prompt_msg] + policy_response.output,
                "cookies": cookies,
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
                "results": new_results,
                "current_step": step_idx + 1,
            }

        async def solve(state):
            # Build plan string with evidence substituted
            plan_with_evidence = ""
            for _plan, step_name, tool, tool_input in state["steps"]:
                for k, v in state["results"].items():
                    tool_input = tool_input.replace(k, v)
                plan_with_evidence += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]\nEvidence: {state['results'].get(step_name, 'N/A')}\n\n"

            prompt = SOLVE_PROMPT.format(plan=plan_with_evidence, task=state["task"])
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

        def route_worker(state):
            if state["current_step"] >= len(state["steps"]):
                return "solve"
            return "worker"

        graph.add_node("plan", plan)
        graph.add_node("worker", worker)
        graph.add_node("solve", solve)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "worker")
        graph.add_conditional_edges("worker", route_worker, {"worker": "worker", "solve": "solve"})
        graph.add_edge("solve", END)

        return graph.compile()

    async def get_initial_state(self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict) -> dict:
        # Extract task text from input
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
            "plan_string": "",
            "steps": [],
            "results": {},
            "current_step": 0,
        }

    def extract_outputs(self, final_state: dict) -> list:
        return final_state["policy_outputs"]

    async def run(self, request: Request, body: ReWOORunRequest) -> ReWOOVerifyResponse:
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

        verify_request = ReWOOVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(resp)}
        )

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resp.cookies,
        )
        await raise_for_status(verify)
        return ReWOOVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    ReWOOAgent.run_webserver()
