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
"""The four LangGraph workflows executed by the sandbox runner.

The algorithms and prompts are retained from this agent's original workflow modules.
Reflection and ReWOO retain their original MIT-licensed tutorial references (LangChain, Inc., 2024):
https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/reflection/reflection.ipynb
https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/rewoo/rewoo.ipynb

Retained notice for the original tutorial material:
MIT License

Copyright (c) 2024 LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Number
from typing import Annotated, TypedDict, cast

from aiohttp import ClientResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, JsonValue

from nemo_gym.client import get_response_json, raise_for_status, request
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputItem,
    NeMoGymResponseUsage,
)


type PolicyItem = NeMoGymResponseInputItem | NeMoGymResponseOutputItem


def response_output_items(policy_outputs: list[PolicyItem]) -> list[NeMoGymResponseOutputItem]:
    """Remove internal input prompts from the returned Responses API output."""
    input_roles = {"user", "system", "developer"}
    return cast(
        list[NeMoGymResponseOutputItem],
        [item for item in policy_outputs if getattr(item, "role", None) not in input_roles],
    )


def _aggregate_usage_values(values: list[object]) -> object:
    values = [value for value in values if value is not None]
    if not values:
        return None
    if all(isinstance(value, bool) for value in values):
        return values[-1]
    if all(isinstance(value, Number) and not isinstance(value, bool) for value in values):
        return sum(values)
    if all(isinstance(value, Mapping) for value in values):
        keys = dict.fromkeys(key for value in values for key in value)
        return {key: _aggregate_usage_values([value.get(key) for value in values]) for key in keys}
    if all(isinstance(value, list) for value in values):
        return [item for value in values for item in value]
    return values[-1]


def aggregate_response_usage(usages: list[NeMoGymResponseUsage | None]) -> NeMoGymResponseUsage | None:
    """Add every model call's usage, preserving unknown usage if any call omits it."""
    if not usages or any(usage is None for usage in usages):
        return None
    usage = _aggregate_usage_values([item.model_dump(mode="python") for item in usages])
    return NeMoGymResponseUsage.model_validate(usage)


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

DEFAULT_PERSPECTIVE_PROMPTS = [
    "Approach this problem step by step using logical deduction.",
    "Consider all possible cases and use process of elimination.",
    "Work backwards from the constraints to determine the answer.",
    "Identify the key relationships and build a truth table or systematic analysis.",
]

AGGREGATE_PROMPT = """You were given a problem and asked to reason about it from multiple perspectives. \
Below are the results of {num_paths} independent reasoning paths.

{paths_text}

Now synthesize these reasoning paths into a single, final answer. Consider where they agree and disagree. \
If there is a consensus, go with it. If they disagree, reason carefully about which path is most sound. \
Wrap your final answer in <answer></answer> tags.

Original problem: {task}"""

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

STEP_REGEX = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"


class WorkflowConfig(BaseModel):
    """Only algorithm settings cross from the host adapter into the graph runtime."""

    max_reflections: int = Field(default=2, ge=0)
    max_subtasks: int = Field(default=5, ge=1)
    num_parallel_paths: int = Field(default=4, ge=1)
    perspective_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_PERSPECTIVE_PROMPTS))


class _Workflow(ABC):
    def __init__(self, *, config: WorkflowConfig, model_url: str) -> None:
        self.config = config
        self.model_url = model_url
        self.graph = self.build_graph()

    async def _request(
        self, request_body: NeMoGymResponseCreateParamsNonStreaming, *, cookies: dict[str, str]
    ) -> ClientResponse:
        # model_url already includes the episode/capture prefix. Never rebuild it from a host config.
        return await request("POST", self.model_url, json=request_body.model_dump(mode="json"), cookies=cookies)

    @abstractmethod
    def build_graph(self) -> CompiledStateGraph:
        """Compile the workflow's existing nodes and routing."""

    @abstractmethod
    async def get_initial_state(
        self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict[str, str]
    ) -> dict[str, object]:
        """Create episode-local graph state."""

    @abstractmethod
    def extract_outputs(self, final_state: dict[str, object]) -> list[NeMoGymResponseOutputItem]:
        """Return response items without graph-internal prompts."""

    async def run(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        """Run the entire graph and aggregate its completed model calls."""
        initial_state = await self.get_initial_state(body.model_copy(deep=True), {})
        final_state = await self.graph.ainvoke(initial_state)
        response = final_state["last_policy_response"]
        response.output = self.extract_outputs(final_state)
        response.usage = aggregate_response_usage(final_state["policy_usages"])
        return response


class ReflectionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    policy_outputs: list[PolicyItem]
    policy_usages: list[NeMoGymResponseUsage | None]
    cookies: dict[str, str]
    model_url_path: str
    reflections: int
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_policy_response: NeMoGymResponse | None


class ReflectionWorkflow(_Workflow):
    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(ReflectionState)

        async def generate(state):
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            input_messages = [
                NeMoGymEasyInputMessage(role=role_map.get(m.type, m.type), content=m.content)
                for m in state["messages"]
            ]

            request_body = state["request_body"].model_copy(update={"input": input_messages})

            resp = await self._request(request_body, cookies=state["cookies"])

            await raise_for_status(resp)
            policy_response = NeMoGymResponse.model_validate(await get_response_json(resp))

            new_outputs = policy_response.output
            all_outputs = state["policy_outputs"] + new_outputs

            text = "".join(
                c.text for o in new_outputs if o.type == "message" for c in o.content if c.type == "output_text"
            )

            return {
                "messages": [AIMessage(content=text)],
                "policy_outputs": all_outputs,
                "policy_usages": state["policy_usages"] + [policy_response.usage],
                "cookies": {key: value.value for key, value in resp.cookies.items()},
                "reflections": state["reflections"],
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
            }

        async def reflect(state):
            reflection_prompt = NeMoGymEasyInputMessage(
                role="user", content="Critique your solution. What could be wrong?"
            )

            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            input_messages = [
                NeMoGymEasyInputMessage(role=role_map.get(m.type, m.type), content=m.content)
                for m in state["messages"]
            ] + [reflection_prompt]

            request_body = state["request_body"].model_copy(update={"input": input_messages})

            resp = await self._request(request_body, cookies=state["cookies"])

            await raise_for_status(resp)
            policy_response = NeMoGymResponse.model_validate(await get_response_json(resp))

            text = "".join(
                c.text
                for o in policy_response.output
                if o.type == "message"
                for c in o.content
                if c.type == "output_text"
            )

            return {
                "messages": [HumanMessage(content=text)],
                "policy_outputs": state["policy_outputs"] + [reflection_prompt] + policy_response.output,
                "policy_usages": state["policy_usages"] + [policy_response.usage],
                "cookies": {key: value.value for key, value in resp.cookies.items()},
                "reflections": state["reflections"] + 1,
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
            }

        def should_continue(state):
            if state["reflections"] >= self.config.max_reflections:
                return END
            last = state["messages"][-1].content if state["messages"] else ""
            return END if "<answer>" in last else "reflect"

        graph.add_node("generate", generate)
        graph.add_node("reflect", reflect)
        graph.set_entry_point("generate")
        graph.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
        graph.add_edge("reflect", "generate")

        return graph.compile()

    async def get_initial_state(
        self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict[str, str]
    ) -> dict[str, object]:
        if isinstance(body.input, str):
            initial_messages = [HumanMessage(content=body.input)]
            policy_outputs = []
        else:
            initial_messages = []
            policy_outputs = []
            for msg in body.input:
                is_output = (
                    hasattr(msg, "type")
                    and msg.type == "message"
                    and hasattr(msg, "role")
                    and msg.role == "assistant"
                    and hasattr(msg, "content")
                    and isinstance(msg.content, list)
                )
                is_function_call = hasattr(msg, "type") and msg.type == "function_call"

                if is_output or is_function_call:
                    policy_outputs.append(msg)
                else:
                    role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "user")
                    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                    if role in ["user", "human"]:
                        initial_messages.append(HumanMessage(content=content))
                    elif role in ["assistant", "ai"]:
                        initial_messages.append(AIMessage(content=content))

        return {
            "messages": initial_messages,
            "policy_outputs": policy_outputs,
            "policy_usages": [],
            "cookies": cookies,
            "model_url_path": "/v1/responses",
            "reflections": 0,
            "request_body": body,
            "last_policy_response": None,
        }

    def extract_outputs(self, final_state: dict[str, object]) -> list[NeMoGymResponseOutputItem]:
        return response_output_items(final_state["policy_outputs"])


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    policy_outputs: list[PolicyItem]
    policy_usages: list[NeMoGymResponseUsage | None]
    cookies: dict[str, str]
    model_url_path: str
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_policy_response: NeMoGymResponse | None
    task: str
    subtasks: list[str]
    subtask_results: dict[str, str]
    current_subtask: int


def _extract_text(outputs: list[NeMoGymResponseOutputItem]) -> str:
    return "".join(c.text for o in outputs if o.type == "message" for c in o.content if c.type == "output_text")


class OrchestratorWorkflow(_Workflow):
    async def _call_model(self, state, prompt):
        input_messages = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        request_body = state["request_body"].model_copy(update={"input": input_messages})
        resp = await self._request(request_body, cookies=state["cookies"])
        await raise_for_status(resp)
        return NeMoGymResponse.model_validate(await get_response_json(resp)), {
            key: value.value for key, value in resp.cookies.items()
        }

    def build_graph(self) -> CompiledStateGraph:
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
                "policy_usages": state["policy_usages"] + [policy_response.usage],
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
                "policy_usages": state["policy_usages"] + [policy_response.usage],
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
                "policy_usages": state["policy_usages"] + [policy_response.usage],
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

    async def get_initial_state(
        self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict[str, str]
    ) -> dict[str, object]:
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
            "policy_usages": [],
            "cookies": cookies,
            "model_url_path": "/v1/responses",
            "request_body": body,
            "last_policy_response": None,
            "task": task,
            "subtasks": [],
            "subtask_results": {},
            "current_subtask": 0,
        }

    def extract_outputs(self, final_state: dict[str, object]) -> list[NeMoGymResponseOutputItem]:
        return response_output_items(final_state["policy_outputs"])


class ParallelThinkingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    policy_outputs: list[PolicyItem]
    policy_usages: list[NeMoGymResponseUsage | None]
    cookies: dict[str, str]
    model_url_path: str
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_policy_response: NeMoGymResponse | None
    task: str
    parallel_results: list[str]


class ParallelThinkingWorkflow(_Workflow):
    async def _call_model(self, state, prompt):
        input_messages = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        request_body = state["request_body"].model_copy(update={"input": input_messages})
        resp = await self._request(request_body, cookies=state["cookies"])
        await raise_for_status(resp)
        return NeMoGymResponse.model_validate(await get_response_json(resp)), {
            key: value.value for key, value in resp.cookies.items()
        }

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(ParallelThinkingState)

        async def parallel_think(state):
            task = state["task"]
            num_paths = self.config.num_parallel_paths
            prompts = self.config.perspective_prompts[:num_paths]
            while len(prompts) < num_paths:
                prompts.append(f"Think carefully about this problem and solve it:\n{task}")

            async def _single_path(perspective):
                full_prompt = f"{perspective}\n\nProblem: {task}"
                policy_response, cookies = await self._call_model(state, full_prompt)
                text = _extract_text(policy_response.output)
                return text, policy_response, cookies

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(_single_path(p)) for p in prompts]
            results = [task.result() for task in tasks]

            parallel_texts = [r[0] for r in results]
            last_response = results[-1][1]
            last_cookies = results[-1][2]

            all_policy_outputs = list(state["policy_outputs"])
            all_policy_usages = list(state["policy_usages"])
            for i, (text, policy_response, _) in enumerate(results):
                prompt_msg = NeMoGymEasyInputMessage(role="user", content=f"{prompts[i]}\n\nProblem: {task}")
                all_policy_outputs.append(prompt_msg)
                all_policy_outputs.extend(policy_response.output)
                all_policy_usages.append(policy_response.usage)

            summary = "\n\n".join(f"[Path {i + 1}]: {t}" for i, t in enumerate(parallel_texts))

            return {
                "messages": [AIMessage(content=summary)],
                "policy_outputs": all_policy_outputs,
                "policy_usages": all_policy_usages,
                "cookies": last_cookies,
                "last_policy_response": last_response,
                "request_body": state["request_body"],
                "task": task,
                "parallel_results": parallel_texts,
            }

        async def aggregate(state):
            task = state["task"]
            paths_text = "\n\n".join(
                f"--- Path {i + 1} ---\n{text}" for i, text in enumerate(state["parallel_results"])
            )
            prompt = AGGREGATE_PROMPT.format(
                num_paths=len(state["parallel_results"]),
                paths_text=paths_text,
                task=task,
            )
            prompt_msg = NeMoGymEasyInputMessage(role="user", content=prompt)

            policy_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(policy_response.output)

            return {
                "messages": [HumanMessage(content=prompt), AIMessage(content=text)],
                "policy_outputs": state["policy_outputs"] + [prompt_msg] + policy_response.output,
                "policy_usages": state["policy_usages"] + [policy_response.usage],
                "cookies": cookies,
                "last_policy_response": policy_response,
                "request_body": state["request_body"],
                "task": state["task"],
                "parallel_results": state["parallel_results"],
            }

        graph.add_node("parallel_think", parallel_think)
        graph.add_node("aggregate", aggregate)
        graph.set_entry_point("parallel_think")
        graph.add_edge("parallel_think", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    async def get_initial_state(
        self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict[str, str]
    ) -> dict[str, object]:
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
            "policy_usages": [],
            "cookies": cookies,
            "model_url_path": "/v1/responses",
            "request_body": body,
            "last_policy_response": None,
            "task": task,
            "parallel_results": [],
        }

    def extract_outputs(self, final_state: dict[str, object]) -> list[NeMoGymResponseOutputItem]:
        return response_output_items(final_state["policy_outputs"])


class ReWOOState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    policy_outputs: list[PolicyItem]
    policy_usages: list[NeMoGymResponseUsage | None]
    cookies: dict[str, str]
    model_url_path: str
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_policy_response: NeMoGymResponse | None
    task: str
    plan_string: str
    steps: list[tuple[str, str, str, str]]
    results: dict[str, str]
    current_step: int


def _substitute_evidence(text: str, results: dict[str, str]) -> str:
    """Replace complete references once, preserving inserted evidence and unresolved references."""
    return re.sub(r"#E\d+", lambda match: results.get(match.group(0), match.group(0)), text)


class ReWOOWorkflow(_Workflow):
    async def _call_model(self, state, prompt):
        input_messages = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        request_body = state["request_body"].model_copy(update={"input": input_messages})
        resp = await self._request(request_body, cookies=state["cookies"])
        await raise_for_status(resp)
        return NeMoGymResponse.model_validate(await get_response_json(resp)), {
            key: value.value for key, value in resp.cookies.items()
        }

    def build_graph(self) -> CompiledStateGraph:
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
                "policy_usages": state["policy_usages"] + [policy_response.usage],
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

            tool_input = _substitute_evidence(tool_input, state["results"])

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
                "policy_usages": state["policy_usages"] + [policy_response.usage],
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
                tool_input = _substitute_evidence(tool_input, state["results"])
                plan_with_evidence += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]\nEvidence: {state['results'].get(step_name, 'N/A')}\n\n"

            prompt = SOLVE_PROMPT.format(plan=plan_with_evidence, task=state["task"])
            prompt_msg = NeMoGymEasyInputMessage(role="user", content=prompt)

            policy_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(policy_response.output)

            return {
                "messages": [HumanMessage(content=prompt), AIMessage(content=text)],
                "policy_outputs": state["policy_outputs"] + [prompt_msg] + policy_response.output,
                "policy_usages": state["policy_usages"] + [policy_response.usage],
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
        graph.add_conditional_edges("plan", route_worker, {"worker": "worker", "solve": "solve"})
        graph.add_conditional_edges("worker", route_worker, {"worker": "worker", "solve": "solve"})
        graph.add_edge("solve", END)

        return graph.compile()

    async def get_initial_state(
        self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict[str, str]
    ) -> dict[str, object]:
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
            "policy_usages": [],
            "cookies": cookies,
            "model_url_path": "/v1/responses",
            "request_body": body,
            "last_policy_response": None,
            "task": task,
            "plan_string": "",
            "steps": [],
            "results": {},
            "current_step": 0,
        }

    def extract_outputs(self, final_state: dict[str, object]) -> list[NeMoGymResponseOutputItem]:
        return response_output_items(final_state["policy_outputs"])


async def run_workflow(
    *,
    workflow: str,
    config: dict[str, JsonValue],
    request_body: NeMoGymResponseCreateParamsNonStreaming,
    model_url: str,
) -> NeMoGymResponse:
    """Run a named workflow using the exact Gym Model Server URL supplied for this episode."""
    workflows = {
        "reflection": ReflectionWorkflow,
        "orchestrator": OrchestratorWorkflow,
        "parallel_thinking": ParallelThinkingWorkflow,
        "rewoo": ReWOOWorkflow,
    }
    if workflow not in workflows:
        raise ValueError("Unknown LangGraph workflow")
    instance = workflows[workflow](config=WorkflowConfig.model_validate(config), model_url=model_url)
    return await instance.run(request_body)
