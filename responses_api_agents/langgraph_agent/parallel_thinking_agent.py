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
Parallel thinking: multiple reasoning paths, then aggregate.

Calls the model N times concurrently with different perspective prompts,
then asks the model to synthesize a final answer from all results.

Graph: parallel_think -> aggregate -> END
"""

import asyncio
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


class ParallelThinkingAgentConfig(LangGraphAgentConfig):
    num_parallel_paths: int = 4
    perspective_prompts: list = DEFAULT_PERSPECTIVE_PROMPTS


class ParallelThinkingRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ParallelThinkingVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ParallelThinkingVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ParallelThinkingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    nemo_outputs: list
    cookies: dict
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_model_response: NeMoGymResponse
    task: str
    parallel_results: List[str]


def _extract_text(outputs):
    return "".join(c.text for o in outputs if o.type == "message" for c in o.content if c.type == "output_text")


class ParallelThinkingAgent(LangGraphAgentAdapter):
    config: ParallelThinkingAgentConfig

    async def _call_model(self, state, prompt):
        input_messages = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        request_body = state["request_body"].model_copy(update={"input": input_messages + state["nemo_outputs"]})
        resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=request_body,
            cookies=state["cookies"],
        )
        await raise_for_status(resp)
        return NeMoGymResponse.model_validate(await resp.json()), resp.cookies

    def build_graph(self):
        graph = StateGraph(ParallelThinkingState)

        async def parallel_think(state):
            task = state["task"]
            num_paths = self.config.num_parallel_paths
            prompts = self.config.perspective_prompts[:num_paths]
            # Pad with the default prompts if fewer perspective_prompts than num_paths
            while len(prompts) < num_paths:
                prompts.append(f"Think carefully about this problem and solve it:\n{task}")

            async def _single_path(perspective):
                full_prompt = f"{perspective}\n\nProblem: {task}"
                nemo_response, cookies = await self._call_model(state, full_prompt)
                text = _extract_text(nemo_response.output)
                return text, nemo_response, cookies

            results = await asyncio.gather(*[_single_path(p) for p in prompts])

            parallel_texts = [r[0] for r in results]
            # Use the last response/cookies for state continuity
            last_response = results[-1][1]
            last_cookies = results[-1][2]

            # Accumulate all nemo_outputs from all paths
            all_nemo_outputs = list(state["nemo_outputs"])
            for i, (text, nemo_response, _) in enumerate(results):
                prompt_msg = NeMoGymEasyInputMessage(role="user", content=f"{prompts[i]}\n\nProblem: {task}")
                all_nemo_outputs.append(prompt_msg)
                all_nemo_outputs.extend(nemo_response.output)

            summary = "\n\n".join(f"[Path {i + 1}]: {t}" for i, t in enumerate(parallel_texts))

            return {
                "messages": [AIMessage(content=summary)],
                "nemo_outputs": all_nemo_outputs,
                "cookies": last_cookies,
                "last_model_response": last_response,
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

            nemo_response, cookies = await self._call_model(state, prompt)
            text = _extract_text(nemo_response.output)

            return {
                "messages": [HumanMessage(content=prompt), AIMessage(content=text)],
                "nemo_outputs": state["nemo_outputs"] + [prompt_msg] + nemo_response.output,
                "cookies": cookies,
                "last_model_response": nemo_response,
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
            "nemo_outputs": [],
            "cookies": cookies,
            "request_body": body,
            "last_model_response": None,
            "task": task,
            "parallel_results": [],
        }

    def extract_outputs(self, final_state: dict) -> list:
        return final_state["nemo_outputs"]

    async def run(self, request: Request, body: ParallelThinkingRunRequest) -> ParallelThinkingVerifyResponse:
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

        verify_request = ParallelThinkingVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(resp)}
        )

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resp.cookies,
        )
        await raise_for_status(verify)
        return ParallelThinkingVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    ParallelThinkingAgent.run_webserver()
