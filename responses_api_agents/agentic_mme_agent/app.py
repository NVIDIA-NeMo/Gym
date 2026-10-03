# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native multi-turn atomic Agentic-MME harness with append-only model history."""

import asyncio
import itertools
import json
from typing import Any

import cv2
from aiohttp import ClientError
from fastapi import Request, Response
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.agentic_mme.app import AgenticMMEVerifyResponse
from responses_api_agents.agentic_mme_agent.retrieval import RETRIEVAL_TOOLS, Retrieval, RetrievalConfig
from responses_api_agents.agentic_mme_agent.tools import IMAGE_TOOLS, ImageWorkspace, function_schema


class AgenticMMEAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # None removes the limit: the episode then ends when the model answers without a tool call,
    # or when the model call fails (e.g. the context is full), recorded as policy_error.
    max_rounds: int | None = Field(default=15, ge=1, le=100)
    max_tool_calls: int | None = Field(default=15, ge=0, le=100)
    max_concurrent_rollouts: int = Field(default=8, ge=1)
    max_image_pixels: int = Field(default=16_000_000, ge=1)
    max_total_image_pixels: int = Field(default=64_000_000, ge=1)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class AgenticMMERunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_replay: list[dict[str, Any]] = Field(default_factory=list)


class AgenticMMEAgent(SimpleResponsesAPIAgent):
    config: AgenticMMEAgentConfig
    _rollouts: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._rollouts = asyncio.Semaphore(self.config.max_concurrent_rollouts)
        cv2.setNumThreads(1)

    def tool_schemas(self) -> list[dict[str, Any]]:
        tools = dict(IMAGE_TOOLS)
        if self.config.retrieval.mode != "disabled":
            tools.update(RETRIEVAL_TOOLS)
        return [function_schema(name, model, description) for name, (model, description) in tools.items()]

    async def _episode(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
        replay: list[dict[str, Any]] | None = None,
    ) -> tuple[NeMoGymResponse, list[dict[str, Any]], Any, str | None]:
        async with self._rollouts:
            workspace = ImageWorkspace(
                max_pixels=self.config.max_image_pixels, max_total_pixels=self.config.max_total_image_pixels
            )
            return await self._episode_with_workspace(params, cookies, replay, workspace)

    async def _episode_with_workspace(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
        replay: list[dict[str, Any]] | None,
        workspace: ImageWorkspace,
    ) -> tuple[NeMoGymResponse, list[dict[str, Any]], Any, str | None]:
        params = params.model_copy(deep=True)
        if isinstance(params.input, str):
            params.input = [NeMoGymEasyInputMessage(role="user", content=params.input)]
        for item in params.input:
            data = item if isinstance(item, dict) else item.model_dump()
            content = data.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "input_image":
                        await asyncio.to_thread(workspace.load, part["image_url"])
        if not workspace.images:
            raise ValueError("Agentic-MME requires at least one input image")
        tools = self.tool_schemas()
        # Inputs contain task prompts only; verifier metadata never enters model context.
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            params.model_dump() | {"tools": tools, "parallel_tool_calls": False}
        )
        retrieval = Retrieval(self.config.retrieval, replay)
        outputs = []
        trace = []
        usage = None
        attempts = 0
        finished = False
        policy_error = None
        generated = None
        # Reserve an additional final-answer-only call when the interaction budget ends.
        max_rounds, max_tool_calls = self.config.max_rounds, self.config.max_tool_calls
        rounds = itertools.count() if max_rounds is None else range(max_rounds + 1)
        for turn in rounds:
            final_only = turn == max_rounds or (max_tool_calls is not None and attempts >= max_tool_calls)
            new_params = params.model_copy(
                update={
                    "input": [*params.input, *outputs],
                    "tool_choice": "none" if final_only else "auto",
                    "tools": [] if final_only else params.tools,
                }
            )
            response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_params,
                cookies=cookies,
            )
            if not response.ok:
                # A refused model call (e.g. an endpoint's request-size limit once many tool images
                # accumulate) ends this episode unanswered instead of failing the whole collection.
                policy_error = f"HTTP {response.status}: {(await response.read()).decode(errors='replace')[:1000]}"
                break
            cookies = response.cookies
            generated = NeMoGymResponse.model_validate(await get_response_json(response))
            usage = accumulate_response_usage(usage, generated.usage)
            # Preserve typed output items, including token IDs/log-probs for training.
            outputs.extend(generated.output)
            calls = [item for item in generated.output if item.type == "function_call"]
            if generated.incomplete_details:
                break
            if not calls and any(item.type == "message" and item.role == "assistant" for item in generated.output):
                finished = True
                break
            image_observations = []
            for call in calls:
                arguments = None
                try:
                    if final_only or (max_tool_calls is not None and attempts >= max_tool_calls):
                        raise ValueError("tool budget exhausted; provide a final answer")
                    attempts += 1  # Invalid calls consume budget too.
                    arguments = json.loads(call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    if call.name in IMAGE_TOOLS:
                        result = await asyncio.to_thread(workspace.apply, call.name, arguments)
                    elif call.name in RETRIEVAL_TOOLS:
                        result = await retrieval.call(call.name, arguments, workspace)
                    else:
                        raise ValueError("unknown tool")
                except (TimeoutError, ClientError):
                    result = {"ok": False, "error": "retrieval provider unavailable or timed out"}
                except (ValueError, TypeError, KeyError, OSError, cv2.error) as exc:
                    result = {"ok": False, "error": str(exc)}
                trace.append(
                    {
                        "index": len(trace) + 1,
                        "turn": turn + 1,
                        "tool_name": call.name,
                        "arguments": arguments,
                        "output": result,
                    }
                )
                image_url = result.get("image_url")
                observation = {key: value for key, value in result.items() if key != "image_url"}
                # Set type explicitly: requests are serialized with exclude_unset, and the model
                # server's converter cannot route an item that has neither type nor role.
                outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output", call_id=call.call_id, output=json.dumps(observation)
                    )
                )
                if image_url:
                    image_observations.append(
                        NeMoGymEasyInputMessage(
                            type="message",
                            role="user",
                            content=[
                                {"type": "input_text", "text": f"Image {result['new_image_index']}"},
                                {"type": "input_image", "image_url": image_url, "detail": "auto"},
                            ],
                        )
                    )
            # Complete all function-call outputs before inserting user images.
            # Some backends emit multiple calls even when parallel_tool_calls=False.
            outputs.extend(image_observations)
            if final_only:
                break
        if generated is None:
            generated = NeMoGymResponse.model_validate(
                {
                    "id": "policy_error",
                    "created_at": 0,
                    "model": "",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [],
                }
            )
        result = generated.model_copy(update={"output": outputs, "usage": usage})
        if not finished and result.incomplete_details is None:
            # No terminal answer must never accidentally score an earlier answer.
            result = NeMoGymResponse.model_validate(
                result.model_dump() | {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
            )
        return result, trace, cookies, policy_error

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        result, _, cookies, _ = await self._episode(body, request.cookies)
        for name, value in (cookies or {}).items():
            response.set_cookie(name, value)
        return result

    async def run(self, request: Request, body: AgenticMMERunRequest) -> AgenticMMEVerifyResponse:
        result, trace, _, policy_error = await self._episode(
            body.responses_create_params, request.cookies, body.retrieval_replay
        )
        verification = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json={
                "responses_create_params": body.responses_create_params.model_dump(),
                "response": result.model_dump(),
                "verifier_metadata": body.verifier_metadata,
            },
            cookies=request.cookies,
        )
        await raise_for_status(verification)
        verified = AgenticMMEVerifyResponse.model_validate(await get_response_json(verification))
        successful = sum(bool(event["output"].get("ok")) for event in trace)
        reference = (
            (body.verifier_metadata.get("process_evaluation") or {}).get("efficiency", {}).get("reference_tool_calls")
        )
        overthink = None
        if isinstance(reference, int) and not isinstance(reference, bool) and reference >= 0:
            overthink = max(0, successful - reference) / (reference + 1)
        return verified.model_copy(
            update={
                "tool_trace": trace,
                "tool_call_count": len(trace),
                "successful_tool_calls": successful,
                "tool_error_count": len(trace) - successful,
                "overthink": overthink,
                "process_scores_available": False,
                "policy_error": policy_error,
                **({"failure_reason": "policy_error"} if policy_error else {}),
            }
        )


if __name__ == "__main__":
    AgenticMMEAgent.run_webserver()
