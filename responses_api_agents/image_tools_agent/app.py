# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from io import BytesIO
from typing import Any, Optional

from fastapi import Request, Response
from PIL import Image
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.image_tools import (
    ImageToolsGymToolConfig,
    ImageToolsGymToolLogic,
    has_malformed_image_tool_markup,
    parse_image_tool_calls,
)


class ImageToolsAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resource_servers_by_agent: dict[str, ResourcesServerRef]
    max_steps: int = 5
    max_output_tokens: Optional[int] = 4096
    stop_strings: list[str] = ["</tool_call>"]
    include_stop_str_in_output: bool = True
    crop_dir: str = "image_tool_outputs"
    crop_format: str = "jpeg"
    crop_jpeg_quality: int = 95
    crop_min_pixels: int = 262144
    crop_max_pixels: int = 1048576
    max_tool_calls: int = 4
    max_tool_calls_per_turn: int = 1
    tool_success_reward: float = 0.02
    tool_success_reward_cap: float = 0.05
    invalid_tool_call_penalty: float = -0.05
    duplicate_tool_call_penalty: float = -0.02
    force_final_after_duplicate: bool = True
    terminate_on_invalid_tool_call: bool = True
    duplicate_iou_threshold: float = 0.5


class ImageToolsAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    image_tools_base_agent_ref: Optional[dict[str, Any]] = None


class ImageToolsAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ImageToolsAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    base_reward: float = 0.0
    image_tools_aux_reward: float = 0.0
    image_tools_call_count: int = 0
    image_tools_error_count: int = 0
    image_tools_output_paths: list[str] = []
    image_tools_generation_image_paths: list[list[str]] = []
    image_tools_base_agent_ref: Optional[dict[str, Any]] = None


def _pil_to_data_url(path: str, fmt: str = "JPEG") -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _extract_text_from_response(response: NeMoGymResponse) -> str:
    texts: list[str] = []
    for output_item in response.output:
        if getattr(output_item, "type", None) != "message":
            continue
        if getattr(output_item, "role", None) != "assistant":
            continue
        content = getattr(output_item, "content", None)
        if isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)
    return "\n".join(texts).strip()


def _extract_image_refs(params: NeMoGymResponseCreateParamsNonStreaming) -> list[str]:
    input_items = params.input
    if isinstance(input_items, str):
        return []
    image_refs: list[str] = []
    for item in input_items:
        if not isinstance(item, dict):
            item = item.model_dump(exclude_unset=True)
        content = item.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("input_image", "image_url", "image"):
                continue
            image_url = part.get("image_url", part.get("image", ""))
            if isinstance(image_url, dict):
                image_url = image_url.get("url", "")
            if image_url:
                image_refs.append(image_url)
    return image_refs


def _response_usage_add(
    running: Optional[NeMoGymResponseUsage],
    latest: Optional[NeMoGymResponseUsage],
) -> Optional[NeMoGymResponseUsage]:
    if latest is None:
        return running
    if running is None:
        return latest
    running.input_tokens += latest.input_tokens
    running.output_tokens += latest.output_tokens
    running.total_tokens += latest.total_tokens
    running.input_tokens_details.cached_tokens = 0
    running.output_tokens_details.reasoning_tokens = 0
    return running


def _final_assistant_response(response: NeMoGymResponse) -> NeMoGymResponse:
    for output_item in reversed(response.output):
        if getattr(output_item, "type", None) == "message" and getattr(output_item, "role", None) == "assistant":
            return response.model_copy(update={"output": [output_item]})
    return response


class ImageToolsAgent(SimpleResponsesAPIAgent):
    config: ImageToolsAgentConfig
    _logic: ImageToolsGymToolLogic = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._logic = ImageToolsGymToolLogic(self._logic_config())

    def _logic_config(self) -> ImageToolsGymToolConfig:
        return {
            "crop_dir": self.config.crop_dir,
            "crop_format": self.config.crop_format,
            "crop_jpeg_quality": self.config.crop_jpeg_quality,
            "crop_min_pixels": self.config.crop_min_pixels,
            "crop_max_pixels": self.config.crop_max_pixels,
            "max_tool_calls": self.config.max_tool_calls,
            "max_tool_calls_per_turn": self.config.max_tool_calls_per_turn,
            "tool_success_reward": self.config.tool_success_reward,
            "tool_success_reward_cap": self.config.tool_success_reward_cap,
            "invalid_tool_call_penalty": self.config.invalid_tool_call_penalty,
            "duplicate_tool_call_penalty": self.config.duplicate_tool_call_penalty,
            "force_final_after_duplicate": self.config.force_final_after_duplicate,
            "terminate_on_invalid_tool_call": self.config.terminate_on_invalid_tool_call,
            "duplicate_iou_threshold": self.config.duplicate_iou_threshold,
            "stop_strings": self.config.stop_strings,
        }

    def _resource_name(self, body: ImageToolsAgentRunRequest) -> str:
        base_agent_ref = body.image_tools_base_agent_ref
        if not base_agent_ref:
            extra = body.model_extra or {}
            base_agent_ref = extra.get("image_tools_base_agent_ref")
        if not base_agent_ref:
            raise ValueError("image_tools_base_agent_ref is required")
        base_agent_name = base_agent_ref["name"]
        if base_agent_name not in self.config.resource_servers_by_agent:
            raise ValueError(f"No resource server mapping for base agent {base_agent_name!r}")
        return self.config.resource_servers_by_agent[base_agent_name].name

    def _tool_user_message(self, observation: dict[str, Any]) -> tuple[NeMoGymEasyInputMessage, list[str]]:
        content = observation.get("content", "")
        if isinstance(content, str):
            return NeMoGymEasyInputMessage(role="user", content=content), []

        converted = []
        crop_paths = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                converted.append({"type": "input_text", "text": part.get("text", "")})
            elif part_type == "image":
                path = part["image"]
                crop_paths.append(path)
                converted.append(
                    {
                        "type": "input_image",
                        "image_url": _pil_to_data_url(path, self.config.crop_format.upper()),
                        "detail": "auto",
                    }
                )
        return NeMoGymEasyInputMessage(role="user", content=converted), crop_paths

    async def _call_model(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
    ):
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=cookies,
        )
        await raise_for_status(response)
        return NeMoGymResponse.model_validate(await get_response_json(response)), response.cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        result, model_cookies, _ = await self._run_image_tools_loop(
            body=body,
            row_metadata={},
            initial_cookies=request.cookies,
        )
        for key, value in model_cookies.items():
            response.set_cookie(key, value)
        return result

    async def _run_image_tools_loop(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        row_metadata: dict[str, Any],
        initial_cookies: Any,
    ) -> tuple[NeMoGymResponse, Any, dict[str, Any]]:
        body = body.model_copy(deep=True)
        if body.max_output_tokens is None and self.config.max_output_tokens is not None:
            body.max_output_tokens = self.config.max_output_tokens

        image_refs = _extract_image_refs(body)
        metadata = {
            "ground_truth": "",
            "image_paths": image_refs,
            "dataset": str(row_metadata.get("dataset", "unknown")),
            "tool_call_count": 0,
            "tool_error_count": 0,
            "crop_paths": [],
            "seen_tool_sigs": [],
            "seen_bboxes": {},
        }

        new_outputs = []
        usage = None
        model_cookies = initial_cookies
        aux_reward = 0.0
        generation_image_paths: list[list[str]] = [[]]
        final_response: Optional[NeMoGymResponse] = None

        for _ in range(self.config.max_steps):
            base_input = (
                [NeMoGymEasyInputMessage(role="user", content=body.input)]
                if isinstance(body.input, str)
                else list(body.input)
            )
            model_body = body.model_copy(update={"input": base_input + new_outputs})
            # The request body type has no top-level `stop` field (extra="forbid"),
            # so route stop strings to vLLM via metadata.extra_body, the same way
            # include_stop_str_in_output is passed.
            extra_body = {}
            if self.config.stop_strings:
                extra_body["stop"] = self.config.stop_strings
            if self.config.include_stop_str_in_output:
                extra_body["include_stop_str_in_output"] = True
            if extra_body:
                metadata_extra = dict(model_body.metadata or {})
                current_extra = json.loads(metadata_extra.get("extra_body", "{}"))
                current_extra.update(extra_body)
                metadata_extra["extra_body"] = json.dumps(current_extra)
                model_body.metadata = metadata_extra

            model_response, model_cookies = await self._call_model(model_body, model_cookies)
            new_outputs.extend(model_response.output)
            usage = _response_usage_add(usage, model_response.usage)
            final_response = model_response

            assistant_text = _extract_text_from_response(model_response)
            tool_calls = parse_image_tool_calls(assistant_text)
            malformed_tool_attempt = has_malformed_image_tool_markup(assistant_text)

            # process_nonterminal_turn executes the image tool: it decodes images,
            # runs PIL transforms and writes crops to disk, and for http(s) image
            # refs _open_image does a blocking requests.get. Calling it directly
            # from this async loop would stall the event loop for every other
            # concurrent rollout, so hand the whole synchronous call to a thread.
            # Keeping base.py synchronous matters -- the verifier imports it too.
            observation, reward, done, _, next_metadata, answer, _ = await asyncio.to_thread(
                self._logic.process_nonterminal_turn,
                [{"role": "assistant", "content": assistant_text}],
                metadata,
            )
            aux_reward += float(reward)
            if next_metadata is not None:
                metadata = next_metadata
            if not tool_calls and not malformed_tool_attempt:
                break
            if done:
                break
            if next_metadata is None:
                break

            user_message, crop_paths = self._tool_user_message(observation)
            new_outputs.append(user_message)
            generation_image_paths.append(crop_paths)
            if crop_paths:
                aux_reward = min(aux_reward, self.config.tool_success_reward_cap)

        if final_response is None:
            raise RuntimeError("ImageToolsAgent did not receive a model response")

        final_response = final_response.model_copy(update={"output": new_outputs, "usage": usage})
        rollout_info = {
            "image_tools_aux_reward": aux_reward,
            "image_tools_call_count": int(metadata.get("tool_call_count", 0)),
            "image_tools_error_count": int(metadata.get("tool_error_count", 0)),
            "image_tools_output_paths": list(metadata.get("crop_paths", [])),
            "image_tools_generation_image_paths": generation_image_paths,
        }
        return final_response, model_cookies, rollout_info

    async def run(self, request: Request, body: ImageToolsAgentRunRequest) -> ImageToolsAgentVerifyResponse:
        cookies = request.cookies
        resource_name = self._resource_name(body)

        seed_session_response = await self.server_client.post(
            server_name=resource_name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        resource_cookies = seed_session_response.cookies

        model_response, model_cookies, rollout_info = await self._run_image_tools_loop(
            body=body.responses_create_params,
            row_metadata=body.model_dump(exclude={"responses_create_params", "response"}),
            initial_cookies=cookies,
        )

        verify_request_body = body.model_dump()
        verify_request_body["response"] = _final_assistant_response(model_response).model_dump()
        verify_request = ImageToolsAgentVerifyRequest.model_validate(verify_request_body)
        verify_response = await self.server_client.post(
            server_name=resource_name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resource_cookies,
        )
        await raise_for_status(verify_response)
        verify_response_json = await get_response_json(verify_response)

        base_reward = float(verify_response_json.get("reward", 0.0))
        aux_reward = float(rollout_info["image_tools_aux_reward"])
        verify_response_json["base_reward"] = base_reward
        verify_response_json["image_tools_aux_reward"] = aux_reward
        verify_response_json["reward"] = base_reward + aux_reward
        verify_response_json["image_tools_base_agent_ref"] = body.image_tools_base_agent_ref
        verify_response_json.update(rollout_info)
        verify_response_json["response"] = model_response.model_dump()
        return ImageToolsAgentVerifyResponse.model_validate(verify_response_json)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return await super().aggregate_metrics(body)


if __name__ == "__main__":
    ImageToolsAgent.run_webserver()
