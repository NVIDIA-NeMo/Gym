# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One vision-model call over public sandbox files, with no model-selected tools."""

import asyncio
import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from uuid import uuid4

from fastapi import Body, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.models.chat_completions_converter import ChatCompletionsConverter
from nemo_gym.sandbox import AsyncSandbox, create_provider, download_bytes, resolve_provider_config, upload_text
from nemo_gym.trajectory.trajectory_record import AgentObservationBundle, ObservationGap
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class DirectVisionAgentConfig(BaseResponsesAPIAgentConfig):
    """Public input manifest and answer artifact, independent of benchmark grading."""

    model_server: ModelServerRef
    input_path: str = "/task/input.json"
    answer_path: str = "/task/answer.txt"
    artifacts_dir: str = "/logs/artifacts"
    max_output_tokens: int = Field(default=32768, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    chat_template_kwargs: dict | None = None
    request_timeout_sec: float = Field(default=14400, gt=0)

    @field_validator("input_path", "answer_path", "artifacts_dir")
    @classmethod
    def absolute_file_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
            raise ValueError("Input and answer paths must be absolute file paths without '..'")
        return value


class VisionInput(BaseModel):
    """Public prompts and ordered image filenames relative to the manifest directory."""

    model_config = ConfigDict(extra="forbid", strict=True)
    system: str = ""
    user: str = Field(min_length=1)
    images: list[str] = Field(min_length=1)

    @field_validator("images")
    @classmethod
    def local_image_names(cls, names: list[str]) -> list[str]:
        if len(set(names)) != len(names):
            raise ValueError("Image filenames must be unique")
        for name in names:
            if not name or PurePosixPath(name).name != name or name in {".", ".."} or "\\" in name:
                raise ValueError("Images must be task-local filenames")
            if not (mimetypes.guess_type(name)[0] or "").startswith("image/"):
                raise ValueError(f"Unsupported image filename: {name}")
        return names


@dataclass
class VisionSession(AgentSessionState):
    sandbox: AsyncSandbox
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    response: NeMoGymResponse | None = None
    answer: str = ""
    raw_response: str = ""


class DirectVisionAgent(SimpleResponsesAPIAgent):
    """Borrow an environment-owned sandbox and answer its public visual question once."""

    config: DirectVisionAgentConfig
    ray_enabled = False

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> VisionSession:
        if body.sandbox_access is None:
            raise AgentSeedRejectedError("Direct vision requires an environment-owned sandbox")
        if self.effective_tool_accesses(body):
            raise AgentSeedRejectedError("Direct vision does not accept tool grants")
        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        return VisionSession(request=body, sandbox=sandbox)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, VisionSession):
            raise TypeError("Expected a direct vision session")
        await state.sandbox.disconnect()
        gaps = [ObservationGap(code="observation_capture_failed")]
        if state.response is not None:
            gaps.append(
                ObservationGap(
                    code="input_media_in_artifact",
                    detail="ATIF text profile: full model input is in policy-input.json",
                )
            )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(source="direct_vision", gaps=gaps),
        )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise AgentSessionConflictError("Seed a direct vision session before inference")
        state = self._require_agent_session(session_id)
        if not isinstance(state, VisionSession):
            raise TypeError("Expected a direct vision session")
        if body.tools:
            raise AgentSeedRejectedError("Direct vision does not accept model tools")
        async with state.lock:
            if state.response is None:
                await self._infer(request, body, state)
            # Retrying an artifact transfer reuses the model response rather than sampling again.
            await upload_text(
                state.sandbox,
                path=str(PurePosixPath(self.config.artifacts_dir) / "policy-response.json"),
                text=state.raw_response,
            )
            await upload_text(state.sandbox, path=self.config.answer_path, text=state.answer)
            return state.response

    async def _infer(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming, state: VisionSession
    ) -> None:
        raw = await download_bytes(state.sandbox, path=self.config.input_path)
        public = VisionInput.model_validate_json(raw)
        content = [{"type": "text", "text": public.user}]
        response_content = [{"type": "input_text", "text": public.user}]
        directory = PurePosixPath(self.config.input_path).parent
        for name in public.images:
            image = await download_bytes(state.sandbox, path=str(directory / name))
            media_type = mimetypes.guess_type(name)[0]
            url = f"data:{media_type};base64,{base64.b64encode(image).decode('ascii')}"
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
            response_content.append({"type": "input_image", "image_url": url, "detail": "auto"})
        messages = []
        if public.system:
            messages.append({"role": "system", "content": public.system})
        messages.append({"role": "user", "content": content})
        chat = NeMoGymChatCompletionCreateParamsNonStreaming(
            model=body.model or self.config.model_server.name,
            messages=messages,
            max_completion_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            tool_choice="none",
            stream=False,
        )
        if self.config.chat_template_kwargs:
            chat.chat_template_kwargs = self.config.chat_template_kwargs
        await upload_text(
            state.sandbox,
            path=str(PurePosixPath(self.config.artifacts_dir) / "policy-input.json"),
            text=json.dumps(chat.model_dump(exclude_unset=True)),
        )
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/chat/completions", request),
            json=chat.model_dump(exclude_unset=True),
            cookies=request.cookies,
            timeout=self.config.request_timeout_sec,
        )
        await raise_for_status(model_response)
        raw_completion = await get_response_json(model_response)
        completion = NeMoGymChatCompletion.model_validate(raw_completion)
        message = completion.choices[0].message
        if message.tool_calls or message.function_call:
            raise ValueError("Policy returned a tool call in a no-tool evaluation")
        converter = ChatCompletionsConverter(return_token_id_information=False)
        inputs = [{"role": "system", "content": public.system}] if public.system else []
        inputs.append({"role": "user", "content": response_content})
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=inputs,
            model=chat.model,
            tool_choice="none",
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        response = converter.chat_completion_to_response(params, completion)
        # Gym's current strict ATIF exporter supports text only. Keep the complete
        # request/response as artifacts rather than dropping images or masking a valid answer.
        transcript = [{"type": "message", "role": "system", "content": public.system}] if public.system else []
        transcript.append(
            {
                "type": "message",
                "role": "user",
                "content": public.user
                + "\n\nImages in order: "
                + ", ".join(str(directory / name) for name in public.images)
                + "\nFull image-bearing model input: "
                + str(PurePosixPath(self.config.artifacts_dir) / "policy-input.json"),
            }
        )
        raw_message = raw_completion["choices"][0]["message"]
        reasoning = raw_message.get("reasoning_content") or raw_message.get("reasoning")
        reasoning_items = []
        if isinstance(reasoning, str) and reasoning:
            reasoning_items.append(
                NeMoGymResponseReasoningItem(
                    id=f"rs_{uuid4().hex}",
                    type="reasoning",
                    status="completed",
                    summary=[{"type": "summary_text", "text": reasoning}],
                )
            )
        state.response = NeMoGymResponse.model_validate(
            response.model_dump()
            | {
                "output": [
                    *transcript,
                    *[item.model_dump() for item in reasoning_items],
                    *[item.model_dump() for item in response.output],
                ]
            }
        )
        state.raw_response = json.dumps(raw_completion)
        state.answer = message.content or ""


if __name__ == "__main__":
    DirectVisionAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DirectVisionAgent.run_webserver()
