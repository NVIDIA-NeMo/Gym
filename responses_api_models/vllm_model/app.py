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
import base64
import json
import os
from copy import deepcopy
from time import time
from typing import Any, ClassVar, Dict, List, Optional, Union

from aiohttp.client_exceptions import ClientResponseError
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    RunTrajectory,
    SimpleResponsesAPIModel,
    TokenIDBufferingMixin,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import (
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,  # noqa: F401
    split_responses_input_output_items,  # noqa: F401
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint


def _chat_completion_to_sse(completion: NeMoGymChatCompletion):
    # re-emit a complete chat completion as SSE chunks (role, content, tool_calls, finish, usage, DONE)
    cc = completion.model_dump()
    choice = (cc.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    base = {k: cc.get(k) for k in ("id", "created", "model", "system_fingerprint")}
    base["object"] = "chat.completion.chunk"

    def _event(delta: Dict[str, Any], finish_reason=None) -> str:
        chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return f"data: {json.dumps(chunk)}\n\n"

    yield _event({"role": "assistant"})
    if message.get("content"):
        yield _event({"content": message["content"]})
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        yield _event(
            {
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")},
                    }
                ]
            }
        )
    yield _event({}, finish_reason=choice.get("finish_reason") or "stop")
    if cc.get("usage"):
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': cc['usage']})}\n\n"
    yield "data: [DONE]\n\n"


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool


    # pin sampling for run-scoped requests so nemo-rl's on-policy assert passes
    on_policy_temperature: float = 1.0
    on_policy_top_p: float = 1.0

    uses_reasoning_parser: bool
    uses_interleaved_reasoning: bool = True
    replace_developer_role_with_system: bool = False
    sequential_reasoning_allowed: bool = True
    is_responses_native: bool = False
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None
    default_headers: Dict[str, str] = Field(default_factory=dict)
    audio_root: Optional[str] = None

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


class VLLMModel(TokenIDBufferingMixin, SimpleResponsesAPIModel):
    config: VLLMModelConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        self.setup_token_id_buffering(app)
        return app

    def get_converter(self) -> "VLLMConverter":
        return VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
            uses_reasoning_parser=self.config.uses_reasoning_parser,
        )

    def model_post_init(self, context):
        self._post_init()
        return super().model_post_init(context)

    def _post_init(self) -> None:
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.api_key,
                default_headers=self.config.default_headers,
            )
            for base_url in self.config.base_url
        ]
        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()
        self._converter = self.get_converter()
        self._trajectory = RunTrajectory(self.token_id_buffer_dir())

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if self.config.is_responses_native:
            return await self._responses_native(request, body)

        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        body.model = self.config.model

        # Chat Completion Create Params -> Chat Completion
        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        return self._converter.chat_completion_to_response(
            responses_create_params=body, chat_completion=chat_completion_response
        )

    async def _responses_native(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        if self.config.return_token_id_information:
            raise NotImplementedError
        if self.config.replace_developer_role_with_system:
            raise NotImplementedError
        if not self.config.sequential_reasoning_allowed:
            raise NotImplementedError

        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.model
        if self.config.chat_template_kwargs:
            body_dict["chat_template_kwargs"] = deepcopy(self.config.chat_template_kwargs)
        if self.config.extra_body:
            body_dict = self.config.extra_body | body_dict

        client = self._resolve_client(request)
        response_dict = await client.create_response(**body_dict)
        return NeMoGymResponse.model_validate(response_dict)

    _AUDIO_EXT_TO_MIME: ClassVar[Dict[str, str]] = {
        ".wav": "wav",
        ".flac": "flac",
        ".mp3": "mpeg",
        ".m4a": "mp4",
        ".ogg": "ogg",
        ".opus": "opus",
    }

    def _resolve_audio_path_to_url(self, audio_path: str) -> str:
        if os.path.isabs(audio_path):
            resolved = audio_path
        elif self.config.audio_root:
            resolved = os.path.join(self.config.audio_root, audio_path)
        else:
            raise ValueError(
                f"metadata.audio_path={audio_path!r} is relative but VLLMModelConfig.audio_root "
                "is unset. set audio_root in the model config or use absolute paths."
            )
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"metadata.audio_path resolved to {resolved!r}, which does not exist.")
        ext = os.path.splitext(resolved)[1].lower()
        mime = self._AUDIO_EXT_TO_MIME.get(ext)
        if mime is None:
            raise ValueError(
                f"unsupported audio extension {ext!r} for {resolved!r}. supported: {sorted(self._AUDIO_EXT_TO_MIME)}."
            )
        with open(resolved, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:audio/{mime};base64,{encoded}"

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.replace_developer_role_with_system:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"

        body_dict["model"] = self.config.model

        chat_template_kwargs = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)

        metadata = body_dict.get("metadata", dict())
        metadata_chat_template_kwargs_str = metadata.get("chat_template_kwargs", "{}")
        chat_template_kwargs.update(json.loads(metadata_chat_template_kwargs_str))
        if chat_template_kwargs:
            body_dict["chat_template_kwargs"] = chat_template_kwargs

        extra_body = {}
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)
        metadata_extra_body_str = metadata.get("extra_body", "{}")
        extra_body.update(json.loads(metadata_extra_body_str))

        if self.config.return_token_id_information:
            body_dict |= dict(
                logprobs=True,
                top_logprobs=0,
                return_tokens_as_token_ids=True,
            )

        if self.config.uses_reasoning_parser:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") != "assistant" or "content" not in message_dict:
                    continue
                content = message_dict["content"]
                if isinstance(content, str):
                    reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(content)
                    message_dict["content"] = remaining_content
                    if reasoning_matches and self.config.uses_interleaved_reasoning:
                        message_dict["reasoning_content"] = reasoning_matches[0]
                        message_dict["reasoning"] = reasoning_matches[0]
                elif isinstance(content, list):
                    reasoning_content = None
                    for content_item_dict in content:
                        reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(
                            content_item_dict["text"]
                        )
                        assert reasoning_content is None or not reasoning_matches, (
                            f"found multiple reasoning matches in a single assistant message content item list!\nmessage: {message_dict}"
                        )
                        content_item_dict["text"] = remaining_content
                        if reasoning_matches and self.config.uses_interleaved_reasoning:
                            message_dict["reasoning_content"] = reasoning_matches[0]
                            message_dict["reasoning"] = reasoning_matches[0]
                elif not content:
                    pass
                else:
                    raise NotImplementedError

        # Drop a null top_logprobs on the non-capture path (caller-supplied logprobs=True).
        # vLLM treats null as "no logprobs" but a missing field as its default (0), so forwarding null is never useful.
        # The capture path above already set it to 0 and is unaffected.
        if body_dict.get("top_logprobs") is None:
            body_dict.pop("top_logprobs", None)

        if extra_body:
            body_dict = extra_body | body_dict

        # audio sidechannel: rows carry audio via metadata (audio_data/audio_path/audio_paths),
        # spliced as audio_url content blocks into the last user message before forwarding to vllm
        audio_keys_present = [k for k in ("audio_data", "audio_path", "audio_paths") if metadata.get(k)]
        if len(audio_keys_present) > 1:
            raise ValueError(
                f"metadata audio keys are mutually exclusive — got {audio_keys_present}. "
                "set exactly one of audio_data / audio_path / audio_paths per row."
            )

        audio_urls: List[str] = []
        if metadata.get("audio_data"):
            audio_urls.append(metadata["audio_data"])
            metadata.pop("audio_data", None)
        elif metadata.get("audio_path"):
            audio_urls.append(self._resolve_audio_path_to_url(metadata["audio_path"]))
            metadata.pop("audio_path", None)
        elif metadata.get("audio_paths"):
            paths = metadata["audio_paths"]
            if not isinstance(paths, list):
                raise ValueError(f"metadata.audio_paths must be a list, got {type(paths).__name__}.")
            audio_urls.extend(self._resolve_audio_path_to_url(p) for p in paths)
            metadata.pop("audio_paths", None)

        if audio_urls:
            if not metadata and "metadata" in body_dict:
                body_dict.pop("metadata", None)
            audio_blocks = [{"type": "audio_url", "audio_url": {"url": url}} for url in audio_urls]
            messages = body_dict.get("messages", []) or []
            for msg in reversed(messages):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = audio_blocks + [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    msg["content"] = audio_blocks + list(content)
                else:
                    msg["content"] = list(audio_blocks)
                break
            else:
                body_dict.setdefault("messages", []).append({"role": "user", "content": list(audio_blocks)})

        return body_dict

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        # streaming clients send stream=true; generate non-streaming so token ids are buffered, re-emit as sse
        completion = await self._generate_chat_completion(request, body)
        if body.stream:
            return StreamingResponse(_chat_completion_to_sse(completion), media_type="text/event-stream")
        return completion

    async def _generate_chat_completion(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict.pop("stream", None)
        body_dict.pop("stream_options", None)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)

        if getattr(request.state, "run_token", None) is not None:
            body_dict["temperature"] = self.config.on_policy_temperature
            body_dict["top_p"] = self.config.on_policy_top_p

        self.attach_tokens_and_logprobs(request, body_dict.get("messages") or [])

        client = self._resolve_client(request)

        if not self.config.sequential_reasoning_allowed:
            last_message = body_dict["messages"][-1]
            if last_message["role"] == "assistant" and not (last_message["content"] or last_message.get("tool_calls")):
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "content_filter"
                return res

        try:
            chat_completion_dict = await client.create_chat_completion(**body_dict)
        except ClientResponseError as e:
            result_content_str = e.response_content.decode()
            is_out_of_context_length = e.status == 400 and (
                "context length" in result_content_str or "max_tokens" in result_content_str
            )
            if is_out_of_context_length:
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "length"
                return res
            else:
                raise e

        choice_dict = chat_completion_dict["choices"][0]
        if self.config.uses_reasoning_parser:
            reasoning_content = choice_dict["message"].get("reasoning_content") or choice_dict["message"].get(
                "reasoning"
            )
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content", None)
                choice_dict["message"].pop("reasoning", None)
                choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                    [reasoning_content]
                ) + (choice_dict["message"].get("content") or "")
        else:
            assert not (choice_dict["message"].get("reasoning_content") or choice_dict["message"].get("reasoning")), (
                f"nemo gym server `{self.config.name}` has uses_reasoning_parser=false but the vllm endpoint returned reasoning content. fix the server config or disable the reasoning parser in vllm."
            )

        if self.config.return_token_id_information and "prompt_token_ids" not in choice_dict["message"]:
            # Check vLLM honored the logprobs request.
            # It returns choice.logprobs=None when it computed none.
            # That happens when a null top_logprobs reached it, or the contract changed across versions.
            # Without this check the code below raises a TypeError or emits empty token ids that zero the loss mask.
            # An empty content list is a valid zero-token generation and passes through.
            logprobs_block = choice_dict.get("logprobs")
            if not logprobs_block or logprobs_block.get("content") is None:
                raise RuntimeError(
                    f"`{self.config.name}` requested per-token logprobs from vLLM "
                    f"(return_token_id_information=True, logprobs=True, top_logprobs=0), but the response "
                    f"had none (choice.logprobs={logprobs_block!r}). Cannot extract token ids or logprobs."
                )
            log_probs = logprobs_block["content"]
            generation_log_probs = [log_prob["logprob"] for log_prob in log_probs]
            generation_token_ids = [log_prob["token"].removeprefix("token_id:") for log_prob in log_probs]

            tokenize_body_dict = dict()
            for key in ("model", "messages", "tools", "chat_template_kwargs"):
                if key in body_dict:
                    tokenize_body_dict[key] = body_dict[key]
            tokenize_response = await client.create_tokenize(**tokenize_body_dict)

            message_dict = choice_dict["message"]
            message_dict.update(
                dict(
                    prompt_token_ids=tokenize_response["tokens"],
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )
            choice_dict.pop("logprobs")

        self.buffer_turn(
            request,
            body_dict.get("messages") or [],
            choice_dict["message"],
            self._converter.postprocess_assistant_message_dict,
        )

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)

    def _create_empty_chat_completion(self) -> NeMoGymChatCompletion:
        return NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=int(time()),
            model=self.config.model,
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=None,
                    ),
                )
            ],
        )

    def _resolve_client(self, request: Request) -> NeMoGymAsyncOpenAI:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._session_id_to_client:
            client_idx = len(self._session_id_to_client) % len(self._clients)
            self._session_id_to_client[session_id] = self._clients[client_idx]
        return self._session_id_to_client[session_id]

if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModel.run_webserver()  # noqa: F401
