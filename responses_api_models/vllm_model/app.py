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
from uuid import uuid4

from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request
from pydantic import Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.responses_converter import (
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,  # noqa: F401
    split_responses_input_output_items,  # noqa: F401
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool

    uses_reasoning_parser: bool
    uses_interleaved_reasoning: bool = True
    replace_developer_role_with_system: bool = False

    # Whether or not the model can generate a reasoning output, and called again to produce additional reasoning output.
    sequential_reasoning_allowed: bool = True

    # As of Feb 2026, we default this to False since majority of open source models aren't responses native with the exception of GPT-OSS
    is_responses_native: bool = False

    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # Corresponds to the extra_body of OpenAI Client.
    extra_body: Optional[Dict[str, Any]] = None

    default_headers: Dict[str, str] = Field(default_factory=dict)
    # Optional prefix for resolving relative ``metadata.audio_path`` (or
    # entries in ``metadata.audio_paths``) against. Absolute paths are used
    # as-is. When unset, relative paths raise. Audio is always inlined as a
    # ``data:audio/<fmt>;base64,...`` URI at request time — keeps the JSONL
    # small without depending on vLLM's ``--allowed-local-media-path``.
    audio_root: Optional[str] = None

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


class VLLMModel(SimpleResponsesAPIModel):
    config: VLLMModelConfig

    def get_converter(self) -> "VLLMConverter":
        """Return the converter used for Responses API <-> Chat Completions mapping.

        Override in subclasses (e.g. GenRMModel) to use a specialized converter.
        """
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

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if self.config.is_responses_native:
            return await self._responses_native(request, body)

        # Response Create Params -> Chat Completion Create Params
        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        body.model = self.config.model

        # Chat Completion Create Params -> Chat Completion
        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        choice = chat_completion_response.choices[0]

        response_output = self._converter.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]

        usage = None
        if chat_completion_response.usage:
            usage = NeMoGymResponseUsage(
                input_tokens=chat_completion_response.usage.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=chat_completion_response.usage.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=chat_completion_response.usage.prompt_tokens
                + chat_completion_response.usage.completion_tokens,
            )

        incomplete_details = None
        if choice.finish_reason == "length":
            incomplete_details = {"reason": "max_output_tokens"}
        elif choice.finish_reason == "content_filter":
            incomplete_details = {"reason": "content_filter"}

        # Chat Completion -> Response
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model,
            object="response",
            output=response_output_dicts,
            tool_choice=body.tool_choice if "tool_choice" in body else "auto",
            parallel_tool_calls=body.parallel_tool_calls,
            tools=body.tools,
            temperature=body.temperature,
            top_p=body.top_p,
            background=body.background,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            previous_response_id=body.previous_response_id,
            prompt=body.prompt,
            reasoning=body.reasoning,
            service_tier=body.service_tier,
            text=body.text,
            top_logprobs=body.top_logprobs,
            truncation=body.truncation,
            metadata=body.metadata,
            instructions=body.instructions,
            user=body.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )

    async def _responses_native(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """
        The following config parameters are effectively no-ops with Responses native models:
        - uses_reasoning_parser: bool (Not applicable)
        """
        # The following parameters could be supported, but have not been supported yet for Responses-native models:
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

    # Mapping from common audio file extensions to MIME subtypes used in the
    # ``data:audio/<subtype>;base64,...`` URI. vLLM-side decoders inspect the
    # subtype to pick a backend (libsndfile, ffmpeg, …); guessing wrong would
    # silently mis-decode, so we keep the table conservative and raise on
    # unknown extensions instead of falling back to ``wav``.
    _AUDIO_EXT_TO_MIME: ClassVar[Dict[str, str]] = {
        ".wav": "wav",
        ".flac": "flac",
        ".mp3": "mpeg",
        ".m4a": "mp4",
        ".ogg": "ogg",
        ".opus": "opus",
    }

    def _resolve_audio_path_to_url(self, audio_path: str) -> str:
        """Turn an ``audio_path`` reference into a ``data:audio/...;base64`` URI.

        Reads the file and inlines it as a base64 data URI at request time
        — same strategy NeMo Skills' ``VLLMMultimodalModel.content_text_to_list``
        uses (read once per request, hand vLLM a self-contained content
        block). Keeps the on-disk JSONL small without requiring any vLLM
        server-side flag.

        Relative paths are resolved against ``config.audio_root``; without
        it, relative paths raise so the failure mode is loud rather than
        silently reading from the server CWD.
        """
        if os.path.isabs(audio_path):
            resolved = audio_path
        elif self.config.audio_root:
            resolved = os.path.join(self.config.audio_root, audio_path)
        else:
            raise ValueError(
                f"metadata.audio_path={audio_path!r} is relative but VLLMModelConfig.audio_root "
                "is unset. Set audio_root in the model config or use absolute paths."
            )

        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"metadata.audio_path resolved to {resolved!r}, which does not exist.")

        ext = os.path.splitext(resolved)[1].lower()
        mime = self._AUDIO_EXT_TO_MIME.get(ext)
        if mime is None:
            raise ValueError(
                f"Unsupported audio extension {ext!r} for {resolved!r}. Supported: {sorted(self._AUDIO_EXT_TO_MIME)}."
            )
        with open(resolved, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:audio/{mime};base64,{encoded}"

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the body dict before issuing a chat completion request.

        Subclasses can override this to apply model-specific transformations
        (e.g. role remapping, extra sampling params).  The base implementation
        handles the features driven by ``VLLMModelConfig``.

        Args:
            request: The originating FastAPI request (available for session /
                client resolution if needed by subclasses).
            body_dict: Mutable dict produced by ``body.model_dump(exclude_unset=True)``.

        Returns:
            The (possibly mutated) ``body_dict`` that will be forwarded to
            ``client.create_chat_completion``.
        """
        if self.config.replace_developer_role_with_system:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"

        body_dict["model"] = self.config.model

        chat_template_kwargs = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)

        metadata = body_dict.get("metadata", dict())

        # Merge global config chat_template_kwargs with per-request overrides in metadata (e.g. per-sample reasoning on/off)
        metadata_chat_template_kwargs_str = metadata.get("chat_template_kwargs", "{}")
        chat_template_kwargs.update(json.loads(metadata_chat_template_kwargs_str))

        if chat_template_kwargs:
            body_dict["chat_template_kwargs"] = chat_template_kwargs

        # Merge global config extra_body with per-request overrides from metadata
        extra_body = {}
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)

        metadata_extra_body_str = metadata.get("extra_body", "{}")
        extra_body.update(json.loads(metadata_extra_body_str))

        if self.config.return_token_id_information:
            body_dict |= dict(
                logprobs=True,
                # Typically passed via OpenAI client extra_body.
                return_tokens_as_token_ids=True,
                # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                # For prompt and generation token IDs
                # return_token_ids=True,
                # For prompt token IDs
                # prompt_logprobs=0,
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

                        # TODO when NeMo RL migrates to vLLM>=0.16.0, remove the reasoning_content support above.
                        # Starting with vLLM 0.16.0, the `reasoning_content` field has been deprecated in favor of just `reasoning`
                        message_dict["reasoning"] = reasoning_matches[0]
                elif isinstance(content, list):
                    reasoning_content = None
                    for content_item_dict in content:
                        reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(
                            content_item_dict["text"]
                        )
                        assert reasoning_content is None or not reasoning_matches, (
                            f"Found multiple reasoning matches in a single assistant message content item list!\nMessage: {message_dict}"
                        )

                        # Even though we set the reasoning content already here, we still loop through all the content item dicts for the assert above.
                        content_item_dict["text"] = remaining_content
                        if reasoning_matches and self.config.uses_interleaved_reasoning:
                            message_dict["reasoning_content"] = reasoning_matches[0]
                            # See the TODO wrt reasoning_content above
                            message_dict["reasoning"] = reasoning_matches[0]
                elif not content:
                    # No content or content None is a no-op
                    pass
                else:
                    raise NotImplementedError

        if extra_body:
            body_dict = extra_body | body_dict

        # Audio sidechannel: rows can carry audio on
        # ``responses_create_params.metadata`` via three mutually exclusive
        # keys, all spliced as ``audio_url`` content blocks into the most
        # recent user message before forwarding to vLLM Chat Completions:
        #
        #   * ``audio_data``  — a single pre-built ``data:audio/...;base64,``
        #                       URI inlined into the JSONL. Self-contained;
        #                       no audio root needed at request time.
        #   * ``audio_path``  — a single file path; resolved against
        #                       ``config.audio_root`` and encoded to a data
        #                       URI at request time.
        #   * ``audio_paths`` — list of file paths; each encoded and spliced
        #                       in order. Mirrors NeMo Skills' ``audios``
        #                       multi-clip schema.
        #
        # OpenAI's Responses API content union has no audio variant (audio
        # types exist as orphans in the SDK but aren't members of
        # ``ResponseInputContentParam``), so audio rows can't ride in
        # ``input.content`` directly — the metadata-sidechannel hop lets
        # audio benchmarks carry audio without a Gym schema change.
        #
        # Audio is placed BEFORE text in the content list (some audio
        # models care). No-op when none of the three keys are present, so
        # non-audio benchmarks are unaffected.
        audio_keys_present = [k for k in ("audio_data", "audio_path", "audio_paths") if metadata.get(k)]
        if len(audio_keys_present) > 1:
            raise ValueError(
                f"metadata audio keys are mutually exclusive — got {audio_keys_present}. "
                "Set exactly one of audio_data / audio_path / audio_paths per row."
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
                    # ``None`` / unexpected shape — replace with a fresh content list
                    msg["content"] = list(audio_blocks)
                break
            else:
                # No user message found — create one with just the audio blocks.
                body_dict.setdefault("messages", []).append({"role": "user", "content": list(audio_blocks)})

        return body_dict

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)

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
            """
            Example messages for out of context length:

            1. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L914
            ```json
            {"object":"error","message":"This model\'s maximum context length is 32768 tokens. However, you requested 32818 tokens in the messages, Please reduce the length of the messages. None","type":"BadRequestError","param":null,"code":400}
            ```
            2. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L940
            3. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L948
            4. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/sampling_params.py#L463
            """
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
            # See the TODO wrt reasoning_content above
            reasoning_content = choice_dict["message"].get("reasoning_content") or choice_dict["message"].get(
                "reasoning"
            )
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content", None)
                # See the TODO wrt reasoning_content above
                choice_dict["message"].pop("reasoning", None)

                # We wrap this here in think tags for Gym's sake and to return a valid OpenAI Chat Completions response.
                choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                    [reasoning_content]
                ) + (choice_dict["message"].get("content") or "")
        else:
            # See the TODO wrt reasoning_content above
            assert not (choice_dict["message"].get("reasoning_content") or choice_dict["message"].get("reasoning")), (
                f"NeMo Gym server `{self.config.name}` config has explicitly been set to not use a reasoning parser i.e. `uses_reasoning_parser: false`. Please do not use a reasoning parser in your vLLM endpoint, or fix the `{self.config.name}` server config!"
            )

        if self.config.return_token_id_information and "prompt_token_ids" not in choice_dict["message"]:
            log_probs = choice_dict["logprobs"]["content"]
            generation_log_probs = [log_prob["logprob"] for log_prob in log_probs]

            """
            START TODO remove this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            """
            # Looks like `"token_id:151667"`
            generation_token_ids = [log_prob["token"].removeprefix("token_id:") for log_prob in log_probs]

            # The tokenize endpoint doesn't accept any sampling parameters
            # The only relevant params are model, messages, and tools.
            #
            # IMPORTANT: pass through chat-template knobs (e.g. enable_thinking)
            # when tokenizing, otherwise `prompt_token_ids` (and therefore logged
            # `prompt_str`) can be built with different chat template settings than
            # the actual generation request.
            tokenize_body_dict = dict()
            for key in ("model", "messages", "tools", "chat_template_kwargs"):
                if key in body_dict:
                    tokenize_body_dict[key] = body_dict[key]

            # The base url has /v1 at the end but vLLM's tokenize endpoint does not have v1, hence the ..
            tokenize_response = await client.create_tokenize(**tokenize_body_dict)
            """
            END
            """

            message_dict = choice_dict["message"]
            message_dict.update(
                dict(
                    # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                    # prompt_token_ids=chat_completion_dict["prompt_token_ids"],
                    prompt_token_ids=tokenize_response["tokens"],
                    # generation_token_ids=choice_dict["token_ids"],
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )

            # Clean the duplicated information
            choice_dict.pop("logprobs")
            # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            # chat_completion_dict.pop("prompt_token_ids")
            # choice_dict.pop("token_ids")

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
            # There is probably a better way to select the endpoint for this request. But this will do for now.
            client_idx = len(self._session_id_to_client) % len(self._clients)
            client = self._clients[client_idx]
            self._session_id_to_client[session_id] = client
        client = self._session_id_to_client[session_id]

        return client


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModel.run_webserver()  # noqa: F401
