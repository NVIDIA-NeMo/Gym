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
import re
from copy import deepcopy
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union
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
)
from nemo_gym.responses_converter import (
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,  # noqa: F401
    split_responses_input_output_items,  # noqa: F401
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.sglang_tool_parsers import (
    normalize_tool_call_arguments,
    parse_qwen3_coder_tool_calls,
)


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

    # Generation engine. "vllm" (default) keeps the original OpenAI /v1/chat/completions
    # marshaling path unchanged. "sglang" switches to SGLang's native /generate endpoint
    # (see VLLMModel._sglang_chat_completion): on the pinned SGLang v0.5.10 the chat endpoint
    # cannot return the exact sampled integer token ids (decoded-string logprobs, no
    # return_tokens_as_token_ids) and /tokenize only accepts a raw prompt string, so the
    # proxy tokenizes locally and reads token ids from /generate's meta_info.output_token_logprobs.
    engine: Literal["vllm", "sglang"] = "vllm"

    # Path to the Jinja chat template the SGLang server was launched with (--chat-template).
    # Used (engine == "sglang") so the proxy renders prompts with the same template the
    # model was served/trained with. If None, the model tokenizer's built-in template is used.
    sglang_chat_template_path: Optional[str] = None

    # Max sequence length of the SGLang server (engine == "sglang"). When a request does not
    # specify a positive max_tokens (e.g. the SWE agent sends max_output_tokens=0 = "unlimited"),
    # the proxy fills the remaining context as max_new_tokens. Without this, SGLang /generate
    # falls back to its default max_new_tokens=128, truncating reasoning before </think> and
    # breaking multi-turn contiguity.
    sglang_max_total_sequence_length: Optional[int] = None

    # Tool-call text format the model emits (engine == "sglang"): the /generate path re-parses
    # tool calls from raw text client-side, so this must match the model's chat template.
    # "hermes": <tool_call>{json}</tool_call> (e.g. Qwen3 thinking models).
    # "qwen3_coder": <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function>
    #                </tool_call> (Qwen3-Coder-style templates).
    sglang_tool_format: Literal["hermes", "qwen3_coder"] = "hermes"

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

        # Lazily-initialised state for the SGLang engine path (see _sglang_chat_completion).
        self._sglang_tokenizer: Any = None
        self._sglang_chat_template: Optional[str] = None
        # Contiguity fix: per-session running token sequence. Each multi-turn rollout's prompt
        # is built by splicing the prior assistant turn's EXACT sampled generation_token_ids
        # (never re-tokenizing them), so nemo_gym.py's `seen == prompt[:len(seen)]` holds by
        # construction. Re-tokenizing prior turns broke this two ways: proxy parse drift
        # (dropped multi-line tool calls, mangled </think>) and BPE retokenization (identical
        # text, different token split). Keyed by SESSION_ID_KEY; cache-miss -> full tokenize.
        self._sglang_session_seq: Dict[str, Dict[str, Any]] = dict()
        self._sglang_eos_nl_ids: Optional[List[int]] = None

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

        return self._converter.chat_completion_to_response(
            responses_create_params=body, chat_completion=chat_completion_response
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
                # Pin top_logprobs=0: capture only needs the chosen token's logprob and id.
                # vLLM computes `logprobs = top_logprobs if logprobs else None`.
                # So an inbound top_logprobs=null yields no logprobs and empties the token ids.
                # Overriding it here makes capture independent of the request.
                top_logprobs=0,
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

        # Drop a null top_logprobs on the non-capture path (caller-supplied logprobs=True).
        # vLLM treats null as "no logprobs" but a missing field as its default (0), so forwarding null is never useful.
        # The capture path above already set it to 0 and is unaffected.
        if body_dict.get("top_logprobs") is None:
            body_dict.pop("top_logprobs", None)

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

        # SGLang engine path: handled entirely by _sglang_chat_completion, which renders the
        # prompt locally (keeping <think> embedded in assistant content, as the SWE chat
        # template expects) and generates via /generate. Dispatched BEFORE the vLLM-specific
        # _preprocess (which would split reasoning out of assistant content for the chat API).
        if self.config.engine == "sglang":
            return await self._sglang_chat_completion(request, body_dict)

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

    # =======================================================
    # SGLang engine path (see VLLMModelConfig.engine == "sglang")
    # =======================================================

    # Hermes tool-call format emitted by SGLang's --tool-call-parser hermes and the SWE
    # chat template: one or more <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
    # blocks. The capture is the JSON object, anchored by the closing tag (so nested braces
    # in arguments are handled without brace-balancing).
    _SGLANG_TOOL_CALL_PATTERN: ClassVar = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    _SGLANG_ARGS_PATTERN: ClassVar = re.compile(r"\"arguments\"\s*:\s*(.*)\}\s*$", re.DOTALL)
    # Turn-terminating special tokens the chat template re-emits after an assistant message.
    # We strip them from the decoded generation so they are not doubled on history re-render.
    _SGLANG_EOS_MARKERS: ClassVar = ("<|im_end|>", "<|endoftext|>")

    def _get_sglang_tokenizer(self) -> Any:
        if self._sglang_tokenizer is None:
            from transformers import AutoTokenizer

            self._sglang_tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        return self._sglang_tokenizer

    def _get_sglang_chat_template(self) -> Optional[str]:
        if self._sglang_chat_template is None and self.config.sglang_chat_template_path:
            with open(self.config.sglang_chat_template_path) as f:
                self._sglang_chat_template = f.read()
        return self._sglang_chat_template

    def _full_sglang_tokenize(
        self, messages: List[Any], tools: Any, chat_template_kwargs: Dict[str, Any]
    ) -> List[int]:
        """Tokenize the full chat prompt via the chat template (the original, non-spliced path)."""
        tokenizer = self._get_sglang_tokenizer()
        # History carries assistant tool-call arguments as JSON strings; the
        # chat template iterates them as a mapping (arguments|items) and
        # 500s on a string. Only this full-render path sees old assistant
        # turns (the splice path reuses their exact tokens).
        encoded = tokenizer.apply_chat_template(
            normalize_tool_call_arguments(messages),
            tools=tools,
            chat_template=self._get_sglang_chat_template(),
            add_generation_prompt=True,
            tokenize=True,
            **chat_template_kwargs,
        )
        # transformers v5's apply_chat_template(tokenize=True) returns a BatchEncoding
        # (dict-like), not a flat list; normalize to a JSON-serializable List[int].
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(t) for t in encoded]

    def _sglang_eos_nl(self) -> List[int]:
        if self._sglang_eos_nl_ids is None:
            enc = self._get_sglang_tokenizer()("<|im_end|>\n", add_special_tokens=False)
            self._sglang_eos_nl_ids = [int(t) for t in enc["input_ids"]]
        return self._sglang_eos_nl_ids

    def _sglang_followup_fragment_ids(
        self, new_msgs: List[Any], chat_template_kwargs: Dict[str, Any]
    ) -> Optional[List[int]]:
        """Token ids for the new (non-assistant) messages + the next generation-prompt header,
        rendered as a standalone fragment that follows a prior assistant turn. Derived by
        differencing two template renders against an anchor assistant turn, then tokenizing the
        suffix. Safe to splice onto the running sequence because the splice boundary is the
        assistant turn's ``<|im_end|>\\n`` (a special token), across which byte-level BPE does
        not merge (validated). Returns None if the template is not splice-friendly -> caller
        falls back to a full re-tokenize."""
        tokenizer = self._get_sglang_tokenizer()
        ct = self._get_sglang_chat_template()
        anchor = [{"role": "assistant", "content": "X"}]
        try:
            full = tokenizer.apply_chat_template(
                anchor + list(new_msgs),
                tools=None,
                chat_template=ct,
                add_generation_prompt=True,
                tokenize=False,
                **chat_template_kwargs,
            )
            base = tokenizer.apply_chat_template(
                anchor,
                tools=None,
                chat_template=ct,
                add_generation_prompt=False,
                tokenize=False,
                **chat_template_kwargs,
            )
        except Exception:
            return None
        if not isinstance(full, str) or not isinstance(base, str) or not full.startswith(base):
            return None
        enc = tokenizer(full[len(base) :], add_special_tokens=False)
        return [int(t) for t in enc["input_ids"]]

    @staticmethod
    def _sglang_msg_sig(m: Dict[str, Any]) -> Tuple[Any, str, str]:
        return (
            m.get("role"),
            json.dumps(m.get("content"), sort_keys=True, default=str),
            json.dumps(m.get("tool_calls"), sort_keys=True, default=str),
        )

    @classmethod
    def _sglang_messages_match(cls, a: List[Any], b: List[Any]) -> bool:
        return len(a) == len(b) and all(cls._sglang_msg_sig(x) == cls._sglang_msg_sig(y) for x, y in zip(a, b))

    def _build_sglang_prompt_ids(
        self,
        request: Request,
        messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], Optional[str]]:
        """Return (prompt_token_ids, session_id). Splices the prior assistant turn's exact
        generation tokens when this is a continuation of a cached session; else full tokenize."""
        try:
            sid = request.session.get(SESSION_ID_KEY)
        except Exception:
            sid = None
        if sid is not None:
            state = self._sglang_session_seq.get(sid)
            if state is not None:
                prev = state["messages"]
                n = len(prev)
                if (
                    len(messages) > n
                    and messages[n].get("role") == "assistant"
                    and all(m.get("role") != "assistant" for m in messages[n + 1 :])
                    and self._sglang_messages_match(messages[:n], prev)
                ):
                    frag = self._sglang_followup_fragment_ids(messages[n + 1 :], chat_template_kwargs)
                    if frag is not None:
                        return state["seq"] + frag, sid
        return self._full_sglang_tokenize(messages, tools, chat_template_kwargs), sid

    def _update_sglang_session_seq(
        self,
        sid: Optional[str],
        messages: List[Any],
        prompt_token_ids: List[int],
        generation_token_ids: List[int],
    ) -> None:
        """Cache the running sequence through this assistant turn (prompt + gen + ``<|im_end|>\\n``)
        for the next turn's splice."""
        if sid is None:
            return
        eos_nl = self._sglang_eos_nl()  # e.g. [151645, 198]
        seq = list(prompt_token_ids) + list(generation_token_ids)
        if not seq or seq[-1] != eos_nl[0]:
            seq = seq + eos_nl
        else:
            seq = seq + eos_nl[1:]  # gen already ended with <|im_end|>; just add the trailing \n
        # Bound memory: refresh this sid's insertion order, evict oldest beyond the cap.
        # Evicted sessions simply fall back to a full tokenize on their next turn (safe).
        self._sglang_session_seq.pop(sid, None)
        while len(self._sglang_session_seq) >= 8192:
            self._sglang_session_seq.pop(next(iter(self._sglang_session_seq)), None)
        self._sglang_session_seq[sid] = {"messages": list(messages), "seq": seq}

    def _parse_sglang_generation(
        self, text: str, tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
        """Reconstruct (reasoning_content, content, tool_calls) from SGLang /generate raw text.

        The thinking-style generation prompt ends with ``<think>\\n``, so the generated text
        begins INSIDE the reasoning block (no opening ``<think>``). Everything up to the first
        ``</think>`` is therefore reasoning; tool calls are parsed out of the remainder in the
        format selected by ``config.sglang_tool_format`` ("hermes" JSON, e.g. Qwen3-thinking;
        or "qwen3_coder" XML-ish, e.g. Qwen3-Coder-style templates — see sglang_tool_parsers.py).
        This mirrors what the serving engine's reasoning/tool parsers would have produced on
        /v1/chat/completions, so downstream Responses marshaling is identical to the vLLM path.
        ``tools`` (the request's tools list) enables schema-aware argument type coercion.
        """
        reasoning_content: Optional[str] = None
        if self.config.uses_reasoning_parser and "</think>" in text:
            reasoning_content, _, remainder = text.partition("</think>")
        else:
            remainder = text

        if self.config.sglang_tool_format == "qwen3_coder":
            tool_calls, content = parse_qwen3_coder_tool_calls(remainder, tools)
            return reasoning_content, content, tool_calls

        tool_calls: List[Dict[str, Any]] = []
        for match in self._SGLANG_TOOL_CALL_PATTERN.finditer(remainder):
            block = match.group(1)
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            # Preserve the model's EXACT arguments serialization (function.arguments is a JSON
            # string in the OpenAI schema). Keeping the raw substring -- rather than
            # re-serializing the parsed dict -- means the chat template re-renders the assistant
            # turn byte-identically, which the nemo_gym.py contiguity assert depends on.
            args_match = self._SGLANG_ARGS_PATTERN.search(block)
            if args_match is not None:
                arguments = args_match.group(1).strip()
            else:
                arguments = json.dumps(parsed.get("arguments", {}))
            tool_calls.append(
                dict(
                    id=f"call_{uuid4().hex}",
                    type="function",
                    function=dict(name=parsed.get("name"), arguments=arguments),
                )
            )

        content = self._SGLANG_TOOL_CALL_PATTERN.sub("", remainder).strip()
        return reasoning_content, content, tool_calls

    def _sglang_length_finish(self, prompt_token_ids: List[int]) -> NeMoGymChatCompletion:
        """Graceful over-context terminal turn (mirrors the vLLM context-length path in
        chat_completions): SGLang /generate hard-rejects input >= context_length, which 500s
        and corrupts the multi-turn trajectory (-> nemo_gym.py contiguity assert -> stalled
        step). Instead, end the turn cleanly with finish_reason="length"."""
        msg: Dict[str, Any] = dict(role="assistant", content=None, tool_calls=None)
        if self.config.return_token_id_information:
            msg.update(dict(prompt_token_ids=list(prompt_token_ids), generation_token_ids=[], generation_log_probs=[]))
        return NeMoGymChatCompletion.model_validate(
            dict(
                id=f"chtcmpl-{uuid4().hex}",
                object="chat.completion",
                created=int(time()),
                model=self.config.model,
                choices=[dict(index=0, finish_reason="length", message=msg, logprobs=None)],
                usage=dict(
                    prompt_tokens=len(prompt_token_ids), completion_tokens=0, total_tokens=len(prompt_token_ids)
                ),
            )
        )

    async def _sglang_chat_completion(self, request: Request, body_dict: Dict[str, Any]) -> NeMoGymChatCompletion:
        """SGLang v0.5.10 generation path (see VLLMModelConfig.engine).

        Tokenizes the chat-templated prompt locally and generates via SGLang's native
        /generate (return_logprob=True) -- the only v0.5.10 source of the exact sampled
        integer token ids AND their logprobs (needed for token-level RL). The decoded text
        is re-parsed into reasoning + hermes tool_calls so the returned object is shaped
        exactly like the vLLM /v1/chat/completions response, keeping every downstream
        Responses-API conversion identical.
        """
        client = self._resolve_client(request)

        messages = body_dict["messages"]
        if self.config.replace_developer_role_with_system:
            for message_dict in messages:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"
        tools = body_dict.get("tools")

        # Merge config chat_template_kwargs with per-request metadata overrides (mirrors
        # _preprocess_chat_completion_create_params so reasoning toggles behave identically).
        chat_template_kwargs: Dict[str, Any] = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)
        metadata = body_dict.get("metadata", dict())
        chat_template_kwargs.update(json.loads(metadata.get("chat_template_kwargs", "{}")))

        tokenizer = self._get_sglang_tokenizer()  # used below to decode generation_token_ids
        # Build prompt token ids with contiguity-preserving splicing across turns (falls back
        # to a full chat-template tokenize on the first turn / cache miss / history condensation).
        prompt_token_ids, _splice_sid = self._build_sglang_prompt_ids(request, messages, tools, chat_template_kwargs)

        # Map the OpenAI sampling knobs onto SGLang /generate sampling_params.
        # spaces_between_special_tokens=False mirrors the NeMo-RL SGLang backend and keeps
        # special tokens (</think>, <tool_call>) tight in the decoded text we parse below.
        sampling_params: Dict[str, Any] = {"spaces_between_special_tokens": False}
        # max_tokens of None OR 0 means "unlimited" (the SWE agent sends max_output_tokens=0).
        # SGLang /generate would otherwise fall back to max_new_tokens=128 and truncate reasoning
        # before </think>; fill the remaining context instead (matches vLLM + the recipe).
        max_new_tokens = body_dict.get("max_tokens") or None
        if max_new_tokens is None and self.config.sglang_max_total_sequence_length:
            # Fill the remaining context, leaving a small margin: SGLang /generate rejects a
            # request whose input + max_new_tokens >= context_length (it requires strictly less,
            # unlike vLLM which allows ==). Reserve a few tokens to stay safely under the limit.
            max_new_tokens = self.config.sglang_max_total_sequence_length - len(prompt_token_ids) - 8
            max_new_tokens = max(1, max_new_tokens)
        if max_new_tokens:
            sampling_params["max_new_tokens"] = max_new_tokens
        for key in ("temperature", "top_p", "top_k", "stop"):
            if body_dict.get(key) is not None:
                sampling_params[key] = body_dict[key]

        # Over-context guard: SGLang /generate hard-rejects input >= context_length. When the
        # spliced multi-turn prompt already fills the window, end cleanly (finish_reason=length)
        # rather than 500 -> trajectory corruption -> contiguity assert -> stalled step.
        _ctx_cap = self.config.sglang_max_total_sequence_length
        if _ctx_cap is not None and len(prompt_token_ids) >= _ctx_cap:
            return self._sglang_length_finish(prompt_token_ids)
        try:
            gen = await client.create_generate(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                return_logprob=True,
            )
        except ClientResponseError as _e:
            _body = ""
            try:
                _body = _e.response_content.decode()
            except Exception:
                _body = str(_e)
            if any(s in _body for s in ("context length", "longer than", "max_total", "is longer")):
                return self._sglang_length_finish(prompt_token_ids)
            raise

        meta_info = gen.get("meta_info") or {}
        # Each tuple is (logprob, token_id, ...). Sourcing both ids and logprobs from the SAME
        # list guarantees they are 1:1 aligned in count and order -- mirrors
        # nemo_rl/models/generation/sglang/sglang_generation.py:generate_one_sample.
        output_token_logprobs = meta_info.get("output_token_logprobs") or []
        generation_token_ids = [item[1] for item in output_token_logprobs]
        generation_log_probs = [item[0] for item in output_token_logprobs]

        # Contiguity fix: cache the running token sequence through this assistant turn so the
        # next turn splices these EXACT generation_token_ids instead of re-tokenizing them.
        self._update_sglang_session_seq(_splice_sid, messages, prompt_token_ids, generation_token_ids)

        # Decode the EXACT sampled ids with skip_special_tokens=False so reasoning markers
        # (</think> is a SPECIAL token, id 151668) survive. SGLang's gen["text"] decodes with
        # skip_special_tokens=True by default, which STRIPS </think> -> the reasoning never gets
        # wrapped -> the <think> wrapper is dropped on history re-render -> the nemo_gym.py
        # contiguity assert fires on every multi-turn rollout. spaces_between_special_tokens=False
        # keeps the markers tight. Then strip the trailing EOS the chat template re-adds.
        generated_text = tokenizer.decode(
            generation_token_ids, skip_special_tokens=False, spaces_between_special_tokens=False
        )
        _stripped = True
        while _stripped:
            _stripped = False
            generated_text = generated_text.rstrip("\n")
            for _eos in self._SGLANG_EOS_MARKERS:
                if generated_text.endswith(_eos):
                    generated_text = generated_text[: -len(_eos)]
                    _stripped = True
        reasoning_content, content, tool_calls = self._parse_sglang_generation(generated_text, tools=tools)

        if (meta_info.get("finish_reason") or {}).get("type") == "length":
            finish_reason = "length"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        # Re-embed reasoning into <think> tags and prepend to content, identical to the vLLM
        # reasoning-parser branch, so postprocess_assistant_message_dict re-extracts it.
        if self.config.uses_reasoning_parser and reasoning_content:
            content = self._converter._wrap_reasoning_in_think_tags([reasoning_content]) + (content or "")

        message_dict: Dict[str, Any] = dict(
            role="assistant",
            content=content or None,
            tool_calls=tool_calls or None,
        )
        if self.config.return_token_id_information:
            message_dict.update(
                dict(
                    prompt_token_ids=prompt_token_ids,
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )

        chat_completion_dict = dict(
            id=f"chtcmpl-{uuid4().hex}",
            object="chat.completion",
            created=int(time()),
            model=self.config.model,
            choices=[dict(index=0, finish_reason=finish_reason, message=message_dict, logprobs=None)],
            usage=dict(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(generation_token_ids),
                total_tokens=len(prompt_token_ids) + len(generation_token_ids),
            ),
        )
        return NeMoGymChatCompletion.model_validate(chat_completion_dict)


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModel.run_webserver()  # noqa: F401
