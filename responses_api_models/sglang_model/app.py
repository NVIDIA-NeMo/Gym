# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym model server backed by SGLang's native ``/generate`` endpoint."""

import json
import re
from dataclasses import dataclass
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from aiohttp.client_exceptions import ClientConnectionError, ClientResponseError
from fastapi import HTTPException, Request
from jinja2 import TemplateError
from pydantic import Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import (
    TOKEN_METADATA_FIELDS,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models import Body
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.sglang_model._logic import extract_generated_tokens_and_logprobs
from responses_api_models.sglang_model.tool_parsers import (
    normalize_tool_call_arguments,
    parse_qwen3_coder_tool_calls,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig, merge_chat_extra_body


class SGLangModelConfig(VLLMModelConfig):
    """Configuration for exact-token SGLang generation."""

    context_length: int = Field(gt=0)
    trust_remote_code: bool = False
    sglang_chat_template: Optional[str] = None
    sglang_chat_template_path: Optional[str] = None
    sglang_tool_format: Literal["hermes", "qwen3_coder"] = "hermes"


@dataclass
class _SampledTurn:
    messages: List[Dict[str, Any]]
    prompt: List[int]
    sampled: List[int]
    rendering: Tuple[str, str, Optional[str]]


class SGLangModel(VLLMModel):
    """Responses-API adapter that preserves exact sampled token IDs."""

    config: SGLangModelConfig

    _SGLANG_TOOL_CALL_PATTERN: ClassVar = re.compile(
        r"<tool_call>((?:(?!</?tool_call>).)*)</tool_call>",
        re.DOTALL,
    )
    _SESSION_CAPACITY: ClassVar = 8192
    _SGLANG_EOS_MARKERS: ClassVar = ("<|im_end|>", "<|endoftext|>")

    def _post_init(self) -> None:
        super()._post_init()
        self._sglang_tokenizer: Any = None
        self._sglang_chat_template: Optional[str] = None
        self._sglang_session_seq: Dict[str, _SampledTurn] = {}

    def _get_sglang_tokenizer(self) -> Any:
        if self._sglang_tokenizer is None:
            from transformers import AutoTokenizer

            self._sglang_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._sglang_tokenizer

    def _get_sglang_chat_template(self) -> Optional[str]:
        if self.config.sglang_chat_template is not None:
            return self.config.sglang_chat_template
        if self._sglang_chat_template is None and self.config.sglang_chat_template_path:
            with open(self.config.sglang_chat_template_path) as template_file:
                self._sglang_chat_template = template_file.read()
        return self._sglang_chat_template

    async def chat_completions(
        self,
        request: Request,
        body: NeMoGymChatCompletionCreateParamsNonStreaming = Body(),
    ) -> NeMoGymChatCompletion:
        """Generate without applying the vLLM-specific request preprocessing."""
        return await self._sglang_chat_completion(
            request,
            body.model_dump(exclude_unset=True),
        )

    def _responses_to_chat_request(
        self, body: NeMoGymResponseCreateParamsNonStreaming, request: Request | None = None
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        # The shared converter omits tool controls when the request has no tools.
        if not body.tools and (body.tool_choice != "auto" or not body.parallel_tool_calls):
            raise HTTPException(422, "SGLang supports only automatic tool choice and parallel tool calls.")
        return super()._responses_to_chat_request(body, request)

    def _full_sglang_tokenize(
        self,
        messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Render and tokenize a complete prompt on a cache miss."""
        encoded = self._get_sglang_tokenizer().apply_chat_template(
            normalize_tool_call_arguments(messages),
            tools=tools,
            chat_template=self._get_sglang_chat_template(),
            add_generation_prompt=True,
            tokenize=True,
            **chat_template_kwargs,
        )
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    def _sglang_followup_ids(
        self,
        state: _SampledTurn,
        new_messages: List[Dict[str, Any]],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Render only the delimiter and new input around the exact sampled IDs."""
        tokenizer = self._get_sglang_tokenizer()
        marker = f"gym_{uuid4().hex}"
        anchor = [*state.messages[:-1], {"role": "assistant", "content": marker}]

        def tail(messages: List[Dict[str, Any]], *, generation_prompt: bool) -> str:
            try:
                rendered = tokenizer.apply_chat_template(
                    normalize_tool_call_arguments(messages),
                    tools=tools,
                    chat_template=self._get_sglang_chat_template(),
                    add_generation_prompt=generation_prompt,
                    tokenize=False,
                    **chat_template_kwargs,
                )
            except (TemplateError, ValueError) as exc:
                raise HTTPException(409, "The chat template cannot render this continuation.") from exc
            if rendered.count(marker) != 1:
                raise HTTPException(409, "The chat template cannot identify the sampled assistant boundary.")
            return rendered.partition(marker)[2]

        boundary = tail(anchor, generation_prompt=False)
        base = boundary
        if state.messages[-1].get("tool_calls"):
            anchor[-1]["tool_calls"] = state.messages[-1]["tool_calls"]
            base = tail(anchor, generation_prompt=False)
            if not base.endswith(boundary):
                raise HTTPException(409, "The chat template has a different boundary for tool calls.")
        full = tail(anchor + new_messages, generation_prompt=True)
        if not full.startswith(base):
            raise HTTPException(409, "The chat template rewrites the sampled assistant boundary.")
        closing = tokenizer(boundary, add_special_tokens=False)["input_ids"]
        fragment = full[len(base) :]
        continuation = tokenizer(boundary + fragment, add_special_tokens=False)["input_ids"]
        overlap = min(len(closing), len(state.sampled))
        while overlap and state.sampled[-overlap:] != closing[:overlap]:
            overlap -= 1
        if continuation[:overlap] == closing[:overlap]:
            continuation = continuation[overlap:]
        else:
            # The sampled boundary prevents a tokenizer merge with the first new message.
            continuation = closing[overlap:] + tokenizer(fragment, add_special_tokens=False)["input_ids"]
        return state.prompt + state.sampled + continuation

    @staticmethod
    def _template_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip training metadata and normalize absent optional message fields for rendering and replay."""
        return [
            {key: value for key, value in message.items() if key not in TOKEN_METADATA_FIELDS and value is not None}
            for message in messages
        ]

    def _sglang_rendering_sig(
        self,
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[str, str, Optional[str]]:
        """Identify inputs that affect the cached prompt rendering."""
        return (
            json.dumps(tools, sort_keys=True, default=str),
            json.dumps(chat_template_kwargs, sort_keys=True, default=str),
            self._get_sglang_chat_template(),
        )

    def _build_sglang_prompt_ids(
        self,
        request: Request,
        messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], str]:
        """Build a first prompt or continue the last sampled turn in this session."""
        session_id = request.session[SESSION_ID_KEY]
        visible = self._template_messages(messages)
        state = self._sglang_session_seq.get(session_id)
        if state is None:
            if request.session.get(self._session_marker) == session_id or any(
                msg.get("generation_token_ids") is not None for msg in messages
            ):
                raise HTTPException(409, "Exact sampled session state is unavailable. Start a new session.")
            return self._full_sglang_tokenize(visible, tools, chat_template_kwargs), session_id
        if state.rendering != self._sglang_rendering_sig(tools, chat_template_kwargs):
            raise HTTPException(409, "SGLang session tools or chat-template inputs changed. Start a new session.")
        prefix_length = len(state.messages)
        if visible[:prefix_length] != state.messages or any(
            msg["role"] == "assistant" for msg in visible[prefix_length:]
        ):
            raise HTTPException(409, "History differs from the sampled session. Start a new session for a branch.")
        return self._sglang_followup_ids(state, visible[prefix_length:], tools, chat_template_kwargs), session_id

    @property
    def _session_marker(self) -> str:
        return f"sglang_sampled_{self.config.name}"

    def _parse_sglang_generation(
        self,
        text: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
        """Reconstruct reasoning, visible content, and tool calls from raw text."""
        reasoning_content: Optional[str] = None
        if self.config.uses_reasoning_parser and "</think>" in text:
            reasoning_content, _, remainder = text.partition("</think>")
            reasoning_content = reasoning_content.removeprefix("<think>")
        else:
            remainder = text

        if self.config.sglang_tool_format == "qwen3_coder":
            tool_calls, content = parse_qwen3_coder_tool_calls(remainder, tools)
            return reasoning_content, content, tool_calls

        tool_calls: List[Dict[str, Any]] = []

        def extract_call(match: re.Match[str]) -> str:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                return match.group()
            if not isinstance(parsed, dict):
                return match.group()
            name = parsed.get("name")
            arguments = parsed.get("arguments", {})
            if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
                return match.group()
            tool_calls.append(
                {
                    "id": f"call_{uuid4().hex}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            )
            return ""

        content = self._SGLANG_TOOL_CALL_PATTERN.sub(extract_call, remainder).strip()
        return reasoning_content, content, tool_calls

    def _sglang_length_finish(
        self,
        prompt_token_ids: List[int],
    ) -> NeMoGymChatCompletion:
        """Terminate an over-context turn without truncating its prompt."""
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": None,
        }
        if self.config.return_token_id_information:
            message.update(
                {
                    "prompt_token_ids": list(prompt_token_ids),
                    "generation_token_ids": [],
                    "generation_log_probs": [],
                }
            )
        return NeMoGymChatCompletion.model_validate(
            {
                "id": f"chtcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time()),
                "model": self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": message,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(prompt_token_ids),
                },
            }
        )

    async def _sglang_chat_completion(
        self,
        request: Request,
        body_dict: Dict[str, Any],
    ) -> NeMoGymChatCompletion:
        """Generate exact training tokens through SGLang's native endpoint."""
        metadata = body_dict.get("metadata") or {}

        def metadata_options(key: str) -> dict:
            try:
                value = json.loads(metadata.get(key) or "{}")
            except json.JSONDecodeError as exc:
                raise HTTPException(422, f"metadata.{key} must encode a JSON object.") from exc
            if not isinstance(value, dict):
                raise HTTPException(422, f"metadata.{key} must encode a JSON object.")
            return value

        extra_body = metadata_options("extra_body")
        defaults = (self.config.extra_body or {}) | extra_body
        body_dict = merge_chat_extra_body(body_dict, extra_body=defaults)
        self._apply_sampling_overrides(body_dict, api="chat")
        supported = {
            "messages",
            "model",
            "metadata",
            "tools",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "user",
        }
        default_controls = {
            "n": 1,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "response_format": {"type": "text"},
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "logit_bias": {},
            "logprobs": False,
            "top_logprobs": 0,
            "stream": False,
            "store": False,
            "modalities": ["text"],
            "service_tier": "auto",
        }
        unsupported = sorted(
            key
            for key, value in body_dict.items()
            if value is not None and key not in supported and value != default_controls.get(key)
        )
        if unsupported:
            raise HTTPException(422, f"Unsupported SGLang request control(s): {', '.join(unsupported)}.")

        messages = body_dict["messages"]
        if self.config.replace_developer_role_with_system:
            for message in messages:
                if message.get("role") == "developer":
                    message["role"] = "system"
        tools = body_dict.get("tools")

        chat_template_kwargs = (self.config.chat_template_kwargs or {}) | metadata_options("chat_template_kwargs")

        tokenizer = self._get_sglang_tokenizer()
        prompt_token_ids, session_id = self._build_sglang_prompt_ids(
            request,
            messages,
            tools,
            chat_template_kwargs,
        )

        remaining_context = self.config.context_length - len(prompt_token_ids)
        if remaining_context <= 0:
            return self._sglang_length_finish(prompt_token_ids)

        sampling_params: Dict[str, Any] = {"spaces_between_special_tokens": False}
        max_new_tokens = body_dict.get("max_completion_tokens")
        if max_new_tokens is None:
            max_new_tokens = body_dict.get("max_tokens")
        if max_new_tokens is None:
            max_new_tokens = remaining_context - 8
            max_new_tokens = max(1, max_new_tokens)
        elif not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise HTTPException(422, "SGLang requires a positive output token limit.")
        else:
            max_new_tokens = min(max_new_tokens, remaining_context)
        sampling_params["max_new_tokens"] = max_new_tokens
        for key in ("temperature", "top_p", "top_k", "stop"):
            if body_dict.get(key) is not None:
                sampling_params[key] = body_dict[key]

        previous_turn = self._sglang_session_seq.get(session_id)
        client = self._resolve_client(request)
        try:
            result = await client.create_generate(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                return_logprob=True,
            )
        except ClientResponseError as error:
            error_body = getattr(error, "response_content", b"").decode(errors="replace") or error.message
            if any(fragment in error_body for fragment in ("context length", "longer than", "max_total", "is longer")):
                self._note_endpoint_success(client)
                return self._sglang_length_finish(prompt_token_ids)
            if error.status >= 500:
                self._note_endpoint_failure(client)
            else:
                self._note_endpoint_success(client)
            raise
        except ClientConnectionError:
            self._note_endpoint_failure(client)
            raise
        self._note_endpoint_success(client)

        meta_info = result.get("meta_info") or {}
        generation_token_ids, generation_log_probs = extract_generated_tokens_and_logprobs(
            result,
        )
        generated_text = tokenizer.decode(
            generation_token_ids,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        stripped = True
        while stripped:
            stripped = False
            generated_text = generated_text.rstrip("\n")
            for eos_marker in (getattr(tokenizer, "eos_token", None), *self._SGLANG_EOS_MARKERS):
                if eos_marker and generated_text.endswith(eos_marker):
                    generated_text = generated_text[: -len(eos_marker)]
                    stripped = True
        reasoning_content, content, tool_calls = self._parse_sglang_generation(
            generated_text,
            tools=tools,
        )

        finish = meta_info.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        if finish == "length":
            finish_reason = "length"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        if self.config.uses_reasoning_parser and reasoning_content:
            content = self._converter._wrap_reasoning_in_think_tags([reasoning_content]) + (content or "")

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls or None,
        }
        if self.config.return_token_id_information:
            message.update(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "generation_token_ids": generation_token_ids,
                    "generation_log_probs": generation_log_probs,
                }
            )

        response = NeMoGymChatCompletion.model_validate(
            {
                "id": f"chtcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time()),
                "model": self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": message,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(generation_token_ids),
                    "total_tokens": len(prompt_token_ids) + len(generation_token_ids),
                },
            }
        )
        if self._sglang_session_seq.get(session_id) is not previous_turn:
            raise HTTPException(409, "The sampled session changed during generation. Replay its accepted turn.")
        self._sglang_session_seq.pop(session_id, None)
        if len(self._sglang_session_seq) >= self._SESSION_CAPACITY:
            del self._sglang_session_seq[next(iter(self._sglang_session_seq))]
        self._sglang_session_seq[session_id] = _SampledTurn(
            self._template_messages([*messages, message]),
            prompt_token_ids,
            generation_token_ids,
            self._sglang_rendering_sig(tools, chat_template_kwargs),
        )
        request.session[self._session_marker] = session_id
        return response


if __name__ == "__main__":
    SGLangModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SGLangModel.run_webserver()  # noqa: F401
