# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ChatCompletionsClient subclass with dynamic max_completion_tokens + graceful length-finish handling.

Stirrup's ``ChatCompletionsClient.generate()`` has two behaviours
that break on long-context models served by vLLM:

1. It sends ``max_completion_tokens = self._max_tokens`` with every call —
   a static value that does not account for the input size.  When the
   prompt consumes a non-trivial fraction of the model's context window,
   the server can return ``finish_reason=length`` with zero output tokens.

2. On any ``finish_reason in ("max_tokens", "length")`` it raises
   ``ContextOverflowError`` unconditionally, even when the response has
   valid partial content.  For reasoning models whose traces can be
   genuinely long, this turns a normal "ran out of output budget" event
   into a fatal error.

This subclass addresses both.  Before each call we tokenize the messages
and size ``max_completion_tokens`` as::

    context_window − tokenized(messages) − completion_token_buffer

clamped to a minimum of ``_MIN_COMPLETION_TOKENS``.  On the response
side, we replicate Stirrup parsing but do *not* raise on
``finish_reason=length`` — the agent loop will either terminate when the
model invokes the ``finish`` tool or exhaust ``max_turns``, yielding a
clean timeout instead of a crash.

``model_id`` selects the HuggingFace tokenizer (or local checkpoint path).
When unset, a conservative character-count fallback is used.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Optional

import stirrup.core.agent as _stirrup_agent_mod
from pydantic import ValidationError as _PydanticValidationError
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.utils import to_openai_tools
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    Reasoning,
    TokenUsage,
    Tool,
    ToolCall,
)

from nemo_gym.openai_utils import NeMoGymChatCompletionMessageParam
from responses_api_agents.stirrup_agent.stirrup_utils import to_provider_openai_messages


LOGGER = logging.getLogger(__name__)


# Monkey-patch stirrup.core.agent.Agent.run_tool to surface pydantic
# ValidationError detail into the ToolResult content. Upstream stirrup
# returns the bare string "Tool arguments are not valid", hiding the
# pydantic error detail (e.g., "paths: Input should be a valid list,
# input_type=str"). Without that detail the agent has no signal to
# self-correct and just retries the same broken shape forever.
#
# Observed on DSv4-Pro GDPVal r5/r7: the model emitted `paths` as a JSON
# string literal ("[]") instead of a JSON array ([]). All ~660 finish
# attempts in r5 failed with the same bare-string error; the agent
# never learned what was wrong.
def _install_tool_arg_error_surfacing() -> None:
    _orig_run_tool = _stirrup_agent_mod.Agent.run_tool
    if getattr(_orig_run_tool, "_gym_surfacing_patched", False):
        return

    async def run_tool_with_error_surfacing(self, tool_call, run_metadata):
        result_msg = await _orig_run_tool(self, tool_call, run_metadata)
        if (not getattr(result_msg, "args_was_valid", True)) and result_msg.content == "Tool arguments are not valid":
            # Mirror upstream stirrup's lookup: self._active_tools is the {name: Tool}
            # dict built from self._tools filtered by isinstance(Tool). Looking up
            # via the dict avoids iterating self._tools which mixes Tool instances
            # with provider objects (e.g. ApptainerCodeExecToolProvider) that don't
            # have a .name attribute and crash a naive `t.name` lookup.
            tool = self._active_tools.get(tool_call.name)
            if tool is not None:
                args = tool_call.arguments if tool_call.arguments and tool_call.arguments.strip() else "{}"
                try:
                    tool.parameters.model_validate_json(args)
                except _PydanticValidationError as exc:
                    errors_str = "; ".join(
                        f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']} (type={e.get('type', '?')})"
                        for e in exc.errors()
                    )
                    args_preview = (tool_call.arguments or "")[:500]
                    detailed = (
                        f"Tool arguments are not valid: {errors_str}. "
                        f"Submitted arguments (first 500 chars): {args_preview!r}"
                    )
                    result_msg = result_msg.model_copy(update={"content": detailed})
                except Exception:
                    pass
        return result_msg

    run_tool_with_error_surfacing._gym_surfacing_patched = True
    _stirrup_agent_mod.Agent.run_tool = run_tool_with_error_surfacing


_install_tool_arg_error_surfacing()


# Replace stirrup's SIMPLE_FINISH_TOOL with a coercing variant whose
# FinishParams accepts `paths` as a JSON-encoded string and normalizes to
# list[str]. vLLM 0.20.0's --tool-call-parser deepseek_v4 forwards DSv4's
# string="false" args as JSON strings (the unwrap landed upstream in vLLM
# PR #41801, merged 2026-05-06, but the wedu image predates the merge).
# See responses_api_agents/stirrup_agent/finish_tool_coercing.py for the
# coerced schema. The override happens at module-import time so any Agent
# constructed after this point picks up the coercing variant via the
# default-arg path in stirrup.core.agent.Agent.__init__.
def _install_coercing_finish_tool() -> None:
    import stirrup.tools as _tools_mod
    import stirrup.tools.finish as _finish_mod

    if getattr(_finish_mod.SIMPLE_FINISH_TOOL, "_gym_coercing_finish_patched", False):
        return

    from responses_api_agents.stirrup_agent.finish_tool_coercing import (
        COERCING_FINISH_TOOL,
    )

    # Tag for idempotency.
    setattr(COERCING_FINISH_TOOL, "_gym_coercing_finish_patched", True)

    # Patch the canonical binding plus every place stirrup or its callers
    # captured a reference via `from ... import SIMPLE_FINISH_TOOL`.
    _finish_mod.SIMPLE_FINISH_TOOL = COERCING_FINISH_TOOL
    if hasattr(_tools_mod, "SIMPLE_FINISH_TOOL"):
        _tools_mod.SIMPLE_FINISH_TOOL = COERCING_FINISH_TOOL
    if hasattr(_stirrup_agent_mod, "SIMPLE_FINISH_TOOL"):
        _stirrup_agent_mod.SIMPLE_FINISH_TOOL = COERCING_FINISH_TOOL


_install_coercing_finish_tool()

# Floor for per-call max_completion_tokens.  Below this the model basically
# cannot produce a useful answer — treat as a hard minimum.
_MIN_COMPLETION_TOKENS = 1024

# Hard cap on per-call max_completion_tokens.  Oversized completion budgets
# on long-context servers can degrade output quality for reasoning models.
_DEFAULT_MAX_COMPLETION_TOKENS_CAP = 64000


def _load_tokenizer(model_id: Optional[str]):
    """Load a HuggingFace tokenizer, tolerating version differences in transformers."""
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        LOGGER.warning(
            "transformers is not installed; dynamic max_tokens sizing will use "
            "a character-count fallback.  `pip install transformers` to enable."
        )
        return None
    # Some tokenizers (Mistral family) expose a ``fix_mistral_regex`` kwarg.
    # Try the richer call first, fall back to the common signature.
    for kwargs in (
        {"use_fast": True, "trust_remote_code": True, "fix_mistral_regex": True},
        {"use_fast": True, "trust_remote_code": True},
    ):
        try:
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except TypeError:
            continue
        except Exception as exc:
            LOGGER.warning(f"Failed to load tokenizer for {model_id!r}: {exc}")
            return None
    return None


class DynamicMaxTokensChatCompletionsClient(ChatCompletionsClient):
    """ChatCompletionsClient that sizes max_completion_tokens per call and
    does not raise on a length-finish response."""

    def __init__(
        self,
        *args: Any,
        model_id: Optional[str] = None,
        completion_token_buffer: int = 1000,
        temperature: float = 1.0,
        top_p: float = 0.95,
        enable_thinking: bool = True,
        max_completion_tokens_cap: int = _DEFAULT_MAX_COMPLETION_TOKENS_CAP,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._completion_token_buffer = completion_token_buffer
        self._temperature = temperature
        self._top_p = top_p
        self._enable_thinking = enable_thinking
        self._max_completion_tokens_cap = max_completion_tokens_cap
        self._tokenizer = _load_tokenizer(model_id)
        if model_id and self._tokenizer is None:
            LOGGER.warning(
                f"model_id={model_id!r} provided but tokenizer could not be loaded. "
                "Dynamic max_tokens will use a character-count fallback."
            )

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def _count_input_tokens(
        self,
        messages: list[NeMoGymChatCompletionMessageParam],
        tools: Optional[dict[str, Tool]] = None,
    ) -> int:
        """Estimate the full prompt token count the server will see.

        ``messages`` must already be serialized for the provider. This keeps
        token accounting aligned with the exact payload sent on the wire,
        including assistant ``tool_calls``, multimodal content blocks, and
        tool-schema injection.

        Counting strategy (in order, best -> worst):

        1. ``tokenizer.apply_chat_template(messages, tools=…)`` — ideal,
           but some chat templates don't support the ``tools`` kwarg.
        2. ``tokenizer.apply_chat_template(messages)`` + tokenise the tool
           JSON blob separately — still captures assistant ``tool_calls``
           via the chat template.
        3. Tokenise the JSON of the serialized messages and tools blob —
           rough but serialises everything.
        4. Character-count fallback when no tokenizer is present.

        Any residual gap is absorbed by ``completion_token_buffer``.
        """
        import json as _json

        if self._tokenizer is None:
            # Pure character-count fallback.
            total = sum(len(str(m.get("content") or "")) for m in messages) // 3
            if tools:
                try:
                    total += len(_json.dumps(to_openai_tools(tools))) // 3
                except Exception:
                    pass
            return total

        oai_tools = None
        if tools:
            try:
                oai_tools = to_openai_tools(tools)
            except Exception as exc:
                LOGGER.warning(f"to_openai_tools failed ({exc}).")

        # Strategy 1: apply_chat_template with tools=
        if oai_tools is not None:
            try:
                text = self._tokenizer.apply_chat_template(
                    messages, tools=oai_tools, tokenize=False, add_generation_prompt=True
                )
                return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception as exc:
                LOGGER.debug(f"apply_chat_template(tools=) unsupported ({exc}); trying separate tool count.")

        # Strategy 2: apply_chat_template on messages only + separate tool JSON count
        try:
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            total = len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            if oai_tools is not None:
                total += len(self._tokenizer(_json.dumps(oai_tools), add_special_tokens=False)["input_ids"])
            return total
        except Exception as exc:
            LOGGER.warning(f"apply_chat_template(messages) failed ({exc}); falling back to JSON count.")

        # Strategy 3: tokenise the full JSON payload
        try:
            blob = _json.dumps(messages)
            total = len(self._tokenizer(blob, add_special_tokens=False)["input_ids"])
            if oai_tools is not None:
                total += len(self._tokenizer(_json.dumps(oai_tools), add_special_tokens=False)["input_ids"])
            return total
        except Exception as exc:
            LOGGER.warning(f"JSON tokenisation failed ({exc}); falling back to character count.")

        # Strategy 4: character count
        total = sum(len(str(m.get("content") or "")) for m in messages) // 3
        return total

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: dict[str, Tool],
    ) -> AssistantMessage:
        provider_messages = to_provider_openai_messages(messages)
        input_tokens = self._count_input_tokens(provider_messages, tools)
        context_window = self._max_tokens
        dynamic_max = max(
            context_window - input_tokens - self._completion_token_buffer,
            _MIN_COMPLETION_TOKENS,
        )
        capped_max = min(dynamic_max, self._max_completion_tokens_cap)

        # ``self._kwargs`` is spread last so explicit per-request kwargs override
        # the agent-level defaults.
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": provider_messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_completion_tokens": capped_max,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self._enable_thinking}},
            **self._kwargs,
        }
        if tools:
            request_kwargs["tools"] = to_openai_tools(tools)
            request_kwargs["tool_choice"] = "auto"
        if self._reasoning_effort:
            request_kwargs["reasoning_effort"] = self._reasoning_effort

        if LOGGER.isEnabledFor(logging.DEBUG):
            _msgs = request_kwargs["messages"]
            _tools = request_kwargs.get("tools") or []
            LOGGER.debug(
                "request: n_messages=%d first_role=%s last_role=%s "
                "msg_content_chars=%d n_tools=%d model=%r max_completion_tokens=%d",
                len(_msgs),
                _msgs[0].get("role") if _msgs else "?",
                _msgs[-1].get("role") if _msgs else "?",
                sum(len(str(m.get("content") or "")) for m in _msgs),
                len(_tools),
                request_kwargs.get("model"),
                request_kwargs.get("max_completion_tokens"),
            )

        request_start_time = perf_counter()
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            LOGGER.error("API call raised %s: %s", type(exc).__name__, exc)
            raise
        request_end_time = perf_counter()

        choice = response.choices[0]
        msg = choice.message
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        answer_tokens = completion_tokens - reasoning_tokens

        LOGGER.debug(
            "response: input_est=%d ctx=%d buf=%d -> max_completion=%d (capped=%d) | "
            "actual prompt=%d completion=%d (reasoning=%d) finish=%s "
            "content_len=%d tool_calls=%d",
            input_tokens,
            context_window,
            self._completion_token_buffer,
            dynamic_max,
            capped_max,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            choice.finish_reason,
            len(msg.content or ""),
            len(msg.tool_calls or []),
        )

        # Upstream raises ContextOverflowError on length/max_tokens; we don't.
        # The agent loop handles termination either via the finish tool or max_turns.

        reasoning: Optional[Reasoning] = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning = Reasoning(content=msg.reasoning_content)
        elif hasattr(msg, "reasoning") and msg.reasoning:
            # vLLM >= 0.16.0 emits `reasoning` (Responses-API convention) instead of
            # `reasoning_content`; e.g. DeepSeek-V4's `--reasoning-parser deepseek_v4`.
            reasoning = Reasoning(content=msg.reasoning)

        tool_calls = [
            ToolCall(
                tool_call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "",
            )
            for tc in (msg.tool_calls or [])
        ]

        return AssistantMessage(
            reasoning=reasoning,
            content=msg.content or "",
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input=prompt_tokens,
                answer=answer_tokens,
                reasoning=reasoning_tokens,
            ),
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )
