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

"""Serve a raw BASE checkpoint behind the Responses API.

A base model has no chat template and no tool calling, so the conversation is
flattened into a `Role: content` transcript and sent to `/v1/completions`. Three
settings do the work, and each one was established by measurement upstream
rather than taste:

  steer    a final instruction naming the action format AND THE REPO ROOT. A
           steer naming the wrong root yields a syntactically perfect command
           against a path that does not exist -- a 0-byte patch that reads as
           model weakness, not misconfiguration.
  primer   "\\nAssistant: " opens the model's turn. Without it the model
           continues the user's turn instead of answering.
  prefill  the opening action fence, pre-opened so that prose continuation is
           ungrammatical. Without it the model echoes the steer's template back,
           literal "<your command>" placeholder and all.

`stop` is load-bearing in this mode: the transcript separates turns with
"\\nUser:" / "\\nAssistant:", so those markers are what END a turn. Without them
the model writes its fence and then hallucinates the next turn, and the harness
cannot extract a single clean action. Note the OpenAI spec caps `stop` at FOUR
entries and some servers enforce it, so the default list is trimmable.
"""

import asyncio
import json
from contextlib import nullcontext
from time import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

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
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


# Turn separators. The transcript emits exactly one newline before each role
# marker, so these match a turn boundary and nothing else. Only the backticks
# wire sends them, and its transcript never contains "\nAssistant (tool call):"
# (only a message with tool_calls renders that), so that marker is left out:
# the OpenAI spec caps `stop` at four entries, and servers that enforce the cap
# reject a fifth with HTTP 400.
DEFAULT_STOP = [
    "\nTool Result:",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
]

# mini-swe-agent backticks mode: no tools, one bash command per turn inside a
# fence the harness parses itself. THE ROOT IS PART OF THE STEER -- /testbed is
# SWE-bench Verified; SWE-bench Pro is /app.
MINI_BASH_STEER = (
    "\nSystem: Your investigation is complete. Fix the bug by editing the "
    "existing source in /testbed. Reply with exactly one bash command in "
    "this exact form and nothing after it:\n"
    "```mswea_bash_command\n"
    "<your command>\n"
    "```"
)

DEFAULT_PRIMER = "\nAssistant: "
DEFAULT_PREFILL = "```mswea_bash_command\n"

# mini-swe-agent function-calling mode: the action is a `bash` tool call taking
# {"command": "..."}, and the repo is /app. A steer describing the backticks
# fence here yields a well-formed reply the agent cannot execute.
BASH_TOOL_STEER = (
    "\nSystem: Your investigation is complete. Fix the bug by editing the "
    "existing source in /app. Reply with exactly one tool call to `bash`, "
    'whose arguments are a JSON object with a single "command" key, and '
    "nothing after it."
)
# With tools in play, assistant turns render as "Assistant (tool call): [...]",
# so the primer must open the same line -- anything else asks the model to
# continue a marker that appears nowhere in the transcript.
TOOL_CALL_PRIMER = "\nAssistant (tool call): "
# Pre-opens the tool-call array, so the model continues a call rather than prose.
TOOL_CALL_PREFILL = '[{"'

# Per-wire defaults. THE STOP LIST INVERTS between them: in backticks mode the
# role markers are what END a turn, but a tool call is JSON, and JSON strings
# routinely contain "\nUser:"-like text -- so on the function-calling wire the
# stop list truncates valid calls and must be OFF.
WIRE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "backticks": {
        "steer": MINI_BASH_STEER,
        "primer": DEFAULT_PRIMER,
        "prefill": DEFAULT_PREFILL,
        "stop": DEFAULT_STOP,
    },
    "function_calling": {
        "steer": BASH_TOOL_STEER,
        "primer": TOOL_CALL_PRIMER,
        "prefill": TOOL_CALL_PREFILL,
        "stop": [],
    },
}

ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}


def message_text(content: Any) -> str:
    """Flatten Responses-API content parts to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
        return "".join(parts)
    return str(content)


def litellm_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """A tool call as litellm re-serializes it on its way back to the server.

    mini-swe-agent keeps each response as litellm's message dump and re-sends
    it, so every call the reference bridge rendered -- replayed or generated --
    arrived as function{arguments, name}, id, type, whatever order the model or
    the trace wrote. Measured by replaying a captured DeepSWE turn through
    mini-swe-agent 2.4.6 / litellm 1.102.1. Key order is visible to a base
    model: it continues the JSON it has been shown.
    """
    function = call.get("function") or {}
    return {
        "function": {"arguments": function.get("arguments"), "name": function.get("name")},
        "id": call.get("id"),
        "type": call.get("type") or "function",
    }


def render_transcript(messages: List[Dict[str, Any]]) -> str:
    """Flatten a conversation to a `Role: content` transcript.

    One newline precedes each role marker and no blank lines separate turns,
    which is what makes the stop markers land exactly on turn boundaries.
    """
    blocks = []
    for message in messages:
        role = message.get("role")
        if message.get("tool_calls"):
            calls = [litellm_tool_call(call) for call in message["tool_calls"]]
            blocks.append(f"\nAssistant (tool call): {json.dumps(calls, ensure_ascii=False)}")
            continue
        label = ROLE_LABELS.get(role)
        if label is None:
            continue
        blocks.append(f"\n{label}: {message_text(message.get('content'))}")
    return "".join(blocks)


def close_json_prefix(text: str) -> Optional[str]:
    """Cut `text` at the end of its first complete JSON value, or close it.

    A base model often stops (or runs on into a hallucinated next turn) with the
    outer brackets unbalanced even though the call itself is complete, so the
    brackets still open are closed -- but ONLY when no string is left
    unterminated. A generation cut inside a string is a truncated command, and a
    truncated command cannot be completed safely: a half heredoc hangs the
    shell on its missing terminator or writes a corrupt file. That must fail
    rather than be guessed at. Returns None on failure.
    """
    stack: List[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if not stack:
                return text[: index + 1]
    if in_string or not stack:
        return None
    return text + "".join(reversed(stack))


def _loads(text: str) -> Any:
    # strict=False admits raw control characters (e.g. a literal newline) inside
    # strings. Heredocs dominate these edits, and a base model sometimes writes a
    # real newline where JSON wants "\n"; that is tolerated, not repaired.
    return json.loads(text, strict=False)


def parse_bash_tool_calls(raw: str, max_calls: int = 3) -> List[Dict[str, Any]]:
    """The `bash` tool calls in a base model's generation, at most `max_calls`.

    The harness executes every call in a turn, in order, so all are returned --
    capped, as the reference caps a base model's calls per turn at 3. Accepts
    each call nested ({"function": {"name", "arguments"}}, the shape the
    transcript shows the model) or flat ({"name", "arguments"}), and arguments
    as a JSON object or a JSON-encoded string. Empty list when none parse.
    """
    start = raw.find("[")
    if start < 0:
        return []
    closed = close_json_prefix(raw[start:])
    if closed is None:
        return []
    try:
        calls = _loads(closed)
    except json.JSONDecodeError:
        return []
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return []
    found: List[Dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call["function"] if isinstance(call.get("function"), dict) else call
        name = function.get("name")
        if not isinstance(name, str) or name.strip() != "bash":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = _loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        command = arguments.get("command")
        if isinstance(command, str) and command.strip():
            found.append({"name": "bash", "arguments": {"command": command}})
            if len(found) >= max_calls:
                break
    return found


class BaseModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str

    # Which action wire the trace was shaped on. THE DRIVER MUST MATCH HOW THE
    # TRACE WAS SHAPED, not which benchmark it is: cross them and every forward
    # produces a well-formed action the agent cannot execute. Sets the defaults
    # for the four fields below; any of them can still be set explicitly.
    wire: Literal["backticks", "function_calling"] = "backticks"

    # Appended after the transcript, before the primer. Names the action format
    # and the repo root; override per benchmark.
    steer: Optional[str] = None
    # Opens the model's turn.
    primer: Optional[str] = None
    # Pre-opens the action envelope; echoed back on the response so the caller
    # sees a complete, parseable action.
    prefill: Optional[str] = None
    # None = the wire's default; [] = send no stop list at all.
    stop: Optional[List[str]] = None

    # The tool schema exactly as the trace's requests carried it, as a JSON
    # string. Rendered verbatim as the transcript's "Tool definitions:" header;
    # kept as a string so key order cannot drift through a dict round-trip.
    tool_definitions_json: Optional[str] = None

    # A base model's calls per turn beyond this are dropped. The reference caps
    # at 3; the harness executes every call it keeps.
    max_tool_calls_per_turn: int = 3

    max_tokens: int = 1024
    temperature: float = 0.6
    top_p: Optional[float] = None

    max_concurrent_requests: Optional[int] = None
    extra_body: Dict[str, Any] = Field(default_factory=dict)

    debug: bool = False


class BaseModelServer(SimpleResponsesAPIModel):
    config: BaseModelServerConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )
        defaults = WIRE_DEFAULTS[self.config.wire]
        self._steer = defaults["steer"] if self.config.steer is None else self.config.steer
        self._primer = defaults["primer"] if self.config.primer is None else self.config.primer
        self._prefill = defaults["prefill"] if self.config.prefill is None else self.config.prefill
        self._stop = list(defaults["stop"] if self.config.stop is None else self.config.stop)
        return super().model_post_init(context)

    def _parse_calls(self, text: str) -> List[Dict[str, Any]]:
        if self.config.wire != "function_calling":
            return []
        return parse_bash_tool_calls(text, self.config.max_tool_calls_per_turn)

    def build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        header = ""
        if self.config.tool_definitions_json:
            header = f"\nTool definitions:\n{self.config.tool_definitions_json}\n\n"
        return header + render_transcript(messages) + self._steer + self._primer + self._prefill

    async def _complete(
        self, messages: List[Dict[str, Any]], body_dict: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any], Optional[str]]:
        """Render the transcript, call /v1/completions, return (text, usage, finish_reason).

        The returned text has the prefill spliced back on: the server never
        echoes it, but it is part of the action the caller has to parse.
        """
        prompt = self.build_prompt(messages)

        kwargs: Dict[str, Any] = dict(self.config.extra_body)
        kwargs.update(
            model=self.config.openai_model,
            prompt=prompt,
            max_tokens=body_dict.get("max_output_tokens") or self.config.max_tokens,
            temperature=(
                body_dict["temperature"] if body_dict.get("temperature") is not None else self.config.temperature
            ),
        )
        if self._stop:
            kwargs["stop"] = self._stop
        top_p = body_dict.get("top_p") if body_dict.get("top_p") is not None else self.config.top_p
        if top_p is not None:
            kwargs["top_p"] = top_p

        async with self._semaphore:
            completion = await self._client.create_completion(**kwargs)

        choices = completion.get("choices") or [{}]
        generated = choices[0].get("text") or ""
        finish_reason = choices[0].get("finish_reason")
        # Enforce the stop list here too. Some servers match stop strings on token
        # boundaries and miss one whose newline shares a token with the text before
        # it ("```\nUser:"), so the model runs on into turns it invents -- each with
        # another action fence, which the harness rejects as a format error. A
        # server that stops correctly never returns a stop string, so this changes
        # nothing there.
        cut = min((i for i in (generated.find(s) for s in self._stop if s) if i >= 0), default=-1)
        if cut >= 0:
            generated, finish_reason = generated[:cut], "stop"
        # The prefill is part of the action the caller must parse, but the
        # server never returned it -- splice it back on.
        text = self._prefill + generated

        if self.config.debug:
            print(f"base_model: prompt_tail={prompt[-200:]!r} finish={finish_reason!r}")

        return text, completion.get("usage") or {}, finish_reason

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = body.model_dump(exclude_unset=True)
        text, usage, finish_reason = await self._complete(body_dict.get("input") or [], body_dict)

        calls = self._parse_calls(text)
        output_items: List[Dict[str, Any]]
        if calls:
            output_items = [
                {
                    "type": "function_call",
                    "id": f"fc_{uuid4().hex}",
                    "call_id": f"call_{uuid4().hex[:12]}",
                    "name": call["name"],
                    # Default json.dumps, as the harness serialises a call it
                    # synthesises, so a live turn and a replayed turn look alike
                    # in the next prompt.
                    "arguments": json.dumps(call["arguments"]),
                    "status": "completed",
                }
                for call in calls
            ]
        else:
            # Backticks: the raw text IS the action, parsed by the agent. Function
            # calling with no parseable call: the raw text goes back so the agent
            # can record the turn and answer it with a format error.
            output_items = [
                {
                    "id": f"msg_{uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ]
        # Surfaced so the agent can tell a cut-off generation from a malformed
        # one: the harness words those two format errors differently.
        truncated = finish_reason == "length"
        return NeMoGymResponse.model_validate(
            {
                "id": f"resp_{uuid4().hex}",
                "created_at": int(time()),
                "model": self.config.openai_model,
                "object": "response",
                "status": "incomplete" if truncated else "completed",
                "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
                "output": output_items,
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        # `max_tokens` is the chat spelling of `max_output_tokens`.
        if body_dict.get("max_tokens") is not None:
            body_dict.setdefault("max_output_tokens", body_dict["max_tokens"])
        text, usage, finish_reason = await self._complete(body_dict.get("messages") or [], body_dict)
        message: Dict[str, Any] = {"role": "assistant", "content": text}
        calls = self._parse_calls(text)
        if calls:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid4().hex[:12]}",
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                    }
                    for call in calls
                ],
            }
        return NeMoGymChatCompletion.model_validate(
            {
                "id": f"chatcmpl_{uuid4().hex}",
                "created": int(time()),
                "model": self.config.openai_model,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if calls else (finish_reason or "stop"),
                        "message": message,
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
        )


if __name__ == "__main__":
    BaseModelServer.run_webserver()
