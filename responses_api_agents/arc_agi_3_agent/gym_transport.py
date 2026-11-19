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
"""The harness's LLM-call layer, routed to the NeMo Gym model server.

The harness (``inference/agent/tool_agent.py``) makes every model call with
``requests.post(f"{base_url}/chat/completions", json=payload, timeout=(connect, read))`` from its game threads. The
agent replaces the module's ``requests`` with :class:`RequestsProxy`, so the harness files stay byte-identical. The
proxy sends the call to the Gym model server's ``/v1/chat/completions`` through Gym's aiohttp ``ServerClient`` on the
agent's event loop, and maps the two formats:

Request (harness, vLLM/SGLang style -> Gym ``NeMoGymChatCompletionCreateParamsNonStreaming``):

- Gym's request model drops unknown fields, so the non-OpenAI fields (``top_k``, ``continue_final_message``, ...) go
  to ``metadata["extra_body"]`` and ``chat_template_kwargs`` to ``metadata["chat_template_kwargs"]`` (JSON strings
  that the Gym vLLM model server merges into the vLLM request). ``continue_final_message`` also sets
  ``add_generation_prompt=false``, which vLLM requires (SGLang does not).
- An assistant history message's ``reasoning_content`` / ``reasoning`` becomes a ``<think>R</think>`` content prefix;
  the Gym model server (``uses_reasoning_parser``) splits it back into ``reasoning_content`` for vLLM.

Response (Gym -> harness): the Gym model server returns the reasoning as a ``<think>R</think>`` content prefix; the
proxy puts it back in ``reasoning_content`` and ``reasoning``, as vLLM/SGLang return it to the harness.

Continuations: the harness continues a final assistant message whose reasoning it already closed (``...</think>``,
turn-compaction records and ARC3_FORCE_ACT). Everything generated after that is content. Some reasoning parsers
(vLLM ``nemotron_v3`` / ``qwen3`` with thinking on) start every reply in the reasoning state whatever the prompt ends
with, and return it as reasoning with empty content; for these requests the proxy moves that text to the content.

Only ``POST .../chat/completions`` is served. Any other POST (an engine-specific endpoint) raises, because the Gym
model server has no such endpoint.

Every call sent is recorded in its game's :class:`~responses_api_agents.arc_agi_3_agent.call_log.GameCallLog` with the
role the harness gave it (``call_role``: ``gameplay``, or ``helper`` with the summary's ``gdn_history_id``), the game's
step and level when it was sent (``bind_game``'s ``progress``), and, for a call that got no answer (timeout, connection
error, cancelled while waiting), its error.
"""

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import wait as wait_for_future
from contextlib import contextmanager
from typing import Any

import requests
from aiohttp import ClientError

from responses_api_agents.arc_agi_3_agent.call_log import GAMEPLAY_ROLE, GameCallLog


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# How often a pending model call checks whether its attempt was cancelled.
CANCEL_POLL_S = 1.0

# Fields of Gym's chat-completion request model (nemo_gym.core.responses_api_types.NeMoGymChatCompletionCreateParamsNonStreaming).
OPENAI_CHAT_FIELDS = frozenset(
    {
        "messages",
        "model",
        "audio",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "seed",
        "service_tier",
        "stop",
        "store",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "web_search_options",
    }
)
# Engine-specific session fields: the switches that send them (ARC3_KV_EVICT, ARC3_GDN_OWNERSHIP) are off under Gym.
ENGINE_ONLY_FIELDS = frozenset({"kv_evict_conversation", "session_id", "gdn_ownership"})
REASONING_KEYS = ("reasoning_content", "reasoning")
# Chat-template kwargs that set the thinking effort (a model's template reads one of them).
EFFORT_KWARGS = ("reasoning_effort", "thinking_effort")


def _message_reasoning(message: dict[str, Any]) -> str | None:
    for key in REASONING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def openai_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """A tool call with the OpenAI fields only. Engine extras that the harness echoes back in its history (SGLang's
    ``index``) are refused by Gym's request model, and chat templates do not render them."""
    function = call.get("function") or {}
    return {
        "id": call.get("id"),
        "type": call.get("type") or "function",
        "function": {"name": function.get("name"), "arguments": function.get("arguments")},
    }


def history_message_to_gym(message: dict[str, Any]) -> dict[str, Any]:
    """Move an assistant message's reasoning into a ``<think>`` content prefix (Gym's history format); its tool calls
    keep their OpenAI fields only."""
    if message.get("role") != "assistant":
        return message
    reasoning = _message_reasoning(message)
    out = {k: v for k, v in message.items() if k not in REASONING_KEYS}
    if out.get("tool_calls"):
        out["tool_calls"] = [openai_tool_call(c) for c in out["tool_calls"]]
    if reasoning is None:
        return out
    prefix = f"<think>{reasoning}</think>"
    content = out.get("content")
    if content is None or isinstance(content, str):
        out["content"] = prefix + (content or "")
    elif isinstance(content, list):
        out["content"] = [{"type": "text", "text": prefix}, *content]
    else:
        raise TypeError(f"Unsupported assistant content type {type(content).__name__}")
    return out


def to_gym_chat_request(
    payload: dict[str, Any],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    reasoning_effort_kwarg: str = "reasoning_effort",
) -> dict[str, Any]:
    """Map a harness chat-completion payload to a Gym ``/v1/chat/completions`` body.

    ``chat_template_kwargs`` and ``extra_body`` are the agent config's static values; the request's own values win.
    ``reasoning_effort`` goes in the chat-template kwarg ``reasoning_effort_kwarg`` of every thinking request
    (``enable_thinking`` not false) that sets no effort of its own (the harness's effort ladder wins).
    """
    if payload.get("stream"):
        raise ValueError("The Gym model server serves non-streaming chat completions only (set provider vllm)")
    engine_only = sorted(ENGINE_ONLY_FIELDS & payload.keys())
    if engine_only:
        raise ValueError(f"Engine-specific request fields {engine_only} have no Gym equivalent; turn their switch off")

    body: dict[str, Any] = {}
    extras: dict[str, Any] = dict(extra_body or {})
    template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
    for key, value in payload.items():
        if key in ("stream", "stream_options"):
            continue
        if key == "chat_template_kwargs":
            template_kwargs.update(value or {})
        elif key in OPENAI_CHAT_FIELDS:
            body[key] = value
        else:
            extras[key] = value
    if (
        reasoning_effort
        and template_kwargs.get("enable_thinking") is not False
        and not any(k in template_kwargs for k in EFFORT_KWARGS)
        and "reasoning_effort" not in body
    ):
        template_kwargs[reasoning_effort_kwarg] = reasoning_effort
    if extras.get("continue_final_message"):
        extras.setdefault("add_generation_prompt", False)
    body["messages"] = [history_message_to_gym(m) for m in payload["messages"]]

    metadata = dict(body.get("metadata") or {})
    if extras:
        metadata["extra_body"] = json.dumps(extras)
    if template_kwargs:
        metadata["chat_template_kwargs"] = json.dumps(template_kwargs)
    if metadata:
        body["metadata"] = metadata
    return body


def split_reasoning(content: str | None) -> tuple[str | None, str | None]:
    """Split a Gym ``<think>R</think>C`` content into (R, C); (None, content) when there is no think block."""
    if not isinstance(content, str):
        return None, content
    matches = THINK_PATTERN.findall(content)
    if not matches:
        return None, content
    return "".join(matches), THINK_PATTERN.sub("", content)


def continues_closed_reasoning(payload: dict[str, Any]) -> bool:
    """True for a continuation of an assistant message whose reasoning is already closed (``...</think>``)."""
    if not payload.get("continue_final_message"):
        return False
    last = (payload.get("messages") or [{}])[-1]
    content = last.get("content")
    return last.get("role") == "assistant" and isinstance(content, str) and "</think>" in content


def from_gym_chat_response(payload: dict[str, Any], reasoning_is_content: bool = False) -> dict[str, Any]:
    """Put the reasoning of every choice back in ``reasoning_content`` / ``reasoning``.

    With ``reasoning_is_content`` (see :func:`continues_closed_reasoning`), a choice with reasoning but no content gets
    the reasoning as its content.
    """
    out = dict(payload)
    choices = []
    for choice in payload.get("choices") or []:
        choice = dict(choice)
        message = dict(choice.get("message") or {})
        reasoning, content = split_reasoning(message.get("content"))
        if reasoning is not None and reasoning_is_content and not (content or "").strip():
            message["content"] = reasoning
        elif reasoning is not None:
            message["content"] = content
            message["reasoning_content"] = reasoning
            message["reasoning"] = reasoning
        choice["message"] = message
        choices.append(choice)
    out["choices"] = choices
    return out


class GymHTTPResponse:
    """The part of ``requests.Response`` the harness reads."""

    def __init__(
        self,
        status_code: int,
        text: str,
        url: str,
        headers: dict[str, str] | None = None,
        reasoning_is_content: bool = False,
    ):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.headers = headers or {}
        self.reasoning_is_content = reasoning_is_content
        self._json: dict[str, Any] | None = None

    def json(self) -> dict[str, Any]:
        if self._json is None:
            payload = json.loads(self.text)
            self._json = (
                from_gym_chat_response(payload, self.reasoning_is_content) if self.status_code < 400 else payload
            )
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error for url: {self.url}", response=None)

    def iter_lines(self, decode_unicode: bool = False):
        raise RuntimeError("The Gym transport returns non-streaming responses")


class GameLogCapture(logging.Handler):
    """Records every log line of level WARNING or above that is emitted on a thread bound to a game (the harness's
    warnings while it plays the game) in that game's log: :meth:`GymChatTransport.record_log`."""

    def __init__(self, transport: "GymChatTransport"):
        super().__init__(logging.WARNING)
        self.transport = transport

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.transport.record_log(record)
        except Exception:
            self.handleError(record)


class GymChatTransport:
    """Sends harness chat completions to a Gym model server from the harness's worker threads."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        post: Callable[[dict[str, Any], dict[str, str] | None, str | None], Any],
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        reasoning_effort_kwarg: str = "reasoning_effort",
    ):
        """``post(body, cookies, game_key)`` is a coroutine function returning ``(status, text, cookies)``."""
        self._loop = loop
        self._post = post
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._extra_body = dict(extra_body or {})
        self._effort = {"reasoning_effort": reasoning_effort, "reasoning_effort_kwarg": reasoning_effort_kwarg}
        self._local = threading.local()
        self._cookies: dict[str, dict[str, str]] = {}
        self._cookies_lock = threading.Lock()
        self._cancelled: set = set()
        self._all_cancelled = threading.Event()
        self._logs: dict[str, GameCallLog] = {}
        self._logs_lock = threading.Lock()
        self._log_capture: GameLogCapture | None = None
        self.requests_sent = 0

    @contextmanager
    def call_role(self, role: str, history_id: str) -> Iterator[None]:
        """Mark this thread's calls inside the block with the harness's role for them."""
        previous = getattr(self._local, "role", None)
        self._local.role = (role, history_id)
        try:
            yield
        finally:
            self._local.role = previous

    def call_log(self, game_key: str) -> GameCallLog:
        with self._logs_lock:
            return self._logs.setdefault(game_key, GameCallLog())

    def take_call_log(self, game_key: str) -> GameCallLog:
        """The game's call log, removed from the transport (call once, after the game ended)."""
        with self._logs_lock:
            return self._logs.pop(game_key, None) or GameCallLog()

    def cancel(self, game_key: str) -> None:
        """Abandon ``game_key``'s pending and future model calls (they raise ``requests.RequestException``)."""
        self._cancelled.add(game_key)

    def cancel_all(self) -> None:
        self._all_cancelled.set()

    def _is_cancelled(self, game_key: str | None) -> bool:
        return self._all_cancelled.is_set() or (game_key is not None and game_key in self._cancelled)

    def bind_game(self, game_key: str | None, progress: Callable[[], tuple[int, int]] | None = None) -> None:
        """Route this thread's calls with ``game_key``'s session cookies (sticky model endpoint per game).
        ``progress()`` returns the game's ``(step, level)`` now; each call records it as of when it is sent."""
        self._local.game_key = game_key
        self._local.progress = progress

    def capture_logs(self) -> None:
        """From now on, keep the log lines (WARNING and above) logged on a bound game's thread in its call log
        (the root logger's handler; once per transport)."""
        if self._log_capture is None:
            self._log_capture = GameLogCapture(self)
            logging.getLogger().addHandler(self._log_capture)

    def record_log(self, record: logging.LogRecord) -> None:
        game_key = getattr(self._local, "game_key", None)
        if game_key is None:
            return
        step, level = self._progress()
        self.call_log(game_key).add_log(
            logged_at=record.created,
            severity=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            step=step,
            level=level,
        )

    def _progress(self) -> tuple[int | None, int | None]:
        progress = getattr(self._local, "progress", None)
        if progress is None:
            return None, None
        try:
            step, level = progress()
            return int(step), int(level)
        except Exception:  # the game's state could not be read: the call is still logged, without it
            return None, None

    def _record(
        self,
        game_key: str | None,
        payload: dict[str, Any],
        sent: tuple[float, int | None, int | None],
        status: int = 0,
        text: str | None = None,
        error: str | None = None,
    ) -> None:
        if game_key is None:
            return
        role, history_id = getattr(self._local, "role", None) or (GAMEPLAY_ROLE, "main")
        try:
            parsed = json.loads(text) if text is not None and 0 < status < 400 else None
        except ValueError:
            parsed = None
        started_at, step, level = sent
        self.call_log(game_key).add(
            role=role,
            history_id=history_id,
            messages=payload.get("messages") or [],
            status=status,
            response=parsed if isinstance(parsed, dict) else None,
            started_at=started_at,
            step=step,
            level=level,
            error=error,
        )

    def post_chat(self, url: str, payload: dict[str, Any], timeout: Any) -> GymHTTPResponse:
        body = to_gym_chat_request(
            payload, chat_template_kwargs=self._chat_template_kwargs, extra_body=self._extra_body, **self._effort
        )
        game_key = getattr(self._local, "game_key", None)
        with self._cookies_lock:
            cookies = self._cookies.get(game_key) if game_key is not None else None
        read_timeout = timeout[1] if isinstance(timeout, (tuple, list)) else timeout
        if self._is_cancelled(game_key):
            raise requests.RequestException(f"attempt {game_key} was cancelled; no model call sent")
        sent = (time.time(), *self._progress())
        future = asyncio.run_coroutine_threadsafe(self._post(body, cookies, game_key), self._loop)
        deadline = None if read_timeout is None else time.monotonic() + float(read_timeout)
        # Wait for the call to finish (polling for cancellation and the deadline), then read its result. The two are
        # separate: an upstream timeout is a TimeoutError stored in a finished future, not a wait that ran out.
        error: requests.RequestException | None = None
        while True:
            wait = CANCEL_POLL_S if deadline is None else min(CANCEL_POLL_S, max(0.0, deadline - time.monotonic()))
            if wait_for_future([future], timeout=wait).done:
                break
            if self._is_cancelled(game_key):
                future.cancel()
                error = requests.RequestException(f"attempt {game_key} was cancelled during a model call")
                break
            if deadline is not None and time.monotonic() >= deadline:
                future.cancel()
                error = requests.exceptions.ReadTimeout(f"Gym model server did not answer in {read_timeout} s")
                break
        if error is None:
            try:
                status, text, new_cookies = future.result()
            except TimeoutError as exc:  # also asyncio.TimeoutError (an alias): the upstream call timed out
                error = requests.exceptions.ReadTimeout(f"Gym model server request timed out: {exc!r}")
                error.__cause__ = exc
            except (OSError, ClientError) as exc:  # e.g. a refused connection, a dropped or truncated response
                error = requests.exceptions.ConnectionError(f"Gym model server request failed: {exc!r}")
                error.__cause__ = exc
        if error is not None:
            self._record(game_key, payload, sent, error=f"{type(error).__name__}: {error}")
            raise error
        if game_key is not None and new_cookies:
            with self._cookies_lock:
                self._cookies[game_key] = dict(new_cookies)
        self.requests_sent += 1
        self._record(game_key, payload, sent, status=status, text=text)
        return GymHTTPResponse(status, text, url, reasoning_is_content=continues_closed_reasoning(payload))


class RequestsProxy:
    """Stands in for the ``requests`` module inside the harness: ``post`` to ``/chat/completions`` goes to Gym."""

    def __init__(self, transport: GymChatTransport):
        self._transport = transport

    def post(self, url: str, *args: Any, json: dict[str, Any] | None = None, timeout: Any = None, **kwargs: Any):
        if not str(url).rstrip("/").endswith("/chat/completions"):
            raise requests.RequestException(f"The Gym transport serves only POST /chat/completions, not {url}")
        if json is None:
            raise requests.RequestException("chat completion POST without a JSON body")
        return self._transport.post_chat(str(url), json, timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(requests, name)


def chat_messages_summary(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Counts of messages and images in a request (for logs and tests)."""
    images = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            images += sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url")
    return {"messages": len(messages), "images": images}
