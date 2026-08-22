# SPDX-License-Identifier: Apache-2.0
"""Dependency-light parser for Cline's newline-delimited JSON event stream."""
import json
import logging
from typing import Any
from uuid import uuid4

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)



LOG = logging.getLogger(__name__)

def _message(index: int, text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def _stringify(value: Any) -> str:
    """Render a tool output/error payload as text for a function_call_output item."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _reasoning_token_count(usage: dict[str, Any], *, running_total: bool = False) -> int:
    """Return Cline's reasoning-token count across its supported usage shapes."""
    direct_keys = (
        ("totalReasoningTokens", "reasoningTokens")
        if running_total
        else (
            "reasoningTokens",
            "totalReasoningTokens",
        )
    )
    for key in direct_keys:
        value = usage.get(key)
        if value is not None:
            return int(value or 0)

    details = usage.get("outputTokenDetails") or usage.get("outputTokensDetails") or {}
    if isinstance(details, dict):
        return int(details.get("reasoningTokens") or details.get("reasoning_tokens") or 0)
    return 0


def parse_cline_events(stdout: str) -> tuple[list[Any], dict[str, Any]]:
    """Convert ``cline --json`` stdout into (output_items, metadata).

    ``cline --json`` writes one JSON object per line. Two record types carry the trajectory:

    - ``{"type": "agent_event", "event": {...}}`` — the agent loop's own events. ``content_start``
      with ``contentType: "text"`` streams assistant text in chunks; the matching ``content_end``
      carries the final text for the turn, so the message is taken from ``content_end`` and the
      chunks only serve as a fallback for a truncated stream. ``contentType: "tool"`` brackets one
      tool call: ``content_start`` has ``toolName``/``toolCallId``/``input``, ``content_end`` has
      ``output`` or ``error``. ``usage`` events carry running token totals, ``done`` the finish
      reason.
    - ``{"type": "run_result", ...}`` — the final summary (``finishReason``, ``iterations``,
      aggregate ``usage``, final ``text``, resolved ``model``).

    ``hook_event`` records duplicate tool-call boundaries the agent events already cover and are
    ignored. Reasoning arrives as ``contentType: "reasoning"`` and becomes a ``<think>`` block on
    the assistant message of the turn that produced it, matching the other CLI agents. Verified
    against cline 3.0.55.
    """
    output_items: list[Any] = []
    metadata: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    # Text/reasoning chunks for the turn being streamed; the matching content_end supersedes them.
    text_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    # True once the current turn's reasoning has been folded into a message, so the trailing
    # content_end for that same reasoning is not attached a second time (see below).
    reasoning_consumed = False
    # toolCallId -> tool name for calls whose content_start was seen. Cline emits start/end pairs,
    # but a stream cut short can leave a start without an end and a malformed one an end without a
    # start, so both halves are tracked rather than assumed.
    open_tool_calls: dict[str, str] = {}
    saw_usage = False

    def flush_text() -> None:
        """Emit the buffered assistant text for the current turn, with any reasoning attached.

        Cline closes the turn's reasoning *after* its text (``content_end`` for text, then
        ``content_end`` for reasoning), so the think block is taken from the reasoning chunks
        streamed so far rather than from that trailing event, which would otherwise land on the
        following turn.
        """
        nonlocal reasoning_consumed
        text = "".join(text_chunks)
        text_chunks.clear()
        think = "".join(reasoning_chunks).strip()
        if not text.strip():
            return
        if think:
            text = f"<think>\n{think}\n</think>\n\n{text}"
            reasoning_chunks.clear()
            reasoning_consumed = True
        output_items.append(_message(len(output_items), text))

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue

        rtype = record.get("type")

        if rtype == "run_result":
            metadata["finish_reason"] = record.get("finishReason")
            metadata["iterations"] = record.get("iterations")
            usage = record.get("aggregateUsage") or record.get("usage") or {}
            if isinstance(usage, dict) and usage:
                # The aggregate is authoritative: Cline computes it from the session rather than
                # summing per-turn events, which a truncated stream can drop.
                metadata["input_tokens"] = int(usage.get("inputTokens") or 0) + int(usage.get("cacheReadTokens") or 0)
                metadata["output_tokens"] = int(usage.get("outputTokens") or 0)
                metadata["reasoning_tokens"] = _reasoning_token_count(usage)
                saw_usage = True
            model = record.get("model")
            if isinstance(model, dict) and model.get("id"):
                metadata["model"] = model["id"]
            elif isinstance(model, str) and model:
                metadata["model"] = model
            continue

        if rtype == "error":
            message = record.get("message")
            if message:
                metadata.setdefault("error", str(message))
                LOG.warning("cline reported an error: %s", str(message)[:500])
            continue

        if rtype != "agent_event":
            # hook_event records mirror tool boundaries the agent events already carry.
            continue

        event = record.get("event")
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        content_type = event.get("contentType")

        if etype == "content_start" and content_type == "text":
            text_chunks.append(event.get("text") or "")

        elif etype == "content_end" and content_type == "text":
            # The turn's final text; it supersedes the streamed chunks.
            final = event.get("text")
            if final is not None:
                text_chunks.clear()
                text_chunks.append(final)
            flush_text()

        elif etype == "content_start" and content_type == "reasoning":
            if not event.get("redacted"):
                reasoning_chunks.append(event.get("reasoning") or "")
                reasoning_consumed = False

        elif etype == "content_end" and content_type == "reasoning":
            # Closes reasoning the turn's message already carries (flush_text ran first), so this
            # only matters when there was no text to attach it to: keep the final text for a
            # reasoning-only turn, which some vLLM parsers produce by routing the whole answer
            # through the reasoning channel.
            if reasoning_consumed:
                reasoning_chunks.clear()
                reasoning_consumed = False
            else:
                final = event.get("reasoning")
                if final:
                    reasoning_chunks.clear()
                    reasoning_chunks.append(final)

        elif etype == "content_start" and content_type == "tool":
            # Text streamed before a tool call belongs to the turn that made it, so it is emitted
            # ahead of the call to keep the trajectory ordered.
            flush_text()
            call_id = str(event.get("toolCallId") or f"call-{uuid4().hex[:8]}")
            name = event.get("toolName") or ""
            tool_input = event.get("input")
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else _stringify(tool_input)
            open_tool_calls[call_id] = name
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=name,
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )

        elif etype == "content_end" and content_type == "tool":
            call_id = str(event.get("toolCallId") or "")
            if call_id and call_id not in open_tool_calls:
                # A result with no recorded call: emit the call so the output is not orphaned.
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments="{}",
                        call_id=call_id,
                        name=event.get("toolName") or "",
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )
            open_tool_calls.pop(call_id, None)
            error = event.get("error")
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id or f"call-{uuid4().hex[:8]}",
                    output=_stringify(error if error else event.get("output")),
                    status="completed",
                )
            )

        elif etype == "usage":
            # Running totals for the session so far; run_result's aggregate wins when the run ends
            # normally, and these are what remains when it does not.
            metadata["input_tokens"] = int(event.get("totalInputTokens") or 0) + int(
                event.get("totalCacheReadTokens") or 0
            )
            metadata["output_tokens"] = int(event.get("totalOutputTokens") or 0)
            metadata["reasoning_tokens"] = _reasoning_token_count(event, running_total=True)
            saw_usage = True

        elif etype == "done":
            metadata.setdefault("finish_reason", event.get("reason"))
            metadata.setdefault("iterations", event.get("iterations"))

        elif etype == "error":
            error = event.get("error")
            text = error.get("message") if isinstance(error, dict) else _stringify(error)
            if text:
                metadata.setdefault("error", text)
                LOG.warning("cline agent error event: %s", str(text)[:500])

    # A stream cut off mid-turn leaves text with no content_end; surface it rather than drop it.
    flush_text()
    trailing_think = "".join(reasoning_chunks).strip()
    if trailing_think:
        # Reasoning with no message to attach it to (a reasoning-only turn, or a truncated
        # stream) is surfaced on its own so the trajectory does not silently lose it.
        output_items.append(_message(len(output_items), f"<think>\n{trailing_think}\n</think>"))

    if not saw_usage:
        LOG.debug("cline stream carried no usage events; token counts reported as 0")

    return output_items, metadata
