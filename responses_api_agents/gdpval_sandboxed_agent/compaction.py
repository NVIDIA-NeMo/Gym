# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-compaction segments of an OpenHands rollout for NeMo RL multi-trace training.

Mirror of ``gdpval_stirrup_agent`` (``policy_client.segments`` / ``segment_messages`` and
``app.export_segment_responses``) for the OpenHands harness. The sandbox runner records
every policy call in ``/trace/completions/*.json`` (the fork's ``NemoGymClient`` completion
log: request ``messages``, the served chat completion in ``response`` with the echoed token
fields on its assistant message, ``kwargs`` with the tools, ``timestamp``). The GDPVal
condenser (``compaction_condenser.py``, uploaded with the runner) marks its summarization
call with ``gdpval_call_kind = "compaction_summary"``. Here the calls are grouped into
context segments and exported as one Gym response per segment:

* segment 0 (``segment_boundary_reason=""``): the calls before the first summary,
* the summary call alone (``"compaction"``),
* the continuation after it (``"post_compaction"``), and so on for later compactions.

Output items cover only the messages after the segment's first request (its context prefix),
so a summary that re-sends earlier assistant text is never a training target twice. The
k-th assistant message after the prefix is the k-th call's reply and carries that reply's
``prompt_token_ids`` / ``generation_token_ids`` / ``generation_log_probs`` as echoed by the
model server (plus the optional sampler top-k fields when score centering is on).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

TOKEN_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
OPTIONAL_TOKEN_KEYS = ("generation_topk_ids", "generation_topk_logprobs", "routed_experts")
CALL_KIND_KEY = "gdpval_call_kind"
SUMMARY_CALL_KIND = "compaction_summary"
# Fallback marker: the first line of the OpenHands LLMSummarizingCondenser prompt, which the
# GDPVal condenser reuses verbatim. Only consulted when the log carries no explicit kind.
SUMMARY_PROMPT_PREFIX = "You are maintaining a context-aware state summary for an interactive agent."


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return str(content)


def _strip_tokens(message: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in message.items() if k not in TOKEN_KEYS + OPTIONAL_TOKEN_KEYS}


def _reply_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Assistant message of a completion log entry with its token bundle (exporter convention)."""
    choices = (payload.get("response") or {}).get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"completion log needs exactly one choice, found {len(choices)}")
    message = dict(choices[0].get("message") or {})
    message.update(payload.get("provider_specific_fields") or {})
    message.setdefault("role", "assistant")
    return message


def call_kind(payload: dict[str, Any]) -> str:
    """``"compaction_summary"`` for the condenser's summarization call, ``"agent"`` otherwise."""
    kind = payload.get(CALL_KIND_KEY)
    if kind:
        return str(kind)
    messages = payload.get("messages") or []
    if (
        len(messages) == 1
        and messages[0].get("role") == "user"
        and _text(messages[0].get("content")).lstrip().startswith(SUMMARY_PROMPT_PREFIX)
    ):
        return SUMMARY_CALL_KIND
    return "agent"


def load_calls(task: Path) -> list[dict[str, Any]]:
    """Completed policy calls of a rollout in dispatch order, from the sandbox completion logs.

    ``task`` is the agent's per-rollout temp dir: the downloaded sandbox trace sits under
    ``task/trace`` and the logs under ``task/trace/completions`` (the ``trajectory.py`` layout).
    """
    paths = list((Path(task) / "trace/completions").glob("*.json"))
    if not paths:
        raise ValueError("OpenHands produced no completion logs")
    payloads = sorted((json.loads(p.read_text()) for p in paths), key=lambda x: x["timestamp"])
    calls = []
    for payload in payloads:
        reply = _reply_from_payload(payload)
        calls.append(
            {
                "kind": call_kind(payload),
                "response_id": (payload.get("response") or {}).get("id"),
                "messages": [dict(m) for m in payload.get("messages") or []],
                "tools": (payload.get("kwargs") or {}).get("tools") or [],
                "reply": reply,
                "finish_reason": (payload["response"]["choices"][0] or {}).get("finish_reason"),
                "timestamp": payload["timestamp"],
            }
        )
    return calls


def segments(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group calls into context segments at the condenser's summarization calls.

    Returns ``[{"index", "reason", "calls": [(position, call), ...]}, ...]`` where ``reason``
    is ``""`` for the first segment, ``"compaction"`` for a single-call summary segment and
    ``"post_compaction"`` for the continuation after one.
    """
    out: list[dict[str, Any]] = []
    for position, call in enumerate(calls):
        entry = (position, call)
        if call["kind"] == SUMMARY_CALL_KIND:
            if not out:
                raise ValueError("compaction summary call before any agent call")
            out.append({"reason": "compaction", "calls": [entry]})
        elif not out:
            out.append({"reason": "", "calls": [entry]})
        elif out[-1]["reason"] == "compaction":
            out.append({"reason": "post_compaction", "calls": [entry]})
        else:
            out[-1]["calls"].append(entry)
    for index, segment in enumerate(out):
        segment["index"] = index
    return out


def segment_messages(segment: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Chat-completions messages of one segment with the echoed token bundle on each reply.

    Returns ``(messages, prefix_message_count)``: the last call's request messages plus its
    reply, where ``prefix_message_count`` is the length of the segment's first request
    (context only, never trained). Token fields that OpenHands re-sends on older assistant
    messages are discarded; each assistant message after the prefix gets the bundle of the
    call that produced it, so the trainer never sees a stale or partial echo.
    """
    calls = [call for _, call in segment["calls"]]
    prefix = len(calls[0]["messages"])
    last = calls[-1]
    messages = [_strip_tokens(dict(m)) for m in last["messages"]]
    messages.append(_strip_tokens(last["reply"]))
    if len(messages) <= prefix:
        raise ValueError(f"segment {segment.get('index')}: request shrank below its own context prefix")
    bundles = [c["reply"] for c in calls]
    assistant_positions = [i for i, m in enumerate(messages) if i >= prefix and m.get("role") == "assistant"]
    if len(assistant_positions) != len(bundles):
        raise ValueError(
            f"segment {segment.get('index')}: {len(assistant_positions)} assistant messages after the "
            f"prefix but {len(bundles)} policy calls"
        )
    echoed = [all(key in bundle for key in TOKEN_KEYS) for bundle in bundles]
    if any(echoed) and not all(echoed):
        raise ValueError("policy replies carry token ids on some calls only; the model server echo is inconsistent")
    if all(echoed):
        for position, bundle in zip(assistant_positions, bundles):
            if len(bundle["generation_token_ids"]) != len(bundle["generation_log_probs"]):
                raise ValueError("generation_token_ids and generation_log_probs are misaligned")
            for key in TOKEN_KEYS + OPTIONAL_TOKEN_KEYS:
                if key in bundle:
                    messages[position][key] = bundle[key]
    return messages, prefix


def export_segment_responses(calls: list[dict[str, Any]], *, model: str, session_id: str) -> list:
    """One Gym response per context segment, metadata per NeMo RL's compaction-trace contract.

    All metadata values are strings: ``training_mode`` "compaction", ``session_id`` /
    ``root_session_id`` (the single main session), ``parent_session_id`` / ``spawn_call_id``
    empty, ``session_role`` "parent", ``segment_index``, ``segment_boundary_reason``,
    ``request_ids`` (JSON list of the served chat-completion ids owned by the segment) and
    ``policy_calls``.
    """
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.responses_converter import VLLMConverter
    from openai.types.responses import FunctionTool

    responses = []
    for segment in segments(calls):
        messages, prefix = segment_messages(segment)
        with_tokens = any("prompt_token_ids" in m for m in messages[prefix:])
        converter = VLLMConverter(return_token_id_information=with_tokens)
        output = converter.chat_completions_messages_to_responses_items(messages[prefix:])
        if not output:
            raise ValueError(f"segment {segment['index']} has no output items")
        tools = segment["calls"][-1][1].get("tools") or []
        responses.append(
            NeMoGymResponse(
                id=f"gdpval-{session_id}-seg{segment['index']}",
                created_at=time.time(),
                model=model,
                object="response",
                output=output,
                tools=[FunctionTool.model_validate(t["function"] | {"type": "function"}) for t in tools],
                tool_choice="auto",
                parallel_tool_calls=True,
                status="completed",
                metadata={
                    "training_mode": "compaction",
                    "session_id": str(session_id),
                    "root_session_id": str(session_id),
                    "parent_session_id": "",
                    "session_role": "parent",
                    "spawn_call_id": "",
                    "segment_index": str(segment["index"]),
                    "segment_boundary_reason": segment["reason"],
                    "request_ids": json.dumps(
                        [
                            str(call.get("response_id") or f"{session_id}:call{position}")
                            for position, call in segment["calls"]
                        ]
                    ),
                    "policy_calls": str(len(segment["calls"])),
                },
            )
        )
    return responses


def export_segments_from_trace(task: Path, *, model: str, session_id: str) -> list:
    """``export_segment_responses`` over the completion logs under ``task/trace/completions``."""
    return export_segment_responses(load_calls(task), model=model, session_id=session_id)
