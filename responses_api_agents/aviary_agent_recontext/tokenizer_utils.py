# Copyright (c) 2025, NVIDIA CORPORATION, PLACEHOLDER.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tokenizer-side helpers for truth->lie recontextualization (system-only splice).

The aviary agent runs the rollout under an honest system prompt, then this
module mutates only the system message at gradient-build time and recomputes
the per-turn ``prompt_token_ids``. ``generation_token_ids`` and
``generation_log_probs`` are not changed by these tokenizer helpers; the agent
may optionally recompute logprobs after the splice.

Single source of truth for tokenization: vLLM's ``/tokenize`` endpoint, exposed
on the model_server via ``vllm_model``'s ``/v1/tokenize`` route. We call it
through ``ServerClient.post`` so the agent never needs a local HF tokenizer
that could drift from vLLM's preprocessing (e.g. vLLM strips ``strict`` from
tool defs, while a naive ``apply_chat_template`` would render
``<strict>True</strict>`` into the system block).

Design:

  For each training-eligible turn K with rollout-produced ``prompt_token_ids[K]``:

    orig_sys_tokens = POST /v1/tokenize {messages: [{role:system, content:ORIG}],
                                          tools, chat_template_kwargs,
                                          add_generation_prompt: False}["tokens"]
    new_sys_tokens  = POST /v1/tokenize {messages: [{role:system, content:SWAP}],
                                          tools, chat_template_kwargs,
                                          add_generation_prompt: False}["tokens"]

  Verify ``prompt_token_ids[K][:len(orig_sys_tokens)] == orig_sys_tokens``,
  then splice ``new_prompt_token_ids[K] = new_sys_tokens + prompt_token_ids[K][L_orig:]``.

  Conversation tail comes verbatim from the rollout — never re-tokenized.
  ``generation_token_ids[K]`` is never mutated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


_TOKENIZER_CACHE: dict[str, Any] = {}


def load_tokenizer(model_name: str) -> Any:
    """Load and cache a HuggingFace tokenizer.

    NOTE: this is used ONLY for ``decode_snippet`` (diagnostic strings on prefix
    mismatch). It is NOT the authoritative tokenization path — that's the
    remote ``/v1/tokenize`` endpoint. If the local tokenizer drifts slightly
    from vLLM's, the diagnostic strings might decode marginally differently
    but the splice math is unaffected.
    """
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def responses_tools_to_chat_completion_tools(
    tools: Sequence[Any],
) -> list[dict[str, Any]]:
    """Convert OpenAI Responses-API flat tool defs into Chat-Completions nested form.

    Aviary (and the broader Responses API) emits tools as
    ``{"type": "function", "name": ..., "parameters": ..., "strict": ...}``, but
    vLLM's ``/tokenize`` chat path validates them against its Chat Completions
    schema (``{"type": "function", "function": {...}}``) and rejects the flat
    shape with a 422. This mirrors the conversion the ``vllm_model`` wrapper
    applies in ``responses_to_chat_completion_create_params`` at rollout time —
    keeping it identical here is what makes the system block tokenize the same
    way it did during generation.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        td = dict(t)
        td.pop("type", None)
        # vLLM Chat Completions tool schema does not accept ``strict``; the
        # rollout-time converter drops it for the same reason.
        td.pop("strict", None)
        out.append({"type": "function", "function": td})
    return out


async def tokenize_system_block(
    server_client: Any,
    model_server_name: str,
    system_text: str,
    tools: Optional[list[Any]] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> list[int]:
    """Render+tokenize a system-only message block by hitting the model server's
    ``/v1/tokenize`` endpoint.

    The endpoint forwards to vLLM's ``/tokenize``, which applies the same
    preprocessing (Pydantic field-stripping, chat-template invocation,
    ``add_special_tokens`` handling) as the rollout-time generation path. So
    these tokens are byte-equivalent to whatever the rollout's first system
    block was — modulo vLLM internal nondeterminism, which there shouldn't be
    for a deterministic chat-template render.
    """
    body: dict[str, Any] = {
        "messages": [{"role": "system", "content": system_text}],
        "add_generation_prompt": False,
    }
    if tools:
        body["tools"] = list(tools)
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs

    response = await server_client.post(
        server_name=model_server_name,
        url_path="/v1/tokenize",
        json=body,
    )
    response.raise_for_status()
    data = await response.json()
    tokens = data.get("tokens")
    if tokens is None:
        raise RuntimeError(f"/v1/tokenize did not return `tokens`; got keys {list(data.keys())}")
    return list(tokens)


@dataclass
class PrefixCheckResult:
    """Outcome of verifying that ``orig_sys_tokens`` is a clean prefix of a
    rollout's ``prompt_token_ids[K]``."""

    ok: bool
    """True iff the entire ``orig_sys_tokens`` matches the leading tokens."""

    divergence_index: int = -1
    """0-indexed position where the two streams first differ; -1 if the
    sequences match for the entire length of ``orig_sys_tokens`` or if the
    prompt is too short to even compare."""

    reason: str = ""
    """Stable, machine-parseable reason. Examples:
    ``"prompt_too_short:N<M"`` or ``"prefix_mismatch_at_index_N"``."""


def verify_system_prefix(
    prompt_token_ids: Sequence[int],
    orig_sys_tokens: Sequence[int],
) -> PrefixCheckResult:
    """Check that ``orig_sys_tokens`` matches the first L_orig tokens of
    ``prompt_token_ids``.

    Returns ``ok=True`` only if every position in ``orig_sys_tokens`` matches
    the corresponding token in ``prompt_token_ids``. If they diverge,
    ``divergence_index`` is the first mismatching index — useful for a
    decoded-snippet diagnostic on the rollout side.
    """
    L = len(orig_sys_tokens)
    if len(prompt_token_ids) < L:
        return PrefixCheckResult(
            ok=False,
            divergence_index=-1,
            reason=f"prompt_too_short:{len(prompt_token_ids)}<{L}",
        )
    for i in range(L):
        if prompt_token_ids[i] != orig_sys_tokens[i]:
            return PrefixCheckResult(
                ok=False,
                divergence_index=i,
                reason=f"prefix_mismatch_at_index_{i}",
            )
    return PrefixCheckResult(ok=True)


def splice_system_block(
    prompt_token_ids: Sequence[int],
    orig_sys_len: int,
    new_sys_tokens: Sequence[int],
) -> list[int]:
    """Replace the leading ``orig_sys_len`` tokens with ``new_sys_tokens`` and
    keep the rest verbatim."""
    if len(prompt_token_ids) < orig_sys_len:
        raise ValueError(
            f"prompt_token_ids has {len(prompt_token_ids)} tokens but "
            f"orig_sys_len is {orig_sys_len}; can't splice"
        )
    return list(new_sys_tokens) + list(prompt_token_ids[orig_sys_len:])


def decode_snippet(
    tokenizer: Any,
    tokens: Sequence[int],
    around: int,
    radius: int = 16,
) -> str:
    """Decode a small window around index ``around`` for diagnostic strings.
    Diagnostic-only — uses a local tokenizer that may drift from vLLM's. If
    it can't decode (bad tokens, missing tokenizer), returns a marker rather
    than crashing."""
    if not tokens or tokenizer is None:
        return ""
    n = len(tokens)
    around = max(0, min(around, n - 1))
    lo = max(0, around - radius)
    hi = min(n, around + radius)
    try:
        return tokenizer.decode(list(tokens[lo:hi]))
    except Exception:
        return f"<decode_failed:{lo}:{hi}>"
