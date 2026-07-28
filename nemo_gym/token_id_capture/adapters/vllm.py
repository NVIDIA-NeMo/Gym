# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""vLLM ``CaptureAdapter``: prefix-in entry, template splice, native extraction.

The engine-specific half of gate-authoritative capture for vLLM's
OpenAI-compatible serving layer, in three pieces:

* ``enter_prefix`` -- prefix ids enter a request through the
  ``required_prefix_token_ids`` field the NeMo-RL async worker's chat/tokenize
  request models already understand; the worker's ``preprocess_chat`` override
  feeds them to :func:`replace_prefix_tokens`.
* :func:`replace_prefix_tokens` -- the template splice, relocated verbatim
  from ``nemo_rl/models/generation/vllm/vllm_worker_async.py`` so every
  framework hosting this adapter shares one implementation. It keeps the
  model's exact prior tokens up to EOS and resumes from the freshly rendered
  template after it, preserving the monotonic-tokens property under
  retokenization drift.
* ``extract_prompt_ids`` / ``extract_generation`` -- read the exact engine
  prompt and the generated ids + logprobs off the final (non-streaming)
  chat-completion payload inside the worker process: no second ``/tokenize``
  round trip and no token transit through gate HTTP. The serving hookup must
  attach the engine's final ``prompt_token_ids`` (known at preprocess time,
  post-splice) onto the response payload before extraction.

This module deliberately imports nothing from vLLM: it manipulates duck-typed
request/response payloads, so it stays importable (and unit-testable) without
an engine present.
"""

from __future__ import annotations

from typing import Any

from nemo_gym.token_id_capture.staging.protocols import CaptureAdapter


# The request field the NeMo-RL async vLLM worker's request models carry and
# its preprocess_chat override consumes (also how today's echoed-token path
# enters); the serving hookup key for the engine's final prompt ids.
PREFIX_IDS_FIELD = "required_prefix_token_ids"
PROMPT_IDS_FIELD = "prompt_token_ids"


def replace_prefix_tokens(
    tokenizer: Any,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> list[int]:
    """Splice the model's exact prior tokens in front of the fresh template suffix.

    Relocated verbatim from NeMo-RL's async vLLM worker (where it ran the
    echoed-token path); see that history for the full rationale. Fixes up the
    chat-template-tokenized message history to match the model's own output
    tokenization up to the last assistant turn, preserving the monotonic
    tokens property for multi-turn training.

    RL training frameworks train on token ids, but an OpenAI-compatible server
    communicates in detokenized text; a prior generation can re-tokenize
    differently (e.g. inconsistent whitespace around tool-call special
    tokens), which mis-aligns token sequences across calls and makes logprobs
    off-policy. The splice keeps the exact prior model tokens up to EOS and
    resumes from the template after that EOS.

    Example (turn-by-turn, concise; eos_token_id = 2):
        Turn 1:
            - prefill_T1 (template prefill) = [11,12,13,40,41]
            - model output = [220,17,2]  # decodes to " 4" + EOS
            - model_prefix_token_ids = prefill_T1 + model output
              => [11,12,13,40,41,220,17,2]

        Turn 2 (template retokenizes prior assistant text differently):
            - template_prefix_token_ids = [11,12,13,40,41,1001,2]  # 1001 decodes to " 4"
            - template_token_ids = [11,12,13,40,41,1001,2,21,22,40,41]

        replace_prefix_tokens keeps the exact prior model tokens up to EOS and
        resumes from the template after that EOS:
            output => [11,12,13,40,41,220,17,2,21,22,40,41]
    """
    if not model_prefix_token_ids:
        return template_token_ids

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids:
        # We are not always guaranteed that the model outputs an EOS token as the stop criteria of the previous model call e.g. when the model reaches max_tokens.
        # And since chat templates will always add one for us, we just cut the model input to right before the EOS token ID (if applicable)
        if model_prefix_token_ids[-1] == eos_token_id:
            model_cut_end -= 1

    # Assert here to prepare for the logic below
    assert len(template_token_ids) > len(
        template_prefix_token_ids
    ), f"""Found possibly non-monotonically increasing trajectory!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}
"""

    # We take everything starting with the EOS token ID.
    template_cut_start = -1
    for pos in reversed(range(len(template_prefix_token_ids))):
        if template_token_ids[pos] == eos_token_id:
            template_cut_start = pos
            break

    # This should never be the case, but
    assert (
        template_cut_start >= 0
    ), f"""No EOS token ID found in the chat-templated messages!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"""

    return model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]


def extract_generation_token_info(choice: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Read generated token ids and logprobs from either supported vLLM
    chat-completion choice shape (ported from the sync prototype's
    ``rollout_writer.py``).

    Prefers the message token fields when a serving layer attached them;
    otherwise falls back to ``choice.logprobs.content`` entries, whose
    ``token`` values are ``"token_id:<id>"`` strings when the server runs
    with ``return_tokens_as_token_ids=True`` (an in-process read of the
    engine's own logprob output -- no detokenized-text round trip).
    """
    message = choice.get("message") or {}
    if "generation_token_ids" in message and "generation_log_probs" in message:
        raw_ids = message["generation_token_ids"]
        raw_logprobs = message["generation_log_probs"]
    else:
        content_logprobs = (choice.get("logprobs") or {}).get("content")
        if content_logprobs is None:
            raise ValueError("vLLM response contained neither message token fields nor choice.logprobs.content")
        raw_ids = [item["token"] for item in content_logprobs]
        raw_logprobs = [item["logprob"] for item in content_logprobs]
    token_ids = [int(str(token_id).removeprefix("token_id:")) for token_id in raw_ids]
    logprobs = [float(value) for value in raw_logprobs]
    if len(token_ids) != len(logprobs):
        raise ValueError(
            f"generated token and log-probability lengths differ: {len(token_ids)} != {len(logprobs)}"
        )
    return token_ids, logprobs


class VLLMCaptureAdapter(CaptureAdapter):
    """The vLLM implementation of the engine seam the capture core drives."""

    def enter_prefix(self, request_payload: dict[str, Any], prefix_ids: list[int]) -> dict[str, Any]:
        """Attach exact prefix ids to a chat request (token-in mode).

        Mutates and returns the payload: ``required_prefix_token_ids`` is the
        field the NeMo-RL worker's request models declare and its
        ``preprocess_chat`` override splices via :func:`replace_prefix_tokens`.
        """
        request_payload[PREFIX_IDS_FIELD] = list(prefix_ids)
        return request_payload

    def extract_prompt_ids(self, response_payload: dict[str, Any]) -> list[int]:
        """The exact prompt ids the engine ran on (post-splice).

        vLLM's OpenAI response does not carry them, so the serving hookup
        attaches the final engine prompt (known at preprocess time) onto the
        payload under ``prompt_token_ids`` before completing the call.
        """
        prompt_ids = response_payload.get(PROMPT_IDS_FIELD)
        if prompt_ids is None:
            raise ValueError(
                "response payload carries no prompt_token_ids; the serving hookup must attach "
                "the engine's final prompt ids before complete_call_from_response"
            )
        return [int(token_id) for token_id in prompt_ids]

    def extract_generation(self, response_payload: dict[str, Any]) -> tuple[list[int], list[float]]:
        """Generated ids + logprobs off the final non-streaming response."""
        choices = response_payload.get("choices") or []
        if len(choices) != 1:
            raise ValueError(f"token capture requires exactly one choice, got {len(choices)}")
        return extract_generation_token_info(choices[0])
