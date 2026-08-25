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
"""Multi-turn text-action agent for the Gym-V resources server.

The agent extracts the model's action from the LAST `\\boxed{...}` token in
the assistant's plain text and posts to Gym-V's `/step` endpoint with
`action_string`. Pairs with `resources_servers/gym_v/`.
"""

import hashlib
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from time import time
from typing import Any, cast

import aiohttp
from pydantic import ConfigDict, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInput,
    NeMoGymResponseOutputItem,
)
from resources_servers.gym_v.schemas import (
    GymVAgentVerifyRequest,
    GymVAgentVerifyResponse,
    GymVEnvStateEasyInputMessage,
    GymVNeMoGymResponse,
    GymVSeedSessionResponse,
    GymVStepRequest,
    GymVStepResponse,
    GymVTaskRow,
)


# Default system prompt: the paired gymv_agent imports this at runtime
# and injects it as the first system message when the incoming request
# doesn't already provide one.
DEFAULT_SYSTEM_PROMPT = """\
You are playing a game. Each turn, you receive an image of the current game
state and a short text description. Read the rules in the first user message
carefully — they specify the action grammar and the goal.

For every turn:
1. Think briefly inside the <think> block in 1-2 short sentences. Just state
   your observation and decision — do NOT enumerate grid rows, list options,
   or restate history.
2. Close the think block with </think>, then immediately write your final
   action wrapped in \\boxed{...} on the LAST line.
   The exact format inside \\boxed{...} is dictated by the env's rules.

Only the LAST \\boxed{...} in your response is sent to the game.
"""


logger = logging.getLogger(__name__)


def _debug_jsonl_path(component: str) -> Path | None:
    debug_dir = os.environ.get("NEMO_RL_DEBUG_RESPONSES_PIPELINE_DIR")
    if not debug_dir:
        return None
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{component}.jsonl"


def _preview_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _seq_summary(values: Any, *, limit: int = 12) -> dict[str, Any]:
    if values is None:
        return {"present": False, "len": 0}
    if not isinstance(values, list):
        return {"present": True, "type": type(values).__name__}
    return {
        "present": True,
        "len": len(values),
        "head": values[:limit],
        "tail": values[-limit:] if len(values) > limit else [],
    }


def _image_url_summary(image_url: Any) -> dict[str, Any]:
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if not isinstance(image_url, str):
        return {"present": False, "type": type(image_url).__name__}
    return {
        "present": True,
        "len": len(image_url),
        "sha256_16": hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16],
        "prefix": image_url[:32],
    }


def _content_summary(content: Any) -> Any:
    if isinstance(content, str):
        return {"kind": "str", "len": len(content), "preview": _preview_text(content)}
    if not isinstance(content, list):
        return {"kind": type(content).__name__, "repr": _preview_text(content)}
    parts = []
    for part in content:
        if hasattr(part, "model_dump"):
            part = part.model_dump(mode="json")
        if not isinstance(part, dict):
            parts.append({"kind": type(part).__name__, "repr": _preview_text(part)})
            continue
        summary = {"type": part.get("type"), "keys": sorted(part.keys())}
        if "text" in part:
            text = part.get("text")
            summary["text_len"] = len(text) if isinstance(text, str) else None
            summary["text_preview"] = _preview_text(text)
        if "image_url" in part:
            summary["image_url"] = _image_url_summary(part.get("image_url"))
        parts.append(summary)
    return {"kind": "list", "len": len(content), "parts": parts}


def _message_summary(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        message = message.model_dump(mode="json")
    if not isinstance(message, dict):
        return {"type": type(message).__name__, "repr": _preview_text(message)}
    return {
        "keys": sorted(message.keys()),
        "id": message.get("id"),
        "role": message.get("role"),
        "type": message.get("type"),
        "content": _content_summary(message.get("content")),
        "summary": _content_summary(message.get("summary")),
        "prompt_token_ids": _seq_summary(message.get("prompt_token_ids")),
        "generation_token_ids": _seq_summary(message.get("generation_token_ids")),
        "generation_log_probs": _seq_summary(message.get("generation_log_probs")),
    }


def _debug_dump(component: str, event: str, payload: dict[str, Any]) -> None:
    path = _debug_jsonl_path(component)
    if path is None:
        return
    row = {"event": event, "created_at": time(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


# Matches the last `\boxed{...}` in the assistant's plain text.
BOXED_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)


class GymVAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int | None = Field(
        default=None,
        description=(
            "Hard cap on rollout turns. Per-env horizon_cap in the task row is "
            "the primary enforcement point; this is a global backstop."
        ),
    )
    return_transitions: bool = Field(
        default=False,
        description="Pinned False — inspector consumes a flat output.",
    )
    max_total_sequence_length: int | None = Field(
        default=None,
        description=(
            "If set, the rollout will stop when the agent state exceeds this "
            "length. If not set, will rely on a vLLM exception to tell us when "
            "we've exceeded the model's token limit. Setting this simply avoids "
            "that exception."
        ),
    )
    done_if_no_boxed_answer: bool = Field(
        default=False,
        description=(
            "When False, a model response with no \\boxed{} match triggers a "
            "client-side recovery user message (no /step call); when True, the "
            "rollout terminates instead."
        ),
    )
    re_emit_rules_each_turn: bool = Field(
        default=False,
        description=(
            "If True, prepend a one-line action-vocabulary summary to every env "
            "user turn obs. Default off matches the pattern of putting rules in "
            "the first turn only."
        ),
    )
    rules_summary_template: str = Field(
        default="Reminder: respond with your reasoning, then write your action as \\boxed{...}.",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description=(
            "System prompt. Injected agent-side (in `responses()`) as the first "
            "message of the conversation if the incoming JSONL row doesn't "
            "already start with a system message. Default is DEFAULT_SYSTEM_PROMPT "
            "from resources_servers/gym_v/app.py — keeping the prompt in agent "
            "code (not per-row JSONL) means a single edit changes every rollout. "
            "Override via the agent's YAML config when you want a per-experiment "
            "variant without regenerating data."
        ),
    )


class GymVAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class GymVAgent(SimpleResponsesAPIAgent):
    config: GymVAgentConfig

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=5))
    async def _seed_session(self, task_idx: int, task_row: GymVTaskRow | None) -> GymVSeedSessionResponse:
        payload = {"task_idx": task_idx}
        if task_row is not None:
            payload["task_row"] = task_row.model_dump(mode="json")
        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=payload,
        )
        reset_response.raise_for_status()
        seed_session_response = GymVSeedSessionResponse.model_validate(await reset_response.json())
        if not seed_session_response.obs:
            raise ValueError("No observations in seed session response")
        return seed_session_response

    @staticmethod
    def _extract_assistant_text(output: list[NeMoGymResponseOutputItem]) -> str:
        """Concatenate assistant text content parts across all assistant
        messages in the response, in order.
        """
        texts: list[str] = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            if getattr(item, "role", None) != "assistant":
                continue
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    text_value = getattr(part, "text", None)
                    if isinstance(text_value, str):
                        texts.append(text_value)
            elif isinstance(content, str):
                texts.append(content)
        return "\n".join(texts)

    @staticmethod
    def _extract_boxed(text: str) -> str | None:
        """Return the LAST `\\boxed{...}` capture in `text`, stripped.

        Reasoning models often write a `\\boxed{example}` earlier in their CoT
        and a final `\\boxed{decision}` at the end, and the last match is the
        one that counts. Returns None for both "no `\\boxed{}` found" and
        "explicitly empty `\\boxed{}`" — the latter is treated as a
        recoverable failure rather than passed verbatim to /step (where it
        would fail the env's parser with a less useful error message).
        """
        matches = BOXED_PATTERN.findall(text)
        if not matches:
            return None
        return matches[-1].strip() or None

    def _no_boxed_recovery(self) -> NeMoGymEasyInputMessage:
        return NeMoGymEasyInputMessage(
            role="user",
            content=(
                "I did not find a \\boxed{...} answer in your response. "
                "Please put your final action inside \\boxed{...} on the last line."
            ),
        )

    def _maybe_prepend_system_prompt(
        self, input_messages: Sequence[NeMoGymEasyInputMessage]
    ) -> list[NeMoGymEasyInputMessage]:
        """Prepend `self.config.system_prompt` as the first system message.

        Idempotent: if `input_messages` already begins with a system-role
        message, returns the list unchanged. This lets per-row JSONLs
        override the agent's default prompt by supplying their own system
        message. If `system_prompt` is empty, no message is prepended.
        """
        prompt = self.config.system_prompt
        if not prompt:
            return list(input_messages)
        existing = list(input_messages)
        if existing and isinstance(existing[0], dict) and existing[0].get("role") == "system":
            return existing
        if existing and hasattr(existing[0], "role") and getattr(existing[0], "role") == "system":
            return existing
        sys_msg = NeMoGymEasyInputMessage(role="system", content=prompt)
        return [sys_msg, *existing]

    def _maybe_inject_rules_summary(self, obs: Sequence[NeMoGymEasyInputMessage]) -> list[NeMoGymEasyInputMessage]:
        if not self.config.re_emit_rules_each_turn:
            return list(obs)
        # Append a fresh reminder AFTER the env-state message rather than
        # mutating it: env-state messages are the single source of truth for
        # env state, and per-turn env text often carries its own action-format
        # hint that would otherwise win on recency over a pre-pended reminder.
        reminder = NeMoGymEasyInputMessage(role="user", content=self.config.rules_summary_template)
        return [*obs, reminder]

    @staticmethod
    def _task_row_from_request(req: GymVAgentRunRequest) -> GymVTaskRow | None:
        payload = req.model_dump(mode="json")
        if "env_id" not in payload or "seed" not in payload:
            return None
        try:
            return GymVTaskRow.model_validate(payload)
        except ValidationError:
            logger.exception("Incoming run request had task-row fields but failed validation.")
            raise

    async def responses(self, req: GymVAgentRunRequest) -> GymVNeMoGymResponse:
        task_row = self._task_row_from_request(req)
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        # Inject the agent's system prompt as the first message of the
        # conversation, unless the caller already provided one. The OpenAI
        # Responses → chat-completions converter in vllm_model silently drops
        # the top-level `instructions` field, so embedding the prompt as a
        # system *message* is the only way to actually deliver it.
        body.input = self._maybe_prepend_system_prompt(body.input)

        seed_session_response = await self._seed_session(req.task_idx, task_row)

        # Apply the rules-summary reminder to turn-1's seed observation too
        # (when re_emit_rules_each_turn is on), not just /step responses;
        # otherwise the very first turn — the most format-sensitive one —
        # sees the env's action-format hint without a counter-reminder.
        seed_obs = self._maybe_inject_rules_summary(seed_session_response.obs)
        agent_state = body.model_copy(
            update={
                "input": body.input + list(seed_obs),
                "tools": list(body.tools or []),
            }
        )
        _debug_dump(
            "gymv_agent",
            "initial_agent_state",
            {
                "task_idx": req.task_idx,
                "env_id": seed_session_response.env_id,
                "body_input": [_message_summary(m) for m in body.input],
                "seed_obs": [_message_summary(m) for m in seed_obs],
                "agent_state_input_len": len(agent_state.input),
                "max_output_tokens": body.max_output_tokens,
            },
        )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        agent_state_history: list[NeMoGymResponseInput] = []
        # The initial observation is returned separately as `seed_obs`. Keep it
        # out of the flat output so downstream multimodal consumers see it once.
        all_messages: list[NeMoGymResponseOutputItem] = []
        model_server_cookies = None

        step = 0
        try:
            while True:
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
                step += 1
                successful_transition = True

                try:
                    raw_model_response = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json=agent_state,
                        cookies=model_server_cookies,
                    )
                    raw_model_response.raise_for_status()
                    model_server_cookies = raw_model_response.cookies
                    model_response_json = await raw_model_response.json()
                    _debug_dump(
                        "gymv_agent",
                        "raw_model_response_json",
                        {
                            "task_idx": req.task_idx,
                            "env_id": env_id,
                            "step": step,
                            "response_keys": sorted(model_response_json.keys())
                            if isinstance(model_response_json, dict)
                            else None,
                            "output": [_message_summary(item) for item in model_response_json.get("output", [])]
                            if isinstance(model_response_json, dict)
                            else None,
                            "usage": model_response_json.get("usage")
                            if isinstance(model_response_json, dict)
                            else None,
                            "incomplete_details": model_response_json.get("incomplete_details")
                            if isinstance(model_response_json, dict)
                            else None,
                        },
                    )
                except (json.JSONDecodeError, aiohttp.ClientResponseError) as e:
                    logger.warning(f"Error calling /v1/responses: {e!r}. Response: {raw_model_response.text!r}.")
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    logger.warning(f"Error validating model response: {e!r}. Response: {model_response_json!r}.")
                    break

                model_output = model_response.output
                assistant_text = self._extract_assistant_text(model_output)
                action_string = self._extract_boxed(assistant_text)
                _debug_dump(
                    "gymv_agent",
                    "validated_model_output",
                    {
                        "task_idx": req.task_idx,
                        "env_id": env_id,
                        "step": step,
                        "output": [_message_summary(item) for item in model_output],
                        "assistant_text_len": len(assistant_text),
                        "assistant_text_preview": _preview_text(assistant_text),
                        "extracted_action": action_string,
                    },
                )

                done = False
                obs: Sequence[GymVEnvStateEasyInputMessage | NeMoGymEasyInputMessage]
                if action_string is None:
                    if self.config.done_if_no_boxed_answer:
                        done = True
                        obs = []
                    else:
                        # Synthesize a recovery user message client-side and do
                        # NOT call /step. Server-side recovery only fires when
                        # /step is reached with garbage; here we have nothing
                        # to send.
                        obs = [self._no_boxed_recovery()]
                        successful_transition = False
                else:
                    step_request = GymVStepRequest(env_id=env_id, action_string=action_string)
                    raw_env_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/step",
                        json=step_request.model_dump(exclude_none=True),
                    )
                    env_response = GymVStepResponse.model_validate(await raw_env_response.json())
                    obs = self._maybe_inject_rules_summary(env_response.obs)
                    done = env_response.done
                    _debug_dump(
                        "gymv_agent",
                        "env_step_response",
                        {
                            "task_idx": req.task_idx,
                            "env_id": env_id,
                            "step": step,
                            "action_string": action_string,
                            "done": done,
                            "obs": [_message_summary(m) for m in obs],
                        },
                    )

                agent_state = agent_state.model_copy(update={"input": agent_state.input + model_output + list(obs)})
                if self.config.return_transitions:
                    agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))
                else:
                    all_messages.extend(model_output)
                    # Skip the terminal env obs: on a done=True step, vLLM
                    # never issued a follow-up /v1/responses call to see this
                    # observation, so including it in response.output creates
                    # a phantom trailing user turn that downstream token-level
                    # reconstructors (e.g. NeMo-RL's _process_full_multimodal_
                    # trajectory) will fail to align against vLLM's tokens.
                    if successful_transition and not done:
                        all_messages.extend(obs)

                if done:
                    break

        finally:
            await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close",
                json={"env_id": env_id},
            )

        assert model_response is not None, (
            "Rollout crashed or terminated before first transition completed, cannot proceed."
        )

        output_overrides = {
            "env_id": env_id,
            "group_id": str(req.task_idx),
            "contains_transitions": self.config.return_transitions,
            # seed_obs is the post-rules-injection initial observation that
            # vLLM saw before the first model call. It stays separate from
            # output so multimodal consumers do not process it twice.
            "seed_obs": (
                [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in seed_obs]
                if not self.config.return_transitions
                else None
            ),
            # Effective initial input as vLLM saw it on the first
            # /v1/responses call: `body.input` here has already been mutated by
            # _maybe_prepend_system_prompt (and any string→message coercion).
            # NeMo-RL uses this to rebuild the exact token sequence for
            # full-trajectory multimodal postprocess — the raw JSONL row's
            # input would be missing the system prompt and produce a
            # token-count mismatch against vLLM.
            "agent_input": [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in body.input],
            "output": (agent_state_history if self.config.return_transitions else all_messages),
        }
        _debug_dump(
            "gymv_agent",
            "final_response_before_validation",
            {
                "task_idx": req.task_idx,
                "env_id": env_id,
                "return_transitions": self.config.return_transitions,
                "seed_obs": [_message_summary(item) for item in (output_overrides["seed_obs"] or [])],
                "output": [_message_summary(item) for item in output_overrides["output"]]
                if isinstance(output_overrides["output"], list)
                else None,
            },
        )
        response = GymVNeMoGymResponse.model_validate(model_response.model_dump() | output_overrides)
        _debug_dump(
            "gymv_agent",
            "final_response_after_validation",
            {
                "task_idx": req.task_idx,
                "env_id": env_id,
                "output": [_message_summary(item) for item in response.model_dump(mode="json").get("output", [])],
            },
        )
        return response

    async def run(self, body: GymVAgentRunRequest) -> GymVAgentVerifyResponse:
        try:
            response = await self.responses(body)
            verify_request = GymVAgentVerifyRequest.model_validate(
                {
                    "responses_create_params": body.responses_create_params,
                    "response": response.model_dump(),
                }
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
            )
            return GymVAgentVerifyResponse.model_validate(await verify_response.json())
        except Exception:
            logger.exception("Error in run")
            raise


if __name__ == "__main__":
    GymVAgent.run_webserver()
