# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic multimodal multi-turn fixtures for context-compaction tests.

The agent deliberately has no task semantics, action parser, or external
resource server. ``computer_use`` emits one meaningless deterministic screen
per turn. ``media_contract`` retains the focused zero/one/multiple/repeated/
reordered image cases used to test media ownership and ordering.
"""

import base64
import json
import logging
import random
import struct
import zlib
from typing import Any, Literal

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.context_compaction import (
    ContextCompactedResponse,
    ContextCompactedTransportResponse,
    ContextCompactionContract,
    ContextCompactionSession,
    PreparedContextCompactionCall,
    build_generation_contract,
    build_transport_response,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInput,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.visual_history import (
    ContextMeasurements,
    FinalizedChunkRecord,
    GuardOutcomeRecord,
    ObservedCompletion,
    PolicyDecisionRecord,
    RewriteBoundaryEvent,
    TransformationLineageDeltaRecord,
    VisualHistoryConfig,
)


logger = logging.getLogger(__name__)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _solid_png_data_url(rgb: tuple[int, int, int], size: int = 32) -> str:
    """Build a small deterministic RGB PNG without an image dependency."""
    scanline = b"\x00" + bytes(rgb) * size
    raw = scanline * size
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _observation(
    text: str,
    colors: list[tuple[int, int, int]],
    *,
    media_mode: Literal["images", "text_padding"] = "images",
) -> NeMoGymEasyInputMessage:
    if media_mode == "text_padding":
        # One 512x512 image becomes 256 projected media tokens for this
        # checkpoint. Preserve approximately the same context length while
        # removing the multimodal path, so log-prob diagnostics can separate
        # long-history effects from media-history effects.
        text += " pad" * (256 * len(colors))
        colors = []
    content: list[dict[str, Any]] = [
        {
            "type": "input_image",
            "image_url": _solid_png_data_url(color),
            "detail": "auto",
        }
        for color in colors
    ]
    content.append({"type": "input_text", "text": text})
    return NeMoGymEasyInputMessage(role="user", content=content)


def scripted_observations(
    media_mode: Literal["images", "text_padding"] = "images",
    *,
    fixture: Literal["computer_use", "media_contract"] = "media_contract",
    num_turns: int = 5,
    reverse_ordered_pair: bool = False,
) -> list[NeMoGymEasyInputMessage]:
    """Return one of the two deterministic context-compaction fixtures."""

    if num_turns < 1:
        raise ValueError("num_turns must be at least 1")
    if fixture == "computer_use":
        rng = random.Random(0)
        return [
            _observation(
                f"step={turn_index + 1}",
                [
                    (
                        rng.randrange(256),
                        rng.randrange(256),
                        rng.randrange(256),
                    )
                ],
                media_mode=media_mode,
            )
            for turn_index in range(num_turns)
        ]

    image_a = (220, 40, 40)
    ordered_pair = [(40, 80, 220), (230, 190, 30)]
    if reverse_ordered_pair:
        ordered_pair.reverse()
    base = [
        ("Turn 1: seed image A.", [image_a]),
        ("Turn 2: new image B.", [(40, 180, 60)]),
        ("Turn 3: ordered same-shaped images C then D.", ordered_pair),
        ("Turn 4: intentionally text only.", []),
        ("Turn 5: image A repeated exactly.", [image_a]),
    ]
    observations = []
    for turn_index in range(num_turns):
        cycle_index, base_index = divmod(turn_index, len(base))
        text, colors = base[base_index]
        if reverse_ordered_pair and base_index == 2:
            text = "Turn 3: ordered same-shaped images D then C."
        if cycle_index:
            text = f"Cycle {cycle_index + 1}, {text.lower()}"
        observations.append(
            _observation(
                text,
                colors,
                media_mode=media_mode,
            )
        )
    return observations


class ScriptedMultimodalAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    reward: float = 1.0
    reward_by_rollout_index: list[float] | None = Field(
        default=None,
        min_length=1,
    )
    fixture: Literal["computer_use", "media_contract"] = "computer_use"
    media_mode: Literal["images", "text_padding"] = "images"
    num_turns: int = Field(default=5, ge=1)
    empty_response_retries: int = Field(default=0, ge=0)
    reverse_ordered_pair: bool = False
    visual_history: VisualHistoryConfig = Field(default_factory=VisualHistoryConfig)


def _has_materializable_assistant_output(output_items: list[Any]) -> bool:
    """Return whether Responses-to-Chat conversion retains this model turn.

    Empty reasoning shells and empty assistant messages are intentionally
    omitted by ``ResponsesConverterState.flush_assistant``. Continuing the
    synthetic rollout after such a call would make its sampled-token evidence
    absent from the next prompt, so it could not form one valid training trace.
    """

    for item in output_items:
        payload = item.model_dump(mode="python") if hasattr(item, "model_dump") else item
        if not isinstance(payload, dict):
            continue
        item_type = payload.get("type")
        if item_type == "function_call":
            return True
        if item_type == "reasoning":
            if any(
                isinstance(part, dict) and str(part.get("text") or "").strip()
                for part in payload.get("summary") or []
            ):
                return True
            continue
        if payload.get("role") != "assistant":
            continue
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list) and any(
            isinstance(part, dict) and str(part.get("text") or "").strip()
            for part in content
        ):
            return True
    return False


class ScriptedMultimodalAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    context_compaction_rollout_id: str | None = None
    context_compaction_group_id: str | None = None
    context_compaction_task_id: str | None = None
    context_compaction_rollout_index: int | None = Field(default=None, ge=0)
    context_compaction_attempt_index: int | None = Field(default=None, ge=0)


class ScriptedMultimodalResponse(NeMoGymResponse):
    agent_input: NeMoGymResponseInput
    seed_obs: NeMoGymResponseInput = Field(default_factory=list)
    media_assets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    completion_evidence: list[ObservedCompletion] = Field(default_factory=list)
    final_policy_decision: PolicyDecisionRecord | None = None
    lineage_deltas: list[TransformationLineageDeltaRecord] = Field(default_factory=list)
    chunk_records: list[FinalizedChunkRecord] = Field(default_factory=list)
    boundary_events: list[RewriteBoundaryEvent] = Field(default_factory=list)
    guard_records: list[GuardOutcomeRecord] = Field(default_factory=list)
    context_compaction_contract: ContextCompactionContract | None = None


class ScriptedMultimodalVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: ContextCompactedTransportResponse | ScriptedMultimodalResponse


def _merge_usage(
    accumulated: NeMoGymResponseUsage | None,
    current: NeMoGymResponseUsage | None,
) -> NeMoGymResponseUsage | None:
    if current is None:
        return accumulated
    if accumulated is None:
        return current.model_copy(deep=True)
    accumulated.input_tokens += current.input_tokens
    accumulated.output_tokens += current.output_tokens
    accumulated.total_tokens += current.total_tokens
    accumulated.input_tokens_details.cached_tokens = 0
    accumulated.output_tokens_details.reasoning_tokens = 0
    return accumulated


def _validated_model_body(
    body: NeMoGymResponseCreateParamsNonStreaming,
    *,
    request_input: list[Any],
    required_prefix_token_ids: list[int] | None,
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Revalidate a materialized history at the Responses API boundary."""

    # Preserve the request's unset-field boundary.  The vLLM Responses adapter
    # serializes with ``exclude_unset=True`` and expects absent optional fields
    # (notably ``metadata``) to remain absent rather than become explicit nulls.
    payload = body.model_dump(mode="python", exclude_unset=True)
    payload.update(
        {
            "input": request_input,
            "required_prefix_token_ids": required_prefix_token_ids,
        }
    )
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)


class ScriptedMultimodalAgent(SimpleResponsesAPIAgent):
    config: ScriptedMultimodalAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> ScriptedMultimodalResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        agent_input = list(body.input)
        observations = scripted_observations(
            self.config.media_mode,
            fixture=self.config.fixture,
            num_turns=self.config.num_turns,
            reverse_ordered_pair=self.config.reverse_ordered_pair,
        )
        seed_obs = [observations[0]]
        transcript = [observations[0]]
        usage = None
        model_cookies = None
        last_response: NeMoGymResponse | None = None
        rollout_id = str(request.cookies.get("session", "scripted-request"))
        generation_contract = build_generation_contract(
            body=body,
            model_server=self.config.model_server,
            visual_history=self.config.visual_history,
        )
        context_session = (
            ContextCompactionSession(
                config=self.config.visual_history,
                rollout_id=rollout_id,
                generation_contract=generation_contract,
                initial_context=agent_input,
                seed_observations=seed_obs,
            )
            if self.config.visual_history.enabled
            else None
        )

        async def measure_context(
            call: PreparedContextCompactionCall,
        ) -> ContextMeasurements:
            nonlocal model_cookies
            prompt_token_count = 0
            guard_config = self.config.visual_history.guards
            if guard_config.max_total_tokens is not None:
                model_body = _validated_model_body(
                    body,
                    request_input=list(call.request_input),
                    required_prefix_token_ids=(
                        list(call.required_prefix_token_ids) if call.required_prefix_token_ids is not None else None
                    ),
                )
                tokenize_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/tokenize",
                    json=model_body,
                    cookies=model_cookies,
                )
                await raise_for_status(tokenize_response)
                tokenize_payload = await get_response_json(tokenize_response)
                tokens = tokenize_payload.get("tokens")
                if not isinstance(tokens, list) or not all(isinstance(token_id, int) for token_id in tokens):
                    raise RuntimeError("Model tokenize preflight returned invalid tokens")
                prompt_token_count = len(tokens)
                if tokenize_response.cookies:
                    model_cookies = tokenize_response.cookies

            active_image_count = len(call.prepared_history.view.media_ids)
            vision_tokens_per_image = guard_config.projected_vision_tokens_per_image or 0
            return ContextMeasurements(
                prompt_token_count=prompt_token_count,
                active_image_count=active_image_count,
                vision_token_count=(active_image_count * vision_tokens_per_image),
            )

        for turn_idx in range(len(observations)):
            legacy_request_input = agent_input + transcript
            request_input = legacy_request_input
            prepared_call = None
            if context_session is not None:
                prepared_call = await context_session.prepare_model_call(
                    legacy_request_input=legacy_request_input,
                    turn_id=turn_idx + 1,
                    measure_context=measure_context,
                )
                request_input = list(prepared_call.request_input)
            model_body = _validated_model_body(
                body,
                request_input=request_input,
                required_prefix_token_ids=(
                    list(prepared_call.required_prefix_token_ids)
                    if prepared_call is not None and prepared_call.required_prefix_token_ids is not None
                    else None
                ),
            )

            for empty_attempt in range(self.config.empty_response_retries + 1):
                model_http_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=model_body,
                    cookies=model_cookies,
                )
                await raise_for_status(model_http_response)
                model_response_json = await get_response_json(model_http_response)
                model_cookies = model_http_response.cookies
                try:
                    candidate_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as exc:
                    raise RuntimeError(
                        "Received an invalid response from model server: " + json.dumps(model_response_json)
                    ) from exc
                usage = _merge_usage(usage, candidate_response.usage)
                if _has_materializable_assistant_output(candidate_response.output):
                    last_response = candidate_response
                    break
                if empty_attempt == self.config.empty_response_retries:
                    raise RuntimeError(
                        "Scripted multimodal model produced no materializable assistant output "
                        f"at turn {turn_idx + 1} after {empty_attempt + 1} attempt(s)"
                    )
                logger.warning(
                    "Retrying empty scripted multimodal response at turn %d (attempt %d)",
                    turn_idx + 1,
                    empty_attempt + 1,
                )

            assert last_response is not None

            if context_session is not None:
                assert prepared_call is not None
                context_session.record_model_response(
                    call=prepared_call,
                    output_items=last_response.output,
                    finish_reason=(
                        last_response.incomplete_details.reason
                        if last_response.incomplete_details is not None
                        else None
                    ),
                )

            transcript.extend(last_response.output)
            if turn_idx + 1 < len(observations):
                next_observation = observations[turn_idx + 1]
                transcript.append(next_observation)
                if context_session is not None:
                    context_session.append_observation(
                        [next_observation],
                        turn_id=turn_idx + 1,
                        conditions_action_turn=turn_idx + 2,
                    )

        if context_session is not None:
            context_session.finalize()

        assert last_response is not None
        for key, value in (*request.cookies.items(), *(model_cookies or {}).items()):
            response.set_cookie(key, value)

        last_response.usage = usage
        if context_session is not None and context_session.authority_mode:
            compacted = context_session.build_response(
                last_response,
                output=transcript[1:],
                agent_input=agent_input,
                seed_obs=seed_obs,
            )
            return ScriptedMultimodalResponse.model_validate(compacted.model_dump())

        result = last_response.model_dump()
        result.update(
            {
                "output": transcript[1:],
                "usage": usage.model_dump() if usage is not None else None,
                "agent_input": agent_input,
                "seed_obs": seed_obs,
                "media_assets": (
                    context_session.semantic_history.media_arena.export() if context_session is not None else {}
                ),
                "completion_evidence": [],
                "final_policy_decision": (
                    context_session.final_policy_decision if context_session is not None else None
                ),
                "lineage_deltas": (context_session.lineage_deltas if context_session is not None else []),
                "chunk_records": [],
                "boundary_events": [],
                "guard_records": [],
                "context_compaction_contract": None,
            }
        )
        return ScriptedMultimodalResponse.model_validate(result)

    async def run(
        self,
        request: Request,
        body: ScriptedMultimodalAgentRunRequest,
    ) -> ScriptedMultimodalVerifyResponse:
        reward = self.config.reward
        if self.config.reward_by_rollout_index is not None:
            rollout_index = body.context_compaction_rollout_index
            if rollout_index is None:
                raise ValueError("reward_by_rollout_index requires context_compaction_rollout_index")
            if rollout_index >= len(self.config.reward_by_rollout_index):
                raise ValueError("context_compaction_rollout_index is outside reward_by_rollout_index")
            reward = self.config.reward_by_rollout_index[rollout_index]

        cookies = dict(request.cookies)
        if body.context_compaction_rollout_id is not None:
            cookies["session"] = body.context_compaction_rollout_id
        model_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(model_response)
        response = ScriptedMultimodalResponse.model_validate(await get_response_json(model_response))
        stamped_response = response.model_copy(
            update={
                "context_compaction_contract": (
                    response.context_compaction_contract.model_copy(
                        update={
                            "group_id": body.context_compaction_group_id,
                            "task_id": body.context_compaction_task_id,
                            "rollout_index": (body.context_compaction_rollout_index),
                            "attempt_index": (body.context_compaction_attempt_index),
                        }
                    )
                    if response.context_compaction_contract is not None
                    else None
                )
            }
        )
        if stamped_response.context_compaction_contract is not None:
            transport_response = build_transport_response(
                ContextCompactedResponse.model_validate(stamped_response.model_dump())
            )
        else:
            transport_response = stamped_response
        return ScriptedMultimodalVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=transport_response,
            reward=reward,
        )


if __name__ == "__main__":
    ScriptedMultimodalAgent.run_webserver()
