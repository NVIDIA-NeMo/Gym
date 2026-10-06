# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequential, token-blind context management composed with Gym's ServerClient."""

from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from inspect import isawaitable
from typing import Any

import orjson

from nemo_gym.config_types import ModelServerRef
from nemo_gym.context_management.config import ContextHistoryConfig, HistoryPolicyConfig
from nemo_gym.context_management.controller import (
    HistoryController,
    TurnChunkedHistoryController,
    evaluate_context_guards,
)
from nemo_gym.context_management.history import (
    ContextMeasurements,
    PreparedHistoryView,
    SemanticHistory,
    _view_digest,
    normalize_semantic_items,
)
from nemo_gym.context_management.policies import build_history_policy
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status, rollout_path_prefix


class ContextGuardRejected(RuntimeError):
    """The next action cannot satisfy the configured guards, even after compaction."""


def _input_items(body: NeMoGymResponseCreateParamsNonStreaming) -> Sequence[Any]:
    return [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input


class ContextManagedResponsesClient:
    """One logical rollout. A transport ambiguity is terminal; it is never retried here.

    Agents may pass their complete append-only history to ``create(body)`` or call
    ``append_observation(items)`` followed by ``create()``. Both use the same history.
    ``select_response`` permits bounded, deliberate resampling of definite responses.
    """

    def __init__(
        self,
        *,
        server_client: ServerClient,
        model_server: ModelServerRef,
        logical_rollout_id: str,
        config: ContextHistoryConfig,
        initial_request: NeMoGymResponseCreateParamsNonStreaming,
        seed_observations: Sequence[Any] = (),
        cookies: Any = None,
    ):
        # Validate identity without creating a second owner/segment namespace.
        if not logical_rollout_id or "/" in logical_rollout_id:
            raise ValueError("A nonempty capture attempt ID is required")
        if initial_request.stream:
            raise ValueError("Context management supports sequential non-streaming Responses only")
        extra = orjson.loads((initial_request.metadata or {}).get("extra_body") or "{}")
        if not isinstance(extra, dict) or extra.get("truncate_prompt_tokens") is not None:
            raise ValueError("CC requires policy-controlled compaction, not engine prompt truncation")
        if initial_request.truncation == "auto" or initial_request.context_management:
            raise ValueError("Provider-managed compaction cannot be combined with semantic context management")
        if initial_request.previous_response_id is not None or initial_request.conversation is not None:
            raise ValueError("Context management requires explicit full semantic history, not provider session state")
        self.server_client = server_client
        self.model_server = model_server
        self.logical_rollout_id = logical_rollout_id
        self.config = config.model_copy(deep=True)
        self._request = initial_request.model_copy(deep=True)
        self._request.input = []
        self.history = SemanticHistory(logical_rollout_id)
        self.history.append_items(
            _input_items(initial_request),
            turn_id=0,
            is_initial_context=True,
            conditions_action_turn=1 if not seed_observations else None,
        )
        self.history.append_items(seed_observations, turn_id=0, conditions_action_turn=1)
        self._seed_event_count = len(self.history.events)
        policy = build_history_policy(self.config.policy if self.config.enabled else HistoryPolicyConfig())
        self.controller = (
            TurnChunkedHistoryController(
                self.history, policy, actions_per_chunk=self.config.schedule.actions_per_chunk
            )
            if self.config.schedule.type == "turn_chunked_recency"
            else HistoryController(self.history, policy)
        )
        self.cookies = dict(cookies or {})
        self._last_accepted_response_id: str | None = None
        self._selected_ids: set[str] = set()
        self._step = 1
        self._model_calls = 0
        self._busy = False
        self._closed = False

    def _check_open(self) -> None:
        if self._closed or self._busy:
            raise RuntimeError("Context client is closed or already has a pending call")

    def append_observation(self, items: Sequence[Any]) -> None:
        self._check_open()
        self.history.append_items(items, turn_id=self._step - 1, conditions_action_turn=self._step)

    def _semantic_source(self) -> list[dict[str, Any]]:
        return [deepcopy(dict(event.item)) for event in self.history.events]

    @property
    def output_items(self) -> list[dict[str, Any]]:
        """Materialize the final conversation once, not a full prefix per action."""
        return self._semantic_source()[self._seed_event_count :]

    def _append_suffix(self, body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        settings = body.model_copy(deep=True)
        settings.input = []
        if settings.model_dump() != self._request.model_dump():
            raise ValueError("Rendering and generation settings must remain unchanged within a rollout")
        # History also accepts raw tool/seed dictionaries. Apply the same request
        # defaults and replay normalization before comparing with a validated body.
        source_request = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            self._request.model_dump() | {"input": self._semantic_source()}
        )
        source = normalize_semantic_items(_input_items(source_request))
        supplied = normalize_semantic_items(_input_items(body))
        if supplied[: len(source)] != source:
            raise ValueError("The agent's source history must be append-only; only the policy may rewrite its view")
        self.append_observation(supplied[len(source) :])

    def _call(self, prepared: PreparedHistoryView) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(
            self._request.model_dump() | {"input": list(prepared.view.items)}
        )

    async def _measure(self, prepared: PreparedHistoryView) -> ContextMeasurements:
        guards = self.config.guards
        prompt_tokens = 0
        if guards.max_total_tokens is not None:
            body = self._call(prepared)
            response = await self.server_client.post(
                server_name=self.model_server.name,
                url_path=f"/context/{self.logical_rollout_id}/measure",
                json=body,
                cookies=self.cookies,
                _retry=False,
            )
            await raise_for_status(response)
            value = await get_response_json(response)
            prompt_tokens = value.get("prompt_token_count")
            if type(prompt_tokens) is not int or prompt_tokens < 0:
                raise ValueError("Invalid context measurement")
        return ContextMeasurements(
            prompt_token_count=prompt_tokens,
            active_image_count=len(prepared.view.image_part_ids),
            vision_token_count=len(prepared.view.image_part_ids) * (guards.projected_vision_tokens_per_image or 0),
        )

    async def _prepare(self) -> PreparedHistoryView:
        prepared = self.controller.prepare(applies_to_step=self._step)
        checks = evaluate_context_guards(self.config.guards, await self._measure(prepared))
        exceeded = next((check for check in checks if check.exceeded), None)
        if exceeded is not None and isinstance(self.controller, TurnChunkedHistoryController):
            if self.controller.close_for_guard(guard_name=exceeded.guard_name):
                prepared = self.controller.prepare(applies_to_step=self._step)
                checks = evaluate_context_guards(self.config.guards, await self._measure(prepared))
        if any(check.exceeded for check in checks):
            raise ContextGuardRejected(f"Context guard rejected action {self._step}: {checks}")
        return prepared

    async def create(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming | None = None,
        *,
        select_response: Callable[[NeMoGymResponse], bool | Awaitable[bool]] | None = None,
    ) -> NeMoGymResponse:
        self._check_open()
        if body is not None:
            self._append_suffix(body)
        self._busy = True
        try:
            prepared = await self._prepare()
            request = self._call(prepared)
            for retry in range(self.config.max_response_retries + 1):
                if self._model_calls >= self.config.max_model_calls:
                    raise RuntimeError("Context model-call limit reached")
                self._model_calls += 1
                http_response = await self.server_client.post(
                    server_name=self.model_server.name,
                    url_path=f"{rollout_path_prefix(self.logical_rollout_id, token_capture=True)}/v1/responses",
                    json=request,
                    cookies=self.cookies,
                    _retry=False,
                )
                await raise_for_status(http_response)
                response = NeMoGymResponse.model_validate(await get_response_json(http_response))
                self.cookies.update(http_response.cookies)
                selected = True if select_response is None else select_response(response)
                if isawaitable(selected):
                    selected = await selected
                if type(selected) is not bool:
                    raise ValueError("Response selector must return bool")
                if selected:
                    self._accept(prepared, response)
                    return response
                if retry == self.config.max_response_retries:
                    raise RuntimeError("Response selection exhausted its bounded resampling budget")
            raise AssertionError("Unreachable response selection state")
        except BaseException:
            # A lost acknowledgement may have generated/captured an action. Never reuse this rollout.
            self._closed = True
            raise
        finally:
            self._busy = False

    def _accept(self, prepared: PreparedHistoryView, response: NeMoGymResponse) -> None:
        if not response.id or response.id in self._selected_ids:
            raise ValueError("Model response identity is missing or reused")
        first_event = len(self.history.events)
        self.history.append_items(response.output, turn_id=self._step)
        appended = self.history.events[first_event:]
        output = normalize_semantic_items(response.output)
        # Acknowledge the completed view, including the selected action. Comparing only
        # requests would silently undo removal of the immediately preceding reasoning.
        completed = replace(
            prepared.view,
            items=(*prepared.view.items, *output),
            descriptor=(*prepared.view.descriptor, *(f"part:{p.part_id}" for e in appended for p in e.parts)),
            image_part_ids=(
                *prepared.view.image_part_ids,
                *(p.part_id for e in appended for p in e.parts if p.kind == "image"),
            ),
        )
        acknowledged = replace(prepared, view=completed, view_digest=_view_digest(completed))
        if isinstance(self.controller, TurnChunkedHistoryController):
            self.controller.acknowledge_action(acknowledged, action_id=response.id, completion_id=response.id)
        else:
            self.controller.acknowledge(acknowledged)
        self._last_accepted_response_id = response.id
        self._selected_ids.add(response.id)
        self._step += 1

    def finish(self, response: NeMoGymResponse) -> None:
        """Validate the final accepted response and close the policy lifecycle."""
        self._check_open()
        if self._last_accepted_response_id is None or response.id != self._last_accepted_response_id:
            raise ValueError("Final logical response must identify the last selected model action")
        if isinstance(self.controller, TurnChunkedHistoryController):
            self.controller.finalize_terminal()
        self._closed = True
