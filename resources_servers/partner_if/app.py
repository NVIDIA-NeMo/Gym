# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared IF verification of replies and live-agent captures, independent of task sources and native task rewards."""

import asyncio
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import time
from typing import Annotated, ClassVar

from fastapi import FastAPI, HTTPException
from pydantic import BeforeValidator, ConfigDict, Field, JsonValue, TypeAdapter, model_validator
from starlette.requests import Request
from starlette.responses import Response

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, current_rollout_id
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    CheckpointMode,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ReverifyMode,
    SimpleResourcesServer,
    observe_verifier,
    reject_partial_typed_seed,
)
from nemo_gym.telemetry import GymSpanGroup, trace_fn
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture, judge_failsafe
from resources_servers.partner_if.checks import register_modules
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import (
    NO_ANSWER,
    TRUNCATED,
    UNSCORABLE,
    UNSIGNED_CONTRACT,
    Evaluation,
    GradingContract,
)
from resources_servers.partner_if.grading_service import JudgeRejection, grade, unanswered
from resources_servers.partner_if.llm import MAX_JUDGE_INPUT_CHARS, PROTOCOL
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import LiveAgentCapture, live_agent_evidence, visible_events
from resources_servers.partner_if.receipt import masked_fields, scoring_fields
from resources_servers.partner_if.replay_protocol import (
    REPLAY_RECEIPT_KEY,
    PrefixReviewRequest,
    PrefixReviewResponse,
    canonical_replay_response,
    json_sha256,
    read_replay_receipt,
    replay_request_sha256,
)
from resources_servers.partner_if.replay_receipts import ReplayReceiptStore
from resources_servers.partner_if.schema import IFResult, VerifierMetadata, validate_input
from resources_servers.partner_if.verifier_fixture import create_fixture_server


class PartnerIFGradingConfig(BaseResourcesServerConfig):
    """The shared grader's settings, which every partner_if server has."""

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=8192, temperature=0
        )
    )
    judge_max_concurrency: int = Field(default=8, gt=0)
    # Characters of the judge's user message (the serialized rubrics and evidence); size it to the judge's
    # context window. A larger input is not sent: the checks that gate reward are judged alone, and any item
    # left out is evidence_too_large.
    judge_max_input_chars: int = Field(default=MAX_JUDGE_INPUT_CHARS, gt=0)
    allow_provisional_rubrics: bool = False
    # Importable modules whose CHECKERS add task-specific verifier IDs (see checks.register_modules).
    task_check_modules: list[str] = Field(default_factory=list)


class PartnerIFResourcesServerConfig(PartnerIFGradingConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    # Grade live_agent rows on the verification_input their agent sends. Rows never carry it, so such a
    # server reports reverify mode UNSUPPORTED.
    accept_verification_input: bool = False
    # Explicit opt-in: only replay envelopes below this root can request verifier-controlled continuation.
    ordering_artifacts_dir: Path | None = None
    ordering_max_policy_calls: int = Field(default=8, ge=1, le=64)
    ordering_max_prefixes: int = Field(default=6, ge=1, le=32)


UNSIGNED = "Rubric audit unsigned; enable allow_provisional_rubrics for provisional development scoring"


class VerifyRequest(BaseVerifyRequest):
    verifier_metadata: VerifierMetadata
    task_name: str | None = Field(default=None, exclude_if=lambda value: value is None)
    # Sent only for live_agent rows and graded with the response; rows never carry it.
    verification_input: LiveAgentCapture | None = Field(default=None, exclude=True)


class VerifyResponse(VerifyRequest, BaseVerifyResponse):
    # Judge failsafe fields and failure reports must survive durable replay receipt reuse.
    model_config = ConfigDict(extra="allow")

    if_result: IFResult
    judge_protocol: str = PROTOCOL
    judge_response: NeMoGymResponse | None = None
    judge_request: dict | None = None
    # Judge replies that failed validation before `judge_response`, each with its reason; the judge was asked again.
    judge_rejected_responses: list[JudgeRejection] = Field(default_factory=list, exclude_if=lambda value: not value)
    grading_result: Evaluation | None = None
    # Whether grading ran to completion, as Gym's evidence gate reads it: a masked rollout's did not.
    evaluation_completed: bool | None = None
    replay_artifacts_dir: str | None = Field(default=None, exclude_if=lambda value: value is None)
    replay_diagnostics: dict[str, JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def completion_follows_masking(self) -> "VerifyResponse":
        if self.evaluation_completed is None:
            self.evaluation_completed = not self.mask_sample
        return self


# Simple Agent marks a provider-completed reply that has neither a message nor a call as incomplete,
# with this reason. The reply is complete: it gives no answer.
AGENT_RELABELED_COMPLETIONS = frozenset({"incomplete_reasoning", "empty_output"})


class NoAnswer(Exception):
    """The reply gives nothing to grade, which is the model's own failure rather than a capture gap."""

    def __init__(self, failure_kind: str, reason: str):
        super().__init__(reason)
        self.failure_kind = failure_kind


def output_cap_cutoff(response: NeMoGymResponse) -> bool:
    details = response.incomplete_details
    return response.error is None and details is not None and details.reason == "max_output_tokens"


def is_complete(response: NeMoGymResponse) -> bool:
    if response.incomplete_details or response.error:
        return False
    reason = (response.metadata or {}).get("ng_termination_reason")
    return response.status == "completed" or (
        response.status == "incomplete" and reason in AGENT_RELABELED_COMPLETIONS
    )


def check_finished(response: NeMoGymResponse) -> None:
    """A reply cut off by the output-token cap is the model's failure; any other unfinished reply is unscorable."""
    if output_cap_cutoff(response):
        raise NoAnswer(TRUNCATED, "The reply was cut off by the output-token cap")
    if not is_complete(response):
        raise ValueError("Actor response is not complete")


def visible_answer(response: NeMoGymResponse) -> list[dict]:
    check_finished(response)
    messages, called = [], False
    for item in response.output:
        if item.type == "reasoning":
            continue  # API reasoning is not user-visible output.
        if item.type == "function_call":
            called = True
            continue
        if item.type != "message" or item.status != "completed":
            raise ValueError("Nontext or incomplete output requires another evidence adapter")
        if any(part.type not in {"output_text", "refusal"} for part in item.content):
            raise ValueError("Unsupported answer content")
        messages.append(item.model_dump(exclude_none=True))
    if called:
        raise NoAnswer(NO_ANSWER, "The reply calls a tool where a text answer is expected")
    if not any(
        (part.get("text") or part.get("refusal") or "").strip() for message in messages for part in message["content"]
    ):
        raise NoAnswer(NO_ANSWER, "The reply has no visible answer")
    return messages


def tool_response_evidence(params: dict, response: NeMoGymResponse) -> EvidenceBundle:
    """Capture one new response, including unexecuted calls, for the shared judge.

    A malformed argument is candidate behavior, not a capture failure. Frozen history stays
    context; reasoning and executed tool results are never substituted for the next response
    being graded.
    """
    check_finished(response)
    for item in response.output:
        if item.type == "reasoning":
            continue
        if item.type not in {"message", "function_call"}:
            raise ValueError("Next-call evidence cannot include executed tool results")
        if item.status not in {None, "completed"}:
            raise ValueError("Actor output item is not complete")
        if item.type == "message" and item.role != "assistant":
            raise ValueError("New response messages must belong to the assistant")
    events = visible_events([item.model_dump(exclude_none=True) for item in response.output])
    if not any(event.kind == "tool_call" or event.text.strip() for event in events):
        raise NoAnswer(NO_ANSWER, "The reply has neither a visible answer nor a tool call")
    return EvidenceBundle(
        context=params["input"],
        instructions=params.get("instructions"),
        tools=params.get("tools") or [],
        answer="\n".join(event.text for event in events if event.kind == "assistant"),
        trajectory=events,
    )


class PartnerIFGradingMixin:
    """The grading boundary every partner_if server shares: its config, its startup and its scoring policy."""

    config: PartnerIFGradingConfig

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        register_modules(self.config.task_check_modules)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)

    async def grade_response(
        self, contract: GradingContract, build_evidence: Callable[[], EvidenceBundle]
    ) -> dict[str, JsonValue]:
        """The Gym verify fields for one rollout. In order: an unsigned contract is masked; a reply that gives
        nothing to grade fails every item without the judge (`build_evidence` raises NoAnswer, TRUNCATED when
        the output-token cap cut the reply off); evidence that cannot be assessed (ValueError) is masked; any
        other evidence is graded."""
        if not contract.audit_signoff and not self.config.allow_provisional_rubrics:
            return masked_fields(contract, UNSIGNED_CONTRACT, UNSIGNED)
        try:
            evidence = build_evidence()
        except NoAnswer as missing:
            receipt = unanswered(contract, missing.failure_kind, str(missing))
        except ValueError as error:
            return masked_fields(contract, UNSCORABLE, str(error))
        else:
            receipt = await grade(
                contract,
                evidence,
                server_client=self.server_client,
                server_name=self.config.judge_model_server.name,
                semaphore=self._judge_semaphore,
                responses_create_params=self.config.judge_responses_create_params,
                max_input_chars=self.config.judge_max_input_chars,
            )
        return scoring_fields(contract, receipt)


class PartnerIFResourcesServer(PartnerIFGradingMixin, PartnerIFMetricsMixin, SimpleResourcesServer):
    # Ordinary verification is stateless. Replay opts into restart_only below: local durable receipts do not
    # make an in-flight replay portable to a replacement server or promise a live checkpoint restore.
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"

    config: PartnerIFResourcesServerConfig

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        if self.config.ordering_artifacts_dir is not None:
            return "restart_only"
        return super().checkpoint_mode_in_effect()

    async def get_reverify_mode(self) -> ReverifyMode:
        # A live agent's capture is not in the row, so reverify cannot grade the row again.
        if self.config.accept_verification_input:
            return ReverifyMode.UNSUPPORTED
        return await super().get_reverify_mode()

    async def seed_session(
        self,
        body: ResourcesSeedSessionRequest
        | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
        *,
        request: Request = None,
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Bind replay cookies durably while ordinary PartnerIF configurations keep stateless lifecycle.

        Implementing the base state hooks would opt every PartnerIF instance into stateful lifecycle, including
        unrelated graders. This opt-in uses the documented seed/close overrides and a benchmark-local store.
        """
        if self.config.ordering_artifacts_dir is None or not isinstance(body, ResourcesSeedSessionRequest):
            return await super().seed_session(body, request=request)
        if request is None:
            raise ResourcesSeedRejectedError("Replay typed seed requires an HTTP session")
        if body.responses_create_params is None:
            raise ResourcesSeedRejectedError("Replay typed seed requires the frozen actor parameters")
        try:
            params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body.responses_create_params)
        except ValueError as error:
            raise ResourcesSeedRejectedError("Replay typed seed actor parameters are invalid") from error
        store = self._replay_store()
        identity = store.session_receipt_id(body.resources_session_id)
        binding = {
            "session_id": body.resources_session_id,
            "episode_id": body.episode_id.model_dump(mode="json"),
            "task_id": body.task_id.model_dump(mode="json"),
            "task_data": body.task_data,
            "responses_create_params": params.model_dump(mode="json", exclude_unset=True, exclude_none=True),
        }
        async with store.locked(identity):
            previous = store.read(identity)
            if previous is not None:
                if previous.get("closed_until", 0) > time():
                    raise ResourcesSessionConflictError("Replay resources session is already closed")
                if not previous.get("closed_until") and previous.get("binding") != binding:
                    raise ResourcesSessionConflictError("Replay resources session is already bound to another seed")
            store.write(identity, {"receipt_id": identity, "state": "seed", "binding": binding})
            request.session[SESSION_ID_KEY] = body.resources_session_id
        return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Fence replay reviews, including close after a lost seed reply, under the review's own session lock."""
        if self.config.ordering_artifacts_dir is None:
            return await super().close_resources_session(body)
        store = self._replay_store()
        identity = store.session_receipt_id(body.resources_session_id)
        episode = body.episode_id.model_dump(mode="json")
        async with store.locked(identity):
            previous = store.read(identity)
            if previous is not None:
                owner = (
                    previous.get("close_episode_id")
                    if previous.get("closed_until", 0) > time()
                    else (previous.get("binding") or {}).get("episode_id") or previous.get("close_episode_id")
                )
                if owner != episode and (not previous.get("closed_until") or previous.get("closed_until", 0) > time()):
                    raise ResourcesSessionConflictError("Replay close belongs to another episode")
                if previous.get("closed_until", 0) > time():
                    return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)
            store.write(
                identity,
                (previous or {"receipt_id": identity, "state": "seed"})
                | {
                    "close_episode_id": episode,
                    "closed_until": time() + self.config.session_close_retry_window_seconds,
                },
            )
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    @staticmethod
    def _require_replay_seed(
        store: ReplayReceiptStore, session_id: str, body: PrefixReviewRequest, *, allow_closed: bool = False
    ) -> None:
        record = store.read(store.session_receipt_id(session_id))
        expected = {
            "session_id": session_id,
            "episode_id": body.episode_id.model_dump(mode="json"),
            "task_id": body.task_id.model_dump(mode="json"),
            "task_data": body.task_data,
            "responses_create_params": body.model_dump(mode="json")["responses_create_params"],
        }
        if record is None or record.get("state") != "seed" or record.get("binding") != expected:
            raise ResourcesSessionConflictError("Replay review differs from its durable typed Resources seed")
        if record.get("closed_until") and not allow_closed:
            raise ResourcesSessionConflictError("Replay resources session is already closed")

    def register_protocol_routes(self, app: FastAPI) -> None:
        """Keep stock lifecycle and ordinary verification, adding the benchmark's private review route."""
        super().register_protocol_routes(app)
        ordinary_route = next(route for route in app.routes if getattr(route, "path", None) == "/verify")
        ordinary_verify = ordinary_route.endpoint
        app.routes.remove(ordinary_route)
        enabled = self._review_observability_enabled()
        cached_verify = trace_fn(GymSpanGroup.VERIFY, "gym.verify")(
            observe_verifier(
                judge_failsafe(self.verify),
                enabled=enabled,
                verifier_name=self.config.name,
                uninstrumented_models=True,
            )
        )

        async def verify_endpoint(request: Request, body: VerifyRequest):
            from resources_servers.partner_if.ordering import is_ordering_response

            if is_ordering_response(body.response) or REPLAY_RECEIPT_KEY in (body.response.model_extra or {}):
                return await cached_verify(body=body, request=request)
            return await ordinary_verify(body=body, request=request)

        app.post("/verify", name="verify")(verify_endpoint)
        app.post("/replay/prefix-review", name="replay_prefix_review")(self.prefix_review)

    def _review_observability_enabled(self) -> bool:
        config = getattr(self.server_client, "global_config_dict", {})
        return TypeAdapter(bool).validate_python(config.get("observability_enabled", False))

    def _replay_store(self) -> ReplayReceiptStore:
        if self.config.ordering_artifacts_dir is None or not self.config.accept_verification_input:
            raise HTTPException(422, "Replay review requires explicitly configured live-agent artifacts")
        return ReplayReceiptStore(self.config.ordering_artifacts_dir)

    @staticmethod
    def _review_verify_body(body: PrefixReviewRequest, response: NeMoGymResponse) -> VerifyRequest:
        return VerifyRequest.model_validate(
            body.task_data
            | {
                "responses_create_params": body.responses_create_params.model_dump(
                    mode="json", exclude_unset=True, exclude_none=True
                ),
                "response": response.model_dump(mode="json"),
            }
        )

    async def _compute_prefix_review(self, body: PrefixReviewRequest) -> dict:
        from resources_servers.partner_if.ordering import verify_ordering

        return await verify_ordering(
            self._review_verify_body(body, body.response),
            artifacts_dir=self.config.ordering_artifacts_dir,
            max_policy_calls=self.config.ordering_max_policy_calls,
            max_prefixes=self.config.ordering_max_prefixes,
            review=body,
            verify=self._verify_response,
        )

    async def prefix_review(self, body: PrefixReviewRequest, request: Request) -> PrefixReviewResponse:
        """Review once, persist the observed reply, and replay it exactly after disconnects or restart."""
        store = self._replay_store()
        session_id = request.session.get(SESSION_ID_KEY)
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(422, "Replay review requires the Resources session cookie")
        rollout_id = current_rollout_id()
        if rollout_id is not None and rollout_id != body.episode_id.capture_key:
            raise HTTPException(409, "Replay review belongs to another episode attempt")
        if REPLAY_RECEIPT_KEY in (body.response.model_extra or {}):
            raise HTTPException(422, "A submitted prefix must not already carry a review receipt")
        verify_body = self._review_verify_body(body, body.response)
        contract = verify_body.verifier_metadata.partner_if
        if contract.evidence_mode != "live_agent":
            raise HTTPException(422, "Replay review task identity does not match its grading contract")
        scope = store.scope(session_id, body)
        receipt_id = store.receipt_id(scope)
        request_sha256 = replay_request_sha256(body)
        async with store.locked(store.session_receipt_id(session_id)), store.locked(receipt_id):
            self._require_replay_seed(store, session_id, body)
            record = store.read(receipt_id)
            if record is not None:
                if record.get("scope") != scope or record.get("request_sha256") != request_sha256:
                    raise HTTPException(409, "This replay prefix was already bound to different input")
                if record.get("state") == "complete":
                    return self._stored_prefix_review(record)
            else:
                if body.prefix_index > 1:
                    self._validate_previous_review(store, scope, body)
                record = {
                    "receipt_id": receipt_id,
                    "scope": scope,
                    "request_sha256": request_sha256,
                    "request": body.model_dump(mode="json"),
                    "state": "in_progress",
                }
                # This marker must reach disk before the first possible judge call.
                store.write(receipt_id, record)
                canonical = canonical_replay_response(body.response, receipt_id, request_sha256)
                observed = observe_verifier(
                    judge_failsafe(self._compute_prefix_review),
                    enabled=self._review_observability_enabled(),
                    verifier_name=self.config.name,
                    # Shared grading deliberately detaches model-call capture; do not infer a programmatic grade.
                    uninstrumented_models=True,
                )
                result = await observed(body.model_copy(update={"response": canonical}))
                if isinstance(result, Response):
                    if result.status_code >= 400:
                        raise HTTPException(result.status_code, "Replay prefix review failed; outcome is uncertain")
                    fields = json.loads(result.body)
                else:
                    fields = result
                if "if_result" not in fields:
                    fields = (
                        fields
                        | masked_fields(
                            contract,
                            fields.get("failure_kind") or "partner_if:replay_judge_error",
                            fields.get("failure_reason") or "Replay judge did not return a usable grading receipt",
                        )
                        | {"evaluation_completed": False, "next_parent_turn_limit": None}
                    )
                return self._complete_prefix_review(store, record, body, fields)

            # An abandoned marker cannot prove whether the judge ran. Mask it without dispatching again.
            async def uncertain_review(body: PrefixReviewRequest) -> dict:
                return masked_fields(
                    contract,
                    "partner_if:replay_review_uncertain",
                    "A prior review dispatch did not durably record its result; start a new episode attempt",
                ) | {"evaluation_completed": False, "next_parent_turn_limit": None}

            fields = await observe_verifier(
                uncertain_review,
                enabled=self._review_observability_enabled(),
                verifier_name=self.config.name,
                uninstrumented_models=True,
            )(body)
            return self._complete_prefix_review(store, record, body, fields)

    @staticmethod
    def _stored_prefix_review(record: dict) -> PrefixReviewResponse:
        result = record.get("result")
        if record.get("result_sha256") != json_sha256(result):
            raise HTTPException(409, "The durable replay review result is corrupt")
        return PrefixReviewResponse.model_validate(result)

    @staticmethod
    def _validate_previous_review(store: ReplayReceiptStore, scope: dict, body: PrefixReviewRequest) -> None:
        previous_scope = scope | {"prefix_index": body.prefix_index - 1}
        previous = store.read(store.receipt_id(previous_scope))
        if previous is None or previous.get("scope") != previous_scope or previous.get("state") != "complete":
            raise HTTPException(409, "Replay prefix has no completed preceding review")
        decision = PartnerIFResourcesServer._stored_prefix_review(previous)
        if decision.next_parent_turn_limit != body.prefix_index:
            raise HTTPException(409, "The preceding Resources review did not authorize this prefix")
        original = PrefixReviewRequest.model_validate(previous["request"])
        if (
            original.task_data != body.task_data
            or original.model_dump(mode="json")["responses_create_params"]
            != body.model_dump(mode="json")["responses_create_params"]
        ):
            raise HTTPException(409, "Replay task data or actor parameters changed between prefixes")

    @staticmethod
    def _complete_prefix_review(
        store: ReplayReceiptStore, record: dict, body: PrefixReviewRequest, fields: dict
    ) -> PrefixReviewResponse:
        receipt_id, request_sha256 = record["receipt_id"], record["request_sha256"]
        canonical = canonical_replay_response(
            body.response,
            receipt_id,
            request_sha256,
            masked=fields["mask_sample"] and not fields.get("evaluation_completed", False),
        )
        result = PrefixReviewResponse.model_validate(
            fields
            | {
                "response": canonical.model_dump(mode="json"),
                "receipt_id": receipt_id,
                "request_sha256": request_sha256,
            }
        )
        data = result.model_dump(mode="json")
        store.write(
            receipt_id,
            record
            | {
                "state": "complete",
                "result": data,
                "result_sha256": json_sha256(data),
                "response_sha256": json_sha256(canonical.model_dump(mode="json")),
            },
        )
        return result

    async def _reuse_replay_review(self, body: VerifyRequest, request: Request | None) -> VerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY) if request is not None else None
        if isinstance(session_id, str) and self.config.ordering_artifacts_dir is not None:
            store = self._replay_store()
            async with store.locked(store.session_receipt_id(session_id)):
                return await self._read_replay_review(body, request)
        return await self._read_replay_review(body, request)

    async def _read_replay_review(self, body: VerifyRequest, request: Request | None) -> VerifyResponse:
        """Final stock /verify validates a terminal receipt and never dispatches another judge."""
        from resources_servers.partner_if.ordering import OrderingPrefix

        try:
            store = self._replay_store()
            reference = read_replay_receipt(body.response)
            record = store.read(reference.id)
            if record is None or record.get("state") != "complete":
                raise ValueError("Terminal replay review receipt is missing or incomplete")
            review = self._stored_prefix_review(record)
            original = PrefixReviewRequest.model_validate(record["request"])
            if (
                reference.request_sha256 != record.get("request_sha256")
                or replay_request_sha256(original) != reference.request_sha256
            ):
                raise ValueError("Terminal replay review request binding differs")
            if review.next_parent_turn_limit is not None:
                raise ValueError("A pending replay prefix cannot be verified as terminal")
            session_id = request.session.get(SESSION_ID_KEY) if request is not None else None
            if session_id != record["scope"]["session_id"]:
                raise ValueError("Terminal replay review belongs to another Resources session")
            rollout_id = current_rollout_id()
            if rollout_id is not None and rollout_id != original.episode_id.capture_key:
                raise ValueError("Terminal replay review belongs to another episode attempt")
            self._require_replay_seed(store, session_id, original, allow_closed=True)
            if store.scope(session_id, original) != record["scope"]:
                raise ValueError("Terminal replay review scope differs")
            if json_sha256(body.response.model_dump(mode="json")) != record.get("response_sha256"):
                raise ValueError("Terminal replay response changed after Resources canonicalized it")
            if (
                body.responses_create_params.model_dump(mode="json", exclude_unset=True, exclude_none=True)
                != original.model_dump(mode="json")["responses_create_params"]
            ):
                raise ValueError("Terminal replay actor parameters changed")
            incoming = await request.json()
            expected = original.task_data
            protocol_keys = {
                "responses_create_params",
                "response",
                "ng_trajectory",
                "ng_agent_observations",
                "failure_report",
            }
            if any(incoming.get(key) != value for key, value in expected.items()) or set(
                incoming
            ) - protocol_keys != set(expected):
                raise ValueError("Terminal replay task data changed")
            if not review.mask_sample:
                OrderingPrefix.read(
                    self._review_verify_body(original, original.response),
                    self.config.ordering_artifacts_dir,
                    review=original,
                )
            fields = review.model_dump(
                mode="json",
                exclude={"request_sha256", "receipt_id", "next_parent_turn_limit", "ng_verifier_invocations"},
            )
            return VerifyResponse.model_validate(body.model_dump() | fields)
        except (OSError, ValueError, TypeError, KeyError, HTTPException) as error:
            return VerifyResponse.model_validate(
                body.model_dump()
                | masked_fields(body.verifier_metadata.partner_if, "partner_if:replay_invalid_receipt", str(error))
                | {"evaluation_completed": False}
            )

    async def verify(self, body: VerifyRequest, request: Request = None) -> VerifyResponse:
        contract = body.verifier_metadata.partner_if
        if contract.evidence_mode == "live_agent" and not self.config.accept_verification_input:
            raise HTTPException(422, "live_agent rows need a server with accept_verification_input: true")
        from resources_servers.partner_if.ordering import is_ordering_response

        if REPLAY_RECEIPT_KEY in (body.response.model_extra or {}) or is_ordering_response(body.response):
            return await self._reuse_replay_review(body, request)
        return await self._verify_response(body)

    async def _verify_response(self, body: VerifyRequest) -> VerifyResponse:
        """The ordinary evidence adapter and shared grading path, unchanged by continuation opt-in."""
        contract = body.verifier_metadata.partner_if

        def evidence() -> EvidenceBundle:
            params = body.responses_create_params.model_dump(exclude_none=True)
            if contract.evidence_mode == "live_agent":
                output = [item.model_dump(exclude_none=True) for item in body.response.output]
                bundle = live_agent_evidence(params, output, body.verification_input)
                # After the capture checks, so a capture fault is masked even in a reply that was cut off. Harbor
                # reports each run it grades as completed, including an OpenHands run stopped at its iteration cap.
                check_finished(body.response)
                return bundle
            validate_input(params, evidence_mode=contract.evidence_mode)
            if contract.evidence_mode == "tool_call_response":
                return tool_response_evidence(params, body.response)
            answer = visible_answer(body.response)
            return EvidenceBundle(
                context=params["input"],
                instructions=params.get("instructions"),
                tools=params.get("tools") or [],
                answer="\n".join(
                    part.get("text") or part.get("refusal") or "" for message in answer for part in message["content"]
                ),
            )

        return VerifyResponse(**body.model_dump(), **await self.grade_response(contract, evidence))


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=partial(create_fixture_server, PartnerIFResourcesServer, PartnerIFResourcesServerConfig),
    request_model=VerifyRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
)


if __name__ == "__main__":
    PartnerIFResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PartnerIFResourcesServer.run_webserver()
