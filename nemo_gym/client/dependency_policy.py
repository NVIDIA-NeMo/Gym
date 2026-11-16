# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed dependency failures adapted to the common outcome-policy selector."""

from typing import Annotated, ClassVar

from aiohttp import ClientConnectionError, ClientConnectorError, ClientPayloadError, ClientResponseError
from pydantic import Field

from nemo_gym.core.failure_kinds import (
    TRANSPORT_PEER_DROP,
    TRANSPORT_TIMEOUT,
    TRANSPORT_UNREACHABLE,
    FailureClassificationConfig,
    FailureStage,
    classify_failure,
)
from nemo_gym.core.outcome_policy import OutcomeAction, OutcomeDecision, OutcomePolicyConfig, choose_action
from nemo_gym.core.typed_exceptions import NgFailureEvent


class DependencyOutcomePolicyConfig(OutcomePolicyConfig):
    """Environment dependency defaults; retry means another episode within the collector's budget."""

    supported_actions: ClassVar[frozenset[OutcomeAction]] = frozenset({OutcomeAction.RETRY, OutcomeAction.FAIL})
    classification: FailureClassificationConfig = Field(default_factory=FailureClassificationConfig)
    by_failure_kind: dict[str, OutcomeAction] = Field(
        default_factory=lambda: dict.fromkeys(
            (TRANSPORT_UNREACHABLE, TRANSPORT_PEER_DROP, TRANSPORT_TIMEOUT), OutcomeAction.RETRY
        )
    )
    by_status_code: dict[Annotated[int, Field(ge=400, le=599)], OutcomeAction] = Field(
        default_factory=lambda: dict.fromkeys((408, 425, 429, *range(500, 600)), OutcomeAction.RETRY)
    )
    default_action: OutcomeAction = OutcomeAction.FAIL


def dependency_error_evidence(error: BaseException) -> tuple[str, int | None]:
    """Read this hop's evidence, without searching causes or constructing unobserved failure events."""
    if isinstance(error, NgFailureEvent):
        # The received event and the HTTP reply carrying it are separate observed facts.
        status = error.http_error.status if error.http_error is not None else error.http_status
        return error.failure_kind, status
    if isinstance(error, ClientResponseError):
        return "unknown", error.status
    if isinstance(error, ClientConnectorError):
        return TRANSPORT_UNREACHABLE, None
    # aiohttp socket/connect timeouts also inherit ClientConnectionError.
    if isinstance(error, TimeoutError):
        return TRANSPORT_TIMEOUT, None
    if isinstance(error, (ClientConnectionError, ClientPayloadError)):
        return TRANSPORT_PEER_DROP, None
    return "unknown", None


def dependency_outcome(
    error: BaseException, *, config: DependencyOutcomePolicyConfig, stage: FailureStage | None = None
) -> OutcomeDecision:
    """Select a dependency disposition; the caller owns retry execution and safety constraints."""
    kind, status = dependency_error_evidence(error)
    observed_stage = error.stage if isinstance(error, NgFailureEvent) and error.stage is not None else stage
    classification = classify_failure(
        failure_kind=kind, stage=observed_stage, http_status=status, config=config.classification
    )
    return OutcomeDecision(
        action=choose_action(failure_kind=kind, http_status=status, failure_class=classification, config=config),
        attribution=classification,
    )
