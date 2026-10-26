# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared failure record for evaluation collectors and other run owners."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nemo_gym.episode_types import EpisodeFailure, EpisodeId


class RolloutFailure(BaseModel):
    """Associate one observed failure with a run and an episode attempt.

    ``source`` identifies who reported the failure. ``delivery`` describes the
    collector's evidence about delivery to the Environment Server: ``not_sent``
    requires evidence that the request was not sent, ``possibly_delivered`` covers
    an uncertain outcome, and ``delivered`` means delivery was established. It
    does not promise exactly-once execution or authorize transport replay. A
    failure reported by the Environment Server requires ``delivery="delivered"``.

    The nested failure uses the wire contract, not a second set of reason/stage
    fields. Protocol-specific diagnostics, partial responses, and training data
    belong outside this record. No reward is inferred from a failure. Persistence,
    conversion from legacy markers, and retry budgets belong to the caller.

    ``schema_version`` identifies the entire saved record, including nested
    contracts. Missing versions are treated as version 1 for pre-version records.
    Changes that older readers cannot parse (including added fields) require a
    version bump. Readers reject unsupported versions and unknown fields; callers
    must report an incompatible record or explicitly migrate it, never silently
    drop fields or reinterpret it as a completed rollout.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    episode_id: EpisodeId
    run_id: str = Field(min_length=1)
    source: Literal["environment", "collector"]
    delivery: Literal["not_sent", "possibly_delivered", "delivered"]
    failure: EpisodeFailure
    http_status: int | None = Field(default=None, ge=100, le=599)
    exception_type: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_delivery(self) -> Self:
        """Require established delivery for failures reported by the environment."""
        if self.source == "environment" and self.delivery != "delivered":
            raise ValueError("An environment-reported failure requires delivery='delivered'")
        return self
