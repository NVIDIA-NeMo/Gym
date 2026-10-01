# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared failure record for evaluation collectors and other run owners."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.episode_types import EpisodeFailure, EpisodeId


class RolloutFailure(BaseModel):
    """Associate one observed failure with a run and an episode attempt.

    ``source`` identifies who reported the failure. ``delivery`` describes the
    collector's evidence about delivery to the Environment Server: ``not_sent``
    requires evidence that the request was not sent, ``possibly_delivered`` covers
    an uncertain outcome, and ``delivered`` means delivery was established. It
    does not promise exactly-once execution or authorize transport replay.

    The nested failure uses the wire contract, not a second set of reason/stage
    fields. Protocol-specific diagnostics, partial responses, and training data
    belong outside this record. No reward is inferred from a failure. Persistence,
    conversion from legacy markers, and retry budgets belong to the caller.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: EpisodeId
    run_id: str = Field(min_length=1)
    source: Literal["environment", "collector"]
    delivery: Literal["not_sent", "possibly_delivered", "delivered"]
    failure: EpisodeFailure
    http_status: int | None = None
    exception_type: str | None = Field(default=None, max_length=2000)
