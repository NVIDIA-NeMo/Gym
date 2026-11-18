# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-reported usage for an agent execution, independent of its harness."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nemo_gym.core.responses_api_types import (
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.rollouts.health.checks import _is_failed_attempt
from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence, model_call_capture_dirs_from_config
from nemo_gym.trajectory.model_call_record import ModelCallRecord, build_model_call_record


logger = logging.getLogger(__name__)


def aggregate_model_usage(calls: Sequence[ModelCallRecord]) -> NeMoGymResponseUsage | None:
    """Sum actual model exchanges, preserving unknown totals and optional details.

    Each record is one exchange, including retries, auxiliary calls and replies
    ending at a token limit. Response IDs need not be unique across exchanges.
    A partial sum is not a complete usage measurement.

    As in the rollout health checks, a failed call that returned no finished response and
    reported no usage contributes nothing, so exchanges that were all rejected sum to zero.
    A failed call that reported usage, such as a reply cut at its token limit, still counts.
    No exchanges at all is no evidence, and stays unknown.
    """

    if not calls or any(call.error_category in {"stream_truncated", "capture_parse_error"} for call in calls):
        return None
    counted = [
        call
        for call in calls
        if not (
            _is_failed_attempt(call.model_dump(mode="json")) and call.tokens_in is None and call.tokens_out is None
        )
    ]

    def total(field: str) -> int | None:
        values = [getattr(call, field) for call in counted]
        if any(type(value) is not int or value < 0 for value in values):
            return None
        return sum(values)

    prompt, completion, tokens = (total(field) for field in ("tokens_in", "tokens_out", "tokens_total"))
    if prompt is None or completion is None or tokens is None:
        return None
    return NeMoGymResponseUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(
            cached_tokens=total("cached_tokens"), cache_write_tokens=total("cache_creation_tokens")
        ),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=total("tokens_reasoning")),
    )


@dataclass(frozen=True)
class ModelUsageCapture:
    """Read usage appended during one agent lifecycle from the shared capture store.

    Start after Resources seeding and read when agent execution has finished,
    including Agent close for session-based lifecycles, before verification.
    This excludes pre-existing calls and judge usage without depending on harness
    transcript IDs, which some harnesses omit. The episode must exclusively own
    its rollout ID, as required by the Environment Server lifecycle.
    """

    rollout_id: str
    store: CaptureStore | None
    start_offset: int = 0

    @classmethod
    async def start(cls, config: Mapping[str, object], *, rollout_id: str) -> "ModelUsageCapture | None":
        """Checkpoint enabled capture; return None when observability is disabled.

        An enabled capture this server cannot locate (for example, no ``model_call_capture_dir``)
        gives unknown usage rather than failing the episode.
        """
        try:
            directories = model_call_capture_dirs_from_config(config)
        except ValueError:
            logger.warning(
                "Model-call capture is enabled but unusable; usage for %s is unknown", rollout_id, exc_info=True
            )
            return cls(rollout_id, None)
        if not directories:
            return None

        def checkpoint() -> ModelUsageCapture:
            try:
                store = CaptureStore(directories[0], create=False)
                return cls(rollout_id, store, store.offset(rollout_id))
            except Exception:
                logger.warning("Could not checkpoint model usage for %s", rollout_id, exc_info=True)
                return cls(rollout_id, None)

        return await asyncio.to_thread(checkpoint)

    async def usage(self) -> NeMoGymResponseUsage | None:
        """Return usage for this execution, or unknown for missing, damaged or still-pending capture."""
        if self.store is None:
            return None
        # A call cancelled at the end of execution finishes its capture write detached; wait for it
        # rather than report a partial total.
        if not await await_model_call_capture_quiescence(self.rollout_id, [self.store.root]):
            return None
        return await asyncio.to_thread(self._read)

    def _read(self) -> NeMoGymResponseUsage | None:
        try:
            if self.store is None or self.store.is_incomplete(self.rollout_id):
                return None
            exchanges, invalid_count = self.store.read_available(self.rollout_id, start_offset=self.start_offset)
            if invalid_count:
                return None
            # A verifier's model calls under this rollout are verification, not agent execution.
            calls = [
                build_model_call_record(exchange, call_index=index)
                for index, exchange in exchanges
                if exchange.get("verifier_invocation_id") is None
            ]
            return aggregate_model_usage(calls)
        except Exception:
            logger.warning("Could not read model usage for %s", self.rollout_id, exc_info=True)
            return None
