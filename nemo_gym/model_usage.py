# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-reported usage for an agent execution, independent of its harness."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nemo_gym.base_responses_api_model import (
    CaptureStore,
    ModelCallRecord,
    build_model_call_record,
    model_call_capture_dirs_from_config,
)
from nemo_gym.openai_utils import (
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)


logger = logging.getLogger(__name__)


def aggregate_model_usage(calls: Sequence[ModelCallRecord]) -> NeMoGymResponseUsage | None:
    """Sum actual model exchanges, preserving unknown totals and optional details.

    Each record is one exchange, including retries, auxiliary calls and replies
    ending at a token limit. Response IDs need not be unique across exchanges.
    A partial sum is not a complete usage measurement.
    """

    if any(call.error_category in {"stream_truncated", "capture_parse_error"} for call in calls):
        return None

    def total(field: str) -> int | None:
        values = [getattr(call, field) for call in calls]
        if not values or any(type(value) is not int or value < 0 for value in values):
            return None
        return sum(values)

    prompt, completion, tokens = (total(field) for field in ("tokens_in", "tokens_out", "tokens_total"))
    if prompt is None or completion is None or tokens is None:
        return None
    return NeMoGymResponseUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=total("cached_tokens")),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=total("tokens_reasoning")),
    )


@dataclass(frozen=True)
class ModelUsageCapture:
    """Read usage appended during one agent lifecycle from the shared capture store.

    Start after Resources seeding and read after Agent close, before verification.
    This excludes pre-existing calls and judge usage without depending on native
    transcript IDs, which some harnesses omit. The episode must exclusively own
    its rollout ID, as required by the Environment Server lifecycle.
    """

    rollout_id: str
    store: CaptureStore | None
    start_offset: int = 0

    @classmethod
    async def start(cls, config: Mapping[str, object], *, rollout_id: str) -> "ModelUsageCapture | None":
        """Checkpoint enabled capture; return None when observability is disabled."""
        directories = model_call_capture_dirs_from_config(config)
        if not directories:
            return None

        def checkpoint() -> ModelUsageCapture:
            try:
                store = CaptureStore(directories[0])
                return cls(rollout_id, store, store.offset(rollout_id))
            except Exception:
                logger.warning("Could not checkpoint model usage for %s", rollout_id, exc_info=True)
                return cls(rollout_id, None)

        return await asyncio.to_thread(checkpoint)

    async def usage(self) -> NeMoGymResponseUsage | None:
        """Return usage for this execution, or unknown for missing/damaged capture."""
        return await asyncio.to_thread(self._read)

    def _read(self) -> NeMoGymResponseUsage | None:
        try:
            if self.store is None or self.store.is_incomplete(self.rollout_id):
                return None
            exchanges, invalid_count = self.store.read_available(self.rollout_id, start_offset=self.start_offset)
            if invalid_count:
                return None
            calls = [build_model_call_record(exchange, call_index=index) for index, exchange in exchanges]
            return aggregate_model_usage(calls)
        except Exception:
            logger.warning("Could not read model usage for %s", self.rollout_id, exc_info=True)
            return None
