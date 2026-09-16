# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic cohort verifier for checkpoint crash/recovery tests."""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Literal

from nemo_gym.base_resources_server import BaseVerifyResponse
from resources_servers.genrm_compare.app import (
    GenRMCompareConfig,
    GenRMCompareResourcesServer,
    GenRMCompareVerifyRequest,
)


class CheckpointTestGenRMConfig(GenRMCompareConfig):
    """Declare the deterministic verifier safe to replay after restart."""

    CHECKPOINT_RECOVERY_MODE: ClassVar[Literal["stateless"]] = "stateless"


_cohorts: dict[
    str,
    list[tuple[GenRMCompareVerifyRequest, asyncio.Future[float]]],
] = defaultdict(list)
_cohort_lock = asyncio.Lock()


def _audit(event: str, **fields: Any) -> None:
    path_value = os.environ.get("NEMO_GYM_CHECKPOINT_TEST_EVENTS")
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "phase": os.environ.get("NEMO_GYM_CHECKPOINT_TEST_PHASE", "unknown"),
        **fields,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class CheckpointTestGenRMResourcesServer(GenRMCompareResourcesServer):
    """Wait for one cohort and score it once without launching a judge model."""

    config: CheckpointTestGenRMConfig

    async def verify(
        self,
        body: GenRMCompareVerifyRequest,
    ) -> BaseVerifyResponse:
        input_messages = getattr(body.responses_create_params, "input", None) or []
        prompt_key = self._get_verify_cohort_key(
            body,
            input_messages if isinstance(input_messages, list) else list(input_messages),
            body.principle,
        )
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        ready: list[tuple[GenRMCompareVerifyRequest, asyncio.Future[float]]] | None = None
        _audit(
            "verify_entered",
            prompt_key=prompt_key,
            task_index=body.task_index,
            rollout_index=body.rollout_index,
        )
        async with _cohort_lock:
            cohort = _cohorts[prompt_key]
            cohort.append((body, future))
            if len(cohort) > self.config.num_rollouts_per_prompt:
                raise RuntimeError("checkpoint test GenRM cohort received more siblings than configured")
            if len(cohort) == self.config.num_rollouts_per_prompt:
                ready = _cohorts.pop(prompt_key)
                _audit(
                    "reward_computed",
                    prompt_key=prompt_key,
                    cohort_size=len(ready),
                )
                for _, waiter in ready:
                    waiter.set_result(1.0)
            else:
                _audit(
                    "verify_waiting",
                    prompt_key=prompt_key,
                    cohort_size=len(cohort),
                )

        reward = await future
        _audit(
            "verify_returned",
            prompt_key=prompt_key,
            task_index=body.task_index,
            rollout_index=body.rollout_index,
        )
        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
        )


if __name__ == "__main__":
    CheckpointTestGenRMResourcesServer.run_webserver()
