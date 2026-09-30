# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic cohort verifier for checkpoint crash/recovery tests."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from nemo_gym.rollout_correlation import current_rollout_id
from resources_servers.genrm_compare.app import (
    GenRMCompareConfig,
    GenRMCompareResourcesServer,
    GenRMCompareVerifyRequest,
    GenRMCompareVerifyResponse,
)


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
    """Run production cohort logic with deterministic checkpoint-test scoring."""

    config: GenRMCompareConfig
    _checkpoint_audit_members: dict[str, set[str]] = PrivateAttr(default_factory=dict)
    _checkpoint_audit_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def verify(
        self,
        body: GenRMCompareVerifyRequest,
    ) -> GenRMCompareVerifyResponse:
        input_messages = getattr(body.responses_create_params, "input", None) or []
        prompt_key = self._get_verify_cohort_key(
            body,
            input_messages if isinstance(input_messages, list) else list(input_messages),
            body.principle,
        )
        capture_rollout_id = current_rollout_id() or body.capture_rollout_id
        _audit(
            "verify_entered",
            prompt_key=prompt_key,
            capture_rollout_id=capture_rollout_id,
            task_index=body.task_index,
            rollout_index=body.rollout_index,
        )
        if capture_rollout_id is not None:
            async with self._checkpoint_audit_lock:
                members = self._checkpoint_audit_members.setdefault(prompt_key, set())
                members.add(capture_rollout_id)
                if len(members) < self.config.num_rollouts_per_prompt:
                    _audit(
                        "verify_waiting",
                        prompt_key=prompt_key,
                        cohort_size=len(members),
                        capture_rollout_ids=sorted(members),
                    )

        result = await super().verify(body)
        _audit(
            "verify_returned",
            prompt_key=prompt_key,
            capture_rollout_id=capture_rollout_id,
            task_index=body.task_index,
            rollout_index=body.rollout_index,
        )
        return result

    async def _evaluate_verify_cohort(
        self,
        prompt_key: str,
        cohort: Any,
        members: dict[int, Any],
    ) -> None:
        """Audit one production cohort evaluation without replacing its logic."""
        await super()._evaluate_verify_cohort(prompt_key, cohort, members)
        if cohort.phase == "completed":
            async with self._checkpoint_audit_lock:
                capture_rollout_ids = sorted(self._checkpoint_audit_members.pop(prompt_key, set()))
                _audit(
                    "reward_computed",
                    prompt_key=prompt_key,
                    cohort_size=len(members),
                    capture_rollout_ids=capture_rollout_ids,
                )

    async def _run_compare(
        self,
        conversation_history: list[dict[str, str]],
        response_objs: list[dict[str, Any]],
        principle: str | None = None,
    ) -> tuple[
        list[float],
        dict[str, float],
        list[tuple[float, float, float]],
        list[tuple[int, int, int]],
    ]:
        """Replace only the expensive judge call with deterministic rewards."""
        del conversation_history, principle
        return [1.0] * len(response_objs), {}, [], []


if __name__ == "__main__":
    CheckpointTestGenRMResourcesServer.run_webserver()
