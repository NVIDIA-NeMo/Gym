# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run-only timing sidecar; preserve Gym dispatch, limits, retries and scores."""

import json
import time
from asyncio import Semaphore
from collections.abc import Awaitable, Iterator
from pathlib import Path
from typing import Any

from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _CompletedRollout
from nemo_gym.server_utils import BaseServerConfig


class TimedCollector(RolloutCollectionHelper):
    """Record completion times while retaining Gym's dispatch and result objects."""

    timing_path: Path

    def _run_examples_with_metadata(
        self,
        examples: list[dict[str, Any]],
        head_server_config: BaseServerConfig | None = None,
        semaphore: Semaphore | None = None,
        route_failures_to_sidecar: bool = False,
    ) -> Iterator[Awaitable[_CompletedRollout]]:
        started = time.time()
        self.timing_path.with_suffix(".start.json").write_text(
            json.dumps(
                {
                    "time_unix": started,
                    "expected_dispatched": len(examples),
                }
            )
        )
        futures = super()._run_examples_with_metadata(
            examples, head_server_config, semaphore, route_failures_to_sidecar
        )

        async def record(future: Awaitable[_CompletedRollout]) -> _CompletedRollout:
            completed = await future
            entry = {
                "task_index": completed.row.get("_ng_task_index"),
                "rollout_index": completed.row.get("_ng_rollout_index"),
                "rollout_id": completed.row.get("_ng_rollout_id"),
                "elapsed_seconds": time.time() - started,
                "rollout_latency_ms": completed.rollout_latency_ms,
                "reward": completed.result.get("reward"),
                "no_persist": bool(completed.result.get("_ng_no_persist")),
            }
            with self.timing_path.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            return completed

        return (record(future) for future in futures)


async def run(config: RolloutCollectionConfig, global_config: dict[str, Any]) -> None:
    """Gym eval driver entrypoint; write timing sidecars beside normal output."""
    collector = TimedCollector(timing_path=Path(config.output_jsonl_fpath).with_suffix(".timings.jsonl"))
    await collector.run_from_config(config)
