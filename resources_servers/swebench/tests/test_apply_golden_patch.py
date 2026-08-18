# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from resources_servers.swebench import apply_golden_patch


class _Response:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    async def json(self) -> dict[str, Any]:
        return self.data


class _TrackingServerClient:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def post(self, **kwargs: Any) -> _Response:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return _Response({"resolved": True, "instance_id": kwargs["json"]["instance_id"]})


class _IncompleteOnceServerClient:
    def __init__(self) -> None:
        self.calls = 0

    async def post(self, **kwargs: Any) -> _Response:
        self.calls += 1
        return _Response(
            {
                "resolved": self.calls > 1,
                "evaluation_completed": self.calls > 1,
                "instance_id": kwargs["json"]["instance_id"],
            }
        )


class _IncompleteThenQueuedServerClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.attempts: dict[str, int] = {}

    async def post(self, **kwargs: Any) -> _Response:
        instance_id = kwargs["json"]["instance_id"]
        self.calls.append(instance_id)
        self.attempts[instance_id] = self.attempts.get(instance_id, 0) + 1
        completed = instance_id != "retry" or self.attempts[instance_id] > 1
        return _Response(
            {
                "resolved": completed,
                "evaluation_completed": completed,
                "instance_id": instance_id,
            }
        )


async def test_main_limits_concurrent_verifications(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _TrackingServerClient()
    monkeypatch.setattr(apply_golden_patch, "server_client", client, raising=False)
    output_path = tmp_path / "results.jsonl"

    await apply_golden_patch.main(
        [{"instance_id": f"instance-{index}"} for index in range(7)],
        max_concurrency=2,
        output_path=output_path,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert client.peak == 2
    assert {row["instance_id"] for row in rows} == {f"instance-{index}" for index in range(7)}


async def test_main_retries_incomplete_evaluation(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _IncompleteOnceServerClient()
    monkeypatch.setattr(apply_golden_patch, "server_client", client, raising=False)
    output_path = tmp_path / "results.jsonl"

    await apply_golden_patch.main(
        [{"instance_id": "instance-1"}],
        max_concurrency=2,
        max_attempts=2,
        retry_delay_s=0,
        output_path=output_path,
    )

    assert client.calls == 2
    assert json.loads(output_path.read_text())["resolved"] is True


async def test_retry_backoff_does_not_occupy_concurrency_slot(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _IncompleteThenQueuedServerClient()
    monkeypatch.setattr(apply_golden_patch, "server_client", client, raising=False)

    await apply_golden_patch.main(
        [{"instance_id": "retry"}, {"instance_id": "queued"}],
        max_concurrency=1,
        max_attempts=2,
        retry_delay_s=0.01,
        output_path=tmp_path / "results.jsonl",
    )

    assert client.calls == ["retry", "queued", "retry"]


@pytest.mark.parametrize("max_concurrency", [0, -1, True])
async def test_main_rejects_invalid_max_concurrency(tmp_path: Path, max_concurrency: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        await apply_golden_patch.main([], max_concurrency=max_concurrency, output_path=tmp_path / "results.jsonl")


@pytest.mark.parametrize("max_attempts", [0, -1, True])
async def test_main_rejects_invalid_max_attempts(tmp_path: Path, max_attempts: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        await apply_golden_patch.main([], max_attempts=max_attempts, output_path=tmp_path / "results.jsonl")


@pytest.mark.parametrize("retry_delay_s", [-1, True, "1"])
async def test_main_rejects_invalid_retry_delay(tmp_path: Path, retry_delay_s: Any) -> None:
    with pytest.raises(ValueError, match="non-negative number"):
        await apply_golden_patch.main([], retry_delay_s=retry_delay_s, output_path=tmp_path / "results.jsonl")
