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
"""Append-only sandbox lifecycle manifest.

One jsonl line per lifecycle transition of a rollout's sandbox. The resources
server appends ``created``/``resumed``/``resume_failed``/``done``; the
client-side pause driver appends ``paused``. Because appends are
chronological, the last record for a ``rollout_key`` is its current state,
which is what the pause driver, the resume path in ``seed_session`` and the
snapshot cleanup driver act on.
"""

from pathlib import Path
from time import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SandboxManifestStatus = Literal["created", "paused", "resumed", "resume_failed", "done"]


class SandboxManifestRecord(BaseModel):
    # "{task_index}-{rollout_index}" without any attempt suffix, so a retried
    # dispatch of the same rollout still matches its paused sandbox.
    rollout_key: str
    sandbox_id: str
    status: SandboxManifestStatus
    instance_id: Optional[str] = None
    rollout_index: Optional[int] = None
    ts: float = Field(default_factory=time)


def append_manifest_record(path: Path, record: SandboxManifestRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json(exclude_none=True) + "\n")


def read_manifest_records(path: Path) -> List[SandboxManifestRecord]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(SandboxManifestRecord.model_validate_json(line))
    return records


def latest_manifest_records(path: Path) -> Dict[str, SandboxManifestRecord]:
    """The current (last-appended) record per rollout_key."""
    return {record.rollout_key: record for record in read_manifest_records(path)}
