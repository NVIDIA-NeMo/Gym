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
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.manifest import (
    SandboxManifestRecord,
    append_manifest_record,
    latest_manifest_records,
    read_manifest_records,
)


def _record(**overrides) -> SandboxManifestRecord:
    defaults = dict(rollout_key="3-1", sandbox_id="sb-1", status="created", instance_id="inst-a", rollout_index=1)
    return SandboxManifestRecord(**(defaults | overrides))


class TestSandboxManifest:
    def test_append_and_read_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "manifest.jsonl"
        record = _record()
        append_manifest_record(path, record)

        assert read_manifest_records(path) == [record]

    def test_missing_file_reads_empty(self, tmp_path: Path) -> None:
        assert read_manifest_records(tmp_path / "missing.jsonl") == []
        assert latest_manifest_records(tmp_path / "missing.jsonl") == {}

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        append_manifest_record(path, _record())
        path.open("a").write("\n\n")

        assert len(read_manifest_records(path)) == 1

    def test_latest_record_wins_per_rollout_key(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        append_manifest_record(path, _record(status="created", ts=1.0))
        append_manifest_record(path, _record(status="paused", ts=2.0))
        append_manifest_record(path, _record(rollout_key="3-2", sandbox_id="sb-2", status="created", ts=3.0))
        append_manifest_record(path, _record(status="resumed", ts=4.0))

        latest = latest_manifest_records(path)
        assert latest["3-1"].status == "resumed"
        assert latest["3-1"].sandbox_id == "sb-1"
        assert latest["3-2"].status == "created"

    def test_unknown_status_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _record(status="restarted")
