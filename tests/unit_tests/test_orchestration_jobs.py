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

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from nemo_gym.orchestration.jobs import (
    SCHEMA_VERSION,
    BenchmarkJob,
    SubmissionRecord,
    dumps,
    load_record,
    local_index_dir,
    new_gym_job_id,
    write_local_index,
)


NOW = datetime(2026, 9, 9, 10, 2, 3, tzinfo=timezone.utc)


def _record(**overrides) -> SubmissionRecord:
    defaults = dict(
        gym_job_id="gym-job-20260909T100203Z-abc123",
        gym_version="0.6.0",
        submitted_at="2026-09-09T10:02:03Z",
        run_dir="/jobs/gym-job-20260909T100203Z-abc123",
        cluster="cluster",
        executor="slurm",
        hostname="login-01",
        submitted_by="wprazuch",
        config_path="/tmp/submit.yaml",
        benchmarks=[
            BenchmarkJob(
                benchmark="gsm8k",
                job_id="12345",
                job_dir="/jobs/gym-job-20260909T100203Z-abc123/gsm8k",
            )
        ],
    )
    return SubmissionRecord(**{**defaults, **overrides})


def test_record_round_trips_through_json():
    record = _record()
    restored = load_record(json.loads(record.model_dump_json()))
    assert restored == record


def test_record_defaults_to_current_schema_version():
    assert _record().schema_version == SCHEMA_VERSION


def test_load_record_refuses_a_newer_schema_version():
    payload = json.loads(_record().model_dump_json())
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        load_record(payload)


def test_gym_job_id_embeds_the_utc_timestamp():
    assert new_gym_job_id(NOW).startswith("gym-job-20260909T100203Z-")


def test_gym_job_ids_minted_in_the_same_second_differ():
    assert new_gym_job_id(NOW) != new_gym_job_id(NOW)


def test_local_index_dir_honours_xdg_cache_home(tmp_path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert local_index_dir() == tmp_path / "nemo-gym" / "jobs"


def test_local_index_dir_falls_back_to_home_cache(tmp_path, monkeypatch: MonkeyPatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert local_index_dir() == tmp_path / ".cache" / "nemo-gym" / "jobs"


def test_write_local_index_writes_the_record(tmp_path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    record = _record()

    path = write_local_index(record)

    assert path == tmp_path / "nemo-gym" / "jobs" / f"{record.gym_job_id}.json"
    assert load_record(json.loads(path.read_text())) == record


def test_write_local_index_returns_none_when_it_cannot_write(tmp_path, monkeypatch: MonkeyPatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", boom)

    assert write_local_index(_record()) is None
    assert "read-only file system" in capsys.readouterr().err


def test_write_local_index_writes_exactly_what_dumps_produces(tmp_path, monkeypatch: MonkeyPatch):
    # The local index and the remote manifest must be byte-identical; both go
    # through dumps(), and this is what holds that true.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    record = _record()

    path = write_local_index(record)

    assert path.read_text() == dumps(record)
