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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pytest import MonkeyPatch

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.jobs import (
    RESOLVED_CONFIG_NAME,
    SCHEMA_VERSION,
    BenchmarkJob,
    SubmissionRecord,
    local_index_dir,
    new_gym_job_id,
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
        benchmarks=[
            BenchmarkJob(
                benchmark="gsm8k",
                job_id="12345",
                job_dir="/jobs/gym-job-20260909T100203Z-abc123/gsm8k",
            )
        ],
    )
    return SubmissionRecord(**{**defaults, **overrides})


def _submit_config(tmp_path) -> SubmitConfig:
    return SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"hsg": {"type": "slurm", "account": "my-account", "hostname": None}},
            "driver": {"container": "gym:latest", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )


def test_record_round_trips_through_json():
    record = _record()
    restored = SubmissionRecord.load(json.loads(record.model_dump_json()))
    assert restored == record


def test_record_defaults_to_current_schema_version():
    assert _record().schema_version == SCHEMA_VERSION


def test_load_record_accepts_an_older_record_after_an_upgrade():
    """The upgrade path: a record written by an older Gym must stay readable.

    Simulated the way it really happens -- the payload predates a field, so the
    key is simply absent and the model's default fills it. Refusing this is what
    would strand every job already on disk the moment nemo-gym is upgraded.
    """
    payload = json.loads(_record().dumps())
    payload["schema_version"] = SCHEMA_VERSION - 1
    payload.pop("hostname")

    record = SubmissionRecord.load(payload)

    assert record.schema_version == SCHEMA_VERSION - 1
    assert record.hostname is None


def test_load_record_rejects_a_payload_with_no_version():
    payload = json.loads(_record().dumps())
    del payload["schema_version"]
    with pytest.raises(ValueError, match="no usable schema_version"):
        SubmissionRecord.load(payload)


def test_load_record_refuses_a_newer_schema_version():
    payload = json.loads(_record().model_dump_json())
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="written by a newer nemo-gym"):
        SubmissionRecord.load(payload)


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

    path = record.write_local_index()

    assert path == tmp_path / "nemo-gym" / "jobs" / f"{record.gym_job_id}.json"
    assert SubmissionRecord.load(json.loads(path.read_text())) == record


def test_write_local_index_returns_none_when_it_cannot_write(tmp_path, monkeypatch: MonkeyPatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", boom)

    assert _record().write_local_index() is None
    assert "read-only file system" in capsys.readouterr().err


def test_write_local_index_writes_exactly_what_dumps_produces(tmp_path, monkeypatch: MonkeyPatch):
    # The local index and the remote manifest must be byte-identical; both go
    # through dumps(), and this is what holds that true.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    record = _record()

    path = record.write_local_index()

    assert path.read_text() == record.dumps()


def test_executor_metadata_carries_whatever_an_executor_needs():
    """The record cannot grow a field per executor, so anything one needs that
    the shared fields cannot express goes here -- a k8s namespace, say. It round
    trips like any other field and defaults to empty for executors with nothing
    to add."""
    assert _record().executor_metadata == {}

    record = _record()
    record.executor_metadata = {"namespace": "frontier-eval", "context": "prod"}

    assert SubmissionRecord.load(json.loads(record.dumps())).executor_metadata == {
        "namespace": "frontier-eval",
        "context": "prod",
    }


def test_persist_writes_the_local_index_even_when_the_manifest_fails(tmp_path, monkeypatch: MonkeyPatch):
    """The ordering is the whole point of persist() living in the base class.

    The index is written before the manifest precisely so that a failed manifest
    write still leaves a parseable record behind -- the by-hand recovery the
    error message asks for has to have something to recover from.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    class _Executor(BaseExecutor):
        def run(self, config, *, dry_run: bool = False):  # pragma: no cover - unused
            raise NotImplementedError

    def _explode(path, text):
        raise OSError("remote is read-only")

    record = _record()
    with pytest.raises(RuntimeError, match="Record these by hand"):
        _Executor().persist(record, _submit_config(tmp_path), _explode)

    index = tmp_path / "nemo-gym" / "jobs" / f"{record.gym_job_id}.json"
    assert SubmissionRecord.load(json.loads(index.read_text())) == record


def test_persist_names_the_queued_jobs_when_the_manifest_fails(tmp_path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    class _Executor(BaseExecutor):
        def run(self, config, *, dry_run: bool = False):  # pragma: no cover - unused
            raise NotImplementedError

    record = _record()
    with pytest.raises(RuntimeError, match=r"Already queued: .*gsm8k=12345"):
        _Executor().persist(
            record, _submit_config(tmp_path), lambda path, text: (_ for _ in ()).throw(OSError("nope"))
        )


def test_persist_writes_the_resolved_config_next_to_the_manifest(tmp_path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    class _Executor(BaseExecutor):
        def run(self, config, *, dry_run: bool = False):  # pragma: no cover - unused
            raise NotImplementedError

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = _record(run_dir=str(run_dir))
    written: dict[Path, str] = {}

    _Executor().persist(record, _submit_config(tmp_path), lambda path, text: written.__setitem__(path, text))

    resolved = yaml.safe_load(written[run_dir / RESOLVED_CONFIG_NAME])
    assert resolved["job"]["output_path"] == str(tmp_path / "jobs")


def test_jobs_module_is_executor_agnostic():
    """`jobs.py` must not drag any executor into a reader's import path.

    A reader that only wants to parse a record -- EFB's collect, `gym eval
    status` -- should not end up importing Slurm, SSH and the sbatch templates.
    Checked in a SUBPROCESS on purpose: this test module imports executors
    itself, so asserting against the already-loaded sys.modules here would pass
    no matter what jobs.py does.
    """
    probe = (
        "import sys;"
        "import nemo_gym.orchestration.jobs;"
        "leaked = sorted(m for m in sys.modules if 'orchestration.executors' in m);"
        "print(','.join(leaked))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "", f"jobs.py pulled in executor modules: {result.stdout.strip()}"


def test_submission_record_executor_is_not_closed_over_todays_executors():
    """A future k8s or local executor must be able to write a record without
    editing this schema, so `executor` is an open str rather than an enum of the
    executors that exist today."""
    record = SubmissionRecord(
        gym_job_id="gym-job-20260101T000000Z-abc123",
        gym_version="0.0.0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir="/runs/gym-job-20260101T000000Z-abc123",
        cluster="some-cluster",
        executor="kubernetes",
        submitted_by="someone",
        benchmarks=[],
    )
    assert record.executor == "kubernetes"
    assert SubmissionRecord.load(json.loads(record.dumps())) == record
