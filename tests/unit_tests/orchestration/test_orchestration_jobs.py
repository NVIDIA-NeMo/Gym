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
    MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    SCHEMA_VERSION,
    BenchmarkJob,
    SubmissionRecord,
    benchmark_artifact_dirs,
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

        def resubmit(self, record, benchmark):  # pragma: no cover - unused
            raise NotImplementedError

    def _explode(path, text, **_):
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

        def resubmit(self, record, benchmark):  # pragma: no cover - unused
            raise NotImplementedError

    record = _record()
    with pytest.raises(RuntimeError, match=r"Already queued: .*gsm8k=12345"):
        _Executor().persist(
            record, _submit_config(tmp_path), lambda path, text, **_: (_ for _ in ()).throw(OSError("nope"))
        )


def test_persist_writes_the_resolved_config_next_to_the_manifest(tmp_path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    class _Executor(BaseExecutor):
        def run(self, config, *, dry_run: bool = False):  # pragma: no cover - unused
            raise NotImplementedError

        def resubmit(self, record, benchmark):  # pragma: no cover - unused
            raise NotImplementedError

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = _record(run_dir=str(run_dir))
    written: dict[Path, str] = {}

    _Executor().persist(record, _submit_config(tmp_path), lambda path, text, **_: written.__setitem__(path, text))

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


def test_a_record_written_before_gym_commit_still_loads():
    """Added with a default, as the module docstring requires of a same-version change."""
    old = json.loads(_record().dumps())
    del old["gym_commit"]

    record = SubmissionRecord.load(old)

    assert record.gym_commit is None
    assert record.schema_version == SCHEMA_VERSION


def test_legacy_record_environment_remains_unknown():
    record = _record()
    assert record.environment is None
    assert "environment" not in json.loads(record.dumps())
    restored = SubmissionRecord.load(json.loads(record.dumps()))
    assert restored.environment is None


def test_submission_environment_records_names_not_values_and_preserves_updates(tmp_path, monkeypatch):
    from nemo_gym.orchestration.jobs import SubmissionEnvironment

    monkeypatch.setenv("SOURCE_KEY", "private-test-value")
    raw = _submit_config(tmp_path).model_dump(mode="json")
    raw["driver"]["env"] = {
        "DEST_KEY": "host:SOURCE_KEY",
        "RUNTIME_ONLY": "runtime:WORKER_KEY",
        "LITERAL": "lit:host:NOT_A_REFERENCE",
    }
    config = SubmitConfig.model_validate(raw)
    observation = BaseExecutor.submission_environment(
        config, required_env=["FORWARDED_KEY"], forwarded_env=["FORWARDED_KEY", "FORWARDED_KEY"]
    )
    assert observation.required_env == ["FORWARDED_KEY", "SOURCE_KEY"]
    assert observation.forwarded_env == ["FORWARDED_KEY"]
    assert observation.telemetry.enabled is True  # actual resolved config default, not a legacy inference
    assert observation.telemetry.token_env == "OTEL_TOKEN"
    assert "private-test-value" not in observation.model_dump_json()
    assert "NOT_A_REFERENCE" not in observation.model_dump_json() and "WORKER_KEY" not in observation.model_dump_json()
    record = _record(environment=observation)
    updated = record.with_benchmark(record.benchmarks[0].model_copy(update={"job_id": "new-job"}))
    assert SubmissionRecord.load(json.loads(updated.dumps())).environment == observation
    for bad in (
        {"required_env": ["BAD=VALUE"], "forwarded_env": [], "telemetry": observation.telemetry},
        {"required_env": [], "forwarded_env": [], "telemetry": {"token_env": "TOKEN"}},
        {"required_env": [], "forwarded_env": [], "telemetry": observation.telemetry, "values": {"TOKEN": "x"}},
    ):
        with pytest.raises(ValueError):
            SubmissionEnvironment.model_validate(bad)


def test_persist_record_keeps_initial_environment_observation(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    observation = BaseExecutor.submission_environment(_submit_config(tmp_path), required_env=[], forwarded_env=[])
    record = _record(environment=observation, run_dir=str(tmp_path / "run"))

    class Executor(BaseExecutor):
        def run(self, config, *, dry_run=False):
            raise NotImplementedError

        def resubmit(self, record, benchmark):
            raise NotImplementedError

    written = {}
    updated = record.with_benchmark(record.benchmarks[0].model_copy(update={"job_id": "successor"}))
    Executor().persist_record(updated, lambda path, text: written.__setitem__(path, text))
    assert SubmissionRecord.load(json.loads(next(iter(written.values())))).environment == observation
    index = tmp_path / "cache/nemo-gym/jobs" / (record.gym_job_id + ".json")
    assert SubmissionRecord.load(json.loads(index.read_text())).environment == observation


def test_submission_environment_includes_nested_service_host_references(tmp_path, monkeypatch):
    for name in ("ROUTER_KEY", "PREFILL_KEY", "DECODE_KEY"):
        monkeypatch.setenv(name, "private-nested-test-value")
    raw = _submit_config(tmp_path).model_dump(mode="json")
    raw["services"] = {
        "policy": {
            "type": "vllm_pd",
            "container": "test/image",
            "model": "test/model",
            "env": {"API_KEY": "host:ROUTER_KEY"},
            "prefill": {"env": {"API_KEY": "host:PREFILL_KEY"}},
            "decode": {"env": {"API_KEY": "host:DECODE_KEY"}},
        }
    }
    raw["compute"]["hsg"]["node_pools"] = {
        name: {"partition": "test", "gpus_per_node": 1} for name in ("prefill", "decode")
    }
    for name in ("prefill", "decode"):
        raw["services"]["policy"][name]["node_pool"] = name
    config = SubmitConfig.model_validate(raw)
    observation = BaseExecutor.submission_environment(config, required_env=[], forwarded_env=[])
    assert observation.required_env == ["DECODE_KEY", "PREFILL_KEY", "ROUTER_KEY"]
    assert observation.forwarded_env == []
    assert "private-nested-test-value" not in observation.model_dump_json()


def test_benchmark_artifact_dirs_resolves_a_job_id_and_a_run_directory(tmp_path: Path, monkeypatch: MonkeyPatch):
    monkeypatch.setattr("nemo_gym.orchestration.jobs.local_index_dir", lambda: tmp_path)
    record = _record()
    (tmp_path / f"{record.gym_job_id}.json").write_text(record.dumps())
    expected = {"gsm8k": Path("/jobs/gym-job-20260909T100203Z-abc123/gsm8k/artifacts")}

    assert benchmark_artifact_dirs(record.gym_job_id) == expected

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / MANIFEST_NAME).write_text(record.dumps())
    assert benchmark_artifact_dirs(str(run_dir)) == expected


def test_benchmark_artifact_dirs_is_none_without_a_record(tmp_path: Path, monkeypatch: MonkeyPatch):
    monkeypatch.setattr("nemo_gym.orchestration.jobs.local_index_dir", lambda: tmp_path)
    (tmp_path / "run").mkdir()

    assert benchmark_artifact_dirs("gym-job-unknown") is None
    assert benchmark_artifact_dirs(str(tmp_path / "run")) is None


def test_persist_includes_safe_submission_provenance(tmp_path, monkeypatch):
    from nemo_gym.orchestration import jobs
    from nemo_gym.package_info import __version__

    class _Executor(BaseExecutor):
        def run(self, config, *, dry_run: bool = False):
            raise NotImplementedError

        def resubmit(self, record, benchmark):
            raise NotImplementedError

    monkeypatch.setattr(jobs, "local_index_dir", lambda: tmp_path / "index")
    config = _submit_config(tmp_path)
    config.driver.benchmarks["gsm8k"].run.update({"policy_model_name": "policy", "policy_api_key": "secret"})
    record = _record()
    written = {}
    _Executor().persist(record, config, lambda path, text, **_: written.__setitem__(path, text))
    loaded = jobs.SubmissionRecord.load(json.loads(written[Path(record.run_dir) / jobs.MANIFEST_NAME]))
    assert loaded.provenance["gym_version"] == __version__
    assert loaded.provenance["materialized_inputs_sha256"] is None
    assert loaded.provenance["models"] == {"driver.benchmarks.gsm8k.run": {"policy_model_name": "policy"}}
    assert "secret" not in json.dumps(loaded.provenance)
    # The resolved config is stored once, beside the manifest, not again inside the record.
    assert "resolved_config" not in loaded.provenance
    assert (
        json.loads((tmp_path / "index" / f"{record.gym_job_id}.json").read_text())["provenance"] == loaded.provenance
    )


def test_old_submission_record_defaults_provenance() -> None:
    payload = _record().model_dump(exclude={"provenance"})
    assert SubmissionRecord.load(payload).provenance == {}


def test_review_provenance_error_does_not_lose_a_queued_submission(tmp_path, monkeypatch, caplog):
    from nemo_gym.orchestration import jobs

    class Executor(BaseExecutor):
        def run(self, config, *, dry_run=False):
            raise NotImplementedError

        def resubmit(self, record, benchmark):
            raise NotImplementedError

    def broken(*args, **kwargs):
        raise ValueError("secret")

    monkeypatch.setattr("nemo_gym.rollouts.provenance.build_provenance", broken)
    monkeypatch.setattr(jobs, "local_index_dir", lambda: tmp_path / "index")
    record = _record()
    written = {}
    Executor().persist(record, _submit_config(tmp_path), lambda path, text, **_: written.__setitem__(path, text))
    assert json.loads((tmp_path / "index" / f"{record.gym_job_id}.json").read_text())["provenance"] == {}
    assert json.loads(written[Path(record.run_dir) / jobs.MANIFEST_NAME])["provenance"] == {}
    assert "ValueError" in caplog.text
    assert "secret" not in caplog.text
