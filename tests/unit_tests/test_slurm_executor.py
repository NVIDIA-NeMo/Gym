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
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import slurm as slurm_module
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm import (
    SlurmExecutor,
    _parse_sbatch_results,
    _sbatch_command,
    _validate_mounts,
)
from nemo_gym.orchestration.jobs import MANIFEST_NAME, load_record


def test_sbatch_command_captures_the_status_of_sbatch_not_the_pipeline():
    command = _sbatch_command("gsm8k", Path("/jobs/run/gsm8k/job.sh"))
    # rc must be captured immediately after sbatch; taking $? after a pipe
    # would report tr's status and call every failure a success.
    assert "rc=$?" in command
    assert command.index("rc=$?") < command.index("tr ")
    assert "sbatch --parsable /jobs/run/gsm8k/job.sh" in command
    assert "__GYM_JOB:gsm8k:$rc:$out" in command


@pytest.mark.parametrize("name", ["has space", "has:colon", "has$dollar", "", "a;b"])
def test_sbatch_command_rejects_a_benchmark_name_that_would_break_the_marker(name):
    with pytest.raises(ValueError, match="benchmark name"):
        _sbatch_command(name, Path("/jobs/run/job.sh"))


@pytest.mark.parametrize("name", ["gsm8k", "gpqa-no-tools", "tau2.airline", "aime_24"])
def test_sbatch_command_accepts_ordinary_benchmark_names(name):
    assert f"__GYM_JOB:{name}:$rc:$out" in _sbatch_command(name, Path("/jobs/run/job.sh"))


def test_parse_sbatch_results_reads_a_successful_submission():
    assert _parse_sbatch_results("__GYM_JOB:gsm8k:0:12345 ") == {"gsm8k": ("12345", None)}


def test_parse_sbatch_results_drops_the_federation_cluster_suffix():
    assert _parse_sbatch_results("__GYM_JOB:gsm8k:0:12345;hsg ") == {"gsm8k": ("12345", None)}


def test_parse_sbatch_results_ignores_unrelated_output():
    output = "Loading modules\n__GYM_JOB:gsm8k:0:12345 \nsome trailing chatter"
    assert _parse_sbatch_results(output) == {"gsm8k": ("12345", None)}


def test_parse_sbatch_results_records_a_failure_with_its_message():
    output = "__GYM_JOB:gsm8k:1:sbatch: error: Invalid account 'nope' "
    assert _parse_sbatch_results(output) == {"gsm8k": (None, "sbatch: error: Invalid account 'nope'")}


def test_parse_sbatch_results_falls_back_to_the_exit_code_when_sbatch_said_nothing():
    assert _parse_sbatch_results("__GYM_JOB:gsm8k:1: ") == {"gsm8k": (None, "sbatch exited 1")}


def test_parse_sbatch_results_treats_a_silent_success_as_a_failure():
    # Exit 0 with no output at all is not a job id; recording it as a "success"
    # with job_id="" would slip past `SubmissionRecord.failed` (which only
    # checks `job_id is None`).
    job_id, error = _parse_sbatch_results("__GYM_JOB:gsm8k:0: ")["gsm8k"]
    assert job_id is None
    assert error is not None


def test_parse_sbatch_results_treats_a_non_numeric_success_payload_as_a_failure():
    # A warning on a successful sbatch (job_submit plugin notices, QOS/ntasks
    # adjustments) merges onto the same line via 2>&1. If it doesn't end in
    # something that looks like a job id, there is no id to trust.
    output = "__GYM_JOB:gsm8k:0:sbatch: Warning: blah "
    assert _parse_sbatch_results(output) == {"gsm8k": (None, "sbatch: Warning: blah")}


def test_parse_sbatch_results_takes_the_trailing_id_off_a_warning_line():
    # The actual C1 regression: a successful sbatch that also warns must not
    # let the warning text end up in job_id.
    output = "__GYM_JOB:gsm8k:0:sbatch: Warning: can't honor --ntasks-per-node 12345 "
    assert _parse_sbatch_results(output) == {"gsm8k": ("12345", None)}


def test_a_failure_mid_list_does_not_shift_the_benchmarks_after_it():
    # The regression test for the positional-zip bug: bench_b fails, and bench_c
    # must still get its own job id rather than inheriting the next one along.
    output = "\n".join(
        [
            "__GYM_JOB:bench_a:0:111 ",
            "__GYM_JOB:bench_b:1:sbatch: error: Invalid account ",
            "__GYM_JOB:bench_c:0:333 ",
        ]
    )

    results = _parse_sbatch_results(output)

    assert results["bench_a"] == ("111", None)
    assert results["bench_b"][0] is None
    assert results["bench_c"] == ("333", None)


def _submit_config(tmp_path, benchmarks):
    return SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"hsg": {"type": "slurm", "account": "my-account", "hostname": None}},
            "driver": {"container": "gym:latest", "benchmarks": {name: {} for name in benchmarks}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )


class _FakeConnection(LocalConnection):
    """A local connection whose `run` answers as a scheduler would."""

    def __init__(self, replies):
        self._replies = replies
        self.commands = []

    def run(self, commands):
        self.commands.append(commands)
        return self._replies.pop(0)


def _install(monkeypatch, conn):
    monkeypatch.setattr(slurm_module, "get_connection", lambda hostname: conn)
    monkeypatch.setattr(slurm_module, "_validate_mounts", lambda config, connection: None)


def test_run_returns_a_record_naming_every_benchmark(tmp_path, monkeypatch):
    conn = _FakeConnection(["__GYM_JOB:bench_a:0:111 \n__GYM_JOB:bench_b:0:222 "])
    _install(monkeypatch, conn)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a", "bench_b"]))

    assert record is not None
    assert [(b.benchmark, b.job_id) for b in record.benchmarks] == [("bench_a", "111"), ("bench_b", "222")]
    assert record.cluster == "hsg"
    assert record.executor == "slurm"
    assert record.hostname is None
    assert record.run_dir.endswith(record.gym_job_id)
    assert record.benchmarks[0].job_dir == f"{record.run_dir}/bench_a"


def test_run_writes_the_manifest_into_the_run_dir(tmp_path, monkeypatch):
    conn = _FakeConnection(["__GYM_JOB:bench_a:0:111 "])
    _install(monkeypatch, conn)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    manifest = Path(record.run_dir) / MANIFEST_NAME
    assert load_record(json.loads(manifest.read_text())) == record


def test_run_writes_the_local_index(tmp_path, monkeypatch):
    conn = _FakeConnection(["__GYM_JOB:bench_a:0:111 "])
    _install(monkeypatch, conn)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    index = tmp_path / "cache" / "nemo-gym" / "jobs" / f"{record.gym_job_id}.json"
    assert load_record(json.loads(index.read_text())) == record


def test_run_records_a_failed_benchmark_without_disturbing_the_others(tmp_path, monkeypatch):
    conn = _FakeConnection(
        ["__GYM_JOB:bench_a:0:111 \n__GYM_JOB:bench_b:1:sbatch: error: bad account \n__GYM_JOB:bench_c:0:333 "]
    )
    _install(monkeypatch, conn)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a", "bench_b", "bench_c"]))

    by_name = {b.benchmark: b for b in record.benchmarks}
    assert by_name["bench_a"].job_id == "111"
    assert by_name["bench_b"].job_id is None
    assert "bad account" in by_name["bench_b"].error
    assert by_name["bench_c"].job_id == "333"
    assert [b.benchmark for b in record.failed] == ["bench_b"]


def test_run_records_a_benchmark_the_scheduler_never_answered_for(tmp_path, monkeypatch):
    conn = _FakeConnection(["__GYM_JOB:bench_a:0:111 "])
    _install(monkeypatch, conn)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a", "bench_b"]))

    by_name = {b.benchmark: b for b in record.benchmarks}
    assert by_name["bench_b"].job_id is None
    assert "no result" in by_name["bench_b"].error


def test_two_runs_in_the_same_second_get_different_run_dirs(tmp_path, monkeypatch):
    frozen = datetime(2026, 9, 9, 10, 2, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(slurm_module, "_utc_now", lambda: frozen)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    run_dirs = []
    for _ in range(2):
        conn = _FakeConnection(["__GYM_JOB:bench_a:0:111 "])
        _install(monkeypatch, conn)
        run_dirs.append(SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"])).run_dir)

    assert run_dirs[0] != run_dirs[1]


def test_a_failed_manifest_write_fails_the_submit_and_names_queued_jobs(tmp_path, monkeypatch):
    class _NoWrite(_FakeConnection):
        def write_text(self, remote, content):
            raise RuntimeError("permission denied")

    conn = _NoWrite(["__GYM_JOB:bench_a:0:111 \n__GYM_JOB:bench_b:0:222 "])
    _install(monkeypatch, conn)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(RuntimeError) as error:
        SlurmExecutor().run(_submit_config(tmp_path, ["bench_a", "bench_b"]))

    # The jobs are already queued; an error that does not say so strands them.
    message = str(error.value)
    assert "111" in message and "222" in message
    assert "permission denied" in message


def test_submitting_locally_runs_the_real_sbatch_command(tmp_path, monkeypatch):
    # The one test that exercises the generated bash end to end, over a real
    # LocalConnection. A command string that only a mocked `run` accepts would
    # pass every other test in this file and still break every local submit.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/bin/bash\necho 4242\n")
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(slurm_module, "_validate_mounts", lambda config, connection: None)

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    assert record.benchmarks[0].job_id == "4242"
    assert (Path(record.run_dir) / MANIFEST_NAME).exists()


def test_submitting_locally_records_a_failing_sbatch(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/bin/bash\necho 'sbatch: error: Invalid account' >&2\nexit 1\n")
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(slurm_module, "_validate_mounts", lambda config, connection: None)

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    assert record.benchmarks[0].job_id is None
    assert "Invalid account" in record.benchmarks[0].error


def test_submitting_locally_does_not_let_a_warning_become_the_job_id(tmp_path, monkeypatch):
    # The end-to-end regression test for C1: a real bash pipeline, not a mocked
    # `conn.run`, proves the generated command's own `2>&1` merge can't leak a
    # warning on a successful sbatch into job_id.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text('#!/bin/bash\necho "sbatch: Warning: can\'t honor --ntasks-per-node" >&2\necho 12345\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(slurm_module, "_validate_mounts", lambda config, connection: None)

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    assert record.benchmarks[0].job_id == "12345"
    assert record.benchmarks[0].error is None


def test_submitting_locally_keeps_a_multiline_error_on_one_marker_line(tmp_path, monkeypatch):
    # Real-bash coverage for the `tr` flattening: a two-line sbatch error must
    # still arrive whole in `error`, not truncated to whichever line survives.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "echo 'sbatch: error: Invalid account' >&2\n"
        "echo 'sbatch: error: try again with a valid account' >&2\n"
        "exit 1\n"
    )
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(slurm_module, "_validate_mounts", lambda config, connection: None)

    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    assert record.benchmarks[0].job_id is None
    assert "Invalid account" in record.benchmarks[0].error
    assert "try again with a valid account" in record.benchmarks[0].error


def test_dry_run_rejects_a_benchmark_name_that_would_break_the_marker(tmp_path):
    # Bad names must fail before staging/copying even happens, and a dry run
    # must not silently print a script listing for a benchmark that could
    # never actually be submitted.
    with pytest.raises(ValueError, match="benchmark name"):
        SlurmExecutor().run(_submit_config(tmp_path, ["bad name"]), dry_run=True)


def test_dry_run_returns_no_record(tmp_path, monkeypatch, capsys):
    record = SlurmExecutor().run(_submit_config(tmp_path, ["bench_a"]), dry_run=True)

    assert record is None
    assert "[dry-run]" in capsys.readouterr().out


def _config_with_driver_mounts(tmp_path, mounts):
    return SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"hsg": {"type": "slurm", "account": "my-account", "hostname": None}},
            "driver": {"container": "gym:latest", "benchmarks": {"bench_a": {}}, "mounts": mounts},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )


class TestValidateMounts:
    """Both connections pipe commands to bash, so mount validation runs one
    shell program either way. These go through a real `LocalConnection`: with
    the old `isinstance` branch the local path never touched the shell, so the
    check that actually runs in production was untested."""

    def test_a_present_mount_src_passes(self, tmp_path):
        src = tmp_path / "data"
        src.mkdir()

        _validate_mounts(_config_with_driver_mounts(tmp_path, [f"{src}:/data"]), LocalConnection())

    def test_a_missing_mount_src_is_named(self, tmp_path):
        src = tmp_path / "absent"

        with pytest.raises(ValueError) as error:
            _validate_mounts(_config_with_driver_mounts(tmp_path, [f"{src}:/data"]), LocalConnection())

        assert str(src) in str(error.value)
        assert "driver" in str(error.value)

    def test_no_mounts_asks_the_connection_nothing(self, tmp_path):
        class _Explodes(LocalConnection):
            def run(self, commands):
                raise AssertionError("no mounts, so there is nothing to check")

        _validate_mounts(_config_with_driver_mounts(tmp_path, []), _Explodes())
