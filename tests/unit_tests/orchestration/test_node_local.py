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

import contextlib
import io
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path
from typing import ClassVar

import pytest

import nemo_gym.orchestration.submit as submit_module
from nemo_gym.orchestration.api import BenchmarkRunConfig, NodeLocalConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.node_local import NODE_LOCAL_DIR_ENV, WATCH_CHECK
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord


_TWO_POOLS = {
    "gpu": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4},
    "aux": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4},
}


def _config(node_local=None, *, services=None, pools=None, policy=None, exports=False) -> SubmitConfig:
    benchmark = {"run": {}, **({"node_local": node_local} if node_local is not None else {})}
    compute = {"type": "slurm", "account": "acct", **({"node_pools": pools} if pools else {})}
    return SubmitConfig.model_validate(
        {
            "services": services
            if services is not None
            else {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": compute},
            "driver": {
                "container": "gym:latest",
                "benchmarks": {"b": benchmark},
                **({"policy_model": policy} if policy else {}),
            },
            "job": {"output_path": "/jobs", **({"export": {"evalrdl": {}}} if exports else {})},
            "otel": {"enabled": False},
        }
    )


def _script(config: SubmitConfig, bench_dir: Path = Path("/jobs/20261116_run/b")) -> str:
    return build_sbatch_script(config, "b", config.driver.benchmarks["b"], config.compute["cluster"], bench_dir)


def _driver_line(script: str) -> str:
    return next(line for line in script.splitlines() if "logs/driver-" in line)


# --- model-call capture ------------------------------------------------------------------------------------------


def test_capture_defaults_to_the_node_local_dir():
    script = _script(_config(True))
    assert "model_call_capture_dir=/tmp/gym-node-local/20261116_run/b/model-calls" in script
    assert "/jobs/20261116_run/b/model-calls" not in script


def test_capture_stays_in_the_benchmark_dir_without_node_local():
    assert "model_call_capture_dir=/jobs/20261116_run/b/model-calls" in _script(_config())


def test_an_explicit_capture_dir_wins_over_the_node_local_default():
    config = _config(True)
    config.driver.benchmarks["b"].run["model_call_capture_dir"] = "/custom/calls"
    script = _script(config)
    assert "model_call_capture_dir=/custom/calls" in script
    assert "/tmp/gym-node-local/20261116_run/b/model-calls" not in script


# --- config ------------------------------------------------------------------------------------------------------


def test_node_local_is_off_unless_set():
    assert BenchmarkRunConfig().node_local_config is None
    assert BenchmarkRunConfig(node_local=True).node_local_config == NodeLocalConfig()
    assert NodeLocalConfig().restore is True
    custom = NodeLocalConfig(restore=False, base_dir="/raid/scratch/")
    assert BenchmarkRunConfig(node_local=custom).node_local_config is custom
    assert custom.base_dir == "/raid/scratch"


@pytest.mark.parametrize(
    "base_dir",
    ["tmp", "/tmp/a,b", "/tmp/a:b", "/tmp/../etc", "/tmp/$HOME", "/tmp/a b", "/tmp\n", "/raid/scratch/\n"],
)
def test_a_base_dir_that_is_not_a_plain_absolute_path_is_refused(base_dir):
    # A trailing newline (a YAML block scalar) would split the rendered path and point cleanup at base_dir itself.
    with pytest.raises(ValueError, match="node_local.base_dir"):
        NodeLocalConfig(base_dir=base_dir)


@pytest.mark.parametrize(("base_dir", "rendered"), [("/tmp", "/tmp"), ("/raid/scratch/", "/raid/scratch")])
def test_a_valid_base_dir_renders_one_per_job_directory_under_it(base_dir, rendered):
    assert f'_gym_nl_dir={rendered}/gym-node-local-"$SLURM_JOB_ID"\n' in _script(_config({"base_dir": base_dir}))


def test_the_stop_window_must_leave_time_to_stop_the_driver_and_archive():
    with pytest.raises(ValueError, match=r"at least archive_timeout_seconds \+ 60 \(160\)"):
        NodeLocalConfig(archive_timeout_seconds=100, stop_before_walltime_seconds=120)
    assert NodeLocalConfig(archive_timeout_seconds=100, stop_before_walltime_seconds=160)
    assert NodeLocalConfig(archive_timeout_seconds=100, stop_before_walltime_seconds=0)


def test_a_run_without_node_local_saves_the_same_config_as_before():
    assert "node_local" not in BenchmarkRunConfig().model_dump()
    assert BenchmarkRunConfig(node_local=True).model_dump()["node_local"] is True
    saved = BenchmarkRunConfig(node_local={"restore": False}).model_dump()
    assert BenchmarkRunConfig.model_validate(saved).node_local_config.restore is False


class _NoNodeLocalExecutor(BaseExecutor):
    name: ClassVar[str] = "stub"

    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        return None

    def resubmit(self, record: SubmissionRecord, benchmark: str) -> BenchmarkJob:  # pragma: no cover - unused
        raise NotImplementedError


def test_submit_refuses_node_local_on_an_executor_without_it(monkeypatch):
    monkeypatch.setitem(submit_module._EXECUTORS, SlurmComputeConfig, _NoNodeLocalExecutor)
    with pytest.raises(ValueError, match=r"b set `node_local`, but _NoNodeLocalExecutor"):
        submit_module.submit(_config(True))
    assert submit_module.submit(_config()) is None
    assert SlurmExecutor.supports_node_local


# --- rendering ---------------------------------------------------------------------------------------------------


def test_without_node_local_the_script_has_none_of_it():
    script = _script(_config())
    assert "gym-node-local" not in script and "_gym_nl_" not in script and NODE_LOCAL_DIR_ENV not in script


def test_without_node_local_a_driver_with_no_service_still_runs_in_the_foreground():
    script = _script(_config(services={}))
    assert _driver_line(script).endswith('"${GYM_CMD[@]}"')
    assert "_gym_run_driver" not in script


def test_the_driver_gets_the_dir_at_a_path_every_job_of_the_run_shares():
    line = _driver_line(_script(_config(True)))
    # Absolute paths the run saves (evidence a later job reads back) stay valid after a restore in another job.
    assert f"{NODE_LOCAL_DIR_ENV}=/tmp/gym-node-local/20261116_run/b srun " in line
    assert (
        '--container-mounts=/jobs/20261116_run/b:/jobs/20261116_run/b,"$_gym_nl_dir":/tmp/gym-node-local/20261116_run/b '
        in line
    )


def test_the_host_dir_is_unique_to_the_slurm_job_and_archives_land_in_the_job_dir():
    script = _script(_config({"base_dir": "/raid/scratch"}))
    assert '_gym_nl_dir=/raid/scratch/gym-node-local-"$SLURM_JOB_ID"' in script
    assert "_gym_nl_archives=/jobs/20261116_run/b/node-local" in script  # pragma: allowlist secret


def test_the_dir_is_set_up_before_the_driver_and_archived_after_it():
    script = _script(_config(True))
    restore = script.index('"$_gym_nl_dir" "$_gym_nl_archives" 1 || exit $?\n')
    driver = script.index(_driver_line(script))
    # The dir goes only once the driver step has ended: exports may read it until then.
    after = script.index(
        '\n_gym_run_driver\n_gym_nl_rc=$?\n_gym_nl_archive\n_gym_nl_cleanup\ntrap - TERM\n(exit "$_gym_nl_rc")'
    )
    assert restore < driver < after
    # The trailing 1: with restore on, the archive step prunes superseded archives.
    assert '"$_gym_nl_dir" "$_gym_nl_archives" "$SLURM_JOB_ID" 120 1\n' in script


def test_restore_can_be_turned_off_and_then_no_archive_is_pruned():
    script = _script(_config({"restore": False}))
    assert '"$_gym_nl_dir" "$_gym_nl_archives" 0 || exit $?\n' in script
    assert '"$_gym_nl_dir" "$_gym_nl_archives" "$SLURM_JOB_ID" 120 0\n' in script


def test_the_watch_loop_stops_the_driver_before_the_time_limit():
    script = _script(_config({"stop_before_walltime_seconds": 400}))
    loop = script[script.index("_gym_run_driver() {") : script.index("\n_gym_run_driver\n")]
    assert WATCH_CHECK in loop
    assert "_gym_nl_deadline=$((BASH_REMATCH[2] - 400))" in script


def test_a_zero_stop_window_never_stops_the_driver_early():
    script = _script(_config({"stop_before_walltime_seconds": 0}))
    assert "scontrol show job" not in script and "_gym_nl_stop_for_walltime()" not in script
    # The loop still runs the check, which an empty deadline turns off; the driver is still backgrounded.
    assert '_gym_nl_deadline=""' in script and WATCH_CHECK in script


def test_a_driver_with_no_service_is_still_watched_so_it_can_be_stopped():
    script = _script(_config(True, services={}))
    loop = script[script.index("_gym_run_driver() {") : script.index("\n_gym_run_driver\n")]
    assert "_gym_service_exited" not in loop.split("_gym_run_driver() {")[1]
    assert WATCH_CHECK in loop


def test_on_the_batch_node_the_dir_is_handled_in_the_batch_shell_and_a_term_archives_it():
    script = _script(_config(True))
    assert '_gym_nl_run() { bash -c "$1" _gym_node_local "${@:2}"; }' in script
    assert "trap _gym_nl_on_term TERM" in script


def test_off_the_batch_node_the_dir_is_handled_by_a_step_on_the_driver_node():
    services = {"policy": {"type": "vllm", "container": "img", "model": "/ckpt", "node_pool": "aux"}}
    config = _config(True, services=services, pools=_TWO_POOLS, policy="policy")
    script = _script(config)
    assert '--nodelist="${gym_nodes[1]}" --nodes=1 --ntasks=1' in _driver_line(script)
    assert (
        '_gym_nl_run() { srun --overlap --nodelist="${gym_nodes[1]}" --nodes=1 --ntasks=1 --account=acct '
        'bash -c "$1" _gym_node_local "${@:2}"; }'
    ) in script
    # A step cannot start once the job is ending, so there is no TERM archive.
    assert "trap _gym_nl_on_term TERM" not in script and "trap - TERM" not in script


def test_a_multi_node_driver_on_node_0_is_handled_in_the_batch_shell():
    services = {"policy": {"type": "vllm", "container": "img", "model": "/ckpt", "node_pool": "gpu"}}
    script = _script(_config(True, services=services, pools=_TWO_POOLS, policy="policy"))
    assert '--nodelist="${gym_nodes[0]}"' in _driver_line(script)
    assert '_gym_nl_run() { bash -c "$1"' in script
    assert "trap _gym_nl_on_term TERM" in script


@pytest.mark.parametrize("node_local", [True, {"restore": False, "stop_before_walltime_seconds": 0}])
def test_the_script_is_valid_bash(node_local):
    result = subprocess.run(["bash", "-n"], input=_script(_config(node_local)), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


# --- the rendered job, run with fake Slurm commands -------------------------------------------------------------

# The fake srun runs the driver on the host, so it points the driver at the mount's host side, as the container does.
_FAKE_SRUN = """\
for arg in "$@"; do
    case "$arg" in --container-mounts=*)
        IFS=, read -ra mounts <<< "${arg#--container-mounts=}"
        for mount in "${mounts[@]}"; do
            [[ "${mount#*:}" != "${GYM_NODE_LOCAL_DIR:-}" ]] || export GYM_NODE_LOCAL_DIR="${mount%%:*}"
        done ;;
    esac
done
case "$*" in
    *logs/driver-*) %(driver)s ;;
    # A node-local step on the driver's node: run it here, on the host.
    *" _gym_node_local "*) while [[ "$1" == --* ]]; do shift; done; exec "$@" ;;
    *) %(service)s ;;
esac
"""


def _start_job(
    tmp_path: Path,
    *,
    driver: str,
    service: str = "exec sleep 600",
    node_local: dict | None = None,
    end_in: int | None = None,
    fakes: dict[str, str] | None = None,
    exports: bool = False,
    job_id: str = "11",
    off_batch: bool = False,
) -> subprocess.Popen:
    """Start the rendered job.sh in its own process group; `end_in`: scontrol reports the job's end that many s on.

    `off_batch` pins the policy, and so the driver, to the second of two nodes: node-local steps then go through srun.
    """
    host = tmp_path / "host"
    host.mkdir(exist_ok=True)
    placement = (
        {
            "services": {"policy": {"type": "vllm", "container": "img", "model": "/ckpt", "node_pool": "aux"}},
            "pools": _TWO_POOLS,
            "policy": "policy",
        }
        if off_batch
        else {}
    )
    node_local = {"base_dir": str(host), "archive_timeout_seconds": 5, **(node_local or {})}
    config = _config(node_local, exports=exports, **placement)
    bench = tmp_path / "run" / "b"
    bench.mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin").mkdir(exist_ok=True)
    all_fakes = {"srun": _FAKE_SRUN % {"driver": driver, "service": service}, "curl": "exit 0", **(fakes or {})}
    job_line = (
        f'echo "JobId=1 TimeLimit=04:00:00 EndTime=$(($(date +%s) + {end_in})) Partition=b"'
        if end_in is not None
        else "exit 1"
    )
    if off_batch:
        all_fakes["scontrol"] = f"case \"$2\" in hostnames) printf 'n0\\nn1\\n' ;; *) {job_line} ;; esac"
    elif end_in is not None:
        all_fakes["scontrol"] = job_line
    for name, body in all_fakes.items():
        (tmp_path / "bin" / name).write_text(f"#!/bin/bash\n{body}\n")
        (tmp_path / "bin" / name).chmod(0o755)
    (bench / "job.sh").write_text(_script(config, bench))
    with (tmp_path / "job.out").open("w") as out:
        return subprocess.Popen(
            ["bash", "job.sh"],
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=bench,
            # COPYFILE_DISABLE keeps macOS tar from adding `._` metadata entries to the archives.
            env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "SLURM_JOB_ID": job_id, "COPYFILE_DISABLE": "1"},
            start_new_session=True,
        )


# As the fake srun's driver: run the rendered entrypoint itself (the export tail), on the host.
_REAL_DRIVER = 'while [[ "$1" != --output=logs/driver-* ]]; do shift; done; shift; exec "$@"'


def _finish(process: subprocess.Popen, timeout: float = 60) -> int:
    try:
        return process.wait(timeout=timeout)
    finally:
        # macOS answers EPERM to a kill of a group whose members have all exited but are not yet reaped.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)


def _wait_for(path: Path, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.1)


def _archive(tmp_path: Path, name: str = "11") -> dict[str, str]:
    with tarfile.open(tmp_path / "run" / "b" / "node-local" / f"{name}.tar.gz") as archive:
        return {
            member.name.removeprefix("./"): archive.extractfile(member).read().decode()
            for member in archive.getmembers()
            if member.isfile()
        }


def _write_archive(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(text.encode())
            archive.addfile(info, io.BytesIO(text.encode()))


_WRITE_EVIDENCE = 'mkdir -p "$GYM_NODE_LOCAL_DIR/ev" && echo row1 > "$GYM_NODE_LOCAL_DIR/ev/1.json"'


def test_a_driver_that_ends_leaves_one_archive_of_the_dir_and_its_status(tmp_path):
    returncode = _finish(_start_job(tmp_path, driver=f"{_WRITE_EVIDENCE}; exit 7"))

    assert returncode == 7
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}
    assert not (tmp_path / "host" / "gym-node-local-11").exists()
    assert [p.name for p in (tmp_path / "run" / "b" / "node-local").iterdir()] == ["11.tar.gz"]


def test_restore_extracts_the_newest_archive_and_the_next_archive_holds_everything(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    # 10 is newer than 9, though it sorts first as text.
    _write_archive(archives / "9.tar.gz", {"ev/old.json": "stale\n"})
    _write_archive(archives / "10.tar.gz", {"ev/0.json": "row0\n"})
    driver = f'cat "$GYM_NODE_LOCAL_DIR/ev/0.json"; test ! -e "$GYM_NODE_LOCAL_DIR/ev/old.json" && {_WRITE_EVIDENCE}'

    returncode = _finish(_start_job(tmp_path, driver=driver))

    assert returncode == 0
    assert "Restored the node-local dir from" in (tmp_path / "job.out").read_text()
    assert _archive(tmp_path) == {"ev/0.json": "row0\n", "ev/1.json": "row1\n"}


def _truncated_archive(path: Path) -> None:
    """A gzip that ends mid-stream: tar may extract its first member before it fails."""
    _write_archive(path, {"ev/partial.json": "p" * 50000, "ev/other.json": bytes(range(256)).hex() * 400})
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


def test_restore_falls_back_to_an_older_archive_when_the_newest_does_not_extract(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    _write_archive(archives / "9.tar.gz", {"ev/0.json": "row0\n"})
    _truncated_archive(archives / "10.tar.gz")
    driver = 'cat "$GYM_NODE_LOCAL_DIR/ev/0.json"; test ! -e "$GYM_NODE_LOCAL_DIR/ev/partial.json"'

    returncode = _finish(_start_job(tmp_path, driver=driver))

    assert returncode == 0
    output = (tmp_path / "job.out").read_text()
    assert "cannot extract" in output and "row0" in output


def test_when_no_archive_restores_the_driver_does_not_start(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    (archives / "10.tar.gz").parent.mkdir(parents=True)
    (archives / "10.tar.gz").write_bytes(b"not a tar")
    _truncated_archive(archives / "9.tar.gz")

    returncode = _finish(_start_job(tmp_path, driver=": > driver-ran"))

    assert returncode == 1
    assert "ERROR: no node-local archive in" in (tmp_path / "job.out").read_text()
    assert not (tmp_path / "run" / "b" / "driver-ran").exists()
    assert sorted(p.name for p in archives.iterdir()) == ["10.tar.gz", "9.tar.gz"]


def test_without_restore_the_dir_starts_empty(tmp_path):
    _write_archive(tmp_path / "run" / "b" / "node-local" / "10.tar.gz", {"ev/0.json": "row0\n"})

    returncode = _finish(
        _start_job(tmp_path, driver='test -z "$(ls -A "$GYM_NODE_LOCAL_DIR")"', node_local={"restore": False})
    )

    assert returncode == 0
    assert _archive(tmp_path) == {}


def _archive_names(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "run" / "b" / "node-local").iterdir())


def test_with_restore_only_the_new_archive_and_the_one_before_it_are_kept(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    for job in (8, 9, 10):
        _write_archive(archives / f"{job}.tar.gz", {f"ev/{job}.json": "row\n"})

    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE))

    assert returncode == 0
    # 11 holds everything 10 held plus this job's files; 10 stays as the fallback a restore would use.
    assert _archive_names(tmp_path) == ["10.tar.gz", "11.tar.gz"]
    assert _archive(tmp_path) == {"ev/10.json": "row\n", "ev/1.json": "row1\n"}


def test_pruning_keeps_the_archive_the_job_restored_from_not_a_newer_broken_one(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    _write_archive(archives / "9.tar.gz", {"ev/0.json": "row0\n"})
    (archives / "10.tar.gz").write_bytes(b"not a tar")

    assert _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE)) == 0

    # Job 11 restored from 9, so 9 stays as the known-good fallback and the broken 10 goes.
    assert _archive_names(tmp_path) == ["11.tar.gz", "9.tar.gz"]
    (archives / "11.tar.gz").write_bytes(b"not a tar")

    returncode = _finish(_start_job(tmp_path, driver='cat "$GYM_NODE_LOCAL_DIR/ev/0.json"', job_id="12"))

    assert returncode == 0
    assert "Restored the node-local dir from" in (tmp_path / "job.out").read_text()
    assert "row0" in (tmp_path / "job.out").read_text()
    assert _archive_names(tmp_path) == ["12.tar.gz", "9.tar.gz"]


def test_without_restore_every_archive_is_kept(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    for job in (8, 9, 10):
        _write_archive(archives / f"{job}.tar.gz", {f"ev/{job}.json": "row\n"})

    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, node_local={"restore": False}))

    assert returncode == 0
    assert _archive_names(tmp_path) == ["10.tar.gz", "11.tar.gz", "8.tar.gz", "9.tar.gz"]


def test_a_failed_archive_prunes_nothing(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    for job in (8, 9, 10):
        _write_archive(archives / f"{job}.tar.gz", {f"ev/{job}.json": "row\n"})
    # Only writing an archive hangs; the restore still extracts.
    hanging_tar = 'case "$1" in -czf) exec sleep 600 ;; *) exec /usr/bin/tar "$@" ;; esac'

    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"tar": hanging_tar}))

    assert returncode == 0
    assert "archiving the node-local dir failed" in (tmp_path / "job.out").read_text()
    assert _archive_names(tmp_path) == ["10.tar.gz", "8.tar.gz", "9.tar.gz"]


def test_a_lost_service_still_leaves_the_archive(tmp_path):
    driver = f"{_WRITE_EVIDENCE}; trap 'exit 143' TERM; sleep 600 & wait"

    # The service dies once the driver has written its evidence.
    evidence = tmp_path / "host" / "gym-node-local-11" / "ev" / "1.json"
    service = f"until [ -e {evidence} ]; do sleep 0.1; done; exit 3"

    returncode = _finish(_start_job(tmp_path, driver=driver, service=service))

    assert returncode == 1
    assert "service vllm_model exited (status 3)" in (tmp_path / "job.out").read_text()
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_before_the_time_limit_the_driver_is_stopped_archived_and_the_job_waits_for_the_limit(tmp_path):
    # The stop window is 65 s and the job ends 68 s after the script asks: the first poll after 3 s stops the driver.
    driver = f"{_WRITE_EVIDENCE}; trap 'echo driver got TERM; exit 143' TERM; sleep 600 & wait"
    process = _start_job(tmp_path, driver=driver, node_local={"stop_before_walltime_seconds": 65}, end_in=68)
    try:
        _wait_for(tmp_path / "run" / "b" / "node-local" / "11.tar.gz")
        _wait_for_text(tmp_path / "job.out", "Waiting for the time limit")
        assert process.poll() is None
        # The time limit: Slurm signals the whole job.
        os.killpg(process.pid, signal.SIGTERM)
        returncode = _finish(process, timeout=30)
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)

    output = (tmp_path / "job.out").read_text()
    assert "stopping the driver and archiving the node-local dir" in output
    assert "driver got TERM" in output
    assert "The job got TERM" not in output and output.count("Archived the node-local dir") == 1
    assert returncode == 143
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_after_the_evaluation_succeeded_the_deadline_leaves_the_exports_running(tmp_path):
    # The evaluation is done at once; the exports run past the deadline (3 s) and the poll that finds it (5 s).
    driver = f'{_WRITE_EVIDENCE}; : > "$GYM_EVAL_SUCCEEDED_MARKER"; sleep 7; echo exports done'
    process = _start_job(
        tmp_path, driver=driver, node_local={"stop_before_walltime_seconds": 65}, end_in=68, exports=True
    )

    returncode = _finish(process)

    output = (tmp_path / "job.out").read_text()
    assert returncode == 0
    assert "The evaluation succeeded: archiving the node-local dir" in output
    assert "stopping the driver" not in output and "exports done" in output
    assert output.count("Archived the node-local dir") == 1
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


@pytest.mark.parametrize("off_batch", [False, True], ids=["batch_node", "off_batch_node"])
def test_a_service_lost_after_the_evaluation_succeeded_still_lets_the_deadline_archive(tmp_path, off_batch):
    # The service dies once the evaluation has succeeded and before the poll that reaches the deadline (3 s on); the
    # exports then run on, so only a watch that keeps polling archives before the driver step ends.
    bench = tmp_path / "run" / "b"
    service = f"until [ -e {bench / 'eval.succeeded.11'} ]; do sleep 0.1; done; exit 3"
    driver = (
        f"trap 'echo driver got TERM; exit 143' TERM; {_WRITE_EVIDENCE}; : > \"$GYM_EVAL_SUCCEEDED_MARKER\"; "
        f"for _ in $(seq 300); do [ -e {bench / 'node-local' / '11.tar.gz'} ] && exit 5; sleep 0.1; done; exit 9"
    )
    process = _start_job(
        tmp_path,
        driver=driver,
        service=service,
        node_local={"stop_before_walltime_seconds": 65},
        end_in=68,
        exports=True,
        off_batch=off_batch,
    )

    returncode = _finish(process)

    output = (tmp_path / "job.out").read_text()
    assert returncode == 5, output
    assert "driver got TERM" not in output and "stopping the driver" not in output
    assert output.count("Archived the node-local dir") == 1
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def _real_driver_fakes(tmp_path: Path, gym_exit: int) -> dict[str, str]:
    """`gym eval run` writes the evidence; the exporter (`python3 -m ...export_step`) reports what the job dir holds."""
    archive = tmp_path / "run" / "b" / "node-local" / "11.tar.gz"
    return {
        "gym": f"{_WRITE_EVIDENCE}; exit {gym_exit}",
        "python3": (
            f"if tar -tzf {archive} 2>/dev/null | grep -q ev/1.json; then echo 'exporter found the evidence archive'; "
            "else echo 'exporter found no evidence archive'; fi"
        ),
    }


def test_the_evidence_is_archived_before_the_exports_read_the_job_dir(tmp_path):
    returncode = _finish(
        _start_job(tmp_path, driver=_REAL_DRIVER, fakes=_real_driver_fakes(tmp_path, 0), exports=True)
    )

    output = (tmp_path / "job.out").read_text()
    assert returncode == 0
    assert "The evaluation succeeded: archiving the node-local dir before the exports" in output
    assert "exporter found the evidence archive" in output
    assert output.count("Archived the node-local dir") == 1
    assert _archive_names(tmp_path) == ["11.tar.gz"]
    assert not (tmp_path / "host" / "gym-node-local-11").exists()


def test_a_failed_evaluation_exports_nothing_and_is_archived_after_the_driver(tmp_path):
    returncode = _finish(
        _start_job(tmp_path, driver=_REAL_DRIVER, fakes=_real_driver_fakes(tmp_path, 3), exports=True)
    )

    output = (tmp_path / "job.out").read_text()
    assert returncode == 3
    assert "exporter" not in output and "The evaluation succeeded" not in output
    assert output.count("Archived the node-local dir") == 1
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_the_exports_wait_for_the_archive_only_with_node_local(tmp_path):
    with_node_local = _script(_config(True, exports=True))
    without = _script(_config(exports=True))
    assert 'GYM_NODE_LOCAL_ARCHIVED=/jobs/20261116_run/b/.gym_node_local_archived."$SLURM_JOB_ID" ' in (
        _driver_line(with_node_local)
    )
    # The bound: archive_timeout_seconds (120) plus the margin for the 5 s poll and the step that archives.
    assert "_gym_i < 180" in with_node_local and "_gym_i <" not in without
    assert ': > /jobs/20261116_run/b/.gym_node_local_archived."$SLURM_JOB_ID"' in with_node_local
    assert "GYM_NODE_LOCAL_ARCHIVED" not in without
    assert "GYM_NODE_LOCAL_ARCHIVED" not in _script(_config(True))


def _wait_for_text(path: Path, text: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while text not in path.read_text():
        assert time.monotonic() < deadline, f"{text!r} never appeared in {path}"
        time.sleep(0.1)


def test_an_unknown_end_time_is_reported_and_the_driver_runs_on(tmp_path):
    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"scontrol": "exit 1"}))

    assert returncode == 0
    assert "the job's end time is unknown" in (tmp_path / "job.out").read_text()
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_a_window_already_past_is_reported_and_the_driver_runs_on(tmp_path):
    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, end_in=10))

    assert returncode == 0
    assert "less than 300s remain" in (tmp_path / "job.out").read_text()
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_a_term_to_the_job_archives_the_dir_first(tmp_path):
    # The driver outlives the TERM, as a step does until Slurm kills it: only the job's own trap can archive.
    driver = f"trap '' TERM; {_WRITE_EVIDENCE}; : > \"$GYM_NODE_LOCAL_DIR/../written\"; sleep 600 & wait"
    process = _start_job(tmp_path, driver=driver, fakes={"scontrol": "exit 1"})
    try:
        _wait_for(tmp_path / "host" / "written")
        os.killpg(process.pid, signal.SIGTERM)
        returncode = _finish(process, timeout=30)
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)

    assert returncode == 143
    assert "The job got TERM: archiving the node-local dir" in (tmp_path / "job.out").read_text()
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_an_archive_that_overruns_its_bound_is_dropped_whole(tmp_path):
    # tar never finishes; the 5 s bound kills it, and no partial archive takes the final name.
    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"tar": "exec sleep 600"}))

    assert returncode == 0
    assert "archiving the node-local dir failed" in (tmp_path / "job.out").read_text()
    assert list((tmp_path / "run" / "b" / "node-local").iterdir()) == []
    assert not (tmp_path / "host" / "gym-node-local-11").exists()
