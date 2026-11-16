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
from nemo_gym.core.failure_kinds import DRIVER_NODE_LOCAL_RESTORE_FAILED
from nemo_gym.orchestration.api import BenchmarkRunConfig, NodeLocalConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.node_local import _RESTORE_SCRIPT, NODE_LOCAL_DIR_ENV, archive_wait_seconds
from nemo_gym.orchestration.executors.script_templates import NODE_LOCAL_ARCHIVE_DONE_MARKER_ENV
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


@pytest.mark.parametrize("node_local", [True, {"restore": False}])
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
    fakes: dict[str, str] | None = None,
    exports: bool = False,
    job_id: str = "11",
    off_batch: bool = False,
    env: dict[str, str] | None = None,
    services: dict | None = None,
) -> subprocess.Popen:
    """Start the rendered job.sh in its own process group.

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
    config = _config(
        node_local, exports=exports, **({"services": services} if services is not None else {}), **placement
    )
    bench = tmp_path / "run" / "b"
    bench.mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin").mkdir(exist_ok=True)
    all_fakes = {"srun": _FAKE_SRUN % {"driver": driver, "service": service}, "curl": "exit 0", **(fakes or {})}
    if off_batch:
        all_fakes["scontrol"] = "case \"$2\" in hostnames) printf 'n0\\nn1\\n' ;; *) exit 1 ;; esac"
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
            env={
                "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
                "SLURM_JOB_ID": job_id,
                "COPYFILE_DISABLE": "1",
                **(env or {}),
            },
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


def _archive(tmp_path: Path, name: str = "11.0") -> dict[str, str]:
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
    assert _archive_names(tmp_path) == ["11.0.tar.gz"]


def _archive_names(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "run" / "b" / "node-local").iterdir())


def test_restore_extracts_every_archive_by_job_id_and_the_new_one_holds_only_what_the_job_changed(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    # 10 ran after 9, though it sorts first as text: its version of a.json wins.
    _write_archive(archives / "9.0.tar.gz", {"ev/a.json": "nine\n", "ev/9.json": "row9\n"})
    _write_archive(archives / "10.0.tar.gz", {"ev/a.json": "ten\n", "ev/10.json": "row10\n"})
    driver = (
        'd="$GYM_NODE_LOCAL_DIR/ev"; [ "$(cat "$d/a.json")" = ten ] && cat "$d/9.json" "$d/10.json" '
        f'&& echo changed >> "$d/10.json" && {_WRITE_EVIDENCE}'
    )

    returncode = _finish(_start_job(tmp_path, driver=driver))

    output = (tmp_path / "job.out").read_text()
    assert returncode == 0, output
    assert output.index("Restored " + str(archives / "9.0.tar.gz")) < output.index(
        "Restored " + str(archives / "10.0")
    )
    # Restored files the job left alone stay out of its archive; the one it changed goes in.
    assert _archive(tmp_path) == {"ev/1.json": "row1\n", "ev/10.json": "row10\nchanged\n"}
    assert _archive_names(tmp_path) == ["10.0.tar.gz", "11.0.tar.gz", "9.0.tar.gz"]


def test_restore_follows_the_job_chain_then_archives_it_does_not_name(tmp_path):
    bench = tmp_path / "run" / "b"
    archives = bench / "node-local"
    # Job ids wrapped at MaxJobId: 20 ran before 9. 15 left no archive; 5 is outside the chain (a judge-only job).
    (bench).mkdir(parents=True)
    (bench / ".gym_job_chain").write_text("20\n15\n9\n11\n")
    _write_archive(archives / "20.0.tar.gz", {"ev/a.json": "first\n"})
    _write_archive(archives / "9.0.tar.gz", {"ev/a.json": "second\n"})
    _write_archive(archives / "5.0.tar.gz", {"ev/judge.json": "judged\n"})
    driver = 'd="$GYM_NODE_LOCAL_DIR/ev"; [ "$(cat "$d/a.json")" = second ] && cat "$d/judge.json"'

    returncode = _finish(_start_job(tmp_path, driver=driver))

    output = (tmp_path / "job.out").read_text()
    assert returncode == 0, output
    assert "judged" in output
    restored = [line.split("/")[-1] for line in output.splitlines() if line.startswith("Restored ")]
    assert restored == ["20.0.tar.gz", "9.0.tar.gz", "5.0.tar.gz"]
    assert "NOTE: job 15 of the run left no node-local archive" in output
    # The running job is in the chain too, and has no archive yet.
    assert "job 11 of the run" not in output


def test_a_requeued_attempt_restores_its_earlier_attempt_and_keeps_its_archive(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    _write_archive(archives / "11.0.tar.gz", {"ev/0.json": "row0\n"})

    returncode = _finish(
        _start_job(
            tmp_path,
            driver=f'cat "$GYM_NODE_LOCAL_DIR/ev/0.json" && {_WRITE_EVIDENCE}',
            env={"SLURM_RESTART_COUNT": "1"},
        )
    )

    assert returncode == 0
    assert _archive_names(tmp_path) == ["11.0.tar.gz", "11.1.tar.gz"]
    assert _archive(tmp_path, "11.1") == {"ev/1.json": "row1\n"}


def _truncated_archive(path: Path) -> None:
    """A gzip that ends mid-stream: tar may extract its first member before it fails."""
    _write_archive(path, {"ev/partial.json": "p" * 50000, "ev/other.json": bytes(range(256)).hex() * 400})
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


def test_an_archive_that_does_not_extract_stops_the_job_before_the_services(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    _write_archive(archives / "9.0.tar.gz", {"ev/0.json": "row0\n"})
    _truncated_archive(archives / "10.0.tar.gz")

    returncode = _finish(_start_job(tmp_path, driver=": > driver-ran", service=": > service-ran; exec sleep 600"))

    assert returncode == 1
    assert f"ERROR: cannot extract {archives / '10.0.tar.gz'}" in (tmp_path / "job.out").read_text()
    assert not (tmp_path / "run" / "b" / "service-ran").exists()
    assert not (tmp_path / "run" / "b" / "driver-ran").exists()
    assert not (tmp_path / "host" / "gym-node-local-11").exists()
    assert _archive_names(tmp_path) == ["10.0.tar.gz", "9.0.tar.gz"]


def test_a_failed_restore_leaves_its_failure_kind_for_the_chain(tmp_path):
    # The batch shell passes the failure file of a resumable job; the next segment's prologue stops on the kind.
    archives = tmp_path / "node-local"
    _truncated_archive(archives / "10.0.tar.gz")
    failure = tmp_path / ".gym_driver_failure.11"
    args = [str(tmp_path / "dir"), str(archives), str(tmp_path / ".gym_job_chain"), "11", "1", str(failure)]

    result = subprocess.run(["bash", "-c", _RESTORE_SCRIPT, "_", *args], capture_output=True, text=True)

    assert result.returncode == 1
    assert failure.read_text() == f"{DRIVER_NODE_LOCAL_RESTORE_FAILED}\n"
    assert not (tmp_path / "dir").exists()


def test_a_resumable_job_names_its_failure_file_to_the_restore():
    config = _config(True)
    config.driver.benchmarks["b"].resumable = True
    assert '"$SLURM_JOB_ID" 1 /jobs/20261116_run/b/.gym_driver_failure."$SLURM_JOB_ID" || exit $?' in _script(config)
    assert '"$SLURM_JOB_ID" 1 "" || exit $?' in _script(_config(True))


def test_a_service_that_fails_its_health_check_leaves_no_node_local_dir(tmp_path):
    _write_archive(tmp_path / "run" / "b" / "node-local" / "10.0.tar.gz", {"ev/0.json": "row0\n"})

    returncode = _finish(_start_job(tmp_path, driver=": > driver-ran", service="exit 3", fakes={"curl": "exit 7"}))

    assert returncode == 1
    assert "died during startup" in (tmp_path / "job.out").read_text()
    assert not (tmp_path / "host" / "gym-node-local-11").exists()
    assert not (tmp_path / "host" / "gym-node-local-11.stamp").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists a directory whatever its mode")
def test_an_archives_dir_that_cannot_be_listed_stops_the_job_before_the_driver(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    _write_archive(archives / "10.0.tar.gz", {"ev/0.json": "row0\n"})
    archives.chmod(0)
    try:
        returncode = _finish(_start_job(tmp_path, driver=": > driver-ran"))
    finally:
        archives.chmod(0o755)

    assert returncode == 1
    assert f"ERROR: cannot list {archives}" in (tmp_path / "job.out").read_text()
    assert not (tmp_path / "run" / "b" / "driver-ran").exists()


def test_without_an_archives_dir_the_dir_starts_empty(tmp_path):
    returncode = _finish(_start_job(tmp_path, driver='test -z "$(ls -A "$GYM_NODE_LOCAL_DIR")"'))

    assert returncode == 0
    assert "No earlier archive in" in (tmp_path / "job.out").read_text()


def test_restore_removes_the_partial_archive_of_a_job_cut_off_mid_archive(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    _write_archive(archives / "10.0.tar.gz", {"ev/0.json": "row0\n"})
    (archives / ".10.0.tar.gz.tmp").write_bytes(b"cut off")

    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE))

    assert returncode == 0
    assert "Removed the partial archive" in (tmp_path / "job.out").read_text()
    assert _archive_names(tmp_path) == ["10.0.tar.gz", "11.0.tar.gz"]


def test_without_restore_the_dir_starts_empty_and_every_archive_is_kept(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    for job in (8, 9, 10):
        _write_archive(archives / f"{job}.0.tar.gz", {f"ev/{job}.json": "row\n"})
    driver = f'test -z "$(ls -A "$GYM_NODE_LOCAL_DIR")" && {_WRITE_EVIDENCE}'

    returncode = _finish(_start_job(tmp_path, driver=driver, node_local={"restore": False}))

    assert returncode == 0
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}
    assert _archive_names(tmp_path) == ["10.0.tar.gz", "11.0.tar.gz", "8.0.tar.gz", "9.0.tar.gz"]


# Only writing an archive hangs; the restore still extracts.
_HANGING_TAR = 'case "$*" in *-czf*) exec sleep 600 ;; *) exec /usr/bin/tar "$@" ;; esac'


def test_a_failed_archive_leaves_no_partial_file_and_every_earlier_archive(tmp_path):
    archives = tmp_path / "run" / "b" / "node-local"
    for job in (9, 10):
        _write_archive(archives / f"{job}.0.tar.gz", {f"ev/{job}.json": "row\n"})

    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"tar": _HANGING_TAR}))

    assert returncode == 0
    assert "archiving the node-local dir failed" in (tmp_path / "job.out").read_text()
    assert _archive_names(tmp_path) == ["10.0.tar.gz", "9.0.tar.gz"]


def test_a_file_removed_before_tar_reads_it_does_not_cost_the_archive(tmp_path):
    # find lists a file that is gone by the time tar reads it, as a temporary file the driver renames can be.
    find = '/usr/bin/find "$@" && printf "./ev/gone.tmp\\0"'

    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"find": find}))

    assert returncode == 0
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_a_lost_service_still_leaves_the_archive(tmp_path):
    driver = f"{_WRITE_EVIDENCE}; trap 'exit 143' TERM; sleep 600 & wait"

    # The service dies once the driver has written its evidence.
    evidence = tmp_path / "host" / "gym-node-local-11" / "ev" / "1.json"
    service = f"until [ -e {evidence} ]; do sleep 0.1; done; exit 3"

    returncode = _finish(_start_job(tmp_path, driver=driver, service=service))

    assert returncode == 1
    assert "service vllm_model exited (status 3)" in (tmp_path / "job.out").read_text()
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


@pytest.mark.parametrize("off_batch", [False, True], ids=["batch_node", "off_batch_node"])
def test_a_service_lost_after_the_evaluation_succeeded_still_lets_the_watch_archive(tmp_path, off_batch):
    # The service dies once the evaluation has succeeded, before the watch's next poll; the exports then run on, so
    # only a watch that keeps polling after the lost service archives before the driver step ends.
    bench = tmp_path / "run" / "b"
    service = f"until [ -e {bench / 'eval.succeeded.11'} ]; do sleep 0.1; done; exit 3"
    driver = (
        f"trap 'echo driver got TERM; exit 143' TERM; {_WRITE_EVIDENCE}; : > \"$GYM_EVAL_SUCCEEDED_MARKER\"; "
        f"for _ in $(seq 300); do [ -e {bench / 'node-local' / '11.0.tar.gz'} ] && exit 5; sleep 0.1; done; exit 9"
    )
    process = _start_job(tmp_path, driver=driver, service=service, exports=True, off_batch=off_batch)

    returncode = _finish(process)

    output = (tmp_path / "job.out").read_text()
    assert returncode == 5, output
    assert "driver got TERM" not in output
    assert output.count("Archived the node-local files") == 1
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_a_requeued_attempt_does_not_take_its_earlier_attempts_markers(tmp_path):
    # The first attempt succeeded and was requeued while its exports ran: both of its markers are still there.
    bench = tmp_path / "run" / "b"
    bench.mkdir(parents=True)
    (bench / "eval.succeeded.11").touch()
    (bench / ".gym_node_local_archive_done.11").touch()
    driver = (
        'test ! -e "$GYM_EVAL_SUCCEEDED_MARKER" && test ! -e "$GYM_NODE_LOCAL_ARCHIVE_DONE_MARKER" && sleep 1 && '
        f"{_WRITE_EVIDENCE}"
    )

    returncode = _finish(_start_job(tmp_path, driver=driver, exports=True, env={"SLURM_RESTART_COUNT": "1"}))

    output = (tmp_path / "job.out").read_text()
    assert returncode == 0, output
    assert "The evaluation succeeded" not in output
    assert _archive(tmp_path, "11.1") == {"ev/1.json": "row1\n"}


def _real_driver_fakes(tmp_path: Path, gym_exit: int) -> dict[str, str]:
    """`gym eval run` writes the evidence; the exporter (`python3 -m ...export_step`) reports what the job dir holds."""
    archive = tmp_path / "run" / "b" / "node-local" / "11.0.tar.gz"
    return {
        "gym": f"{_WRITE_EVIDENCE}; exit {gym_exit}",
        "python3": (
            f"if tar -tzf {archive} 2>/dev/null | grep -q ev/1.json; then echo 'exporter found the evidence archive'; "
            "else echo 'exporter found no evidence archive'; fi; "
            f"! ls -A {archive.parent.parent} | grep -q archive_done || echo 'exporter found the marker'"
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
    assert output.count("Archived the node-local files") == 1
    assert _archive_names(tmp_path) == ["11.0.tar.gz"]
    assert not (tmp_path / "host" / "gym-node-local-11").exists()
    # The wait takes the marker before the exports run, so they never upload it.
    assert "exporter found the marker" not in output
    assert not list((tmp_path / "run" / "b").glob(".gym_node_local_archive_done*"))


def test_a_failed_evaluation_exports_nothing_and_is_archived_after_the_driver(tmp_path):
    returncode = _finish(
        _start_job(tmp_path, driver=_REAL_DRIVER, fakes=_real_driver_fakes(tmp_path, 3), exports=True)
    )

    output = (tmp_path / "job.out").read_text()
    assert returncode == 3
    assert "exporter" not in output and "The evaluation succeeded" not in output
    assert output.count("Archived the node-local files") == 1
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_only_a_job_with_node_local_and_exports_waits_for_the_archive():
    assert NODE_LOCAL_ARCHIVE_DONE_MARKER_ENV in _driver_line(_script(_config(True, exports=True)))
    assert NODE_LOCAL_ARCHIVE_DONE_MARKER_ENV not in _script(_config(exports=True))
    assert NODE_LOCAL_ARCHIVE_DONE_MARKER_ENV not in _script(_config(True))
    # The bound: archive_timeout_seconds plus the margin for the watch's 5 s poll and the step that archives.
    assert archive_wait_seconds(NodeLocalConfig(archive_timeout_seconds=120)) == 180


def test_a_term_to_the_job_archives_the_dir_first(tmp_path):
    # The driver outlives the TERM, as a step does until Slurm kills it: only the job's own trap can archive.
    driver = f"trap '' TERM; {_WRITE_EVIDENCE}; : > \"$GYM_NODE_LOCAL_DIR/../written\"; sleep 600 & wait"
    process = _start_job(tmp_path, driver=driver)
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


def test_a_term_that_lands_while_the_archive_runs_does_not_cut_it_short(tmp_path):
    # Slurm's TERM reaches the whole job, the archive's tar included, while the archive after the driver step runs.
    started = tmp_path / "tar-started"
    slow_tar = f'case "$*" in *-czf*) : > {started}; sleep 3 ;; esac; exec /usr/bin/tar "$@"'
    process = _start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"tar": slow_tar})
    try:
        _wait_for(started)
        os.killpg(process.pid, signal.SIGTERM)
        returncode = _finish(process, timeout=30)
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)

    output = (tmp_path / "job.out").read_text()
    assert returncode == 143, output
    assert "archiving the node-local dir failed" not in output
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


@pytest.mark.parametrize(("tar_exit", "kept"), [(1, True), (2, False)])
def test_a_tar_that_exits_1_keeps_the_archive_and_any_other_failure_drops_it(tmp_path, tar_exit, kept):
    # GNU tar exits 1 when a file changed while it was read; the archive it wrote is still complete.
    tar = f'case "$*" in *-czf*) /usr/bin/tar "$@" && exit {tar_exit} ;; esac; exec /usr/bin/tar "$@"'

    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"tar": tar}))

    assert returncode == 0
    assert _archive_names(tmp_path) == (["11.0.tar.gz"] if kept else [])
    assert ("archiving the node-local dir failed" in (tmp_path / "job.out").read_text()) is not kept


def test_a_term_to_a_job_without_services_archives_at_once(tmp_path):
    # The watch keeps the driver in the background, so the batch shell's TERM trap does not wait for the driver.
    driver = f"trap '' TERM; {_WRITE_EVIDENCE}; : > \"$GYM_NODE_LOCAL_DIR/../written\"; sleep 600 & wait"
    process = _start_job(tmp_path, driver=driver, services={})
    try:
        _wait_for(tmp_path / "host" / "written")
        os.killpg(process.pid, signal.SIGTERM)
        returncode = _finish(process, timeout=30)
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)

    assert returncode == 143
    assert _archive(tmp_path) == {"ev/1.json": "row1\n"}


def test_an_archive_that_overruns_its_bound_is_dropped_whole(tmp_path):
    # tar never finishes; the 5 s bound kills it, and no partial archive takes the final name.
    returncode = _finish(_start_job(tmp_path, driver=_WRITE_EVIDENCE, fakes={"tar": _HANGING_TAR}))

    assert returncode == 0
    assert "archiving the node-local dir failed" in (tmp_path / "job.out").read_text()
    assert list((tmp_path / "run" / "b" / "node-local").iterdir()) == []
    assert not (tmp_path / "host" / "gym-node-local-11").exists()
