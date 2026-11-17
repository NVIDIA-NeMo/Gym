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
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import script_templates
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.lustre_md_stats import analyze


_SERVICE = {"type": "vllm", "container": "vllm:latest", "model": "org/model"}


def _config(tmp_path: Path, *, md_stats: bool = True, nodes: int = 0, services=None, resumable=False, otel=False):
    pools = {"gpu": {"partition": "batch", "nodes": nodes}} if nodes else {}
    return SubmitConfig.model_validate(
        {
            "services": {"vllm_model": _SERVICE} if services is None else services,
            "compute": {"cluster": {"type": "slurm", "account": "acct", "node_pools": pools}},
            "driver": {"container": "gym:latest", "benchmarks": {"b": {"run": {}, "resumable": resumable}}},
            "job": {"output_path": str(tmp_path / "jobs"), **({"lustre_md_stats": True} if md_stats else {})},
            "otel": {"enabled": otel},
        }
    )


def _script(tmp_path: Path, **kwargs) -> str:
    config = _config(tmp_path, **kwargs)
    return build_sbatch_script(config, "b", config.driver.benchmarks["b"], config.compute["cluster"], tmp_path)


def _snapshot_lines(script: str) -> list[int]:
    lines = script.splitlines()
    return [i for i, line in enumerate(lines) if re.fullmatch(r"_gym_md_stats \w+", line)]


def _without_snapshots(script: str) -> str:
    """The script with the snapshot function and every call to it taken out, blank-line runs collapsed."""
    script = re.sub(r"# --- Lustre metadata stats.*?\n_gym_md_stats start\n", "", script, flags=re.DOTALL)
    script = re.sub(r"^_gym_md_stats \w+\n?", "", script, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", script).rstrip("\n")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_off_by_default_the_script_has_no_snapshot(tmp_path):
    assert _config(tmp_path, md_stats=False).job.lustre_md_stats is False
    script = _script(tmp_path, md_stats=False)
    assert "md_stats" not in script
    assert "lctl" not in script


@pytest.mark.parametrize(
    "layout",
    [{}, {"nodes": 2, "otel": True}, {"resumable": True}, {"resumable": True, "otel": True}, {"services": {}}],
    ids=["one_node", "two_nodes_observed", "resumable", "resumable_observed", "no_services"],
)
def test_on_adds_only_the_snapshots(tmp_path, layout):
    on, off = _script(tmp_path, **layout), _script(tmp_path, md_stats=False, **layout)
    assert _without_snapshots(on) == re.sub(r"\n{3,}", "\n\n", off).rstrip("\n")
    assert [on.splitlines()[i] for i in _snapshot_lines(on)] == [
        "_gym_md_stats start",
        "_gym_md_stats services_ready",
        "_gym_md_stats driver_end",
        "_gym_md_stats end",
    ]


def test_start_follows_the_resume_prologue_and_precedes_every_service(tmp_path):
    script = _script(tmp_path, resumable=True, otel=True)
    start = script.index("\n_gym_md_stats start\n")
    assert script.index("# --- Auto-resume chain ---") < start
    assert start < script.index("# service: otel_collector") < script.index("# service: vllm_model")


def test_services_ready_follows_the_health_checks_and_precedes_the_driver(tmp_path):
    script = _script(tmp_path, otel=True)
    ready = script.index("\n_gym_md_stats services_ready\n")
    assert script.rindex("did not become healthy") < ready < script.index("logs/driver-")


def test_driver_end_follows_the_driver_and_end_is_last_before_the_exit(tmp_path):
    lines = _script(tmp_path, resumable=True, otel=True).splitlines()
    assert lines[lines.index("_gym_md_stats driver_end") - 1] == "_gym_run_driver"
    # The epilogue reads the driver's status after the snapshot, which returns it.
    after = lines[lines.index("_gym_md_stats driver_end") + 1]
    assert after == "# --- A job that finishes the run ends its own chain ---"
    assert lines[-2:] == ["_gym_md_stats end", "exit $DRIVER_RC"]


def test_without_the_collector_end_is_the_last_line(tmp_path):
    assert _script(tmp_path).rstrip("\n").splitlines()[-2:] == ["_gym_md_stats driver_end", "_gym_md_stats end"]


def test_a_multi_node_job_reads_every_node_in_one_host_step(tmp_path):
    script = _script(tmp_path, nodes=3)
    function = script[script.index("_gym_md_stats() {") : script.index("\n}\n", script.index("_gym_md_stats() {"))]
    [step] = [line for line in function.splitlines() if "srun" in line]
    assert (
        "timeout -k 5 30 srun --overlap --nodes=3 --ntasks=3 --ntasks-per-node=1 --account=acct "
        'bash -c "$_gym_md_stats_node" _gym_md_stats "$1"' in step
    )
    assert "--container" not in step


def test_a_one_node_job_reads_in_the_batch_shell(tmp_path):
    script = _script(tmp_path)
    function = script[script.index("_gym_md_stats() {") : script.index("\n}\n", script.index("_gym_md_stats() {"))]
    assert "srun" not in function
    assert 'timeout -k 5 30 bash -c "$_gym_md_stats_node" _gym_md_stats "$1"' in function


def test_the_rows_go_to_one_file_per_slurm_job_in_the_logs_dir(tmp_path):
    assert f'_gym_md_stats_file={tmp_path}/logs/lustre-md-stats-"$SLURM_JOB_ID".tsv' in _script(tmp_path)


# ---------------------------------------------------------------------------
# Running the rendered job.sh with fake srun, lctl and timeout
# ---------------------------------------------------------------------------

# Two MDTs; each call returns larger counts (per node, scaled by the node number). getattr is 0, so omitted, on the
# first call, as lctl omits every op with a zero count.
_FAKE_LCTL = r"""#!/bin/bash
[ "$1 $2" = "get_param mdc.*.md_stats" ] || exit 2
node=${SLURMD_NODENAME:-node1}
f=${node#node}
state="$FAKE_STATE/lctl-$node"
n=$(( $(cat "$state" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$state"
cat <<EOF
mdc.scratch-MDT0000-mdc-ffff8b5a1c2e3000.md_stats=
snapshot_time             1763380000.123456789 secs.nsecs
start_time                1763000000.000000000 secs.nsecs
elapsed_time              380000.123456789 secs.nsecs
close                     $((10 * f * n)) samples [reqs]
EOF
[ "$n" -eq 1 ] || echo "getattr                   $((5 * f * (n - 1))) samples [reqs]"
cat <<EOF
mdc.scratch-MDT000b-mdc-ffff8b5a1c2e3000.md_stats=
snapshot_time             1763380000.123456789 secs.nsecs
start_time                1763000000.000000000 secs.nsecs
elapsed_time              380000.123456789 secs.nsecs
intent_lock               $((1000 * f * n)) samples [reqs]
EOF
"""

# The snapshot step (the only one with --ntasks-per-node=1) runs its command once per node, as nodeN.
_FAKE_SRUN = """#!/bin/bash
case "$*" in
    *--ntasks-per-node=1*)
        for arg in "$@"; do case "$arg" in --nodes=*) nodes=${{arg#--nodes=}} ;; esac; done
        while [[ "$1" == --* ]]; do shift; done
        echo "md step on $nodes nodes" >> "$FAKE_STATE/srun.log"
        for i in $(seq 1 "$nodes"); do SLURMD_NODENAME=node$i "$@" || rc=$?; done
        exit "${{rc:-0}}" ;;
    *logs/driver-*) {driver} ;;
    *) {service} ;;
esac
"""


def _run_job(
    tmp_path: Path,
    *,
    nodes: int = 0,
    services=None,
    driver: str = "exit 0",
    service: str = "exec sleep 600",
    lctl: str | None = _FAKE_LCTL,
    fake_timeout: bool = True,
) -> subprocess.CompletedProcess:
    """Run a rendered job.sh with `lustre_md_stats` on; `lctl=None` leaves lctl off the PATH."""
    config = _config(tmp_path, nodes=nodes, services={} if services is None else services)
    script = build_sbatch_script(config, "b", config.driver.benchmarks["b"], config.compute["cluster"], tmp_path)
    (tmp_path / "logs").mkdir()
    state = tmp_path / "state"
    state.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fakes = {
        "srun": _FAKE_SRUN.format(driver=driver, service=service),
        "curl": "#!/bin/bash\nexit 0\n",
        "scontrol": "#!/bin/bash\nfor i in $(seq 1 4); do echo node$i; done\n",
    }
    if lctl is not None:
        fakes["lctl"] = lctl
    if fake_timeout:
        fakes["timeout"] = '#!/bin/bash\n[ "$1" = -k ] && shift 2\nshift\nexec "$@"\n'
    else:
        (bin_dir / "timeout").symlink_to(shutil.which("timeout"))
    for name, body in fakes.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    (tmp_path / "job.sh").write_text(script)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "SLURM_JOB_ID": "77", "FAKE_STATE": str(state)}
    if not nodes:
        env["SLURMD_NODENAME"] = "node1"
    with (tmp_path / "job.out").open("w") as out:
        process = subprocess.Popen(
            ["bash", "job.sh"], stdout=out, stderr=subprocess.STDOUT, cwd=tmp_path, env=env, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=60)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    return subprocess.CompletedProcess(["job.sh"], returncode, (tmp_path / "job.out").read_text())


def _rows(tmp_path: Path) -> list[tuple[str, str, str, str, int]]:
    """The stats file's rows without their epoch, which is checked to be whole seconds."""
    rows = []
    for line in (tmp_path / "logs" / "lustre-md-stats-77.tsv").read_text().splitlines():
        phase, epoch, node, mdt, op, count = line.split("\t")
        assert epoch.isdigit()
        rows.append((phase, node, mdt, op, int(count)))
    return rows


def test_a_one_node_job_records_every_phase_with_omitted_ops_left_out(tmp_path):
    result = _run_job(tmp_path, driver="exit 7")

    assert result.returncode == 7
    assert "WARNING" not in result.stdout
    assert _rows(tmp_path) == [
        ("start", "node1", "MDT0000", "close", 10),
        ("start", "node1", "MDT000b", "intent_lock", 1000),
        ("services_ready", "node1", "MDT0000", "close", 20),
        ("services_ready", "node1", "MDT0000", "getattr", 5),
        ("services_ready", "node1", "MDT000b", "intent_lock", 2000),
        ("driver_end", "node1", "MDT0000", "close", 30),
        ("driver_end", "node1", "MDT0000", "getattr", 10),
        ("driver_end", "node1", "MDT000b", "intent_lock", 3000),
        ("end", "node1", "MDT0000", "close", 40),
        ("end", "node1", "MDT0000", "getattr", 15),
        ("end", "node1", "MDT000b", "intent_lock", 4000),
    ]
    # The snapshots ran in the batch shell, not in a step.
    assert not (tmp_path / "state" / "srun.log").exists()


def test_a_one_node_job_with_a_service_snapshots_around_it_and_keeps_the_driver_status(tmp_path):
    result = _run_job(tmp_path, services={"vllm_model": _SERVICE}, driver="exit 3")

    assert result.returncode == 3
    assert list(dict.fromkeys(row[0] for row in _rows(tmp_path))) == ["start", "services_ready", "driver_end", "end"]
    assert "vllm_model ready." in result.stdout


def test_a_two_node_job_records_each_node_through_one_step_per_phase(tmp_path):
    result = _run_job(tmp_path, nodes=2)

    assert result.returncode == 0
    rows = _rows(tmp_path)
    assert {(phase, node) for phase, node, *_ in rows} == {
        (phase, node) for phase in ("start", "services_ready", "driver_end", "end") for node in ("node1", "node2")
    }
    assert ("driver_end", "node2", "MDT000b", "intent_lock", 6000) in rows
    assert (tmp_path / "state" / "srun.log").read_text().splitlines() == ["md step on 2 nodes"] * 4
    # Read back: node2 makes twice node1's requests.
    services_to_driver = analyze(tmp_path / "logs" / "lustre-md-stats-77.tsv")["intervals"][1]
    assert services_to_driver["requests"] == (10 + 5 + 1000) * 3
    assert services_to_driver["by_mdt"] == {"MDT000b": 3000, "MDT0000": 45}


@pytest.mark.parametrize("driver_status", [0, 9])
def test_a_failing_lctl_warns_once_per_phase_and_keeps_the_job_status(tmp_path, driver_status):
    result = _run_job(tmp_path, driver=f"exit {driver_status}", lctl="#!/bin/bash\necho broken >&2\nexit 1\n")

    assert result.returncode == driver_status
    warnings = [line for line in result.stdout.splitlines() if line.startswith("WARNING: lustre_md_stats")]
    assert [line.split()[3] for line in warnings] == ["start", "services_ready", "driver_end", "end"]
    assert not (tmp_path / "logs" / "lustre-md-stats-77.tsv").exists()


@pytest.mark.skipif(Path("/usr/sbin/lctl").exists() or Path("/sbin/lctl").exists(), reason="this host has lctl")
def test_a_missing_lctl_warns_and_keeps_the_job_status(tmp_path):
    result = _run_job(tmp_path, driver="exit 4", lctl=None)

    assert result.returncode == 4
    assert result.stdout.count("WARNING: lustre_md_stats") == 4


def test_one_node_that_fails_keeps_the_rows_of_the_others(tmp_path):
    node = "node=${SLURMD_NODENAME:-node1}\n"
    lctl = _FAKE_LCTL.replace(node, f'{node}[ "$node" = node2 ] && exit 1\n')
    result = _run_job(tmp_path, nodes=2, lctl=lctl)

    assert result.returncode == 0
    assert result.stdout.count("WARNING: lustre_md_stats") == 4
    assert {node for _, node, *_ in _rows(tmp_path)} == {"node1"}


@pytest.mark.skipif(shutil.which("timeout") is None, reason="needs coreutils timeout")
def test_a_hung_read_is_cut_off_and_the_job_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr(script_templates, "LUSTRE_MD_STATS_TIMEOUT_S", 1)
    began = time.monotonic()
    result = _run_job(tmp_path, driver="exit 5", lctl="#!/bin/bash\nexec sleep 60\n", fake_timeout=False)

    assert result.returncode == 5
    assert result.stdout.count("WARNING: lustre_md_stats") == 4
    assert time.monotonic() - began < 30
