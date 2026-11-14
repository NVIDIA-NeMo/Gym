# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The heavy remote readers in a Slurm step (NEMO_GYM_REMOTE_READER_SRUN), with fake srun, squeue and scancel."""

import inspect
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

import nemo_gym.orchestration.executors.slurm as slurm
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.recovery import (
    READER_PLACEMENT_FIELDS,
    CollectedEvidence,
    decide_from_collected,
    gate_summary_from_collected,
)
from tests.unit_tests.orchestration.test_reader_cleanup import _token_processes, _wait_until, needs_setsid
from tests.unit_tests.orchestration.test_remote_managed_recovery import ProjectionConnection, config


pytestmark = pytest.mark.skipif(shutil.which("timeout") is None, reason="coreutils timeout not installed")

_STARTED_SRUN = """#!/bin/bash
echo "$@" >> "$FAKE_LOG/srun"
while [[ $# -gt 0 && $1 != bash ]]; do shift; done
export SLURM_JOB_ID=4242
exec "$@"
"""
_REFUSED_SRUN = """#!/bin/bash
echo "$@" >> "$FAKE_LOG/srun"
echo "srun: error: Unable to allocate resources: Requested node configuration is not available" >&2
exit 1
"""
# Lists the configured reader jobs; a lookup by name (the wait after a cancel) finds nothing.
_SQUEUE = """#!/bin/bash
[[ -n $FAKE_SQUEUE_FAILS ]] && exit 1
[[ " $* " == *" -n "* && -n $FAKE_SQUEUE_STEP_FAILS ]] && exit 1
[[ " $* " == *" -n "* && -n $FAKE_SQUEUE_STEP_QUEUED ]] && { echo 99; exit 0; }
[[ " $* " == *" -n "* ]] && exit 0
cat "$FAKE_LOG/queue" 2>/dev/null
exit 0
"""
# The step runs on another host: a signal to srun's process group does not reach it, and only scancel ends it.
_DETACHED_SRUN = """#!/bin/bash
echo "$@" >> "$FAKE_LOG/srun"
while [[ $# -gt 0 && $1 != bash ]]; do shift; done
export SLURM_JOB_ID=4242
setsid "$@" <&0 &
echo $! > "$FAKE_LOG/step.pid"
wait $!
"""
_SCANCEL = """#!/bin/bash
echo "$@" >> "$FAKE_LOG/scancel"
[[ -f $FAKE_LOG/step.pid ]] && kill -TERM -- -"$(cat "$FAKE_LOG/step.pid")" 2>/dev/null
exit 0
"""
HOST_REPORT = "__GYM_READER_HOST:cpu:99\n__GYM_READER_RC:0\n"


@pytest.fixture
def fake_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A bin dir with fake srun, squeue and scancel, first on PATH; returns a function that picks the srun."""
    bin_dir, log = tmp_path / "bin", tmp_path / "log"
    bin_dir.mkdir()
    log.mkdir()
    for name, body in (("squeue", _SQUEUE), ("scancel", _SCANCEL)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("USER", "tester")
    monkeypatch.delenv("FAKE_SQUEUE_FAILS", raising=False)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN", "-p cpu -c 4")
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN_WAIT_S", "1")
    # These tests pin the login-host fallback where it is allowed; test_reader_login_fallback covers the default.
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK", "allow")
    # The cap tests set their own queue wait.
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S", "0")

    def use(srun: str) -> Path:
        (bin_dir / "srun").write_text(srun)
        (bin_dir / "srun").chmod(0o755)
        return log

    return use


def _reader(tmp_path: Path, payload: str = "PAYLOAD-LINE") -> list[str]:
    # The reader is its own process and prints what it reads on FD 3, as the managed Store reader reads its input.
    return slurm._reader_command(
        "bash -c 'cat <&3'", tmp_path, payload, bound_s=300, factor=1, token="tok", step=slurm.reader_step()
    )


def test_unset_keeps_the_reader_commands_as_they_were(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_GYM_REMOTE_READER_SRUN", raising=False)
    assert slurm.reader_step() is None
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN", "  ")
    assert slurm.reader_step() is None

    command = slurm._reader_command("ENTRY", tmp_path, "PAYLOAD", bound_s=300, factor=1, token="tok", step=None)

    assert command == slurm._reader_command("ENTRY", tmp_path, "PAYLOAD", bound_s=300, factor=1, token="tok")
    assert not any(word in "\n".join(command) for word in ("srun", "squeue", "scancel", "__GYM_READER_HOST"))
    assert 'timeout --kill-after=10s "${_gym_bound}s" ENTRY 3<<\'GYM_MANAGED_INPUT\' 2>"$_gym_err"\nPAYLOAD\n' in (
        "\n".join(command)
    )


def test_the_step_command_carries_the_limits_the_name_and_the_fd3_redirect(fake_slurm, tmp_path: Path) -> None:
    script = "\n".join(_reader(tmp_path))
    srun = next(line for line in script.splitlines() if " srun " in line)

    assert "srun -p cpu -c 4 --ntasks=1 --immediate=1" in srun
    # The time limit is the reader bound plus 60 s, in whole minutes rounded up; the name carries the token.
    assert '-t "$(( (_gym_bound + 119) / 60 ))" -J gym-reader-tok' in srun
    # Inside the step, stdin becomes FD 3 and the reader keeps its own bound, passed in as $1.
    assert 'exec 3<&0 </dev/null; timeout --kill-after=10s "${1}s" bash -c' in srun
    assert 'gym-reader "$_gym_bound" <"$_gym_payload"' in srun


def test_a_started_step_reads_its_payload_on_fd3_and_reports_its_job(fake_slurm, tmp_path: Path) -> None:
    log = fake_slurm(_STARTED_SRUN)

    output = LocalConnection().run(_reader(tmp_path))

    assert "PAYLOAD-LINE" in output.splitlines() and "__GYM_READER_RC:0" in output
    assert slurm.reader_placement(output) == ("cpu:4242", None)
    assert (log / "srun").exists() and not (log / "scancel").exists()


def test_a_step_that_does_not_start_is_cancelled_and_read_on_the_login_host(fake_slurm, tmp_path: Path) -> None:
    log = fake_slurm(_REFUSED_SRUN)

    output = LocalConnection().run(_reader(tmp_path))

    assert (log / "scancel").read_text().split() == ["--name", "gym-reader-tok", "--user", "tester"]
    assert slurm.reader_placement(output) == (
        "login",
        "the reader step did not start within 1 s and was cancelled "
        "(srun: error: Unable to allocate resources: Requested node configuration is not available)",
    )
    # The login-host reader got the same payload on FD 3.
    assert "PAYLOAD-LINE" in output.splitlines() and "__GYM_READER_RC:0" in output


def test_a_reader_in_a_step_stops_at_its_bound_as_a_timeout(fake_slurm, tmp_path: Path) -> None:
    fake_slurm(_STARTED_SRUN)
    command = slurm._reader_command(
        "sleep 30", tmp_path, "{}", bound_s=1, factor=1, token="tok", step=slurm.reader_step()
    )

    output = LocalConnection().run(command)

    # Exit code 124 at the bound, as on the login host, so the one retry with twice the bound still applies.
    assert "__GYM_READER_RC:124" in output
    assert slurm.reader_placement(output) == ("cpu:4242", None)
    assert slurm._reader_failure(output, 1.0)[1] is True


@pytest.mark.parametrize(
    ("queue", "fails", "reason"),
    [
        ("gym-reader-a\ngym-reader-b\nother-job\n", False, "2 reader steps are already running or pending"),
        ("", True, "squeue failed, so the running reader steps cannot be counted"),
    ],
)
def test_the_cap_and_a_failed_squeue_keep_the_reader_on_the_login_host(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queue: str, fails: bool, reason: str
) -> None:
    log = fake_slurm(_STARTED_SRUN)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_MAX", "2")
    (log / "queue").write_text(queue)
    if fails:
        monkeypatch.setenv("FAKE_SQUEUE_FAILS", "1")

    output = LocalConnection().run(_reader(tmp_path))

    assert slurm.reader_placement(output) == ("login", reason)
    assert not (log / "srun").exists()
    assert "PAYLOAD-LINE" in output.splitlines()


@pytest.mark.parametrize(("srun", "host"), [(_STARTED_SRUN, "cpu:4242"), (_REFUSED_SRUN, "login")])
def test_the_collector_keeps_its_own_output_and_reports_where_it_ran(fake_slurm, tmp_path: Path, srun, host) -> None:
    fake_slurm(srun)
    command = slurm.collector_command("echo '{\"collected\": true}'", tmp_path, slurm.reader_step(), "tok")

    output = LocalConnection().run(command)

    assert slurm.reader_placement(output)[0] == host
    assert slurm.without_placement(output) == '{"collected": true}'


@pytest.mark.parametrize(
    ("srun_args", "report", "reader_host", "echoed"),
    [
        pytest.param(None, "", None, False, id="unset-reader-predates-the-fields"),
        pytest.param("-p cpu", HOST_REPORT, "cpu:99", True, id="step-reader-echoes-them-empty"),
    ],
)
def test_the_host_keeps_the_placement_and_the_reader_payload_carries_none_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, srun_args, report: str, reader_host, echoed: bool
) -> None:
    if srun_args is None:
        monkeypatch.delenv("NEMO_GYM_REMOTE_READER_SRUN", raising=False)
    else:
        monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN", srun_args)
    collected = CollectedEvidence(
        materialized=None,
        results=[],
        failures=[],
        source_job_identity=(str(tmp_path), 1, 2),
        collector_host="cpu:7",
    )
    # A pinned reader echoes the evidence it received: without the placement fields when its model predates them,
    # else with them empty, since the payload carries none.
    projection = collected.model_dump(mode="json", exclude=set(READER_PLACEMENT_FIELDS))
    if echoed:
        projection |= dict.fromkeys(READER_PLACEMENT_FIELDS)
    conn = ProjectionConnection(f"{report}__GYM_MANAGED:{json.dumps(projection)}\n")

    result = slurm._remote_managed_evidence(conn, collected, tmp_path, config())

    assert result.managed_reader_error is None
    assert (result.collector_host, result.reader_host) == ("cpu:7", reader_host)
    assert not any(name in "\n".join(conn.commands) for name in READER_PLACEMENT_FIELDS)


def test_the_gate_task_reports_where_the_readers_ran_only_with_a_reader_step() -> None:
    plain = CollectedEvidence.model_validate_json(
        (Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json").read_text()
    )
    placed = plain.model_copy(
        update={"collector_host": "cpu:7", "reader_host": "login", "reader_fallback_reason": "squeue failed"}
    )

    def task(collected: CollectedEvidence) -> dict:
        decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
        return gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)

    assert {name: task(placed).get(name) for name in READER_PLACEMENT_FIELDS} == {
        "collector_host": "cpu:7",
        "collector_fallback_reason": None,
        "reader_host": "login",
        "reader_fallback_reason": "squeue failed",
    }
    assert "collector_fallback_reason" not in task(placed)
    assert not set(READER_PLACEMENT_FIELDS) & set(task(plain))


def _sleeping_readers(token: str) -> list[int]:
    """The test reader (``time.sleep``) and its wrappers: processes that carry the token and the sleep in argv."""
    found = []
    for pid in _token_processes(token):
        try:
            if b"time.sleep" in Path(f"/proc/{pid}/cmdline").read_bytes():
                found.append(pid)
        except OSError:
            continue
    return found


@needs_setsid
@pytest.mark.parametrize(
    ("srun", "host"),
    [pytest.param(_STARTED_SRUN, "cpu:4242", id="step"), pytest.param(_REFUSED_SRUN, "login", id="login-fallback")],
)
def test_a_watched_reader_with_a_step_reports_as_before_over_a_held_input(
    fake_slurm, tmp_path: Path, srun: str, host: str
) -> None:
    fake_slurm(srun)
    command = slurm._reader_command(
        "bash -c 'cat <&3'",
        tmp_path,
        "PAYLOAD-LINE",
        bound_s=300,
        factor=1,
        token="tok",
        watchdog=True,
        step=slurm.reader_step(),
    )
    result: dict[str, str] = {}
    runner = threading.Thread(target=lambda: result.update(output=LocalConnection().run_with_input_held(command)))
    runner.start()
    runner.join(timeout=30)

    assert not runner.is_alive(), "the watched reader did not end the shell"
    assert "PAYLOAD-LINE" in result["output"].splitlines() and "__GYM_READER_RC:0" in result["output"]
    assert slurm.reader_placement(result["output"])[0] == host


@needs_setsid
@pytest.mark.parametrize(
    ("srun", "cancelled"),
    [pytest.param(_DETACHED_SRUN, True, id="step"), pytest.param(_REFUSED_SRUN, False, id="login-fallback")],
)
def test_the_watcher_ends_a_reader_with_a_step_when_the_shell_input_closes(
    fake_slurm, tmp_path: Path, srun: str, cancelled: bool
) -> None:
    log = fake_slurm(srun)
    token = f"token-{uuid.uuid4().hex}"
    command = slurm._reader_command(
        f"{sys.executable} -c 'import time; time.sleep(300)'",
        tmp_path,
        "{}",
        bound_s=300,
        factor=1,
        token=token,
        watchdog=True,
        step=slurm.reader_step(),
    )
    shell = subprocess.Popen(
        ["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        shell.stdin.write("\n".join(command) + "\n")
        shell.stdin.flush()
        assert _wait_until(lambda: _sleeping_readers(token), 15), "the reader never started"

        shell.stdin.close()  # The client, or its SSH connection, is gone.
        shell.communicate(timeout=30)

        assert _wait_until(lambda: not _sleeping_readers(token), 15), "the reader outlived the closed input"
        if cancelled:
            # Only the cancel by name reaches a step on another host.
            assert (log / "scancel").read_text().split() == ["--name", f"gym-reader-{token}", "--user", "tester"]
    finally:
        shell.kill()
        for pid in _token_processes(token):
            os.kill(pid, signal.SIGKILL)


def test_only_the_two_heavy_reader_sites_use_the_reader_step() -> None:
    """The collector and the managed Store reader; the lock, fill, accounting and requeue steps stay on the login
    host."""
    source = inspect.getsource(slurm)
    calls = [m.start() for m in re.finditer(r"(?<!def )\breader_step\(\)", source)]
    assert len(calls) == 2
    assert "step=reader_step()" in inspect.getsource(slurm._remote_managed_evidence)
    assert "step = reader_step()" in inspect.getsource(slurm.SlurmExecutor.recovery_evidence)


@pytest.mark.parametrize(("raw", "expected"), [("", 120), ("30", 30), ("0", 120), ("x", 120)])
def test_the_wait_and_the_cap_take_their_defaults_for_unset_or_invalid_values(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN", "-p cpu")
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN_WAIT_S", raw)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_MAX", raw)
    step = slurm.reader_step()
    assert step.wait_s == expected and step.max_readers == (8 if expected == 120 else expected)


@pytest.mark.parametrize("env", ["FAKE_SQUEUE_STEP_QUEUED", "FAKE_SQUEUE_STEP_FAILS"])
def test_an_unconfirmed_cancellation_starts_no_second_reader(fake_slurm, tmp_path: Path, monkeypatch, env) -> None:
    fake_slurm(_REFUSED_SRUN)
    monkeypatch.setattr(slurm, "_READER_CANCEL_WAIT_S", 2)
    monkeypatch.setenv(env, "1")

    output = LocalConnection().run(_reader(tmp_path))

    # The step may still start, so the login host does not read too: the reader fails and says why.
    assert "PAYLOAD-LINE" not in output.splitlines()
    assert "__GYM_READER_RC:1" in output
    assert "was not confirmed gone" in output
    assert slurm.reader_placement(output)[0] == "unconfirmed"


def test_a_reader_at_the_cap_queues_and_runs_in_its_step_once_a_reader_ends(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = fake_slurm(_STARTED_SRUN)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_MAX", "2")
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S", "60")
    (log / "queue").write_text("gym-reader-a\ngym-reader-b\n")
    threading.Timer(3, lambda: (log / "queue").write_text("gym-reader-a\n")).start()

    result = subprocess.run(["bash", "-s"], input="\n".join(_reader(tmp_path)), text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert slurm.reader_placement(result.stdout) == ("cpu:4242", None)
    assert (log / "srun").exists()
    assert "gym-reader progress: queued behind 2 reader steps (cap 2), waited 0 s of 60 s" in result.stderr


def test_a_reader_still_at_the_cap_after_its_queue_wait_says_how_long_it_queued(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = fake_slurm(_STARTED_SRUN)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_MAX", "2")
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S", "4")
    (log / "queue").write_text("gym-reader-a\ngym-reader-b\ngym-reader-c\n")

    output = LocalConnection().run(_reader(tmp_path))

    assert slurm.reader_placement(output) == (
        "login",
        "2 reader steps are already running or pending (still 3 after queueing 4 s)",
    )
    assert not (log / "srun").exists()


@pytest.mark.parametrize(("raw", "expected"), [("", 600), ("0", 0), ("90", 90), ("-5", 600), ("x", 600)])
def test_the_queue_wait_defaults_to_ten_minutes_and_zero_does_not_wait(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN", "-p cpu")
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S", raw)
    assert slurm.reader_step().queue_wait_s == expected


_REFUSE_THEN_START_SRUN = """#!/bin/bash
echo "$@" >> "$FAKE_LOG/srun"
n=$(cat "$FAKE_LOG/refusals" 2>/dev/null || echo 0)
if (( n > 0 )); then
    echo $(( n - 1 )) > "$FAKE_LOG/refusals"
    echo "srun: error: Unable to allocate resources: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)" >&2
    exit 1
fi
while [[ $# -gt 0 && $1 != bash ]]; do shift; done
export SLURM_JOB_ID=4242
exec "$@"
"""


@pytest.mark.parametrize("watched", [False, True], ids=["plain", "watched"])
def test_a_reader_step_refused_by_a_submit_limit_is_submitted_again_until_it_starts(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, watched: bool
) -> None:
    log = fake_slurm(_REFUSE_THEN_START_SRUN)
    (log / "refusals").write_text("2")
    monkeypatch.setattr(slurm, "_READER_SRUN_RETRY_S", 1)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S", "30")
    command = slurm._reader_command(
        "bash -c 'cat <&3'",
        tmp_path,
        "PAYLOAD-LINE",
        bound_s=300,
        factor=1,
        token="tok",
        step=slurm.reader_step(),
        watchdog=watched,
    )

    if watched:
        # The watcher ends the reader when the shell's input closes: hold it open, as the recovery reader does.
        result: dict[str, str] = {}
        runner = threading.Thread(target=lambda: result.update(output=LocalConnection().run_with_input_held(command)))
        runner.start()
        runner.join(timeout=60)
        assert not runner.is_alive(), "the watched reader did not end the shell"
        stdout, stderr = result["output"], ""
    else:
        done = subprocess.run(["bash", "-s"], input="\n".join(command), text=True, capture_output=True)
        stdout, stderr = done.stdout, done.stderr

    assert slurm.reader_placement(stdout) == ("cpu:4242", None), stderr
    assert len((log / "srun").read_text().splitlines()) == 3
    assert "PAYLOAD-LINE" in stdout.splitlines()
    if not watched:
        refusals = [line for line in stderr.splitlines() if "the reader step was refused" in line]
        assert len(refusals) == 2 and "job submit limit" in refusals[0] and "retrying in 1 s" in refusals[0]


def test_a_refusal_that_outlasts_the_queue_wait_gives_up_with_how_long_it_resubmitted(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = fake_slurm(_REFUSE_THEN_START_SRUN)
    (log / "refusals").write_text("99")
    monkeypatch.setattr(slurm, "_READER_SRUN_RETRY_S", 1)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S", "3")

    output = LocalConnection().run(_reader(tmp_path))

    host, reason = slurm.reader_placement(output)
    assert host == "login" and reason.startswith(
        "the reader step did not start within 1 s and was cancelled after 3 s"
    )
    assert "job submit limit" in reason
    # 1 + 2 s of back-off fit in the 3 s wait: three submissions, then the fallback.
    assert len((log / "srun").read_text().splitlines()) == 3


def test_an_srun_failure_that_is_not_a_refusal_is_not_submitted_again(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = fake_slurm(_REFUSED_SRUN)  # "Requested node configuration is not available": a config error, not a limit
    monkeypatch.setattr(slurm, "_READER_SRUN_RETRY_S", 1)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S", "30")

    output = LocalConnection().run(_reader(tmp_path))

    assert slurm.reader_placement(output)[0] == "login"
    assert len((log / "srun").read_text().splitlines()) == 1


@pytest.mark.parametrize(("raw", "expected"), [(None, "1800"), ("3600", "3600"), ("0", "1800"), ("x", "1800")])
@pytest.mark.parametrize("in_step", [False, True], ids=["login", "step"])
def test_a_remote_reader_takes_the_callers_uv_lock_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str | None, expected: str, in_step: bool
) -> None:
    if raw is None:
        monkeypatch.delenv("UV_LOCK_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("UV_LOCK_TIMEOUT", raw)
    if in_step:
        monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN", "-p cpu")
    else:
        monkeypatch.delenv("NEMO_GYM_REMOTE_READER_SRUN", raising=False)
    command = slurm._reader_command(
        "true", tmp_path, "{}", bound_s=300, factor=1, token="tok", step=slurm.reader_step()
    )
    exports = [line for line in command if line.startswith("export UV_LOCK_TIMEOUT=")]
    assert exports == [f"export UV_LOCK_TIMEOUT={expected}"]
    # Exported after the cleared variables and before the reader runs (in its step or on the login host).
    assert command.index(exports[0]) > command.index("unset VIRTUAL_ENV UV_PYTHON PYTHONPATH PYTHONHOME")


class _PlainLocalConnection(LocalConnection):
    """A connection that does not hold the shell's input open (no watchdog), as a plain ``run`` does."""

    holds_input = False


@needs_setsid
@pytest.mark.parametrize("separator", ["\r", "\n", "\r\n"], ids=["CR", "LF", "CRLF"])
@pytest.mark.parametrize("started_step", [False, True], ids=["login-host", "step"])
@pytest.mark.parametrize("connection", [LocalConnection, _PlainLocalConnection], ids=["held-input", "plain"])
def test_reader_stderr_keeps_its_tail_whatever_line_ending_it_uses(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, connection, started_step: bool, separator: str
) -> None:
    # Both connections decode the report with text=True, which turns a CR into a newline: the report command must
    # flatten CR and LF itself, or the stderr line ends at the first line ending.
    fake_slurm(_STARTED_SRUN)
    if not started_step:
        monkeypatch.delenv("NEMO_GYM_REMOTE_READER_SRUN")
    stderr = separator.join(["progress", "error: Required uv version does not match"])
    script = f"printf '%s' {shlex.quote(stderr)} >&2; exit 2"
    monkeypatch.setattr(slurm, "render_driver_entrypoint", lambda **_: "bash -c " + shlex.quote(script))
    job = tmp_path / "job"
    (job / "artifacts").mkdir(parents=True)
    (job / "artifacts" / "rollouts.jsonl").write_bytes(b"")
    (job / "artifacts" / "rollouts_failures.jsonl").write_bytes(b"")
    stat = job.stat()
    collected = CollectedEvidence(
        materialized=[],
        results=[],
        failures=[],
        managed_store=True,
        source_job_identity=(str(job.resolve()), stat.st_dev, stat.st_ino),
    )

    result = slurm._remote_managed_evidence(connection(), collected, job, config())

    assert result.managed_reader_unavailable is True
    assert re.search(r"stderr: progress +error: Required uv version does not match$", result.managed_reader_error)
    assert result.managed_reader_error.endswith("does not match")
