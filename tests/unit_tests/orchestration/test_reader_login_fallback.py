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
"""With a reader step configured, a reader whose step never starts reads nothing on the login host unless allowed."""

import shlex
import shutil
from pathlib import Path

import pytest

import nemo_gym.orchestration.executors.slurm as slurm
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.recovery import decide_from_collected
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_managed_recovery_selection import prepare_managed
from tests.unit_tests.orchestration.test_reader_srun import fake_slurm as fake_slurm
from tests.unit_tests.orchestration.test_remote_managed_recovery import config
from tests.unit_tests.orchestration.test_remote_reader_transport import _collected, _install, _job


pytestmark = pytest.mark.skipif(shutil.which("timeout") is None, reason="coreutils timeout not installed")

# The cpu queue is full: the step waits for resources until its --immediate wait ends, and never starts.
_NEVER_STARTS = """#!/bin/bash
echo "$@" >> "$FAKE_LOG/srun"
echo "srun: job 77 queued and waiting for resources" >&2
sleep 1
echo "srun: error: Unable to allocate resources: Job/step not started within the immediate wait" >&2
exit 1
"""
NOT_STARTED = (
    "the reader step did not start within 1 s and was cancelled "
    "(srun: error: Unable to allocate resources: Job/step not started within the immediate wait)"
)


@pytest.mark.parametrize("setting", [None, "forbid", "allow"])
def test_a_store_reader_whose_step_never_starts_reads_on_the_login_host_only_when_allowed(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting
) -> None:
    fake_slurm(_NEVER_STARTS)
    if setting is None:
        monkeypatch.delenv("NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK", raising=False)
    else:
        monkeypatch.setenv("NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK", setting)
    job = _job(tmp_path)
    collected = _collected(job)
    ran = tmp_path / "ran-on-the-login-host"
    projection = "__GYM_MANAGED:" + collected.model_dump_json()
    _install(monkeypatch, f"touch {shlex.quote(str(ran))}; cat <&3 >/dev/null; echo {shlex.quote(projection)}")

    result = slurm._remote_managed_evidence(LocalConnection(), collected, job, config())
    decision = decide_from_collected(result, max_infra_retries=3, max_walltime_s=None, live_job=False)

    if setting == "allow":
        # Today's behaviour: the read falls back to the login host and is used.
        assert ran.exists() and result.managed_reader_error is None
        assert (result.reader_host, result.reader_fallback_reason) == ("login", NOT_STARTED)
        return
    assert not ran.exists()
    assert (result.managed_selection, result.managed_reader_unavailable) == (None, True)
    assert result.reader_host == "none"
    assert result.reader_fallback_reason == (
        f"{NOT_STARTED}; the login-host fallback is forbidden (NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK=forbid), "
        "so nothing was read"
    )
    assert decision.action.value == "unknown" and "could not run" in decision.reason


@pytest.mark.parametrize("setting", [None, "allow"])
def test_a_collector_whose_step_never_starts_collects_on_the_login_host_only_when_allowed(
    fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting
) -> None:
    fake_slurm(_NEVER_STARTS)
    if setting is None:
        monkeypatch.delenv("NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK", raising=False)
    else:
        monkeypatch.setenv("NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK", setting)
    command = slurm.collector_command("echo '{\"collected\": true}'", tmp_path, slurm.reader_step(), "tok")

    output = LocalConnection().run(command)

    if setting == "allow":
        assert slurm.reader_placement(output) == ("login", NOT_STARTED)
        assert slurm.without_placement(output) == '{"collected": true}'
    else:
        assert slurm.reader_placement(output)[0] == "none"
        assert slurm.without_placement(output) == ""


def test_a_collector_that_could_not_run_decides_nothing(tmp_path: Path, slurm_bin, monkeypatch) -> None:
    from nemo_gym.orchestration.submit import inspect_recovery

    job, _, _ = _resumable_run(tmp_path)
    prepare_managed(job, "missing", n=2)
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_SRUN", "-p cpu")
    reason = f"{NOT_STARTED}; the login-host fallback is forbidden"
    report = f"__GYM_READER_HOST:none\n__GYM_READER_FALLBACK:{reason}\n"
    monkeypatch.setattr(slurm, "collector_command", lambda *args: [f"printf %s {shlex.quote(report)}"])

    out = inspect_recovery(*_handle(job))

    assert (out.decision.action.value, out.ok, out.dispatched) == ("unknown", False, None)
    assert "evidence collector could not run" in out.decision.reason and reason in out.decision.reason
    assert out.decision.reason.endswith("gym eval resubmit {} {} --recover --inspect".format(*_handle(job)))
    assert not (slurm_bin / "env").exists()
