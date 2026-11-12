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
"""`gym eval resubmit --recover` exports a complete run that lacks its required MLflow export receipt."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from nemo_gym.orchestration import submit
from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import slurm as sm
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.recovery import CollectedEvidence, RecoveryAction, RecoveryDecision


RECEIPT = {"run_id": "post-run", "current": True, "uploaded_rollouts": True, "rollout_projection_version": 1}


class FakeExecutor:
    """A complete run whose saved MLflow evidence the test sets; ``export_tracking`` stands for the executor's export."""

    name = "fake"
    calls: list[str] = []
    receipt: dict | str | None = None
    configured: bool | None = True
    tracking_required: bool | None = None
    export_state = "succeeded"

    def recovery_evidence(self, record, benchmark, **kwargs):
        FakeExecutor.calls.append("decide")
        decision = RecoveryDecision(
            action=RecoveryAction.NOOP,
            reason="every identity is settled",
            expected=1,
            counts={"scored": 1},
            identities={},
            evidence=[],
        )
        facts = {"configured": FakeExecutor.configured}
        if FakeExecutor.tracking_required is not None:
            facts["tracking_required"] = FakeExecutor.tracking_required
        collected = CollectedEvidence(
            materialized=[],
            results=[],
            failures=[],
            quality={"provenance": {"mlflow": facts}},
            mlflow={"live_run_id": "live-run", "receipt": FakeExecutor.receipt},
        )
        return decision, collected

    def export_tracking(self, record, benchmark, evidence):
        FakeExecutor.calls.append("export")
        if FakeExecutor.export_state == "unsupported":
            return BaseExecutor.export_tracking(self, record, benchmark, evidence)
        if FakeExecutor.export_state != "succeeded":
            return {"exporter": "mlflow", "host": "login", "state": "failed", "error": "tracking server refused"}
        FakeExecutor.receipt = RECEIPT
        return {"exporter": "mlflow", "host": "login", "state": "succeeded", "run_id": "post-run"}


@pytest.fixture
def fake(monkeypatch):
    FakeExecutor.calls = []
    FakeExecutor.receipt = None
    FakeExecutor.configured = True
    FakeExecutor.tracking_required = None
    FakeExecutor.export_state = "succeeded"
    monkeypatch.setitem(submit._EXECUTORS_BY_NAME, "fake", FakeExecutor)
    monkeypatch.setattr(submit, "load_submission_record", lambda job_id: SimpleNamespace(executor="fake"))
    return FakeExecutor


def test_recover_exports_a_complete_run_without_its_receipt_and_says_so(fake) -> None:
    out = submit.recover("gym-job-a", "bench")
    assert fake.calls == ["decide", "export", "decide"]
    assert out.ok and out.dispatched == "export" and out.decision.action == RecoveryAction.NOOP
    assert "MLflow post-run export performed: receipt written for run post-run" in out.decision.reason
    assert out.mlflow["export"] == {"exporter": "mlflow", "host": "login", "state": "succeeded", "run_id": "post-run"}
    # The outcome reports the evidence collected after the export: the receipt it wrote.
    assert out.mlflow["local"]["receipt"] == RECEIPT
    payload = json.loads(out.model_dump_json(exclude={"decision": {"evidence"}}))
    assert payload["dispatched"] == "export" and payload["mlflow"]["export"]["state"] == "succeeded"


@pytest.mark.parametrize("receipt", [RECEIPT, {**RECEIPT, "current": False}, "malformed"])
def test_recover_with_a_receipt_stays_a_noop_and_never_replaces_it(fake, receipt) -> None:
    fake.receipt = receipt
    out = submit.recover("gym-job-b", "bench")
    assert out.ok and out.dispatched is None and out.decision.action == RecoveryAction.NOOP
    assert fake.calls == ["decide"] and "export" not in out.mlflow


@pytest.mark.parametrize(("configured", "required"), [(False, None), (None, None), (True, False)])
def test_recover_never_exports_a_run_without_required_tracking(fake, configured, required) -> None:
    fake.configured, fake.tracking_required = configured, required
    out = submit.recover("gym-job-c", "bench")
    assert out.dispatched is None and fake.calls == ["decide"] and "export" not in out.mlflow


def test_inspect_never_exports(fake) -> None:
    out = submit.inspect_recovery("gym-job-d", "bench")
    assert out.dispatched is None and fake.calls == ["decide"]


@pytest.mark.parametrize("state", ["failed", "unsupported"])
def test_a_failed_or_unsupported_export_is_reported_not_hidden(fake, state) -> None:
    fake.export_state = state
    out = submit.recover("gym-job-e", "bench")
    assert not out.ok and out.decision.action == RecoveryAction.NOOP
    assert fake.calls == ["decide", "export"] and out.mlflow["export"]["state"] == state
    assert out.dispatched == (None if state == "unsupported" else "export")
    assert f"MLflow post-run export {'failed' if state == 'failed' else 'not performed'}" in out.decision.reason


@pytest.mark.parametrize("case", ["succeeded", "failed", "live", "purged", "purged-live", "squeue-down", "sacct-down"])
def test_slurm_export_runs_gym_export_on_the_login_host_with_the_saved_settings(tmp_path, monkeypatch, case) -> None:
    job = tmp_path / "run" / "bench"
    (job / "artifacts").mkdir(parents=True)
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"test": {"type": "slurm", "account": "test", "walltime": "00:05:00"}},
            "driver": {
                "gym_install": {"ref": "a" * 40},
                "env": {"TRACKING_VALUE": "runtime:FORWARDED_VALUE"},
                "benchmarks": {
                    "bench": {
                        "run": {
                            "config_paths": ["fixture.yaml"],
                            "mlflow_tracking_uri": "https://tracking.example",
                            "mlflow_experiment_name": "experiment",
                            "mlflow_run_name": "run",
                            "mlflow_run_tags": {"gym_sha": "'abc'"},
                            "upload_rollouts": True,
                        }
                    }
                },
            },
            "job": {"output_path": str(tmp_path)},
            "otel": {"enabled": False},
        }
    )
    bins = tmp_path / "bin"
    bins.mkdir()
    gym = bins / "gym"
    status = "succeeded" if case != "failed" else "failed"
    gym.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" > {tmp_path}/args\n'
        f'echo "$TRACKING_VALUE" > {tmp_path}/forwarded\n'
        f"test -e {sm._RESUBMIT_LOCK} && echo locked > {tmp_path}/lock\n"
        f'echo "Export mlflow: {status}"\n'
        + ('echo "  run_id: post-run"\n' if status == "succeeded" else 'echo "Error: refused" >&2; exit 1\n')
    )
    # Slurm purges a finished job from the controller: `squeue -j` then exits 1 and sacct still knows the job.
    purged = "echo 'slurm_load_jobs error: Invalid job id specified' >&2; exit 1\n"
    squeue = bins / "squeue"
    squeue.write_text(
        "#!/bin/bash\n"
        + {
            "live": "echo 10\n",
            "purged": purged,
            "purged-live": purged,
            "sacct-down": purged,
            "squeue-down": "echo 'slurm_load_jobs error: Unable to contact slurm controller' >&2; exit 1\n",
        }.get(case, "true\n")
    )
    sacct = bins / "sacct"
    sacct.write_text(
        "#!/bin/bash\n"
        + {"purged-live": "echo '10|PENDING|0'\n", "sacct-down": "exit 1\n"}.get(case, "echo '10|COMPLETED|0'\n")
    )
    for f in (gym, squeue, sacct):
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bins}:{os.environ['PATH']}")
    monkeypatch.setenv("FORWARDED_VALUE", "forwarded")
    entrypoints = []

    def entrypoint(**kwargs):
        entrypoints.append(kwargs)
        return kwargs["command"]

    class Conn(LocalConnection):
        def run(self, commands):
            result = subprocess.run(["bash", "-c", "\n".join(commands)], text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            return result.stdout

    monkeypatch.setattr(sm, "render_driver_entrypoint", entrypoint)
    monkeypatch.setattr(sm, "get_connection", lambda _: Conn())
    monkeypatch.setattr(sm.SlurmExecutor, "_saved_config", staticmethod(lambda conn, run_dir: config))
    record = SimpleNamespace(
        hostname=None, run_dir=str(job.parent), benchmark_job=lambda _: SimpleNamespace(job_dir=str(job))
    )
    evidence = CollectedEvidence(materialized=[], results=[], failures=[], job_chain=["10"])
    result = sm.SlurmExecutor().export_tracking(record, "bench", evidence)
    assert not (job / sm._RESUBMIT_LOCK).exists()
    if case in ("live", "purged-live"):
        assert result["state"] == "failed" and "still live" in result["error"]
        assert not (tmp_path / "args").exists()
        return
    if case in ("squeue-down", "sacct-down"):
        assert result["state"] == "failed" and "scheduler unknown" in result["error"]
        assert not (tmp_path / "args").exists()
        return
    assert entrypoints[0]["ref"] == "a" * 40 and entrypoints[0]["extras"] == ("mlflow",)
    args = (tmp_path / "args").read_text().splitlines()
    # Only the run's MLflow settings reach the export; the rest of the run config is the driver's.
    assert args == [
        "eval",
        "export",
        "mlflow",
        f"+rollouts_jsonl_fpath={job}/artifacts/rollouts.jsonl",
        "+mlflow_tracking_uri=https://tracking.example",
        "+mlflow_experiment_name=experiment",
        "+mlflow_run_name=run",
        "+mlflow_run_tags.gym_sha='abc'",
        "+upload_rollouts=True",
    ]
    assert (tmp_path / "forwarded").read_text().strip() == "forwarded"
    assert (tmp_path / "lock").read_text().strip() == "locked"
    if case in ("succeeded", "purged"):
        assert result == {"exporter": "mlflow", "host": "login", "state": "succeeded", "run_id": "post-run"}
    else:
        assert result["state"] == "failed" and "exit code 1" in result["error"]


@pytest.mark.parametrize(
    ("accounting", "live"),
    [
        ("10|COMPLETED|0\n11|CANCELLED by 4242|0", False),
        ("10|COMPLETED|0\n11|RUNNING|0", True),
        ("10|PENDING+|0", True),
        ("10|REQUEUED|0", True),
        # sacct -D lists every incarnation of a requeued job: only the latest one says whether it is live.
        ("10|REQUEUED|0\n10|COMPLETED|1", False),
        ("10|REQUEUED|0\n10|RUNNING|1", True),
        ("", False),
    ],
)
def test_a_chain_purged_from_squeue_is_judged_by_its_accounting(tmp_path, accounting, live) -> None:
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "squeue").write_text("#!/bin/bash\necho 'slurm_load_jobs error: Invalid job id specified' >&2; exit 1\n")
    (bins / "sacct").write_text(f"#!/bin/bash\ncat <<'EOF'\n{accounting}\nEOF\n")
    for f in bins.iterdir():
        f.chmod(0o755)
    script = [
        'fail() { echo "Error: $*" >&2; exit 1; }',
        *sm._chain_live_check(["10", "11"], "still live"),
        "echo free",
    ]
    env = {**os.environ, "PATH": f"{bins}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", "-c", "\n".join(script)], text=True, capture_output=True, env=env)
    assert (result.returncode, result.stdout.strip(), "still live" in result.stderr) == (
        (1, "", True) if live else (0, "free", False)
    )
