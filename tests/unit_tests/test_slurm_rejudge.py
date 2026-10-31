import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import slurm as sm
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.recovery import CollectedEvidence, RecoveryAction, RecoveryDecision


@pytest.mark.parametrize("case", ["ok", "exhausted", "unknown", "stale", "live"])
def test_cpu_rejudge_existing_accounting(tmp_path, monkeypatch, case):
    job = tmp_path / "run" / "judge"
    (job / "artifacts").mkdir(parents=True)
    (job / "logs").mkdir()
    cfg = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {
                "test": {
                    "type": "slurm",
                    "account": "test",
                    "walltime": "00:05:00",
                    "node_pools": {"cpu": {"partition": "cpu", "gpus_per_node": 0}},
                }
            },
            "driver": {
                "gym_install": {"ref": "a" * 40},
                "benchmarks": {
                    "judge": {
                        "run": {"config_paths": ["fixture.yaml"]},
                        "resumable": {"max_retries": 1, "max_walltime": "00:10:00"},
                    }
                },
            },
            "job": {"output_path": str(tmp_path)},
            "otel": {"enabled": False},
        }
    )
    (job.parent / "resolved_config.yaml").write_text("fixture: true\n")
    # Real source/accounting files; the only process-level fakes are the scheduler executables.
    (job / "artifacts/rollouts.jsonl").write_text("fixture\n")
    for n, v in {".gym_job_chain": "10\n", ".gym_infra_retries": "0\n", ".gym_accumulated_walltime": "60\n"}.items():
        (job / n).write_text(v)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    snapshot = {n: digest(job / n) for n in (".gym_job_chain", ".gym_infra_retries", ".gym_accumulated_walltime")}
    snapshot["resume_config"] = cfg.driver.benchmarks["judge"].resume_config.model_dump_json()
    evidence = CollectedEvidence(
        materialized=[],
        results=[],
        failures=[],
        infra_retries_used=0,
        job_chain=["10"],
        accounted_walltime_s=600 if case == "exhausted" else None if case == "unknown" else 60,
        accounting_snapshot=snapshot,
        source_digests={"rollouts.jsonl": digest(job / "artifacts/rollouts.jsonl")},
    )
    if case == "stale":
        (job / "artifacts/rollouts.jsonl").write_text("changed\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    for name, body in {
        "squeue": ("echo 10" if case == "live" else "true"),
        "sbatch": f"echo call >> {tmp_path}/calls; echo 11",
    }.items():
        f = bins / name
        f.write_text("#!/bin/bash\n" + body + "\n")
        f.chmod(0o755)
    import os

    monkeypatch.setenv("PATH", str(bins) + ":" + os.environ["PATH"])

    class Conn(LocalConnection):
        def run(self, commands):
            # Config file name is Gym's own constant, made here from the same fixture.
            return subprocess.check_output(["bash", "-c", "\n".join(commands)], text=True, stderr=subprocess.STDOUT)

    record = SimpleNamespace(
        cluster="test",
        hostname=None,
        run_dir=str(job.parent),
        benchmark_job=lambda _: SimpleNamespace(
            job_dir=str(job), model_copy=lambda **kw: SimpleNamespace(job_dir=str(job), job_id=kw["update"]["job_id"])
        ),
        with_benchmark=lambda _: None,
    )
    (job.parent / sm.RESOLVED_CONFIG_NAME).write_text("fixture: true\n")
    monkeypatch.setattr(sm, "get_connection", lambda _: Conn())
    monkeypatch.setattr(sm, "installed_gym_commit", lambda: "a" * 40)
    monkeypatch.setattr(sm.SlurmExecutor, "_saved_config", staticmethod(lambda *a: cfg))
    decision = RecoveryDecision(
        action=RecoveryAction.REVERIFY,
        reason="judge only",
        expected=1,
        counts={"judge_recoverable": 1},
        identities={},
        evidence=[],
    )
    monkeypatch.setattr(sm.SlurmExecutor, "recovery_evidence", lambda *a: (decision, evidence))
    monkeypatch.setattr(sm.SlurmExecutor, "persist_record", lambda *a: None)
    if case in ("stale", "live"):
        with pytest.raises(subprocess.CalledProcessError):
            sm.SlurmExecutor().reverify_judge_failures(record, "judge", decision)
    else:
        out = sm.SlurmExecutor().reverify_judge_failures(record, "judge", decision)
        assert out.ok is (case == "ok")
    assert (tmp_path / "calls").exists() is (case == "ok")
    assert (job / ".gym_infra_retries").read_text() == "0\n"
    if case == "ok":
        assert (job / ".gym_job_chain").read_text() == "10\n11\n"
        assert float((job / ".gym_accumulated_walltime").read_text()) == 60
        script = next(job.glob("reverify-*.sh")).read_text()
        assert "gym eval reverify" in script and "--settle-into-source" in script
        assert "gym eval run" not in script and "afternotok" not in script
        assert "#SBATCH --time=00:05:00" in script
