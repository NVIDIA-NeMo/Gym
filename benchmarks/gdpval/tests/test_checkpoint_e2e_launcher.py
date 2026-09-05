# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "benchmarks" / "gdpval" / "hsg" / "checkpoint_e2e"
LAUNCHER = PACKAGE / "run_checkpoint_e2e.sh"


def _write(path: Path, data: str = "fixture\n", *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    checkpoint = tmp_path / "model" / "iter_0002000" / "hf"
    checkpoint.mkdir(parents=True)
    _write(checkpoint / "config.json", '{"model_type":"fixture"}\n')
    _write(checkpoint / "tokenizer.json", '{"version":"1"}\n')
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    reference = _write(tmp_path / "reference.txt")
    dataset = tmp_path / "gdpval.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"task-{index:03d}",
                    "prompt": f"task {index}",
                    "reference_file_urls": [str(reference.resolve())],
                }
            )
            + "\n"
            for index in range(220)
        ),
        encoding="utf-8",
    )

    owner = tmp_path / "owner"
    aav2 = owner / "gdpval_colo" / "aav2"
    unified = owner / "gdpval_colo" / "unified"
    gym = owner / "gdpval_integ"
    rollout = _write(unified / "gdpval_rollout.sbatch")
    serve = _write(unified / "serve" / "serve_vllm_replica.sh")
    parser = _write(aav2 / "parsers" / "ultra_v3_reasoning_parser.py")
    overlay = _write(aav2 / "refs.yaml", "reference_models: {}\n")
    env_file = _write(aav2 / "aav2.env", "")
    container = _write(tmp_path / "vllm.sqsh")
    sif = _write(tmp_path / "gdpval.sqsh")
    agent_sif = _write(tmp_path / "agent.sif")
    apptainer = _write(tmp_path / "apptainer" / "apptainer", "#!/bin/sh\n", executable=True)
    _write(gym / ".venv" / "bin" / "gym", "#!/bin/sh\n", executable=True)
    for relative in (
        "benchmarks/gdpval/config.yaml",
        "benchmarks/gdpval/prepare.py",
        "benchmarks/gdpval/run_gdpval_rollouts.sh",
        "nemo_gym/deliverables.py",
        "nemo_gym/rollout_collection.py",
        "nemo_gym/rollout_reverification.py",
        "resources_servers/gdpval/app.py",
        "resources_servers/gdpval/comparison.py",
        "resources_servers/gdpval/judge_panel.py",
        "resources_servers/gdpval/multistage_elo.py",
        "resources_servers/gdpval/multistage_orchestrator.py",
        "resources_servers/gdpval/preconvert.py",
        "resources_servers/gdpval/scoring.py",
        "responses_api_agents/stirrup_agent/app.py",
        "responses_api_agents/stirrup_agent/file_reader.py",
        "responses_api_agents/stirrup_agent/stirrup_utils.py",
        "responses_api_models/openai_model/app.py",
        "responses_api_models/openai_model/client.py",
        "responses_api_models/vllm_model/configs/vllm_model.yaml",
    ):
        _write(gym / relative)

    environment = {
        **os.environ,
        "CHECKPOINT_E2E_PYTHON": sys.executable,
        "CHECKPOINT_E2E_OWNER_ROOT": str(owner),
        "CHECKPOINT_E2E_AAV2_ROOT": str(aav2),
        "CHECKPOINT_E2E_UNIFIED_ROOT": str(unified),
        "CHECKPOINT_E2E_ROOT": str(tmp_path / "campaigns"),
        "CHECKPOINT_E2E_GYM_ROOT": str(gym),
        "CHECKPOINT_E2E_EXPECTED_GYM_REVISION": "unversioned",
        "CHECKPOINT_E2E_ROLLOUT_GYM_ROOT": str(gym),
        "CHECKPOINT_E2E_EXPECTED_ROLLOUT_GYM_REVISION": "unversioned",
        "CHECKPOINT_E2E_DATASET": str(dataset),
        "CHECKPOINT_E2E_REFERENCE_OVERLAY": str(overlay),
        "CHECKPOINT_E2E_ENV_FILE": str(env_file),
        "CHECKPOINT_E2E_ROLLOUT_SBATCH": str(rollout),
        "CHECKPOINT_E2E_SERVE_SCRIPT": str(serve),
        "CHECKPOINT_E2E_JUDGE_SBATCH": str(PACKAGE / "judge.sbatch"),
        "CHECKPOINT_E2E_PARSER_ROOT": str(parser.parent),
        "CHECKPOINT_E2E_PARSER_PLUGIN": str(parser),
        "CHECKPOINT_E2E_CONTAINER": str(container),
        "CHECKPOINT_E2E_GDPVAL_SIF": str(sif),
        "CHECKPOINT_E2E_AGENT_SIF": str(agent_sif),
        "CHECKPOINT_E2E_APPTAINER_BIN": str(apptainer.parent),
    }
    return checkpoint.resolve(), environment


def _run(action: str, checkpoint: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), action, str(checkpoint)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _controller_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    run_dir = tmp_path / "run"
    e2e_dir = tmp_path / "e2e"
    fake_bin = tmp_path / "fake-bin"
    deliverables = run_dir / "deliverables"
    for path in (run_dir / "logs", deliverables, e2e_dir, fake_bin):
        path.mkdir(parents=True, exist_ok=True)

    dataset = _write(tmp_path / "dataset.jsonl", "{}\n")
    judge_sbatch = _write(e2e_dir / "judge.sbatch")
    _write(e2e_dir / "slurm_receipts.sh", (PACKAGE / "slurm_receipts.sh").read_text(encoding="utf-8"))
    _write(e2e_dir / "judge_process_group.sh", (PACKAGE / "judge_process_group.sh").read_text(encoding="utf-8"))
    transport_views = _write(e2e_dir / "transport_views.py")
    apptainer = _write(e2e_dir / "apptainer" / "apptainer", "#!/bin/sh\nexit 0\n", executable=True)
    sif = _write(tmp_path / "fixture.sif")
    reference_roots = []
    for index in range(9):
        root = tmp_path / "references" / f"model_{index}"
        root.mkdir(parents=True)
        reference_roots.append(root)
    reference_overlay = _write(
        tmp_path / "reference.yaml",
        "gdpval:\n  reference_models:\n"
        + "".join(
            f"    model_{index}:\n      deliverables_dir: {root}\n" for index, root in enumerate(reference_roots)
        ),
    )
    transport_root = run_dir / "judge_transport_views"
    manifest = _write(transport_root / "manifest.json", "{}\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _write(
        run_dir / "TRANSPORT_PREBUILD_PASS",
        f"schema=gdpval.transport-prebuild.v1\nmanifest={manifest}\nmanifest_sha256={manifest_sha}\n",
    )
    _write(
        run_dir / "settings.env",
        "\n".join(
            (
                "RUN_ID=fixture-run",
                f"DATASET={dataset}",
                f"DELIVERABLES={deliverables}",
                "EXPECTED_TASKS=220",
                "ACCOUNT=fixture",
                "CPU_PARTITION=fixture",
                "CPU_QOS=fixture",
                "JUDGE_WALL=00:10:00",
                f"JUDGE_SBATCH={judge_sbatch}",
                f"TRANSPORT_VIEW_ROOT={transport_root}",
                f"TRANSPORT_VIEWS_PY={transport_views}",
                f"REFERENCE_OVERLAY={reference_overlay}",
                f"APPTAINER_BIN={apptainer.parent}",
                f"GDPVAL_SIF={sif}",
                "MODEL_NAME=fixture-model",
                f"AAV2_ROOT={tmp_path}",
            )
        )
        + "\n",
    )
    _write(run_dir / "FLEET_JOBS.tsv", "shard-0\t90\n")
    _write(e2e_dir / "run_checkpoint_e2e.sh", "#!/bin/sh\nexit 0\n", executable=True)

    remaining = _write(tmp_path / "remaining", "0\n")
    fake_python = _write(
        fake_bin / "python",
        """#!/bin/bash
remaining=$(<"$FAKE_REMAINING_FILE")
if [[ ${1:-} == -c ]]; then
    cat >/dev/null
    if [[ ${2:-} == *remaining_produced* ]]; then
        printf '%s\n' "$remaining"
    else
        echo 220
    fi
elif [[ ${2:-} == coverage ]]; then
    printf 'coverage\n' >> "$FAKE_EVENT_LOG"
    exit 0
elif [[ ${2:-} == result ]]; then
    exit 1
elif [[ ${2:-} == inventory ]]; then
    status=OPEN
    (( remaining > 0 )) || status=CLOSED
    printf '{"status":"%s","remaining_produced":%s,"closure_fingerprint":"fixture","produced_pairs":[]}\n' \
        "$status" "$remaining"
    exit 0
elif [[ ${1:-} == - ]]; then
    cat >/dev/null
    exit 0
else
    exit 0
fi
""",
        executable=True,
    )
    _write(fake_bin / "flock", "#!/bin/sh\nexit 0\n", executable=True)
    _write(
        fake_bin / "date",
        """#!/bin/sh
if [ "${1:-}" = -Is ]; then
    echo 2026-08-22T00:00:00+00:00
else
    /bin/date "$@"
fi
""",
        executable=True,
    )
    _write(fake_bin / "sleep", "#!/bin/sh\nprintf 'sleep\n' >> \"$FAKE_EVENT_LOG\"\n", executable=True)
    _write(fake_bin / "scancel", "#!/bin/sh\nexit 0\n", executable=True)
    _write(
        fake_bin / "squeue",
        """#!/bin/bash
jid=
while (( $# )); do
    if [[ $1 == -j ]]; then jid=$2; break; fi
    shift
done
printf 'squeue:%s\n' "$jid" >> "$FAKE_EVENT_LOG"
""",
        executable=True,
    )
    _write(
        fake_bin / "sacct",
        """#!/bin/bash
jid=
while (( $# )); do
    if [[ $1 == -j ]]; then jid=$2; break; fi
    shift
done
printf 'sacct:%s\n' "$jid" >> "$FAKE_EVENT_LOG"
IFS=, read -ra ids <<<"$jid"
for id in "${ids[@]}"; do
    printf '%s|COMPLETED|0:0|00:00:01|fixture|fixture|fixture\n' "$id"
done
""",
        executable=True,
    )
    counter = _write(tmp_path / "job-counter", "200\n")
    _write(
        fake_bin / "sbatch",
        """#!/bin/bash
next=$(( $(<"$FAKE_JOB_COUNTER") + 1 ))
printf '%s\n' "$next" > "$FAKE_JOB_COUNTER"
printf 'sbatch:%s\n' "$next" >> "$FAKE_EVENT_LOG"
printf '%s\n' "$next"
""",
        executable=True,
    )
    events = _write(tmp_path / "events", "")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUN_DIR": str(run_dir),
        "E2E_DIR": str(e2e_dir),
        "CHECKPOINT_E2E_PYTHON": str(fake_python),
        "CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS": "false",
        "CHECKPOINT_E2E_POLL_SECONDS": "1",
        "CHECKPOINT_E2E_MAX_PRECONVERT_ATTEMPTS": "4",
        "FAKE_JOB_COUNTER": str(counter),
        "FAKE_EVENT_LOG": str(events),
        "FAKE_REMAINING_FILE": str(remaining),
        "SLURM_JOB_ID": "700",
    }
    return run_dir, fake_bin, events, environment


def _run_controller(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PACKAGE / "controller.sbatch")],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )


def _intent_comment(path: Path) -> str:
    fields = dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)
    return fields["comment"]


def test_prepare_is_checkpoint_only_idempotent_and_status_is_read_only(tmp_path: Path) -> None:
    checkpoint, environment = _fixture(tmp_path)
    first = _run("prepare", checkpoint, environment)
    assert first.returncode == 0, first.stderr
    run_dir = Path(next(line.split("=", 1)[1] for line in first.stdout.splitlines() if line.startswith("RUN_DIR=")))
    assert run_dir.name.startswith("iter_0002000-")
    assert len(list((run_dir / "shards").glob("shard_*_of_06.jsonl"))) == 6
    assert [sum(1 for _ in path.open()) for path in sorted((run_dir / "shards").glob("shard_*_of_06.jsonl"))] == [
        37,
        37,
        37,
        37,
        36,
        36,
    ]
    for path in (run_dir / "campaign.json", run_dir / "settings.env", run_dir / "model_profile.env"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o400

    second = _run("prepare", checkpoint, environment)
    assert second.returncode == 0, second.stderr
    assert f"RUN_DIR={run_dir}" in second.stdout

    before = sorted((path.relative_to(run_dir), path.stat().st_mtime_ns) for path in run_dir.rglob("*"))
    status_result = _run("status", checkpoint, environment)
    after = sorted((path.relative_to(run_dir), path.stat().st_mtime_ns) for path in run_dir.rglob("*"))
    assert status_result.returncode == 0, status_result.stderr
    assert "STATE=PREPARED" in status_result.stdout
    assert "ROLLOUT=0/220" in status_result.stdout
    assert before == after


def test_compute_preflight_runs_under_nounset_and_publishes_receipt(tmp_path: Path) -> None:
    checkpoint, environment = _fixture(tmp_path)
    prepared = _run("prepare", checkpoint, environment)
    assert prepared.returncode == 0, prepared.stderr
    run_dir = Path(next(line.split("=", 1)[1] for line in prepared.stdout.splitlines() if line.startswith("RUN_DIR=")))

    preflight = subprocess.run(
        ["bash", str(LAUNCHER), "_compute-preflight", str(run_dir)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert preflight.returncode == 0, (preflight.stdout, preflight.stderr)
    assert "PREFLIGHT_PASS" in preflight.stdout
    receipt = run_dir / "PREFLIGHT_PASS"
    assert receipt.read_text(encoding="utf-8").startswith("campaign=")
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o400


def test_compute_preflight_receipts_post_rollout_vllm_drift_only(tmp_path: Path) -> None:
    checkpoint, environment = _fixture(tmp_path)
    prepared = _run("prepare", checkpoint, environment)
    assert prepared.returncode == 0, prepared.stderr
    run_dir = Path(next(line.split("=", 1)[1] for line in prepared.stdout.splitlines() if line.startswith("RUN_DIR=")))
    settings_before = (run_dir / "settings.env").read_bytes()

    container = Path(environment["CHECKPOINT_E2E_CONTAINER"])
    container.write_text("replacement vLLM image\n", encoding="utf-8")
    before_coverage = subprocess.run(
        ["bash", str(LAUNCHER), "_compute-preflight", str(run_dir)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert before_coverage.returncode == 64
    assert "vLLM container changed before immutable rollout coverage" in before_coverage.stderr

    for index in range(220):
        marker = run_dir / "deliverables" / f"task_task-{index:03d}" / "repeat_0" / "finish_params.json"
        _write(marker, "{}\n")
    coverage_receipt = _write(run_dir / "ROLLOUT_COVERAGE_PASS", "")
    coverage_receipt.chmod(0o400)

    after_coverage = subprocess.run(
        ["bash", str(LAUNCHER), "_compute-preflight", str(run_dir)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert after_coverage.returncode == 0, (after_coverage.stdout, after_coverage.stderr)
    assert "VLLM_CONTAINER_POST_ROLLOUT_DRIFT" in after_coverage.stdout
    assert (run_dir / "settings.env").read_bytes() == settings_before
    drift_receipt = run_dir / "VLLM_CONTAINER_DRIFT_AFTER_ROLLOUTS"
    fields = dict(line.split("=", 1) for line in drift_receipt.read_text(encoding="utf-8").splitlines())
    assert fields["schema"] == "gdpval.vllm-container-post-rollout-drift.v1"
    assert fields["path"] == str(container)
    assert fields["expected_signature"] != fields["observed_signature"]
    assert stat.S_IMODE(drift_receipt.stat().st_mode) == 0o400
    preflight = (run_dir / "PREFLIGHT_PASS").read_text(encoding="utf-8")
    assert preflight.startswith(f"campaign={run_dir.name}\n")
    assert drift_receipt.exists()

    Path(environment["CHECKPOINT_E2E_GDPVAL_SIF"]).write_text("changed judge image\n", encoding="utf-8")
    judging_input_drift = subprocess.run(
        ["bash", str(LAUNCHER), "_compute-preflight", str(run_dir)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert judging_input_drift.returncode == 64
    assert "GDPVal container changed" in judging_input_drift.stderr


def test_launch_contract_keeps_fast_shards_exact_recovery_and_strict_judging() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    controller = (PACKAGE / "controller.sbatch").read_text(encoding="utf-8")
    judge = (PACKAGE / "judge.sbatch").read_text(encoding="utf-8")
    rejudge_controller = (PACKAGE / "rejudge_controller.sbatch").read_text(encoding="utf-8")

    assert "CHECKPOINT_E2E_ROLLOUT_SBATCH:-$SCRIPT_DIR/gdpval_rollout.sbatch" in launcher
    assert 'ROLLOUT_LIFECYCLE_SH="$SCRIPT_DIR/rollout_lifecycle.sh"' in launcher
    assert 'ROLLOUT_SHARD_COVERAGE_PY="$SCRIPT_DIR/rollout_shard_coverage.py"' in launcher
    assert 'ROLLOUT_PACKAGE_DIR="$SCRIPT_DIR"' in launcher
    assert launcher.count('"$ROLLOUT_LIFECYCLE_SH"') >= 2
    assert launcher.count('"$ROLLOUT_SHARD_COVERAGE_PY"') >= 2
    assert 'ROLLOUT_CONCURRENCY="${CHECKPOINT_E2E_ROLLOUT_CONCURRENCY:-20}"' in launcher
    assert "SHARD_COUNT=6" in launcher
    assert 'TREE="$ROLLOUT_GYM_ROOT"' in launcher
    assert "recovery_r${round}" in launcher
    assert "MAX_JUDGE_ATTEMPTS=4" in controller
    assert "JUDGE_CONCURRENCIES=(16 16 8 4)" in controller
    assert "while (( attempt < MAX_JUDGE_ATTEMPTS ))" in controller
    assert '--concurrency "$CONCURRENCY"' in judge
    assert 'stirrup_agent.concurrency="$CONCURRENCY"' not in judge
    assert "scientific fingerprint stays" in judge
    assert '"$CAMPAIGN_PY" coverage' in controller
    assert '"$CAMPAIGN_PY" result' in controller
    assert "partial_completion: {min_success_fraction: 0.9" in controller
    assert "partial_completion: {min_success_fraction: 0.9" in judge
    for script in (controller, judge, rejudge_controller):
        assert "waivable_failure_classes: [timeout_exceeded, transient]" in script
        assert "{num_tasks: 220, num_models: 4}" in script
    assert "JUDGE_ONLY=true" in judge
    assert "RERUN_INCOMPLETE=true" in judge
    assert "num_comparison_trials=4" in judge
    assert "JUDGE_PORT_BASE=12000" in judge
    assert "JUDGE_PORT_SLOT_WIDTH=20" in judge
    assert "JUDGE_PORT_SLOT_COUNT=1000" in judge
    assert "flock -n 9" in judge
    assert "candidate_offset < JUDGE_PORT_SLOT_WIDTH" in judge
    assert "/dev/tcp/127.0.0.1/$candidate_port" in judge
    assert "++head_server.host=127.0.0.1" in judge
    assert '++head_server.port="$JUDGE_HEAD_PORT"' in judge
    assert '++port_range_low="$JUDGE_PORT_RANGE_LOW"' in judge
    assert '++port_range_high="$JUDGE_PORT_RANGE_HIGH"' in judge
    assert "dataset overlay drift" in judge
    assert "final receipt no longer matches the strict result" in launcher
    assert '--dataset "$DATASET" --expected-tasks "$EXPECTED_TASKS"' in launcher
    assert '--dataset "$DATASET" --expected-tasks "$EXPECTED_TASKS"' in controller
    assert '--dataset "$DATASET" --expected-tasks "$EXPECTED_TASKS"' in judge


def test_slurm_lifecycle_is_locked_revalidated_and_bounded() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    controller = (PACKAGE / "controller.sbatch").read_text(encoding="utf-8")
    judge = (PACKAGE / "judge.sbatch").read_text(encoding="utf-8")

    submit_body = launcher.split("submit_campaign() {", 1)[1].split("\n}", 1)[0]
    resume_body = launcher.split("resume_campaign() {", 1)[1].split("\n}", 1)[0]
    assert "acquire_launcher_lock" in submit_body
    assert "acquire_launcher_lock" in resume_body
    assert 'flock -n 8 || fail "another submit/resume process owns this campaign' in launcher
    assert "normalize_job_id" in launcher
    assert "job_liveness" in launcher
    assert "validate_dependency_policy" in launcher
    assert "DependencyParameters=kill_invalid_depend" in launcher
    assert "--kill-on-invalid-dep=yes" not in launcher
    assert "no live or terminal Slurm evidence; refusing resume" in launcher

    assert controller.count('"$E2E_DIR/run_checkpoint_e2e.sh" _compute-preflight "$RUN_DIR"') == 2
    assert "[[ -f $RUN_DIR/PREFLIGHT_PASS ]]" not in controller
    assert "CHECKPOINT_E2E_MAX_CONTROLLER_REQUEUES" in controller
    assert "CONTROLLER_ROTATION >= MAX_CONTROLLER_REQUEUES" in controller
    assert "MAX_CONTROLLER_REQUEUES <= 4" in controller
    assert "CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS" in controller
    assert "JUDGE_NO_PROGRESS_SECONDS <= 7200" in controller
    assert "judge_progress_signature" in controller
    assert "job_is_terminal" in controller
    assert "jobs_all_terminal" in controller
    assert "absent from squeue but not fully terminal in accounting; retrying" in controller
    wait_jobs_body = controller.split("wait_jobs() {", 1)[1].split("\n}", 1)[0]
    assert "fast_coverage_count" in wait_jobs_body
    assert "CAMPAIGN_PY" not in wait_jobs_body
    assert 'scancel --signal=TERM --batch "$jid"' in controller
    assert "attempt=$(latest_judge_attempt)" in controller
    assert "judge receipt count exceeds campaign-global maximum" in controller
    assert "judge receipt sequence has a gap" in controller
    assert 'newest_receipt="$RUN_DIR/JUDGE_JOB_${attempt}"' in controller
    assert 'judge_job_requires_wait "$newest_jid"' in controller
    assert 'wait_judge_job "judge_attempt${attempt}_reattach" "$newest_jid"' in controller
    assert 'job_file="$RUN_DIR/JUDGE_JOB_${attempt}"' in controller
    assert "refusing to replace existing judge receipt" in controller
    assert 'wait_judge_job "judge_attempt${attempt}_c${concurrency}" "$jid"' in controller
    assert "attempt=$(latest_preconvert_attempt)" in controller
    assert "attempt <= MAX_PRECONVERT_ATTEMPTS" in controller
    assert "while (( before > 0 && attempt < MAX_PRECONVERT_ATTEMPTS ))" in controller
    assert 'job_file="$RUN_DIR/PRECONVERT_JOB_${attempt}"' in controller
    assert 'wait_jobs "preconvert_${attempt}_reattach" "$jid"' in controller
    assert "preconvert receipt count exceeds configured maximum" in controller

    assert "#SBATCH --no-requeue" in judge
    assert "preserving partial state for the next concurrency" in judge
    assert "scontrol requeue" not in judge
    assert '"$CAMPAIGN_PY" verify --run-dir "$RUN_DIR"' in launcher

    for script in (LAUNCHER, PACKAGE / "controller.sbatch", PACKAGE / "judge.sbatch"):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_nested_recovery_submission_scrubs_parent_memory_limits(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    captured_environment = tmp_path / "sbatch.env"
    captured_arguments = tmp_path / "sbatch.args"
    _write(
        fake_bin / "sbatch",
        """#!/bin/sh
env > "$FAKE_SBATCH_ENV"
printf '%s\n' "$@" > "$FAKE_SBATCH_ARGS"
printf '901;fixture-cluster\n'
""",
        executable=True,
    )
    receipt = tmp_path / "run" / "recovery_r1" / "rollout_s00" / "JOBID"
    inherited_limits = {
        "SLURM_MEM_PER_NODE": "8G",
        "SLURM_MEM_PER_CPU": "4G",
        "SLURM_MEM_PER_GPU": "2G",
        "SBATCH_MEM": "8G",
        "SBATCH_MEM_PER_NODE": "8G",
        "SBATCH_MEM_PER_CPU": "4G",
        "SBATCH_MEM_PER_GPU": "2G",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; slurm_submit_or_adopt "$2" recovery-r1-s0 recovery-job '
            "--parsable -N 1 --mem=900G fixture.sbatch",
            "nested-recovery",
            str(PACKAGE / "slurm_receipts.sh"),
            str(receipt),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            **inherited_limits,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "RUN_DIR": str(tmp_path / "run"),
            "FAKE_SBATCH_ENV": str(captured_environment),
            "FAKE_SBATCH_ARGS": str(captured_arguments),
            "UNRELATED_PARENT_VALUE": "preserved",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "901"
    assert receipt.read_text(encoding="utf-8") == "901\n"
    submitted_environment = dict(
        line.split("=", 1) for line in captured_environment.read_text(encoding="utf-8").splitlines() if "=" in line
    )
    assert inherited_limits.keys().isdisjoint(submitted_environment)
    assert submitted_environment["UNRELATED_PARENT_VALUE"] == "preserved"
    assert "--mem=900G" in captured_arguments.read_text(encoding="utf-8").splitlines()


def test_controller_waits_for_every_job_to_have_terminal_accounting(tmp_path: Path) -> None:
    run_dir, fake_bin, events, environment = _controller_fixture(tmp_path)
    _write(run_dir / "FLEET_JOBS.tsv", "shard-0\t90\nshard-1\t91\n")
    squeue_calls = _write(tmp_path / "squeue-calls", "0\n")
    sacct_calls = _write(tmp_path / "sacct-calls", "0\n")
    _write(
        fake_bin / "squeue",
        """#!/bin/bash
call=$(( $(<"$FAKE_SQUEUE_CALLS") + 1 ))
printf '%s\n' "$call" > "$FAKE_SQUEUE_CALLS"
printf 'squeue:%s\n' "$call" >> "$FAKE_EVENT_LOG"
(( call == 1 )) && exit 1
exit 0
""",
        executable=True,
    )
    _write(
        fake_bin / "sacct",
        """#!/bin/bash
jid=
while (( $# )); do
    if [[ $1 == -j ]]; then jid=$2; break; fi
    shift
done
call=$(( $(<"$FAKE_SACCT_CALLS") + 1 ))
printf '%s\n' "$call" > "$FAKE_SACCT_CALLS"
printf 'sacct:%s:%s\n' "$call" "$jid" >> "$FAKE_EVENT_LOG"
printf '90|COMPLETED|0:0|00:00:01|fixture|fixture|fixture\n'
(( call == 1 )) || printf '91|COMPLETED|0:0|00:00:01|fixture|fixture|fixture\n'
""",
        executable=True,
    )

    result = _run_controller(
        {
            **environment,
            "FAKE_SQUEUE_CALLS": str(squeue_calls),
            "FAKE_SACCT_CALLS": str(sacct_calls),
        }
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    event_lines = events.read_text(encoding="utf-8").splitlines()
    assert event_lines[:6] == [
        "squeue:1",
        "sacct:1:90,91",
        "sleep",
        "squeue:2",
        "sacct:2:90,91",
        "sacct:3:90,91",
    ]
    assert event_lines.index("coverage") > event_lines.index("sacct:2:90,91")
    assert event_lines.count("coverage") == 2
    assert "queue query failed; retaining ownership and retrying" in result.stdout


def test_controller_accepts_terminal_accounting_when_squeue_rejects_expired_job(tmp_path: Path) -> None:
    _run_dir, fake_bin, events, environment = _controller_fixture(tmp_path)
    _write(
        fake_bin / "squeue",
        "#!/bin/bash\nprintf 'squeue:expired\\n' >> \"$FAKE_EVENT_LOG\"\nexit 1\n",
        executable=True,
    )

    result = _run_controller(environment)

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "is absent from squeue and terminal in accounting" in result.stdout
    assert "sleep" not in events.read_text(encoding="utf-8").splitlines()


def test_preconvert_restart_reattaches_latest_then_uses_remaining_global_budget(tmp_path: Path) -> None:
    run_dir, fake_bin, events, environment = _controller_fixture(tmp_path)
    _write(Path(environment["FAKE_REMAINING_FILE"]), "1\n")
    _write(Path(environment["FAKE_JOB_COUNTER"]), "202\n")
    _write(run_dir / "PRECONVERT_JOB_1", "201\n")
    _write(run_dir / "PRECONVERT_JOB_2", "202;fixture-cluster\n")
    live_once = _write(tmp_path / "preconvert-live-once", "yes\n")
    _write(
        fake_bin / "squeue",
        """#!/bin/bash
jid=
while (( $# )); do
    if [[ $1 == -j ]]; then jid=$2; break; fi
    shift
done
printf 'squeue:%s\n' "$jid" >> "$FAKE_EVENT_LOG"
if [[ $jid == 202 && -e $FAKE_LIVE_ONCE ]]; then
    rm -f "$FAKE_LIVE_ONCE"
    echo RUNNING
fi
""",
        executable=True,
    )
    _write(
        fake_bin / "sbatch",
        """#!/bin/bash
next=$(( $(<"$FAKE_JOB_COUNTER") + 1 ))
printf '%s\n' "$next" > "$FAKE_JOB_COUNTER"
printf 'sbatch:%s\n' "$next" >> "$FAKE_EVENT_LOG"
printf '0\n' > "$FAKE_REMAINING_FILE"
printf '%s;fixture-cluster\n' "$next"
""",
        executable=True,
    )

    result = _run_controller({**environment, "FAKE_LIVE_ONCE": str(live_once)})

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (run_dir / "PRECONVERT_JOB_1").read_text(encoding="utf-8") == "201\n"
    assert (run_dir / "PRECONVERT_JOB_2").read_text(encoding="utf-8") == "202;fixture-cluster\n"
    assert (run_dir / "PRECONVERT_JOB_3").read_text(encoding="utf-8") == "203\n"
    assert not (run_dir / "PRECONVERT_JOB_4").exists()
    event_lines = events.read_text(encoding="utf-8").splitlines()
    assert event_lines.index("squeue:202") < event_lines.index("sbatch:203")
    assert "reattaching newest preconvert attempt=2 job=202" in result.stdout


def test_preconvert_global_attempt_limit_survives_controller_restart(tmp_path: Path) -> None:
    run_dir, _fake_bin, events, environment = _controller_fixture(tmp_path)
    _write(Path(environment["FAKE_REMAINING_FILE"]), "1\n")
    for attempt in range(1, 5):
        _write(run_dir / f"PRECONVERT_JOB_{attempt}", f"{200 + attempt}\n")

    result = _run_controller(environment)

    assert result.returncode != 0
    assert not any(line.startswith("sbatch:") for line in events.read_text(encoding="utf-8").splitlines())
    assert "model-produced Office files remain unrendered" in result.stdout
    assert not (run_dir / "PRECONVERT_JOB_5").exists()


def test_controller_adopts_lost_preconvert_receipt_without_resubmitting(tmp_path: Path) -> None:
    run_dir, fake_bin, events, environment = _controller_fixture(tmp_path)
    remaining = Path(environment["FAKE_REMAINING_FILE"])
    _write(remaining, "1\n")
    job_name = "gdp-fixture-run-pc1"
    receipt = run_dir / "PRECONVERT_JOB_1"
    seeded = subprocess.run(
        [
            "bash",
            "-c",
            'source "$E2E_DIR/slurm_receipts.sh"; '
            'slurm_submit_or_adopt "$RUN_DIR/PRECONVERT_JOB_1" preconvert-a1 "$1" '
            '--parsable -J "$1" fixture.sbatch',
            "seed-preconvert",
            job_name,
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert seeded.returncode == 0, seeded.stderr
    assert receipt.read_text(encoding="utf-8") == "201\n"
    receipt.unlink()
    intent = run_dir / ".slurm_submit_intents" / "PRECONVERT_JOB_1.intent"
    comment = _intent_comment(intent)
    events.write_text("", encoding="utf-8")

    _write(
        fake_bin / "squeue",
        """#!/bin/bash
name=
jid=
while (( $# )); do
    case "$1" in
        -n) name=$2; shift 2 ;;
        -j) jid=$2; shift 2 ;;
        *) shift ;;
    esac
done
printf 'squeue:name=%s:jid=%s\n' "$name" "$jid" >> "$FAKE_EVENT_LOG"
[[ $name == gdp-fixture-run-pc1 ]] && echo 201
exit 0
""",
        executable=True,
    )
    _write(
        fake_bin / "scontrol",
        """#!/bin/bash
printf 'JobId=201 JobName=gdp-fixture-run-pc1 Comment=%s State=COMPLETED\n' "$FAKE_ADOPT_COMMENT"
""",
        executable=True,
    )
    _write(
        fake_bin / "sacct",
        """#!/bin/bash
if [[ " $* " == *" --name "* ]]; then
    printf '201|%s|\n' "$FAKE_ADOPT_COMMENT"
    exit 0
fi
jid=
while (( $# )); do
    if [[ $1 == -j ]]; then jid=$2; break; fi
    shift
done
printf 'sacct:%s\n' "$jid" >> "$FAKE_EVENT_LOG"
[[ $jid == 201 ]] && printf '0\n' > "$FAKE_REMAINING_FILE"
IFS=, read -ra ids <<<"$jid"
for id in "${ids[@]}"; do
    printf '%s|COMPLETED|0:0|00:00:01|fixture|fixture|fixture\n' "$id"
done
""",
        executable=True,
    )

    result = _run_controller({**environment, "FAKE_ADOPT_COMMENT": comment})

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert receipt.read_text(encoding="utf-8") == "201\n"
    event_lines = events.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("sbatch:") for line in event_lines)
    assert "ADOPTED_SLURM_JOB role=preconvert-a1 job=201" in result.stderr


def test_transport_prebuild_attempt_budget_is_global_across_restarts(tmp_path: Path) -> None:
    run_dir, _fake_bin, events, environment = _controller_fixture(tmp_path)
    (run_dir / "TRANSPORT_PREBUILD_PASS").unlink()

    first = _run_controller(environment)

    assert first.returncode != 0, (first.stdout, first.stderr)
    assert [
        (run_dir / f"TRANSPORT_PREBUILD_JOB_{attempt}").read_text(encoding="utf-8").strip() for attempt in range(1, 3)
    ] == ["201", "202"]
    assert not (run_dir / "TRANSPORT_PREBUILD_JOB_3").exists()
    first_events = events.read_text(encoding="utf-8").splitlines()
    assert [event for event in first_events if event.startswith("sbatch:")] == ["sbatch:201", "sbatch:202"]
    assert "transport views failed validation after 2 attempts" in first.stdout

    events.write_text("", encoding="utf-8")
    second = _run_controller({**environment, "SLURM_JOB_ID": "701"})

    assert second.returncode != 0, (second.stdout, second.stderr)
    assert not (run_dir / "TRANSPORT_PREBUILD_JOB_3").exists()
    assert not any(event.startswith("sbatch:") for event in events.read_text(encoding="utf-8").splitlines())
    assert "reattaching newest transport prebuild attempt=2 job=202" in second.stdout
    assert "transport views failed validation after 2 attempts" in second.stdout

    events.write_text("", encoding="utf-8")
    third = _run_controller(
        {
            **environment,
            "SLURM_JOB_ID": "702",
            "CHECKPOINT_E2E_MAX_TRANSPORT_PREBUILD_ATTEMPTS": "3",
        }
    )

    assert third.returncode != 0, (third.stdout, third.stderr)
    assert (run_dir / "TRANSPORT_PREBUILD_JOB_3").read_text(encoding="utf-8").strip() == "203"
    assert [event for event in events.read_text(encoding="utf-8").splitlines() if event.startswith("sbatch:")] == [
        "sbatch:203"
    ]
    assert "transport views failed validation after 3 attempts" in third.stdout


def test_submit_and_resume_fail_closed_when_campaign_lock_is_owned(tmp_path: Path) -> None:
    checkpoint, environment = _fixture(tmp_path)
    prepared = _run("prepare", checkpoint, environment)
    assert prepared.returncode == 0, prepared.stderr
    run_dir = Path(next(line.split("=", 1)[1] for line in prepared.stdout.splitlines() if line.startswith("RUN_DIR=")))

    fake_bin = tmp_path / "fake-bin"
    fake_flock = _write(fake_bin / "flock", "#!/bin/sh\nexit 1\n", executable=True)
    locked_environment = {**environment, "PATH": f"{fake_flock.parent}:{environment['PATH']}"}
    for action in ("submit", "resume"):
        result = _run(action, checkpoint, locked_environment)
        assert result.returncode == 64
        assert "another submit/resume process owns this campaign" in result.stderr

    assert not (run_dir / "FLEET_SUBMITTED").exists()
    assert not (run_dir / "CONTROLLER_JOBID").exists()


def test_submit_requires_cluster_kill_invalid_dependency_policy(tmp_path: Path) -> None:
    def submit_with_policy(case_root: Path, policy: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        checkpoint, environment = _fixture(case_root)
        prepared = _run("prepare", checkpoint, environment)
        assert prepared.returncode == 0, prepared.stderr
        fake_bin = case_root / "slurm-bin"
        events = _write(case_root / "sbatch-events", "")
        counter = _write(case_root / "sbatch-counter", "800\n")
        _write(fake_bin / "flock", "#!/bin/sh\nexit 0\n", executable=True)
        _write(
            fake_bin / "scontrol",
            """#!/bin/bash
if [[ $1 == show && $2 == config ]]; then
    printf 'DependencyParameters = %s\n' "$FAKE_DEPENDENCY_PARAMETERS"
    exit 0
fi
exit 1
""",
            executable=True,
        )
        _write(
            fake_bin / "sbatch",
            """#!/bin/bash
next=$(( $(<"$FAKE_JOB_COUNTER") + 1 ))
printf '%s\n' "$next" > "$FAKE_JOB_COUNTER"
printf '%s\n' "$next" >> "$FAKE_EVENT_LOG"
printf '%s\n' "$next"
""",
            executable=True,
        )
        result = _run(
            "submit",
            checkpoint,
            {
                **environment,
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "FAKE_DEPENDENCY_PARAMETERS": policy,
                "FAKE_JOB_COUNTER": str(counter),
                "FAKE_EVENT_LOG": str(events),
            },
        )
        return result, events.read_text(encoding="utf-8").splitlines()

    accepted, accepted_events = submit_with_policy(
        tmp_path / "accepted",
        "after_corr, kill_invalid_depend, disable_remote_singleton",
    )
    assert accepted.returncode == 0, (accepted.stdout, accepted.stderr)
    assert "SUBMITTED" in accepted.stdout
    assert accepted_events == [str(job) for job in range(801, 809)]

    rejected, rejected_events = submit_with_policy(tmp_path / "rejected", "after_corr")
    assert rejected.returncode == 64
    assert "Slurm must enable DependencyParameters=kill_invalid_depend" in rejected.stderr
    assert rejected_events == []


def test_resume_requires_terminal_evidence_and_normalizes_cluster_job_id(tmp_path: Path) -> None:
    checkpoint, environment = _fixture(tmp_path)
    prepared = _run("prepare", checkpoint, environment)
    assert prepared.returncode == 0, prepared.stderr
    run_dir = Path(next(line.split("=", 1)[1] for line in prepared.stdout.splitlines() if line.startswith("RUN_DIR=")))
    _write(run_dir / "FLEET_SUBMITTED", "")
    _write(run_dir / "FLEET_JOBS.tsv", "0\t600\t37\t/run\n")
    _write(run_dir / "CONTROLLER_JOBID", "700\n")

    fake_bin = tmp_path / "resume-bin"
    _write(fake_bin / "flock", "#!/bin/sh\nexit 0\n", executable=True)
    _write(fake_bin / "squeue", "#!/bin/sh\nexit 0\n", executable=True)
    _write(fake_bin / "sacct", "#!/bin/sh\nexit 0\n", executable=True)
    events = _write(tmp_path / "resume-events", "")
    _write(
        fake_bin / "sbatch",
        '#!/bin/sh\nprintf "sbatch\\n" >> "$FAKE_EVENT_LOG"\nprintf "701;hsg-cluster\\n"\n',
        executable=True,
    )
    resume_environment = {
        **environment,
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "FAKE_EVENT_LOG": str(events),
    }

    unknown = _run("resume", checkpoint, resume_environment)
    assert unknown.returncode == 64
    assert "no live or terminal Slurm evidence; refusing resume" in unknown.stderr
    assert events.read_text() == ""
    assert (run_dir / "CONTROLLER_JOBID").read_text() == "700\n"

    _write(fake_bin / "squeue", "#!/bin/sh\nexit 1\n", executable=True)
    _write(
        fake_bin / "sacct",
        "#!/bin/sh\nprintf '700|COMPLETED|\\n'\n",
        executable=True,
    )
    terminal = _run("resume", checkpoint, resume_environment)
    assert terminal.returncode == 0, terminal.stderr
    assert "RESUMED controller=701" in terminal.stdout
    assert (run_dir / "CONTROLLER_JOBID").read_text() == "701\n"
    assert events.read_text() == "sbatch\n"

    generation_receipt = run_dir / "controller_submissions" / "after-700.jobid"
    generation_receipt.unlink()
    (run_dir / "CONTROLLER_JOBID").unlink()
    _write(run_dir / "CONTROLLER_JOBID", "700\n")
    intent = run_dir / "controller_submissions" / ".slurm_submit_intents" / "after-700.jobid.intent"
    comment = _intent_comment(intent)
    events.write_text("", encoding="utf-8")
    _write(
        fake_bin / "squeue",
        """#!/bin/bash
if [[ " $* " == *" -n "* ]]; then
    echo 701
fi
exit 0
""",
        executable=True,
    )
    _write(
        fake_bin / "scontrol",
        """#!/bin/bash
printf 'JobId=701 JobName=fixture Comment=%s State=COMPLETED\n' "$FAKE_ADOPT_COMMENT"
""",
        executable=True,
    )
    _write(
        fake_bin / "sacct",
        """#!/bin/bash
if [[ " $* " == *" --name "* ]]; then
    printf '701|%s|\n' "$FAKE_ADOPT_COMMENT"
else
    printf '700|COMPLETED|\n'
fi
""",
        executable=True,
    )
    adopted = _run("resume", checkpoint, {**resume_environment, "FAKE_ADOPT_COMMENT": comment})
    assert adopted.returncode == 0, adopted.stderr
    assert "RESUMED controller=701" in adopted.stdout
    assert "ADOPTED_SLURM_JOB role=controller-after-700 job=701" in adopted.stderr
    assert generation_receipt.read_text(encoding="utf-8") == "701\n"
    assert (run_dir / "CONTROLLER_JOBID").read_text(encoding="utf-8") == "701\n"
    assert events.read_text(encoding="utf-8") == ""


def test_controller_judge_attempt_budget_is_global_across_restarts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    e2e_dir = tmp_path / "e2e"
    fake_bin = tmp_path / "fake-bin"
    deliverables = run_dir / "deliverables"
    for path in (run_dir / "logs", deliverables, e2e_dir, fake_bin):
        path.mkdir(parents=True, exist_ok=True)

    dataset = _write(tmp_path / "dataset.jsonl", "{}\n")
    judge_sbatch = _write(e2e_dir / "judge.sbatch")
    _write(e2e_dir / "slurm_receipts.sh", (PACKAGE / "slurm_receipts.sh").read_text(encoding="utf-8"))
    _write(e2e_dir / "judge_process_group.sh", (PACKAGE / "judge_process_group.sh").read_text(encoding="utf-8"))
    transport_views = _write(e2e_dir / "transport_views.py")
    apptainer = _write(e2e_dir / "apptainer" / "apptainer", "#!/bin/sh\nexit 0\n", executable=True)
    sif = _write(tmp_path / "fixture.sif")
    reference_roots = []
    for index in range(9):
        root = tmp_path / "references" / f"model_{index}"
        root.mkdir(parents=True)
        reference_roots.append(root)
    reference_overlay = _write(
        tmp_path / "reference.yaml",
        "gdpval:\n  reference_models:\n"
        + "".join(
            f"    model_{index}:\n      deliverables_dir: {root}\n" for index, root in enumerate(reference_roots)
        ),
    )
    transport_root = run_dir / "judge_transport_views"
    manifest = _write(transport_root / "manifest.json", "{}\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _write(
        run_dir / "TRANSPORT_PREBUILD_PASS",
        f"schema=gdpval.transport-prebuild.v1\nmanifest={manifest}\nmanifest_sha256={manifest_sha}\n",
    )
    _write(
        run_dir / "settings.env",
        "\n".join(
            (
                "RUN_ID=fixture-run",
                f"DATASET={dataset}",
                f"DELIVERABLES={deliverables}",
                "EXPECTED_TASKS=220",
                "ACCOUNT=fixture",
                "CPU_PARTITION=fixture",
                "CPU_QOS=fixture",
                "JUDGE_WALL=00:10:00",
                f"JUDGE_SBATCH={judge_sbatch}",
                f"TRANSPORT_VIEW_ROOT={transport_root}",
                f"TRANSPORT_VIEWS_PY={transport_views}",
                f"REFERENCE_OVERLAY={reference_overlay}",
                f"APPTAINER_BIN={apptainer.parent}",
                f"GDPVAL_SIF={sif}",
                "MODEL_NAME=fixture-model",
                f"AAV2_ROOT={tmp_path}",
            )
        )
        + "\n",
    )
    _write(run_dir / "FLEET_JOBS.tsv", "shard-0\t90\n")
    _write(e2e_dir / "run_checkpoint_e2e.sh", "#!/bin/sh\nexit 0\n", executable=True)

    fake_python = _write(
        fake_bin / "python",
        """#!/bin/bash
if [[ ${1:-} == -c ]]; then
    cat >/dev/null
    if [[ ${2:-} == *remaining_produced* ]]; then
        echo 0
    else
        echo 220
    fi
elif [[ ${2:-} == coverage ]]; then
    if [[ " $* " == *" --json "* ]]; then
        echo '{"completed": 220}'
    fi
    exit 0
elif [[ ${2:-} == result ]]; then
    exit 1
elif [[ ${2:-} == inventory ]]; then
    echo '{"status":"CLOSED","remaining_produced":0,"closure_fingerprint":"fixture","produced_pairs":[]}'
    exit 0
elif [[ ${1:-} == - ]]; then
    exit 0
else
    exit 0
fi
""",
        executable=True,
    )
    _write(fake_bin / "flock", "#!/bin/sh\nexit 0\n", executable=True)
    _write(
        fake_bin / "date",
        """#!/bin/sh
if [ "${1:-}" = -Is ]; then
    echo 2026-08-22T00:00:00+00:00
else
    /bin/date "$@"
fi
""",
        executable=True,
    )
    _write(
        fake_bin / "sbatch",
        """#!/bin/bash
next=$(( $(<"$FAKE_JOB_COUNTER") + 1 ))
printf '%s\n' "$next" > "$FAKE_JOB_COUNTER"
printf 'sbatch:%s\n' "$next" >> "$FAKE_EVENT_LOG"
printf '%s\n' "$next"
""",
        executable=True,
    )
    _write(
        fake_bin / "squeue",
        """#!/bin/bash
jid=
name=
while (( $# )); do
    case "$1" in
        -j) jid=$2; shift 2 ;;
        -n) name=$2; shift 2 ;;
        *) shift ;;
    esac
done
if [[ -n $name ]]; then
    [[ -n ${FAKE_ADOPT_COMMENT:-} ]] && echo 101
    exit 0
fi
printf 'squeue:%s\n' "$jid" >> "$FAKE_EVENT_LOG"
if [[ -n ${FAKE_LIVE_JID:-} && $jid == "$FAKE_LIVE_JID" && -e $FAKE_LIVE_ONCE ]]; then
    rm -f "$FAKE_LIVE_ONCE"
    echo PENDING
fi
""",
        executable=True,
    )
    _write(
        fake_bin / "sacct",
        """#!/bin/bash
if [[ " $* " == *" --name "* ]]; then
    printf '101|%s|\n' "$FAKE_ADOPT_COMMENT"
    exit 0
fi
jid=
while (( $# )); do
    if [[ $1 == -j ]]; then
        jid=$2
        break
    fi
    shift
done
printf '%s|COMPLETED|0:0|00:00:01|fixture|fixture|fixture\n' "$jid"
""",
        executable=True,
    )
    _write(
        fake_bin / "scontrol",
        """#!/bin/bash
printf 'JobId=101 JobName=gdp-fixture-run-j16 Comment=%s State=COMPLETED\n' "$FAKE_ADOPT_COMMENT"
""",
        executable=True,
    )
    _write(fake_bin / "sleep", "#!/bin/sh\nexit 0\n", executable=True)
    _write(fake_bin / "scancel", "#!/bin/sh\nexit 0\n", executable=True)

    counter = _write(tmp_path / "job-counter", "100\n")
    events = _write(tmp_path / "events", "")
    live_once = tmp_path / "live-once"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUN_DIR": str(run_dir),
        "E2E_DIR": str(e2e_dir),
        "CHECKPOINT_E2E_PYTHON": str(fake_python),
        "CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS": "true",
        "CHECKPOINT_E2E_POLL_SECONDS": "1",
        "FAKE_JOB_COUNTER": str(counter),
        "FAKE_EVENT_LOG": str(events),
        "FAKE_LIVE_ONCE": str(live_once),
        "SLURM_JOB_ID": "700",
    }

    seeded = subprocess.run(
        [
            "bash",
            "-c",
            'source "$E2E_DIR/slurm_receipts.sh"; '
            'slurm_submit_or_adopt "$RUN_DIR/JUDGE_JOB_1" judge-a1 gdp-fixture-run-j16 '
            "--parsable -J gdp-fixture-run-j16 fixture.sbatch",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert seeded.returncode == 0, seeded.stderr
    (run_dir / "JUDGE_JOB_1").unlink()
    comment = _intent_comment(run_dir / ".slurm_submit_intents" / "JUDGE_JOB_1.intent")
    events.write_text("", encoding="utf-8")
    environment = {**environment, "FAKE_ADOPT_COMMENT": comment}

    first = subprocess.run(
        ["bash", str(PACKAGE / "controller.sbatch")],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert first.returncode != 0, (first.stdout, first.stderr)
    assert "ADOPTED_SLURM_JOB role=judge-a1 job=101" in first.stderr
    assert (run_dir / "JUDGE_JOB_1").exists(), (first.stdout, first.stderr)
    assert [(run_dir / f"JUDGE_JOB_{attempt}").read_text(encoding="utf-8").strip() for attempt in range(1, 5)] == [
        "101",
        "102",
        "103",
        "104",
    ]
    first_events = events.read_text(encoding="utf-8").splitlines()
    assert "sbatch:101" not in first_events
    assert [event for event in first_events if event.startswith("sbatch:")] == [
        "sbatch:102",
        "sbatch:103",
        "sbatch:104",
    ]

    events.write_text("", encoding="utf-8")
    live_once.touch()
    second = subprocess.run(
        ["bash", str(PACKAGE / "controller.sbatch")],
        cwd=ROOT,
        env={**environment, "FAKE_LIVE_JID": "104", "SLURM_JOB_ID": "701"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert second.returncode != 0, (second.stdout, second.stderr)
    assert not (run_dir / "JUDGE_JOB_5").exists(), (second.stdout, second.stderr)
    assert [(run_dir / f"JUDGE_JOB_{attempt}").read_text(encoding="utf-8").strip() for attempt in range(1, 5)] == [
        "101",
        "102",
        "103",
        "104",
    ]
    second_events = events.read_text(encoding="utf-8").splitlines()
    assert not any(event.startswith("sbatch:") for event in second_events)
    assert "reattaching newest judge attempt=4 job=104" in second.stdout
    assert "failed after concurrency 16->16->8->4" in second.stdout
