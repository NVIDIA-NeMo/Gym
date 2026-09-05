# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import json
import stat
import subprocess
from pathlib import Path

import pytest

import benchmarks.gdpval.hsg.checkpoint_e2e.prepare_existing_campaign as existing_import
from benchmarks.gdpval.hsg.checkpoint_e2e.prepare_existing_campaign import (
    ImportError,
    identify,
    identify_package,
    prepare,
    prepare_input,
    publish_envelope,
    verify,
    verify_envelope,
)


def _fixture(tmp_path: Path, *, marker_values: tuple[object, ...] = (None, {"reason": "done"}, {})):
    dataset = tmp_path / "dataset.jsonl"
    rows = [
        {
            "task_id": f"task-{index}",
            "responses_create_params": {"input": []},
            "prompt": f"prompt {index}",
        }
        for index in range(len(marker_values))
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    source = tmp_path / "external"
    for index, marker_value in enumerate(marker_values):
        repeat = source / f"task_task-{index}" / "repeat_0"
        repeat.mkdir(parents=True)
        (repeat / "finish_params.json").write_text(json.dumps(marker_value) + "\n", encoding="utf-8")
        (repeat / f"artifact-{index}.txt").write_text(f"artifact {index}\n", encoding="utf-8")

    package = tmp_path / "package"
    package.mkdir()
    runner = package / "judge.sbatch"
    runner.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    runner.chmod(0o755)
    (package / "true3_transport.yaml").write_text("profile: test\n", encoding="utf-8")
    pycache = package / "__pycache__"
    pycache.mkdir()
    (pycache / "unstable.pyc").write_bytes(b"ignored")

    run_dir = tmp_path / "runs" / "fresh"
    (run_dir / "deliverables").mkdir(parents=True)
    settings = run_dir / "settings.env"
    settings.write_text(f"RUN_DIR={run_dir}\nDATASET={dataset}\n", encoding="utf-8")
    settings.chmod(0o400)
    return dataset.resolve(), source.resolve(), package.resolve(), run_dir.resolve()


def test_import_snapshots_external_tree_and_is_idempotent(tmp_path: Path) -> None:
    dataset, source, package, run_dir = _fixture(tmp_path)
    source_marker = source / "task_task-0" / "repeat_0" / "finish_params.json"
    source_before = (source_marker.read_bytes(), source_marker.stat().st_mode, source_marker.stat().st_mtime_ns)
    identity = identify(source, dataset, expected_tasks=3)

    first = prepare(
        run_dir,
        source,
        dataset,
        package,
        expected_tasks=3,
        expected_import_id=identity["import_id"],
    )
    second = prepare(
        run_dir,
        source,
        dataset,
        package,
        expected_tasks=3,
        expected_import_id=identity["import_id"],
    )

    assert first == second
    assert first["completed"] == 3
    assert (
        source_marker.read_bytes(),
        source_marker.stat().st_mode,
        source_marker.stat().st_mtime_ns,
    ) == source_before
    snapshot_marker = run_dir / "deliverables" / "task_task-0" / "repeat_0" / "finish_params.json"
    assert snapshot_marker.read_bytes() == source_marker.read_bytes()
    assert stat.S_IMODE(snapshot_marker.stat().st_mode) == 0o400
    assert stat.S_IMODE((run_dir / "existing_import_receipt.json").stat().st_mode) == 0o400
    assert stat.S_IMODE((run_dir / "EXISTING_IMPORT_READY").stat().st_mode) == 0o400
    assert not (run_dir / "existing_judge_package" / "__pycache__").exists()
    assert stat.S_IMODE((run_dir / "existing_judge_package" / "judge.sbatch").stat().st_mode) == 0o500
    assert first["judge_package_inventory_sha256"]
    assert identify_package(package)["inventory_sha256"]


def test_import_rejects_nonterminal_and_inexact_task_coverage(tmp_path: Path) -> None:
    dataset, source, _, _ = _fixture(tmp_path, marker_values=([], {"ok": True}, None))
    with pytest.raises(ImportError, match="neither a JSON object nor null"):
        identify(source, dataset, expected_tasks=3)

    (source / "task_task-0" / "repeat_0" / "finish_params.json").write_text("null\n")
    (source / "task_extra" / "repeat_0").mkdir(parents=True)
    (source / "task_extra" / "repeat_0" / "finish_params.json").write_text("{}\n")
    with pytest.raises(ImportError, match="task directory coverage mismatch"):
        identify(source, dataset, expected_tasks=3)


def test_import_rejects_symlinks_without_modifying_source(tmp_path: Path) -> None:
    dataset, source, _, _ = _fixture(tmp_path)
    target = source / "task_task-0" / "repeat_0" / "artifact-0.txt"
    (source / "task_task-0" / "repeat_0" / "link.txt").symlink_to(target)
    with pytest.raises(ImportError, match="symlinks are forbidden"):
        identify(source, dataset, expected_tasks=3)
    assert target.read_text() == "artifact 0\n"


def test_transient_io_retries_dataset_marker_and_copy_without_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, source, package, run_dir = _fixture(tmp_path)
    marker = source / "task_task-0" / "repeat_0" / "finish_params.json"
    artifact = source / "task_task-0" / "repeat_0" / "artifact-0.txt"
    remaining_reads = {dataset: 1, marker: 1}
    real_read_bytes = Path.read_bytes

    def flaky_read_bytes(path: Path) -> bytes:
        if remaining_reads.get(path, 0):
            remaining_reads[path] -= 1
            raise OSError(errno.ENOTCONN, "injected transient read failure")
        return real_read_bytes(path)

    real_copy = existing_import._copy_file_once
    copy_failures = 1

    def flaky_copy(source_path: Path, target: Path, row: dict[str, object]) -> None:
        nonlocal copy_failures
        if source_path == artifact and copy_failures:
            copy_failures -= 1
            raise OSError(errno.EIO, "injected transient copy failure")
        real_copy(source_path, target, row)

    monkeypatch.setattr(existing_import.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(Path, "read_bytes", flaky_read_bytes)
    monkeypatch.setattr(existing_import, "_copy_file_once", flaky_copy)

    identity = identify(source, dataset, expected_tasks=3)
    prepared = prepare(
        run_dir,
        source,
        dataset,
        package,
        expected_tasks=3,
        expected_import_id=identity["import_id"],
    )
    assert prepared["status"] == "PASS"
    assert remaining_reads == {dataset: 0, marker: 0}
    assert copy_failures == 0
    assert verify(run_dir, strict_snapshot=True)["source_inventory_sha256"] == identity["source_inventory_sha256"]


def test_persistent_eio_exhausts_the_bound_and_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, source, _, _ = _fixture(tmp_path)
    marker = source / "task_task-0" / "repeat_0" / "finish_params.json"
    real_read_bytes = Path.read_bytes
    attempts = 0

    def persistent_eio(path: Path) -> bytes:
        nonlocal attempts
        if path == marker:
            attempts += 1
            raise OSError(errno.EIO, "injected persistent read failure")
        return real_read_bytes(path)

    monkeypatch.setattr(existing_import.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(Path, "read_bytes", persistent_eio)
    with pytest.raises(OSError) as failure:
        identify(source, dataset, expected_tasks=3)
    assert failure.value.errno == errno.EIO
    assert attempts == existing_import.IO_RETRY_ATTEMPTS

    required_names = {
        "EIO",
        "ESTALE",
        "ETIMEDOUT",
        "EAGAIN",
        "ENOTCONN",
        "ESHUTDOWN",
        "ENETDOWN",
        "ENETUNREACH",
        "ECONNRESET",
        "EHOSTDOWN",
        "EHOSTUNREACH",
    }
    supported = {getattr(errno, name) for name in required_names if hasattr(errno, name)}
    assert supported <= existing_import.TRANSIENT_IO_ERRNOS


def test_verify_allows_derived_sidecars_but_rejects_imported_file_or_dataset_drift(tmp_path: Path) -> None:
    dataset, source, package, run_dir = _fixture(tmp_path)
    prepare(run_dir, source, dataset, package, expected_tasks=3, expected_import_id=None)
    derived = run_dir / "deliverables" / "task_task-0" / "repeat_0" / "artifact-0.pdf"
    derived.write_bytes(b"%PDF-1.4\n")
    assert verify(run_dir)["status"] == "PASS"
    with pytest.raises(ImportError, match="derived or missing paths"):
        verify(run_dir, strict_snapshot=True)

    imported = run_dir / "deliverables" / "task_task-0" / "repeat_0" / "artifact-0.txt"
    imported.chmod(0o600)
    imported.write_text("tampered\n")
    with pytest.raises(ImportError, match="imported file content drift"):
        verify(run_dir)

    imported.write_text("artifact 0\n")
    dataset.write_text(dataset.read_text().replace("prompt 0", "changed 0"), encoding="utf-8")
    with pytest.raises(ImportError, match="dataset identity drift"):
        verify(run_dir)


def test_prepare_input_is_frozen_and_agent_aligned(tmp_path: Path) -> None:
    dataset, source, package, run_dir = _fixture(tmp_path)
    prepare(run_dir, source, dataset, package, expected_tasks=3, expected_import_id=None)
    output = run_dir / "judge_existing" / "preprocessed_datasets" / "benchmark.jsonl"
    first = prepare_input(run_dir, output)
    second = prepare_input(run_dir, output)
    assert first == second
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert {json.dumps(row["agent_ref"], sort_keys=True) for row in rows} == {
        json.dumps({"type": "responses_api_agents", "name": "gdpval_stirrup_agent"}, sort_keys=True)
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o400

    output.chmod(0o600)
    output.write_text("{}\n")
    output.chmod(0o400)
    with pytest.raises(ImportError, match="immutable receipt drift"):
        prepare_input(run_dir, output)


def test_final_envelope_binds_every_import_only_result_input(tmp_path: Path) -> None:
    dataset, source, package, run_dir = _fixture(tmp_path)
    prepare(run_dir, source, dataset, package, expected_tasks=3, expected_import_id=None)
    runtime = run_dir / "judge_runtime_overlay_existing"
    transport = run_dir / "judge_transport_views_existing"
    runtime.mkdir()
    transport.mkdir()
    (run_dir / "campaign.json").write_text('{"schema":"campaign"}\n')
    (run_dir / "checkpoint_fingerprint.json").write_text('{"sha256":"checkpoint"}\n')
    (runtime / "runtime_manifest.json").write_text('{"schema":"runtime"}\n')
    (transport / "manifest.json").write_text('{"schema":"transport"}\n')
    fingerprint = run_dir / "fingerprint_existing.json"
    fingerprint.write_text(
        json.dumps(
            {
                "schema": "gdpval.multistage-fingerprint-probe.v1",
                "status": "PASS",
                "fingerprint": "a" * 64,
            }
        )
        + "\n"
    )
    result = run_dir / "final_receipt_existing.json"
    result.write_text('{"elo":1234.5}\n')

    published = publish_envelope(run_dir, "existing")
    assert published == verify_envelope(run_dir, "existing")
    envelope = json.loads((run_dir / "final_envelope_existing.json").read_text())
    assert envelope["import_receipt"]["sha256"]
    assert envelope["campaign_manifest"]["sha256"]
    assert envelope["checkpoint_fingerprint"]["sha256"]
    assert envelope["campaign_settings"]["sha256"]
    assert envelope["runtime_manifest"]["sha256"]
    assert envelope["transport_manifest"]["sha256"]
    assert envelope["fingerprint_receipt"]["fingerprint"] == "a" * 64
    assert envelope["strict_result"]["sha256"]

    result.write_text('{"elo":9999}\n')
    with pytest.raises(ImportError, match="final import envelope drift"):
        verify_envelope(run_dir, "existing")


def test_import_only_shell_entrypoints_parse_and_never_submit_rollouts_or_gpus() -> None:
    package = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"
    scripts = [
        package / "run_existing_judge.sh",
        package / "prepare_existing_fingerprint.sh",
        package / "existing_judge_controller.sbatch",
        package / "existing_judge_bootstrap.sbatch",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
    launcher = scripts[0].read_text()
    controller = scripts[2].read_text()
    assert "existing_judge_controller.sbatch" in launcher
    assert "all) all_run" in launcher
    assert "bootstrap) launch_bootstrap" in launcher
    assert "existing_judge_bootstrap.sbatch" in launcher
    assert 'CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS="$authorize"' in launcher
    assert "[[ $ACCOUNT == nemotron_n3_post ]]" in launcher
    assert "[[ $CPU_PARTITION == cpu && $CPU_QOS == cpu-normal ]]" in launcher
    assert "gdpval_rollout.sbatch" not in launcher + controller
    assert "GPU_PARTITION" not in launcher + controller
    assert "--gres" not in launcher + controller
    assert "FLEET_JOBS" not in launcher + controller
    assert "--export=ALL" not in launcher + controller
    slurm_bin = "/cm/local/apps/slurm/current/bin"
    for slurm_script in (
        scripts[0],
        scripts[2],
        scripts[3],
        package / "preconvert_closure.sbatch",
        package / "transport_prebuild.sbatch",
        package / "judge.sbatch",
    ):
        assert slurm_bin in slurm_script.read_text()
    assert 'PATH="$SAFE_PATH"' in launcher
    assert controller.count('PATH="$SAFE_PATH"') >= 3
    preconvert = (package / "preconvert_closure.sbatch").read_text()
    assert preconvert.index("REQUESTED_ACTIVE_PACKAGE=") < preconvert.index('source "$MARS_JOB_ROOT/settings.env"')
    assert ': "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"' in preconvert
    assert 'REQUESTED_ACTIVE_PACKAGE != "$CAMPAIGN_E2E_DIR"' in preconvert
    assert 'CHECKPOINT_E2E_ACTIVE_PACKAGE="$DURABLE_ACTIVE_PACKAGE"' in controller


def test_import_controller_gates_every_scientific_phase_and_preserves_null_markers() -> None:
    package = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"
    controller = (package / "existing_judge_controller.sbatch").read_text()
    fingerprint = (package / "prepare_existing_fingerprint.sh").read_text()
    assert 'prepare_existing_campaign.py" verify' in controller
    assert "PRECONVERT_PASS_EXISTING" in controller
    assert "TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX" in controller
    assert "prepare_existing_fingerprint.sh" in controller
    assert controller.index("prepare_existing_fingerprint.sh") < controller.index("if [[ $AUTHORIZE != true ]]")
    assert controller.index("if [[ $AUTHORIZE != true ]]") < controller.index("existing-judge-a${attempt}")
    assert "null finish markers remain untouched" in controller
    assert "JUDGE_CONCURRENCIES=(16 8 4 1 16 8 4 1)" in controller
    assert "JUDGE_NO_PROGRESS_SECONDS" in controller
    assert "scancel --signal=TERM" in controller
    assert "queue_state != RUNNING" in controller
    assert 'publish_immutable "$RUN_DIR/final_receipt_existing.tmp" "$FINAL_RECEIPT"' in controller
    assert "publish-envelope" in controller
    assert "verify-envelope" in controller
    assert 'publish_immutable "$marker_tmp" "$FINAL_MARKER"' in controller
    assert "prepare-input" in fingerprint
    assert "fingerprint discovery failed" in fingerprint


def test_bootstrap_submit_line_contains_only_paths_and_authorization() -> None:
    package = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"
    launcher = (package / "run_existing_judge.sh").read_text()
    bootstrap = (package / "existing_judge_bootstrap.sbatch").read_text()
    export_line = next(line for line in launcher.splitlines() if 'export_spec="CHECKPOINT=' in line)
    assert "API_KEY" not in export_line
    assert "TOKEN" not in export_line
    assert "PASSWORD" not in export_line
    assert "AUTHORIZE_PROVIDER_CALLS" in export_line
    assert "ACTIVE_PACKAGE" in export_line
    assert "CHECKPOINT_E2E_MODEL_NAME" in export_line
    assert '--export="$export_spec"' in launcher
    assert "CHECKPOINT_E2E_GYM_ROOT" in launcher
    assert "CHECKPOINT_E2E_EXPECTED_GYM_REVISION" in launcher
    assert 'run_existing_judge.sh" prepare' in bootstrap
    assert 'run_existing_judge.sh" submit' in bootstrap
    assert ': "${CHECKPOINT_E2E_MODEL_NAME:?set CHECKPOINT_E2E_MODEL_NAME}"' in bootstrap
    assert "EXPECTED_IMPORT_ID" in export_line
    assert "EXPECTED_PACKAGE_SOURCE_SHA256" in export_line
    assert "EXPECTED_RUN_ID" in export_line
    assert "identify-package" in launcher
    assert "attempt_[0-9]*.jobid" in launcher
    assert "job_liveness" in launcher
    assert "bootstrap submission bound reached" in launcher
