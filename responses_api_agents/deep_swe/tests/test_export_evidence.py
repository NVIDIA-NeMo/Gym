import copy
import gc
import hashlib
import json
import os
import shutil
import stat
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.deep_swe import export_evidence as exporter
from responses_api_agents.deep_swe.export_evidence import EvidenceExportError, export_evidence


GYM_SOURCE = {
    "repository_url": "https://github.com/NVIDIA-NeMo/Gym",
    "commit": "1" * 40,
    "uv_lock_sha256": "2" * 64,
    "working_tree_clean": True,
}


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, _, file_names in os.walk(root, topdown=True, followlinks=False):
        Path(directory).chmod(0o700)
        for name in file_names:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o600)


@pytest.fixture(autouse=True)
def _restore_tmp_permissions(tmp_path: Path):
    yield
    _make_tree_writable(tmp_path)


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400, follow_symlinks=False)
    root.chmod(0o500)


def _make_job(source_root: Path, name: str = "job-one") -> tuple[Path, Path, dict]:
    job = source_root / name
    trial = job / "trial-one"
    trajectory = trial / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_bytes(b'{"schema_version":"ATIF-v1.7","steps":[{}]}')
    runtime = {
        "schema_version": 2,
        "gym_source": dict(GYM_SOURCE),
        "provider": "modal",
        "sha256": "a" * 64,
    }
    (job / "gym-runtime-provenance.json").write_text(json.dumps(runtime), encoding="utf-8")
    (job / "gym-pier-stdout.log").write_text("complete\n", encoding="utf-8")
    _seal_tree(job)
    return job, trial, runtime


def _row(job: Path, trial: Path, runtime: dict, *, status: str = "success") -> dict:
    trajectory = trial / "agent" / "trajectory.json"
    payload = trajectory.read_bytes()
    row = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "task_id": "task-one",
        "status": status,
        "benchmark_metadata": {
            "benchmark": "datacurve-ai/deep-swe",
            "job_dir": str(job),
            "trial_uri": trial.as_uri(),
            "sandbox_runtime_path": str(job / "gym-runtime-provenance.json"),
            "sandbox_runtime": runtime,
            "gym_source": dict(GYM_SOURCE),
        },
        "artifacts": [
            {
                "path": "agent/trajectory.json",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    if status != "success":
        row["error_type"] = "HarnessError"
    return row


def _write_rollouts(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    path.write_bytes(payload)
    return payload


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "private-jobs"
    source_root.mkdir(mode=0o700)
    results = tmp_path / "results"
    results.mkdir()
    return source_root, results / "deep_swe_jobs"


def test_exports_only_successful_referenced_job_and_preserves_paths(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    unreferenced, _, _ = _make_job(source_root, "unreferenced")
    row = _row(job, trial, runtime)
    original_row = json.loads(json.dumps(row))
    rollouts = tmp_path / "rollouts.jsonl"
    payload = _write_rollouts(rollouts, [row])

    receipt = export_evidence(rollouts, export_root, source_root=source_root)

    assert row == original_row
    assert receipt["rollouts_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["exported_jobs"] == 1
    assert receipt["jobs"][0]["original_job_dir"] == str(job)
    assert receipt["jobs"][0]["original_trial_uri"] == trial.as_uri()
    assert (export_root / job.name / "trial-one/agent/trajectory.json").read_bytes().startswith(b"{")
    assert not (export_root / unreferenced.name).exists()
    assert stat.S_IMODE(export_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((export_root / job.name).stat().st_mode) == 0o500
    assert stat.S_IMODE((export_root / job.name / "gym-pier-stdout.log").stat().st_mode) == 0o400


def test_resume_is_idempotent_after_private_source_disappears(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    first = export_evidence(rollouts, export_root, source_root=source_root)
    _make_tree_writable(job)
    shutil.rmtree(job)

    second = export_evidence(rollouts, export_root, source_root=source_root)

    assert second == first


def test_finalized_harness_error_is_exported_but_unreferenced_job_is_not(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    successful_job, successful_trial, successful_runtime = _make_job(source_root)
    failed_job, failed_trial, failed_runtime = _make_job(source_root, "failed-job")
    failed_row = _row(failed_job, failed_trial, failed_runtime, status="harness_error")
    failed_row["_ng_task_index"] = 1
    failed_row["task_id"] = "failed-task"
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(
        rollouts,
        [
            _row(successful_job, successful_trial, successful_runtime),
            failed_row,
        ],
    )

    receipt = export_evidence(rollouts, export_root, source_root=source_root)

    assert receipt["rollout_rows"] == 2
    assert receipt["exported_jobs"] == 2
    assert (export_root / failed_job.name).is_dir()


def test_unsafe_row_without_finalized_evidence_paths_is_not_exported(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    row = _row(job, trial, runtime)
    unsafe = _row(job, trial, runtime, status="harness_error")
    unsafe["_ng_task_index"] = 1
    unsafe["task_id"] = "unsafe-task"
    unsafe["benchmark_metadata"] = {"benchmark": "datacurve-ai/deep-swe"}
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [row, unsafe])

    receipt = export_evidence(rollouts, export_root, source_root=source_root)

    assert receipt["rollout_rows"] == 2
    assert receipt["exported_jobs"] == 1


def test_rejects_symlink_and_does_not_publish_partial_job(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    job.chmod(0o700)
    unsafe_link = job / "unsafe-link"
    unsafe_link.symlink_to(tmp_path / "outside")
    job.chmod(0o500)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])

    with pytest.raises(EvidenceExportError, match="not a safe regular file"):
        export_evidence(rollouts, export_root, source_root=source_root)

    assert not (export_root / job.name).exists()
    job.chmod(0o700)
    unsafe_link.unlink()
    _make_tree_writable(job)


def test_rejects_rollout_manifest_mismatch(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    row = _row(job, trial, runtime)
    row["artifacts"][0]["sha256"] = "0" * 64
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [row])

    with pytest.raises(EvidenceExportError, match="does not match rollout manifest"):
        export_evidence(rollouts, export_root, source_root=source_root)

    assert not (export_root / job.name).exists()


def test_rejects_job_outside_private_source_root(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    outside_root = tmp_path / "outside"
    outside_root.mkdir(mode=0o700)
    job, trial, runtime = _make_job(outside_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])

    with pytest.raises(EvidenceExportError, match="not a direct safe child"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_rejects_corrupted_existing_export(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    export_evidence(rollouts, export_root, source_root=source_root)
    exported = export_root / job.name / "trial-one/agent/trajectory.json"
    exported.chmod(0o600)
    exported.write_bytes(b"changed")
    exported.chmod(0o400)

    with pytest.raises(EvidenceExportError, match="does not match rollout manifest"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_rejects_corrupted_existing_receipt_manifest_summary(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    export_evidence(rollouts, export_root, source_root=source_root)
    receipt_path = export_root / "export-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["jobs"][0]["manifest_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)

    with pytest.raises(EvidenceExportError, match="failed receipt validation"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_rejects_symlinked_export_ancestor(tmp_path: Path) -> None:
    source_root, _ = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    actual = tmp_path / "actual"
    actual.mkdir()
    symlink = tmp_path / "linked-results"
    symlink.symlink_to(actual, target_is_directory=True)

    with pytest.raises(EvidenceExportError, match="missing or symlinked ancestor"):
        export_evidence(rollouts, symlink / "deep_swe_jobs", source_root=source_root)


def test_rejects_attacker_writable_export_ancestor(tmp_path: Path) -> None:
    source_root, _ = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    loose = tmp_path / "loose-results"
    loose.mkdir(mode=0o700)
    loose.chmod(0o777)

    with pytest.raises(EvidenceExportError, match="attacker-writable ancestor"):
        export_evidence(
            rollouts,
            loose / "deep_swe_jobs",
            source_root=source_root,
        )

    loose.chmod(0o700)


def test_export_root_cannot_be_filesystem_root() -> None:
    with pytest.raises(EvidenceExportError, match="cannot be the filesystem root"):
        exporter._ensure_export_root(Path("/"))


def test_export_root_rejects_missing_ancestor_and_final_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(EvidenceExportError, match="missing ancestor"):
        exporter._ensure_export_root(tmp_path / "missing" / "deep_swe_jobs")

    root = tmp_path / "deep_swe_jobs"
    root.mkdir(mode=0o700)
    original_lstat = Path.lstat

    def swapped_lstat(path: Path):
        metadata = original_lstat(path)
        if path == root:
            return SimpleNamespace(st_dev=metadata.st_dev, st_ino=metadata.st_ino + 1)
        return metadata

    monkeypatch.setattr(Path, "lstat", swapped_lstat)
    with pytest.raises(EvidenceExportError, match="changed while it was being validated"):
        exporter._ensure_export_root(root)


def test_owner_only_tree_validation_contract(tmp_path: Path) -> None:
    not_directory = tmp_path / "file"
    not_directory.write_text("x", encoding="utf-8")
    with pytest.raises(EvidenceExportError, match="not a current-user-owned directory"):
        exporter._validate_owner_only_tree(not_directory, ())

    root = tmp_path / "tree"
    root.mkdir(mode=0o700)
    root.chmod(0o755)
    with pytest.raises(EvidenceExportError, match="tree is not owner-only"):
        exporter._validate_owner_only_tree(root, ())
    root.chmod(0o700)

    linked = root / "linked"
    linked.symlink_to(not_directory)
    with pytest.raises(EvidenceExportError, match="evidence entry is unsafe"):
        exporter._validate_owner_only_tree(root, ("linked",))
    linked.unlink()

    loose = root / "loose"
    loose.write_text("x", encoding="utf-8")
    loose.chmod(0o644)
    with pytest.raises(EvidenceExportError, match="entry is not owner-only"):
        exporter._validate_owner_only_tree(root, ("loose",))


def test_rejects_symlinked_existing_receipt(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    export_root.mkdir(mode=0o700)
    outside = tmp_path / "outside-receipt.json"
    outside.write_text("{}", encoding="utf-8")
    receipt_link = export_root / "export-receipt.json"
    receipt_link.symlink_to(outside)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])

    with pytest.raises(EvidenceExportError, match="receipt is unsafe"):
        export_evidence(rollouts, export_root, source_root=source_root)
    receipt_link.unlink()


def test_existing_receipt_contract_failures(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    valid_entry = {
        "destination": "job",
        "manifest_sha256": "a" * 64,
        "files": 1,
        "bytes": 1,
    }
    cases = [
        ({"schema_version": 2, "jobs": []}, "unsupported schema"),
        ({"schema_version": 1, "jobs": ["bad"]}, "invalid job entry"),
        (
            {
                "schema_version": 1,
                "jobs": [valid_entry, dict(valid_entry)],
            },
            "duplicate destinations",
        ),
        (
            {
                "schema_version": 1,
                "jobs": [{**valid_entry, "manifest_sha256": "bad"}],
            },
            "invalid manifest summary",
        ),
    ]
    for payload, message in cases:
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        receipt_path.chmod(0o600)
        with pytest.raises(EvidenceExportError, match=message):
            exporter._load_receipt(receipt_path)


def test_receipt_read_limit_and_write_progress_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path = tmp_path / "export-receipt.json"
    receipt_path.write_text(json.dumps({"schema_version": 1, "jobs": []}), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setattr(
        exporter,
        "read_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(exporter.ArtifactLimitError("too large")),
    )
    with pytest.raises(EvidenceExportError, match="receipt is unsafe"):
        exporter._load_receipt(receipt_path)

    monkeypatch.undo()
    monkeypatch.setattr(exporter.os, "write", lambda *args, **kwargs: 0)
    with pytest.raises(EvidenceExportError, match="write made no progress"):
        exporter._write_receipt(receipt_path, {"schema_version": 1, "jobs": []})
    for temporary in tmp_path.glob(".export-receipt.json.*.tmp"):
        temporary.unlink()


def test_rejects_loose_export_root_and_source_job_modes(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    export_root.mkdir(mode=0o700)
    export_root.chmod(0o755)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    with pytest.raises(EvidenceExportError, match="mode 0700"):
        export_evidence(rollouts, export_root, source_root=source_root)
    export_root.chmod(0o700)
    job.chmod(0o755)
    with pytest.raises(EvidenceExportError, match="source job is not sealed mode 0500"):
        export_evidence(rollouts, export_root, source_root=source_root)
    job.chmod(0o500)
    trajectory = trial / "agent" / "trajectory.json"
    trajectory.chmod(0o600)
    with pytest.raises(EvidenceExportError, match="source evidence file is not sealed mode 0400"):
        export_evidence(rollouts, export_root, source_root=source_root)
    trajectory.chmod(0o400)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text('{"status":"success","status":"success"}\n', encoding="utf-8")
    with pytest.raises(EvidenceExportError, match="duplicate JSON object key"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_rejects_non_object_json() -> None:
    with pytest.raises(EvidenceExportError, match="must be a JSON object"):
        exporter._load_json_object(b"[]", "test payload")


def test_parse_file_uri_rejects_relative_file_uri() -> None:
    with pytest.raises(EvidenceExportError, match="absolute path"):
        exporter._parse_file_uri("file:relative/path", 1)


def test_record_contract_rejects_invalid_provenance_fields(tmp_path: Path) -> None:
    source_root, _ = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    baseline = _row(job, trial, runtime)

    def set_metadata(row: dict, key: str, value) -> None:
        row["benchmark_metadata"][key] = value

    cases = [
        (lambda row: row.update(status=None), "status must be"),
        (lambda row: row.update(status="running"), "status is not a terminal"),
        (lambda row: row.update(task_id=""), "task_id must be"),
        (lambda row: row.update(_ng_task_index=True), "_ng_task_index"),
        (lambda row: row.update(_ng_rollout_index=-1), "_ng_rollout_index"),
        (lambda row: set_metadata(row, "job_dir", "relative"), "normalized absolute"),
        (lambda row: set_metadata(row, "trial_uri", "https://example"), "local file"),
        (lambda row: set_metadata(row, "trial_uri", job.as_uri()), "inside job_dir"),
        (
            lambda row: set_metadata(row, "trial_uri", (tmp_path / "outside").as_uri()),
            "outside job_dir",
        ),
        (lambda row: set_metadata(row, "sandbox_runtime_path", "relative"), "normalized absolute"),
        (
            lambda row: set_metadata(row, "sandbox_runtime_path", str(job / "other.json")),
            "job runtime provenance",
        ),
        (lambda row: set_metadata(row, "sandbox_runtime", None), "must be an object"),
        (lambda row: set_metadata(row, "gym_source", None), "gym_source must be an object"),
        (
            lambda row: row["benchmark_metadata"]["gym_source"].update(repository_url="https://example.invalid"),
            "not the NeMo Gym origin",
        ),
        (
            lambda row: row["benchmark_metadata"]["gym_source"].update(commit="short"),
            "not a full Git SHA",
        ),
        (
            lambda row: row["benchmark_metadata"]["gym_source"].update(uv_lock_sha256="bad"),
            "uv_lock_sha256 is invalid",
        ),
        (
            lambda row: row["benchmark_metadata"]["gym_source"].update(working_tree_clean=False),
            "working_tree_clean is not true",
        ),
        (
            lambda row: row["benchmark_metadata"]["sandbox_runtime"]["gym_source"].update(commit="3" * 40),
            "source provenance copies differ",
        ),
        (lambda row: row.update(artifacts=[]), "nonempty list"),
        (
            lambda row: row.update(artifacts=[row["artifacts"][0]] * (exporter.MAX_ARTIFACT_ENTRIES_PER_ROW + 1)),
            "artifacts exceed",
        ),
        (lambda row: row.update(artifacts=["bad"]), "every artifact"),
        (lambda row: row["artifacts"][0].update(path="../bad"), "unsafe or duplicate"),
        (lambda row: row["artifacts"][0].update(bytes=True), "invalid size"),
        (lambda row: row["artifacts"][0].update(sha256="bad"), "invalid SHA-256"),
        (
            lambda row: row.update(status="s" * (exporter.MAX_TASK_STATUS_UTF8_BYTES + 1)),
            "status exceeds",
        ),
        (
            lambda row: row.update(task_id="é" * (exporter.MAX_TASK_STATUS_UTF8_BYTES // 2 + 1)),
            "task_id exceeds",
        ),
        (lambda row: row.update(task_id="\ud800"), "task_id is not valid UTF-8"),
        (
            lambda row: set_metadata(row, "job_dir", "/" + "j" * exporter.MAX_PATH_URI_UTF8_BYTES),
            "job_dir exceeds",
        ),
        (
            lambda row: set_metadata(row, "trial_uri", "file:///" + "t" * exporter.MAX_PATH_URI_UTF8_BYTES),
            "trial_uri exceeds",
        ),
        (
            lambda row: set_metadata(
                row,
                "sandbox_runtime_path",
                "/" + "r" * exporter.MAX_PATH_URI_UTF8_BYTES,
            ),
            "sandbox_runtime_path exceeds",
        ),
        (
            lambda row: row["artifacts"][0].update(path="a" * (exporter.MAX_PATH_URI_UTF8_BYTES + 1)),
            r"artifacts\[\]\.path exceeds",
        ),
        (
            lambda row: set_metadata(row, "sandbox_runtime", {"invalid": float("nan")}),
            "sandbox_runtime is not canonical JSON",
        ),
    ]
    for mutation, message in cases:
        row = copy.deepcopy(baseline)
        mutation(row)
        with pytest.raises(EvidenceExportError, match=message):
            exporter._parse_record(row, 1, source_root)


def test_record_contract_rejects_oversized_runtime_and_aggregate_artifact_paths(tmp_path: Path) -> None:
    source_root, _ = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    baseline = _row(job, trial, runtime)

    oversized_runtime = copy.deepcopy(baseline)
    oversized_runtime["benchmark_metadata"]["sandbox_runtime"] = {
        "padding": "r" * exporter.MAX_SANDBOX_RUNTIME_CANONICAL_BYTES
    }
    with pytest.raises(EvidenceExportError, match="canonical benchmark_metadata.sandbox_runtime exceeds"):
        exporter._parse_record(oversized_runtime, 1, source_root)

    oversized_paths = copy.deepcopy(baseline)
    base_artifact = oversized_paths["artifacts"][0]
    path_count = exporter.MAX_ARTIFACT_PATHS_UTF8_BYTES // exporter.MAX_PATH_URI_UTF8_BYTES + 1
    oversized_paths["artifacts"] = [
        {
            **base_artifact,
            "path": f"{index:04x}" + "a" * (exporter.MAX_PATH_URI_UTF8_BYTES - 4),
        }
        for index in range(path_count)
    ]
    with pytest.raises(EvidenceExportError, match="artifact paths exceed"):
        exporter._parse_record(oversized_paths, 1, source_root)


def test_loads_339_maximum_compact_records_with_bounded_memory(tmp_path: Path) -> None:
    source_root, _ = _roots(tmp_path)
    rollouts = tmp_path / "maximum-compact-rollouts.jsonl"
    canonical_options = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False}
    sandbox_runtime = {"gym_source": dict(GYM_SOURCE), "padding": ""}
    runtime_overhead = len(json.dumps(sandbox_runtime, **canonical_options).encode("utf-8"))
    sandbox_runtime["padding"] = "r" * (exporter.MAX_SANDBOX_RUNTIME_CANONICAL_BYTES - runtime_overhead)
    assert (
        len(json.dumps(sandbox_runtime, **canonical_options).encode("utf-8"))
        == exporter.MAX_SANDBOX_RUNTIME_CANONICAL_BYTES
    )
    artifact_count = exporter.MAX_ARTIFACT_ENTRIES_PER_ROW
    artifact_path_bytes = exporter.MAX_ARTIFACT_PATHS_UTF8_BYTES // artifact_count
    artifact_paths = [f"{index:04x}" + "a" * (artifact_path_bytes - 4) for index in range(artifact_count)]
    assert sum(len(path.encode("utf-8")) for path in artifact_paths) == exporter.MAX_ARTIFACT_PATHS_UTF8_BYTES
    runtime_suffix_bytes = len(f"/{exporter.RUNTIME_PROVENANCE_RELATIVE.as_posix()}".encode("utf-8"))
    job_path_bytes = exporter.MAX_PATH_URI_UTF8_BYTES - runtime_suffix_bytes
    job_prefix_bytes = len(f"{source_root}/".encode("utf-8"))

    with rollouts.open("w", encoding="utf-8") as stream:
        for index in range(exporter.MAX_ROLLOUT_ROWS):
            job_name_prefix = f"job-{index:03d}-"
            job_name = job_name_prefix + "j" * (job_path_bytes - job_prefix_bytes - len(job_name_prefix))
            job = source_root / job_name
            trial = job / "trial"
            runtime_path = job / exporter.RUNTIME_PROVENANCE_RELATIVE
            assert len(str(runtime_path).encode("utf-8")) == exporter.MAX_PATH_URI_UTF8_BYTES
            assert len(trial.as_uri().encode("utf-8")) <= exporter.MAX_PATH_URI_UTF8_BYTES
            task_prefix = f"task-{index:03d}-"
            row = {
                "_ng_task_index": index,
                "_ng_rollout_index": 0,
                "task_id": task_prefix + "t" * (exporter.MAX_TASK_STATUS_UTF8_BYTES - len(task_prefix)),
                "status": "success",
                "benchmark_metadata": {
                    "benchmark": "datacurve-ai/deep-swe",
                    "job_dir": str(job),
                    "trial_uri": trial.as_uri(),
                    "sandbox_runtime_path": str(runtime_path),
                    "sandbox_runtime": sandbox_runtime,
                    "gym_source": dict(GYM_SOURCE),
                },
                "artifacts": [{"path": path, "bytes": 0, "sha256": "0" * 64} for path in artifact_paths],
            }
            json.dump(row, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
    del row
    gc.collect()

    tracemalloc.start()
    try:
        records, _, row_count = exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=1024 * 1024,
            allow_incomplete_trailing_line=False,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert row_count == exporter.MAX_ROLLOUT_ROWS
    assert len(records) == exporter.MAX_ROLLOUT_ROWS
    assert peak_bytes < 256 * 1024 * 1024


def test_rollout_stream_contract_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, _ = _roots(tmp_path)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_bytes(b'{"benchmark_metadata":{}}')
    with pytest.raises(EvidenceExportError, match="end each row with a newline"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=1024,
            allow_incomplete_trailing_line=False,
        )

    hardlink_target = tmp_path / "hardlink-target.jsonl"
    hardlink_target.write_text("{}\n", encoding="utf-8")
    hardlink = tmp_path / "hardlink.jsonl"
    os.link(hardlink_target, hardlink)
    with pytest.raises(EvidenceExportError, match="regular single-link"):
        exporter._load_rollouts(
            hardlink,
            source_root,
            max_line_bytes=1024,
            allow_incomplete_trailing_line=False,
        )


def test_rollout_stream_rejects_duplicate_destinations_and_task_keys(
    tmp_path: Path,
) -> None:
    source_root, _ = _roots(tmp_path)
    first_job, first_trial, first_runtime = _make_job(source_root)
    first = _row(first_job, first_trial, first_runtime)
    duplicate_destination = copy.deepcopy(first)
    duplicate_destination["task_id"] = "other-task"
    duplicate_destination["_ng_rollout_index"] = 1
    rollouts = tmp_path / "duplicate-destination.jsonl"
    _write_rollouts(rollouts, [first, duplicate_destination])
    with pytest.raises(EvidenceExportError, match="duplicate job directory"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=4096,
            allow_incomplete_trailing_line=False,
        )

    second_job, second_trial, second_runtime = _make_job(source_root, "job-two")
    duplicate_key = _row(second_job, second_trial, second_runtime)
    _write_rollouts(rollouts, [first, duplicate_key])
    with pytest.raises(EvidenceExportError, match="duplicate task/repeat keys"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=4096,
            allow_incomplete_trailing_line=False,
        )

    inconsistent_source = _row(second_job, second_trial, second_runtime)
    inconsistent_source["task_id"] = "other-task"
    inconsistent_source["_ng_task_index"] = 1
    inconsistent_source["benchmark_metadata"]["gym_source"]["commit"] = "3" * 40
    inconsistent_source["benchmark_metadata"]["sandbox_runtime"]["gym_source"]["commit"] = "3" * 40
    _write_rollouts(rollouts, [first, inconsistent_source])
    with pytest.raises(EvidenceExportError, match="inconsistent Gym source provenance"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=4096,
            allow_incomplete_trailing_line=False,
        )


def test_final_rollout_check_rejects_disappeared_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, _ = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    original_lstat = Path.lstat

    def missing_at_recheck(path: Path):
        if path == rollouts:
            raise FileNotFoundError(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", missing_at_recheck)
    with pytest.raises(EvidenceExportError, match="changed during final export"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=4096,
            allow_incomplete_trailing_line=False,
        )
    monkeypatch.undo()
    rollouts.write_bytes(b"\n")
    with pytest.raises(EvidenceExportError, match="blank JSONL row"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=1024,
            allow_incomplete_trailing_line=False,
        )
    rollouts.write_text('{"benchmark_metadata":{}}\n', encoding="utf-8")
    with pytest.raises(EvidenceExportError, match="no finalized"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=1024,
            allow_incomplete_trailing_line=False,
        )
    monkeypatch.setattr(exporter, "MAX_ROLLOUT_ROWS", 1)
    rollouts.write_text("{}\n{}\n", encoding="utf-8")
    with pytest.raises(EvidenceExportError, match="exceeds 1 rows"):
        exporter._load_rollouts(
            rollouts,
            source_root,
            max_line_bytes=1024,
            allow_incomplete_trailing_line=False,
        )


def test_streams_rollouts_without_read_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    original = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == rollouts:
            raise AssertionError("rollout JSONL must be streamed")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    receipt = export_evidence(rollouts, export_root, source_root=source_root)
    assert receipt["exported_jobs"] == 1


def test_rejects_symlinked_rollout_jsonl(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    actual = tmp_path / "actual-rollouts.jsonl"
    _write_rollouts(actual, [_row(job, trial, runtime)])
    linked = tmp_path / "linked-rollouts.jsonl"
    linked.symlink_to(actual)

    with pytest.raises(EvidenceExportError, match="rollout JSONL is unavailable"):
        export_evidence(linked, export_root, source_root=source_root)


def test_final_export_rejects_rollout_metadata_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    original_parse = exporter._parse_record

    def touch_during_parse(*args, **kwargs):
        result = original_parse(*args, **kwargs)
        metadata = rollouts.stat()
        os.utime(
            rollouts,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
        )
        return result

    monkeypatch.setattr(exporter, "_parse_record", touch_during_parse)
    with pytest.raises(EvidenceExportError, match="changed during final export"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_rejects_oversized_rollout_line_before_json_decode(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_bytes(b'{"padding":"' + b"x" * 128 + b'"}\n')
    with pytest.raises(EvidenceExportError, match="rollout row exceeds 64 bytes"):
        export_evidence(
            rollouts,
            export_root,
            source_root=source_root,
            rollout_max_line_bytes=64,
        )


def test_accepts_scaled_near_limit_row_with_duplicated_trajectory(
    tmp_path: Path,
) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    row = _row(job, trial, runtime)
    large_trajectory = "t" * (256 * 1024)
    row["raw_rollout"] = {"trajectory": large_trajectory}
    row["response"] = {"output_text": large_trajectory}
    row["model_patch"] = "p" * (64 * 1024)
    payload = json.dumps(row, sort_keys=True).encode() + b"\n"
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_bytes(payload)

    with pytest.raises(EvidenceExportError, match="rollout row exceeds"):
        export_evidence(
            rollouts,
            export_root,
            source_root=source_root,
            rollout_max_line_bytes=len(payload) - 1,
        )
    receipt = export_evidence(
        rollouts,
        export_root,
        source_root=source_root,
        rollout_max_line_bytes=len(payload),
    )
    assert receipt["exported_jobs"] == 1


def test_growing_jsonl_exports_complete_prefix_then_resumes_superset(
    tmp_path: Path,
) -> None:
    source_root, export_root = _roots(tmp_path)
    first_job, first_trial, first_runtime = _make_job(source_root)
    second_job, second_trial, second_runtime = _make_job(source_root, "job-two")
    first_row = _row(first_job, first_trial, first_runtime)
    second_row = _row(second_job, second_trial, second_runtime)
    second_row["_ng_task_index"] = 1
    second_row["task_id"] = "task-two"
    first_payload = json.dumps(first_row, sort_keys=True).encode() + b"\n"
    second_payload = json.dumps(second_row, sort_keys=True).encode() + b"\n"
    split = len(second_payload) // 2
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_bytes(first_payload + second_payload[:split])

    partial = export_evidence(
        rollouts,
        export_root,
        source_root=source_root,
        allow_incomplete_trailing_line=True,
    )
    assert partial["rollout_rows"] == 1
    assert partial["exported_jobs"] == 1

    with rollouts.open("ab") as stream:
        stream.write(second_payload[split:])
    complete = export_evidence(
        rollouts,
        export_root,
        source_root=source_root,
        allow_incomplete_trailing_line=True,
    )
    assert complete["rollout_rows"] == 2
    assert complete["exported_jobs"] == 2
    assert {entry["destination"] for entry in complete["jobs"]} == {
        first_job.name,
        second_job.name,
    }


def test_growing_mode_still_rejects_malformed_completed_row(
    tmp_path: Path,
) -> None:
    source_root, export_root = _roots(tmp_path)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_bytes(b'{"broken":}\n')
    with pytest.raises(EvidenceExportError, match="invalid rollout row"):
        export_evidence(
            rollouts,
            export_root,
            source_root=source_root,
            allow_incomplete_trailing_line=True,
        )


def test_resume_removes_only_safe_exporter_temporary_entries(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    export_root.mkdir(mode=0o700)
    stale_partial = export_root / ".partial-interrupted"
    stale_partial.mkdir(mode=0o700)
    (stale_partial / "copied.txt").write_text("partial", encoding="utf-8")
    (stale_partial / "copied.txt").chmod(0o600)
    stale_receipt = export_root / ".export-receipt.json.interrupted.tmp"
    stale_receipt.write_text("partial", encoding="utf-8")
    stale_receipt.chmod(0o600)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])

    receipt = export_evidence(rollouts, export_root, source_root=source_root)

    assert receipt["exported_jobs"] == 1
    assert not stale_partial.exists()
    assert not stale_receipt.exists()


def test_rejects_unsafe_stale_temporary_entries(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    export_root.mkdir(mode=0o700)
    unsafe_partial = export_root / ".partial-unsafe"
    unsafe_partial.mkdir(mode=0o755)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    with pytest.raises(EvidenceExportError, match="stale partial export is unsafe"):
        export_evidence(rollouts, export_root, source_root=source_root)
    unsafe_partial.chmod(0o700)
    unsafe_partial.rmdir()


def test_rejects_unsafe_stale_receipt_temporary(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    export_root.mkdir(mode=0o700)
    unsafe = export_root / ".export-receipt.json.unsafe.tmp"
    unsafe.write_text("x", encoding="utf-8")
    unsafe.chmod(0o644)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])

    with pytest.raises(EvidenceExportError, match="stale receipt temporary is unsafe"):
        export_evidence(rollouts, export_root, source_root=source_root)
    unsafe.unlink()


def test_rejects_missing_or_mismatched_runtime_provenance(tmp_path: Path) -> None:
    for case in ("missing", "mismatch"):
        case_root = tmp_path / case
        case_root.mkdir()
        source_root, export_root = _roots(case_root)
        job, trial, runtime = _make_job(source_root)
        row = _row(job, trial, runtime)
        if case == "missing":
            job.chmod(0o700)
            (job / "gym-runtime-provenance.json").unlink()
            job.chmod(0o500)
            message = "provenance is unavailable"
        else:
            row["benchmark_metadata"]["sandbox_runtime"] = {
                **runtime,
                "provider": "different",
            }
            message = "provenance differs"
        rollouts = case_root / "rollouts.jsonl"
        _write_rollouts(rollouts, [row])
        with pytest.raises(EvidenceExportError, match=message):
            export_evidence(rollouts, export_root, source_root=source_root)


def test_record_evidence_rejects_duplicate_manifest_paths(tmp_path: Path) -> None:
    source_root, _ = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    record = exporter._parse_record(_row(job, trial, runtime), 1, source_root)
    assert record is not None
    duplicate = {
        "path": "trial-one/agent/trajectory.json",
        "bytes": 1,
        "sha256": "a" * 64,
    }
    with pytest.raises(EvidenceExportError, match="duplicate paths"):
        exporter._validate_record_evidence(
            record,
            job,
            [duplicate, dict(duplicate)],
        )


def test_incremental_pass_skips_old_hash_but_final_pass_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, export_root = _roots(tmp_path)
    first_job, first_trial, first_runtime = _make_job(source_root)
    second_job, second_trial, second_runtime = _make_job(source_root, "job-two")
    first_row = _row(first_job, first_trial, first_runtime)
    second_row = _row(second_job, second_trial, second_runtime)
    second_row["_ng_task_index"] = 1
    second_row["task_id"] = "task-two"
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [first_row])
    export_evidence(rollouts, export_root, source_root=source_root)
    _write_rollouts(rollouts, [first_row, second_row])
    original_snapshot = exporter._snapshot

    def reject_old_snapshot(root: Path, limits):
        if root.name == first_job.name:
            raise AssertionError("incremental export re-hashed an old job")
        return original_snapshot(root, limits)

    monkeypatch.setattr(exporter, "_snapshot", reject_old_snapshot)
    receipt = export_evidence(
        rollouts,
        export_root,
        source_root=source_root,
        incremental=True,
    )
    assert receipt["exported_jobs"] == 2

    monkeypatch.setattr(exporter, "_snapshot", original_snapshot)
    retained = export_root / first_job.name / "trial-one/agent/trajectory.json"
    retained.chmod(0o600)
    retained.write_bytes(b"mutated")
    retained.chmod(0o400)
    export_evidence(
        rollouts,
        export_root,
        source_root=source_root,
        incremental=True,
    )
    with pytest.raises(EvidenceExportError, match="does not match rollout manifest"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_export_resume_contract_failures(tmp_path: Path) -> None:
    missing_source = tmp_path / "missing-source"
    with pytest.raises(EvidenceExportError, match="source root does not exist"):
        export_evidence(
            tmp_path / "missing.jsonl",
            tmp_path / "export",
            source_root=missing_source,
        )

    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    row = _row(job, trial, runtime)
    _write_rollouts(rollouts, [row])
    with pytest.raises(EvidenceExportError, match="must not overlap"):
        export_evidence(rollouts, source_root, source_root=source_root)

    receipt = export_evidence(rollouts, export_root, source_root=source_root)
    changed_row = copy.deepcopy(row)
    changed_row["task_id"] = "changed-task"
    _write_rollouts(rollouts, [changed_row])
    with pytest.raises(EvidenceExportError, match="receipt does not match"):
        export_evidence(rollouts, export_root, source_root=source_root)
    _write_rollouts(rollouts, [row])

    exported_job = export_root / receipt["jobs"][0]["destination"]
    _make_tree_writable(exported_job)
    shutil.rmtree(exported_job)
    with pytest.raises(EvidenceExportError, match="receipt names a missing"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_export_rejects_missing_source_stale_receipt_and_unreferenced_entry(
    tmp_path: Path,
) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    row = _row(job, trial, runtime)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [row])
    _make_tree_writable(job)
    shutil.rmtree(job)
    with pytest.raises(EvidenceExportError, match="referenced private job is missing"):
        export_evidence(rollouts, export_root, source_root=source_root)

    job, trial, runtime = _make_job(source_root)
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    export_evidence(rollouts, export_root, source_root=source_root)
    receipt_path = export_root / "export-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["jobs"].append(
        {
            "destination": "stale-job",
            "manifest_sha256": "a" * 64,
            "files": 1,
            "bytes": 1,
        }
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    with pytest.raises(EvidenceExportError, match="not referenced by current"):
        export_evidence(rollouts, export_root, source_root=source_root)

    receipt["jobs"].pop()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    extra = export_root / "unexpected"
    extra.write_text("x", encoding="utf-8")
    with pytest.raises(EvidenceExportError, match="unreferenced entry"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_incremental_rejects_loose_existing_destination(tmp_path: Path) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    export_evidence(rollouts, export_root, source_root=source_root)
    destination = export_root / job.name
    destination.chmod(0o755)
    with pytest.raises(EvidenceExportError, match="not a private directory"):
        export_evidence(
            rollouts,
            export_root,
            source_root=source_root,
            incremental=True,
        )
    destination.chmod(0o500)


def test_unreceipted_export_is_recovered_only_when_source_manifest_matches(
    tmp_path: Path,
) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    export_evidence(rollouts, export_root, source_root=source_root)
    receipt_path = export_root / "export-receipt.json"
    receipt_path.unlink()

    recovered = export_evidence(rollouts, export_root, source_root=source_root)
    assert recovered["exported_jobs"] == 1

    receipt_path.unlink()
    destination = export_root / job.name
    destination.chmod(0o700)
    extra = destination / "extra.txt"
    extra.write_text("extra", encoding="utf-8")
    extra.chmod(0o600)
    destination.chmod(0o500)
    with pytest.raises(EvidenceExportError, match="differs from its private source"):
        export_evidence(rollouts, export_root, source_root=source_root)

    destination.chmod(0o700)
    extra.unlink()
    destination.chmod(0o500)
    _make_tree_writable(job)
    shutil.rmtree(job)
    with pytest.raises(EvidenceExportError, match="unreceipted export"):
        export_evidence(rollouts, export_root, source_root=source_root)


def test_publication_fsyncs_nested_directories_and_parent_after_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, export_root = _roots(tmp_path)
    job, trial, runtime = _make_job(source_root)
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, [_row(job, trial, runtime)])
    events: list[tuple[str, Path]] = []
    fsync_calls = 0
    original_fsync = exporter.os.fsync
    original_fsync_directory = exporter._fsync_directory
    original_rename = exporter.os.rename
    original_replace = exporter.os.replace

    def counted_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        original_fsync(descriptor)

    def tracked_directory_sync(path: Path) -> None:
        events.append(("fsync-directory", Path(path)))
        original_fsync_directory(path)

    def tracked_rename(source, destination) -> None:
        events.append(("rename-job", Path(destination)))
        original_rename(source, destination)

    def tracked_replace(source, destination) -> None:
        events.append(("replace-receipt", Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(exporter.os, "fsync", counted_fsync)
    monkeypatch.setattr(exporter, "_fsync_directory", tracked_directory_sync)
    monkeypatch.setattr(exporter.os, "rename", tracked_rename)
    monkeypatch.setattr(exporter.os, "replace", tracked_replace)
    export_evidence(rollouts, export_root, source_root=source_root)

    rename_index = events.index(("rename-job", export_root / job.name))
    replace_index = events.index(("replace-receipt", export_root / "export-receipt.json"))
    assert events[rename_index + 1] == ("fsync-directory", export_root)
    assert events[replace_index + 1] == ("fsync-directory", export_root)
    assert fsync_calls >= 10


def test_main_renders_receipt_and_reports_export_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        exporter,
        "export_evidence",
        lambda *args, **kwargs: {"schema_version": 1, "jobs": []},
    )
    assert (
        exporter.main(
            [
                "--rollouts",
                str(tmp_path / "rollouts.jsonl"),
                "--exported-artifacts-root",
                str(tmp_path / "artifacts"),
                "--source-root",
                str(tmp_path / "source"),
                "--allow-incomplete-trailing-line",
                "--incremental",
            ]
        )
        == 0
    )
    assert '"schema_version": 1' in capsys.readouterr().out

    def fail(*args, **kwargs):
        raise EvidenceExportError("boom")

    monkeypatch.setattr(exporter, "export_evidence", fail)
    with pytest.raises(SystemExit, match="2"):
        exporter.main(
            [
                "--rollouts",
                str(tmp_path / "rollouts.jsonl"),
                "--exported-artifacts-root",
                str(tmp_path / "artifacts"),
            ]
        )
