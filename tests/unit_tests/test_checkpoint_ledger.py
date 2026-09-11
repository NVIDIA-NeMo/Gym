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
"""Checkpoint token-free model custody without copying staged token arrays."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from nemo_gym._checkpoint import (
    LEDGER_MANIFEST_NAME,
    MODEL_ADMISSION_URL_PREFIX,
    MODEL_CHECKPOINT_URL_PREFIX,
    MODEL_LEDGER_SUBDIR,
    AdmissionLimiter,
    AgentContinuationRoot,
    CaptureLedgerCheckpointer,
    CheckpointArtifactReference,
    ControlCapabilities,
    ControlFence,
    ExternalStorageReference,
    LedgerMismatchError,
    MultiProcessCapability,
    StaleAttemptError,
    install_control_plane,
    install_model_admission,
    install_model_checkpoint,
    read_jsonl_artifact,
)
from nemo_gym._checkpoint.artifacts import write_jsonl_artifact
from nemo_gym.token_id_capture.lineage import FileLineageStore


AUTH_TOKEN = "checkpoint-token"
AUTH_HEADERS = {"authorization": f"Bearer {AUTH_TOKEN}"}


def _write_custody(root, rollout_id: str, call_count: int = 2) -> bytes:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model_call_id": f"{rollout_id}-call-{index}",
            "staging_key": f"opaque-{rollout_id}-{index}",
            "staging_digest": f"digest-{index}",
            "parent_call_id": None if index == 0 else f"{rollout_id}-call-{index - 1}",
            "staging_chain": [f"opaque-{rollout_id}-{parent}" for parent in range(index)],
        }
        for index in range(call_count)
    ]
    payload = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    (root / f"{rollout_id}.lineage.jsonl").write_bytes(payload)
    return payload


def _continuation_root(
    rollout_id: str,
    *,
    last_call_index: int = 1,
) -> AgentContinuationRoot:
    return AgentContinuationRoot(
        rollout_id=rollout_id,
        attempt_index=0,
        capture_key=rollout_id,
        last_committed_model_call_id=f"{rollout_id}-call-{last_call_index}",
    )


def _write_continuation_index(
    checkpoint: Path,
    roots: list[AgentContinuationRoot],
) -> CheckpointArtifactReference:
    return write_jsonl_artifact(
        checkpoint,
        "agent/continuations.jsonl",
        roots,
    )


def test_commit_restore_preserves_only_token_free_custody(tmp_path) -> None:
    root = tmp_path / "ledger-a"
    expected = _write_custody(root, "rollout-a")
    _write_custody(root, "rollout-b-a2")

    summary = CaptureLedgerCheckpointer(root).commit(
        tmp_path / "checkpoint",
        checkpoint_id="checkpoint-1",
        tombstones=[("rollout-b", 2)],
        continuation_roots=[_continuation_root("rollout-a")],
    )
    assert summary == {
        "rollouts": 1,
        "rows": 2,
        "excluded_tombstoned": 1,
        "excluded_inactive": 0,
        "manifest_digest": summary["manifest_digest"],
        "storage_reference_index": summary["storage_reference_index"],
    }

    ledger_dir = tmp_path / "checkpoint" / MODEL_LEDGER_SUBDIR
    assert (ledger_dir / "rollout-a.lineage.jsonl").read_bytes() == expected
    assert not (ledger_dir / "rollout-b-a2.lineage.jsonl").exists()
    assert not list(ledger_dir.glob("*.tokens.*"))

    restored_root = tmp_path / "ledger-b"
    result = CaptureLedgerCheckpointer(restored_root).restore(tmp_path / "checkpoint")
    assert result["tombstones"] == [{"rollout_id": "rollout-b", "attempt_index": 2}]
    assert (restored_root / "rollout-a.lineage.jsonl").read_bytes() == expected


def test_commit_packages_only_active_continuations_without_scanning_store(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    expected = _write_custody(source, "rollout-a", call_count=3)
    _write_custody(source, "rollout-b", call_count=2)
    checkpoint = tmp_path / "checkpoint"
    root = AgentContinuationRoot(
        rollout_id="rollout-a",
        attempt_index=0,
        capture_key="rollout-a",
        last_committed_model_call_id="rollout-a-call-1",
    )
    original_glob = Path.glob

    def reject_store_glob(path: Path, pattern: str):
        if path == source:
            raise AssertionError(f"checkpoint commit must not scan the lineage store with {pattern!r}")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", reject_store_glob)

    summary = CaptureLedgerCheckpointer(source).commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        source_attempts=[("rollout-a", 0), ("rollout-b", 0)],
        continuation_roots=[root],
    )

    assert summary["rollouts"] == 1
    assert summary["rows"] == 3
    assert summary["excluded_inactive"] == 1
    ledger_dir = checkpoint / MODEL_LEDGER_SUBDIR
    assert (ledger_dir / "rollout-a.lineage.jsonl").read_bytes() == expected
    assert not (ledger_dir / "rollout-b.lineage.jsonl").exists()
    references = read_jsonl_artifact(
        checkpoint,
        CheckpointArtifactReference.model_validate(summary["storage_reference_index"]),
        ExternalStorageReference,
    )
    assert [reference.key for reference in references] == [
        "opaque-rollout-a-0",
        "opaque-rollout-a-1",
    ]
    assert {reference.boundary_model_call_id for reference in references} == {"rollout-a-call-1"}


def test_commit_rejects_a_requested_continuation_without_lineage(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "checkpoint"

    with pytest.raises(LedgerMismatchError, match="continuation roots have no model lineage"):
        CaptureLedgerCheckpointer(source).commit(
            checkpoint,
            checkpoint_id="checkpoint-1",
            tombstones=[],
            source_attempts=[("rollout-missing", 0)],
            continuation_roots=[_continuation_root("rollout-missing")],
        )

    assert not (checkpoint / MODEL_LEDGER_SUBDIR / LEDGER_MANIFEST_NAME).exists()


def test_restore_rejects_corrupt_storage_reference_index_before_install(tmp_path) -> None:
    source = tmp_path / "source"
    _write_custody(source, "rollout-a")
    checkpoint = tmp_path / "checkpoint"
    summary = CaptureLedgerCheckpointer(source).commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        continuation_roots=[_continuation_root("rollout-a")],
    )
    reference_path = checkpoint / summary["storage_reference_index"]["relative_path"]
    reference_path.write_text("corrupt\n")

    restored = tmp_path / "restored"
    with pytest.raises(LedgerMismatchError, match="storage-reference index"):
        CaptureLedgerCheckpointer(restored).restore(checkpoint)
    assert not restored.exists()


def test_restore_rejects_manifest_without_storage_reference_index(tmp_path) -> None:
    source = tmp_path / "source"
    _write_custody(source, "rollout-a")
    checkpoint = tmp_path / "checkpoint"
    CaptureLedgerCheckpointer(source).commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        continuation_roots=[_continuation_root("rollout-a")],
    )
    manifest_path = checkpoint / MODEL_LEDGER_SUBDIR / LEDGER_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["storage_reference_index"]
    manifest_path.write_text(json.dumps(manifest))

    restored = tmp_path / "restored"
    with pytest.raises(LedgerMismatchError, match="missing its storage-reference index"):
        CaptureLedgerCheckpointer(restored).restore(checkpoint)
    assert not restored.exists()


def test_continuation_scope_rejects_duplicate_and_retired_roots(tmp_path) -> None:
    source = tmp_path / "source"
    _write_custody(source, "rollout-a")
    root = AgentContinuationRoot(
        rollout_id="rollout-a",
        attempt_index=0,
        capture_key="rollout-a",
        last_committed_model_call_id="rollout-a-call-1",
    )

    with pytest.raises(LedgerMismatchError, match="duplicate continuation roots"):
        CaptureLedgerCheckpointer(source).commit(
            tmp_path / "duplicate-checkpoint",
            checkpoint_id="checkpoint-1",
            tombstones=[],
            continuation_roots=[root, root],
        )
    with pytest.raises(LedgerMismatchError, match="retired model attempts"):
        CaptureLedgerCheckpointer(source).commit(
            tmp_path / "retired-checkpoint",
            checkpoint_id="checkpoint-1",
            tombstones=[("rollout-a", 0)],
            continuation_roots=[root],
        )


def test_commit_rejects_malformed_lineage_before_manifest_publication(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "rollout-a.lineage.jsonl").write_text('{"model_call_id":"call-1"}\nnot-json\n')
    checkpoint = tmp_path / "checkpoint"

    with pytest.raises(LedgerMismatchError, match="invalid lineage JSON"):
        CaptureLedgerCheckpointer(source).commit(
            checkpoint,
            checkpoint_id="checkpoint-1",
            tombstones=[],
            continuation_roots=[
                AgentContinuationRoot(
                    rollout_id="rollout-a",
                    attempt_index=0,
                    capture_key="rollout-a",
                    last_committed_model_call_id="call-1",
                )
            ],
        )
    assert not (checkpoint / MODEL_LEDGER_SUBDIR / LEDGER_MANIFEST_NAME).exists()


def test_model_commit_accepts_agent_continuation_index_and_returns_reference_index(tmp_path) -> None:
    client, limiter = _participant(tmp_path / "ledger")
    _write_custody(tmp_path / "ledger", "rollout-a")
    checkpoint = tmp_path / "checkpoint"
    continuation_index = write_jsonl_artifact(
        checkpoint,
        "agent/continuations.jsonl",
        [
            AgentContinuationRoot(
                rollout_id="rollout-a",
                attempt_index=0,
                capture_key="rollout-a",
                last_committed_model_call_id="rollout-a-call-1",
            )
        ],
    )
    limiter.release(limiter.admit(rollout_id="rollout-a", attempt_index=0))
    control = {"checkpoint_id": "checkpoint-1", "deadline_ts": 4e9}
    pause = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause",
        json=control,
        headers=AUTH_HEADERS,
    )
    assert pause.status_code == 200
    commit = client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json={
            **control,
            "checkpoint_dir": str(checkpoint),
            "continuation_indexes": [continuation_index.model_dump(mode="json")],
        },
        headers=AUTH_HEADERS,
    )
    assert commit.status_code == 200
    assert commit.json()["excluded_inactive"] == 0
    references = read_jsonl_artifact(
        checkpoint,
        CheckpointArtifactReference.model_validate(commit.json()["storage_reference_index"]),
        ExternalStorageReference,
    )
    assert {reference.key for reference in references} == {
        "opaque-rollout-a-0",
        "opaque-rollout-a-1",
    }

    restored_client, _ = _participant(tmp_path / "restored")
    restored = restored_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={
            "checkpoint_id": "restore-1",
            "deadline_ts": 4e9,
            "checkpoint_dir": str(checkpoint),
        },
        headers=AUTH_HEADERS,
    )
    assert restored.status_code == 200
    assert restored.json()["storage_reference_index"] == commit.json()["storage_reference_index"]


def test_restore_validates_all_files_before_installing_any(tmp_path) -> None:
    source = tmp_path / "source"
    _write_custody(source, "rollout-a")
    _write_custody(source, "rollout-b")
    checkpoint = tmp_path / "checkpoint"
    CaptureLedgerCheckpointer(source).commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        continuation_roots=[
            _continuation_root("rollout-a"),
            _continuation_root("rollout-b"),
        ],
    )
    (checkpoint / MODEL_LEDGER_SUBDIR / "rollout-b.lineage.jsonl").write_text("corrupt")

    restored = tmp_path / "restored"
    with pytest.raises(LedgerMismatchError):
        CaptureLedgerCheckpointer(restored).restore(checkpoint)
    assert not restored.exists()


def test_restore_rejects_uncommitted_and_nonfresh_namespaces(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    ledger_dir = checkpoint / MODEL_LEDGER_SUBDIR
    ledger_dir.mkdir(parents=True)
    _write_custody(ledger_dir, "rollout-a")
    with pytest.raises(LedgerMismatchError):
        CaptureLedgerCheckpointer(tmp_path / "restored").restore(checkpoint)

    source = tmp_path / "source"
    _write_custody(source, "rollout-a")
    CaptureLedgerCheckpointer(source).commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        continuation_roots=[_continuation_root("rollout-a")],
    )
    restored = tmp_path / "nonfresh"
    _write_custody(restored, "old-rollout")
    with pytest.raises(LedgerMismatchError):
        CaptureLedgerCheckpointer(restored).restore(checkpoint)


def _participant(root) -> tuple[TestClient, AdmissionLimiter]:
    app = FastAPI()
    limiter = AdmissionLimiter()
    fence = ControlFence()
    ledger = FileLineageStore(root)
    install_control_plane(
        app,
        capabilities=ControlCapabilities(
            component="responses_api_models",
            name="policy",
            multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
            instance_role="policy",
        ),
        fence=fence,
    )
    install_model_admission(
        app,
        limiter=limiter,
        fence=fence,
        instance_role="policy",
        auth_token=AUTH_TOKEN,
    )
    install_model_checkpoint(
        app,
        fence=fence,
        limiter=limiter,
        ledger_provider=lambda: ledger,
        file_ledger_root_provider=lambda: ledger.checkpoint_root,
        instance_role="policy",
        server_name="policy",
        auth_token=AUTH_TOKEN,
    )
    return TestClient(app), limiter


def test_commit_requires_completed_drain_and_restore_stays_paused(tmp_path) -> None:
    source_client, source_limiter = _participant(tmp_path / "ledger-a")
    _write_custody(tmp_path / "ledger-a", "rollout-a")
    held = source_limiter.admit(rollout_id="rollout-a", attempt_index=0)

    pause_body = {"checkpoint_id": "checkpoint-1", "deadline_ts": 4e9}
    pause = source_client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause",
        json=pause_body,
        headers=AUTH_HEADERS,
    )
    assert pause.json()["state"] == "draining"
    missing_indexes = source_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json={**pause_body, "checkpoint_dir": str(tmp_path / "checkpoint")},
        headers=AUTH_HEADERS,
    )
    assert missing_indexes.status_code == 422
    commit_body = {
        **pause_body,
        "checkpoint_dir": str(tmp_path / "checkpoint"),
        "continuation_indexes": [],
    }
    early = source_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json=commit_body,
        headers=AUTH_HEADERS,
    )
    assert early.status_code == 409
    assert early.json()["error"]["code"] in {"invalid_phase", "ledger_not_quiescent"}

    source_limiter.release(held)
    status = source_client.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status",
        params=pause_body,
        headers=AUTH_HEADERS,
    )
    assert status.json()["state"] == "paused"
    commit = source_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json=commit_body,
        headers=AUTH_HEADERS,
    )
    assert commit.status_code == 200
    assert commit.json()["storage_reference_index"]["records"] == 0
    assert commit.json()["excluded_inactive"] == 1
    retry = source_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json=commit_body,
        headers=AUTH_HEADERS,
    )
    assert retry.json() == commit.json()
    committed_status = source_client.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status",
        params=pause_body,
        headers=AUTH_HEADERS,
    )
    assert committed_status.status_code == 200
    assert committed_status.json()["state"] == "paused"

    restored_client, restored_limiter = _participant(tmp_path / "ledger-b")
    restore = restored_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={
            "checkpoint_id": "restore-1",
            "deadline_ts": 4e9,
            "checkpoint_dir": str(tmp_path / "checkpoint"),
        },
        headers=AUTH_HEADERS,
    )
    assert restore.status_code == 200
    assert restore.json()["storage_reference_index"] == commit.json()["storage_reference_index"]
    assert restored_limiter.counts()["state"] == "paused"
    restored_status = restored_client.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status",
        params={"checkpoint_id": "restore-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )
    assert restored_status.status_code == 200
    assert restored_status.json()["state"] == "paused"
    with pytest.raises(StaleAttemptError):
        restored_limiter.admit(rollout_id="rollout-a", attempt_index=0)


def test_restored_tombstone_fences_exact_attempt(tmp_path) -> None:
    source = tmp_path / "source"
    _write_custody(source, "run-a1")
    CaptureLedgerCheckpointer(source).commit(
        tmp_path / "checkpoint",
        checkpoint_id="checkpoint-1",
        tombstones=[("run-a1", 0)],
        continuation_roots=[],
    )
    restored = tmp_path / "restored"
    result = CaptureLedgerCheckpointer(restored).restore(tmp_path / "checkpoint")
    limiter = AdmissionLimiter()
    for tombstone in result["tombstones"]:
        limiter.install_tombstone(tombstone["rollout_id"], tombstone["attempt_index"])

    with pytest.raises(StaleAttemptError):
        limiter.admit(rollout_id="run-a1", attempt_index=0)
    limiter.release(limiter.admit(rollout_id="run", attempt_index=1))


def test_checkpoint_routes_require_control_bearer(tmp_path) -> None:
    client, _ = _participant(tmp_path / "ledger")
    response = client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "restore-1", "deadline_ts": 4e9, "checkpoint_dir": str(tmp_path)},
    )
    assert response.status_code == 401


def test_manifest_is_published_last(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    _write_custody(source, "rollout-a")
    checkpoint = tmp_path / "checkpoint"

    import nemo_gym._checkpoint.ledger as ledger_module

    real_replace = ledger_module.os.replace

    def fail_manifest_replace(source_path, target_path) -> None:
        if target_path.name == LEDGER_MANIFEST_NAME:
            raise RuntimeError("injected manifest publication failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(ledger_module.os, "replace", fail_manifest_replace)
    with pytest.raises(RuntimeError, match="injected"):
        CaptureLedgerCheckpointer(source).commit(
            checkpoint,
            checkpoint_id="checkpoint-1",
            tombstones=[],
            continuation_roots=[_continuation_root("rollout-a")],
        )
    assert not (checkpoint / MODEL_LEDGER_SUBDIR / LEDGER_MANIFEST_NAME).exists()


def test_model_checkpoint_artifacts_are_namespaced_by_server(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_custody(first, "rollout-a")
    _write_custody(second, "rollout-b")
    checkpoint = tmp_path / "checkpoint"

    CaptureLedgerCheckpointer(first, server_name="policy-a").commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        continuation_roots=[_continuation_root("rollout-a")],
    )
    CaptureLedgerCheckpointer(second, server_name="policy-b").commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        continuation_roots=[_continuation_root("rollout-b")],
    )

    root = checkpoint / MODEL_LEDGER_SUBDIR
    assert (root / "policy-a" / "rollout-a.lineage.jsonl").exists()
    assert (root / "policy-b" / "rollout-b.lineage.jsonl").exists()
    with pytest.raises(ValueError, match="model server name"):
        CaptureLedgerCheckpointer(first, server_name="../policy")

    shutil.copytree(root / "policy-a", root / "policy-copy")
    with pytest.raises(LedgerMismatchError, match="different model server"):
        CaptureLedgerCheckpointer(tmp_path / "restore-copy", server_name="policy-copy").restore(checkpoint)


@pytest.mark.parametrize(
    ("retry_tombstones", "retry_source_attempts", "message"),
    [
        ([("rollout-b", 1)], [("rollout-a", 0)], "abort exclusions"),
        ([], [("rollout-a", 0), ("rollout-c", 2)], "source attempts"),
    ],
)
def test_commit_retry_rejects_changed_semantic_sets_after_final_fsync_failure(
    tmp_path,
    monkeypatch,
    retry_tombstones,
    retry_source_attempts,
    message,
) -> None:
    source = tmp_path / "source"
    _write_custody(source, "rollout-a")
    checkpoint = tmp_path / "checkpoint"
    checkpointer = CaptureLedgerCheckpointer(source, server_name="policy")

    import nemo_gym._checkpoint.ledger as ledger_module

    real_fsync_dir = ledger_module._fsync_dir
    calls = 0

    def fail_final_fsync(path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected final directory fsync failure")
        real_fsync_dir(path)

    monkeypatch.setattr(ledger_module, "_fsync_dir", fail_final_fsync)
    with pytest.raises(OSError, match="injected"):
        checkpointer.commit(
            checkpoint,
            checkpoint_id="checkpoint-1",
            tombstones=[],
            source_attempts=[("rollout-a", 0)],
            continuation_roots=[_continuation_root("rollout-a")],
        )
    assert (checkpoint / MODEL_LEDGER_SUBDIR / "policy" / LEDGER_MANIFEST_NAME).exists()

    with pytest.raises(LedgerMismatchError, match=message):
        checkpointer.commit(
            checkpoint,
            checkpoint_id="checkpoint-1",
            tombstones=retry_tombstones,
            source_attempts=retry_source_attempts,
            continuation_roots=[_continuation_root("rollout-a")],
        )


def test_restored_source_ledger_survives_the_next_commit(tmp_path) -> None:
    source_client, source_limiter = _participant(tmp_path / "source")
    expected = _write_custody(tmp_path / "source", "rollout-a")
    source_limiter.release(source_limiter.admit(rollout_id="rollout-a", attempt_index=0))
    pause = {"checkpoint_id": "checkpoint-1", "deadline_ts": 4e9}
    source_client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json=pause, headers=AUTH_HEADERS)
    first_checkpoint = tmp_path / "checkpoint-1"
    first_index = _write_continuation_index(
        first_checkpoint,
        [_continuation_root("rollout-a")],
    )
    assert (
        source_client.post(
            f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
            json={
                **pause,
                "checkpoint_dir": str(first_checkpoint),
                "continuation_indexes": [first_index.model_dump(mode="json")],
            },
            headers=AUTH_HEADERS,
        ).status_code
        == 200
    )

    restored_client, restored_limiter = _participant(tmp_path / "restored")
    restore = {"checkpoint_id": "restore-1", "deadline_ts": 4e9}
    assert (
        restored_client.post(
            f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
            json={**restore, "checkpoint_dir": str(first_checkpoint)},
            headers=AUTH_HEADERS,
        ).status_code
        == 200
    )
    assert (
        restored_client.post(
            f"{MODEL_ADMISSION_URL_PREFIX}/resume",
            json=restore,
            headers=AUTH_HEADERS,
        ).status_code
        == 200
    )
    with pytest.raises(StaleAttemptError):
        restored_limiter.admit(rollout_id="rollout-a", attempt_index=0)

    second = {"checkpoint_id": "checkpoint-2", "deadline_ts": 4e9}
    restored_client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json=second, headers=AUTH_HEADERS)
    second_checkpoint = tmp_path / "checkpoint-2"
    second_index = _write_continuation_index(
        second_checkpoint,
        [_continuation_root("rollout-a")],
    )
    assert (
        restored_client.post(
            f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
            json={
                **second,
                "checkpoint_dir": str(second_checkpoint),
                "continuation_indexes": [second_index.model_dump(mode="json")],
            },
            headers=AUTH_HEADERS,
        ).status_code
        == 200
    )
    assert (second_checkpoint / MODEL_LEDGER_SUBDIR / "policy" / "rollout-a.lineage.jsonl").read_bytes() == expected


def test_recovered_parent_manifest_survives_checkpoint_restore(tmp_path) -> None:
    source_capture_key = "rollout-a"
    recovered_capture_key = "rollout-a-a1"
    parent = {
        "capture_key": source_capture_key,
        "model_call_id": "call-a",
        "parent_call_id": None,
        "prev_len": 0,
        "delta_len": 2,
        "cum_len": 2,
        "weight_version": 1,
        "digest": "1" * 64,
        "extras_digest": "2" * 64,
        "staging_key": f"{source_capture_key}/call-a",
        "mode": "text",
        "chain_hash": "3" * 64,
        "cumulative_hash": "4" * 64,
        "response_id": "response-a",
    }
    child = {
        "model_call_id": "call-b",
        "parent_call_id": "call-a",
        "prev_len": 2,
        "delta_len": 1,
        "cum_len": 3,
        "weight_version": 1,
        "staging_digest": "5" * 64,
        "extras_digest": "6" * 64,
        "staging_key": f"{recovered_capture_key}/call-b",
        "mode": "token_in",
        "staging_chain": [f"{source_capture_key}/call-a"],
        "chain_hash": "7" * 64,
        "cumulative_hash": "8" * 64,
        "response_id": "response-b",
        "parent_manifest": [parent],
    }
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / f"{recovered_capture_key}.lineage.jsonl"
    source_file.write_text(json.dumps(child, sort_keys=True) + "\n")

    checkpoint = tmp_path / "checkpoint"
    CaptureLedgerCheckpointer(source).commit(
        checkpoint,
        checkpoint_id="checkpoint-1",
        tombstones=[],
        continuation_roots=[
            AgentContinuationRoot(
                rollout_id="rollout-a",
                attempt_index=1,
                capture_key=recovered_capture_key,
                last_committed_model_call_id="call-b",
            )
        ],
    )

    restored = tmp_path / "restored"
    CaptureLedgerCheckpointer(restored).restore(checkpoint)
    restored_manifest = asyncio.run(FileLineageStore(restored).manifest(recovered_capture_key))

    assert [record["model_call_id"] for record in restored_manifest["records"]] == ["call-a", "call-b"]
    assert [record["capture_key"] for record in restored_manifest["records"]] == [
        source_capture_key,
        recovered_capture_key,
    ]


def test_failed_restore_is_paused_observable_and_recoverable(tmp_path) -> None:
    client, limiter = _participant(tmp_path / "restored")
    body = {
        "checkpoint_id": "restore-1",
        "deadline_ts": 4e9,
        "checkpoint_dir": str(tmp_path / "missing"),
    }
    failed = client.post(f"{MODEL_CHECKPOINT_URL_PREFIX}/restore", json=body, headers=AUTH_HEADERS)
    assert failed.status_code == 409
    assert limiter.state.value == "paused"
    capabilities = client.get("/ng-control/v1/capabilities").json()
    assert capabilities["phase"] == "restore_failed_paused"
    status = client.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status",
        params={"checkpoint_id": "restore-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )
    assert status.status_code == 200
    assert status.json()["state"] == "paused"
    resumed = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/resume",
        json={"checkpoint_id": "restore-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )
    assert resumed.status_code == 200
    assert limiter.state.value == "accepting"
