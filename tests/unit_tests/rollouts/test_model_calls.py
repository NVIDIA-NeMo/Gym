# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest
from omegaconf import OmegaConf

from nemo_gym.models.model_call_capture import CaptureStore


def test_observability_enabled_from_config(tmp_path):
    from nemo_gym.rollouts.model_calls import observability_enabled_from_config

    assert observability_enabled_from_config({}) is False

    global_config = OmegaConf.create({"observability_enabled": True, "model_call_capture_dir": str(tmp_path)})
    assert observability_enabled_from_config(global_config) is True

    with pytest.raises(ValueError, match="required"):
        observability_enabled_from_config({"observability_enabled": True})


def _capture_exchange(dialect, model_server, usage, response):
    return {
        "model_call_id": f"call-{model_server}",
        "dialect": dialect,
        "model_ref": {"type": "responses_api_models", "name": model_server},
        "started_at": 100.0,
        "completed_at": 100.01,
        "latency_ms": 1.0,
        "status_code": 200,
        "error_category": None,
        "request": {"input": "hi"},
        "response": {"model": "m", "usage": usage, **response},
    }


def test_merge_capture_attaches_metrics_without_raw_payloads(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    exchange = _capture_exchange(
        "responses",
        "A",
        {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        {
            "id": "resp-A",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        },
    )
    exchange["request_raw"] = "malformed request"
    exchange["response_raw"] = "malformed response"
    store.record("0-0", exchange)

    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "reward": 1.0,
        "response": {"harness": "A"},
        "ng_agent_observations": {
            "source": "test",
            "records": [
                {
                    "kind": "agent_invocation",
                    "invocation_id": "root",
                    "model_calls": [
                        {
                            "model_ref": {"type": "responses_api_models", "name": "A"},
                            "response_id": "resp-A",
                        }
                    ],
                }
            ],
            "gaps": [{"code": "model_call_ownership_unavailable", "detail": "capture:stale"}],
        },
    }
    merge_model_call_capture_into_record(record, [tmp_path])

    capture = record["ng_model_call_capture"]
    assert set(capture) == {"rollout_id", "metrics", "calls"}
    assert capture["rollout_id"] == "0-0"
    assert capture["metrics"]["num_calls"] == 1
    attached_call = capture["calls"][0]
    assert attached_call["model_call_id"] == "call-A"
    assert attached_call["response_id"] == "resp-A"
    assert attached_call["model_ref"] == {"type": "responses_api_models", "name": "A"}
    assert attached_call["model"] == "m"
    assert attached_call["response_status"] == "completed"
    assert attached_call["started_at"] == 100.0 and attached_call["completed_at"] == 100.01
    assert attached_call["tokens_in"] == 3
    assert {"request", "response", "request_raw", "response_raw"}.isdisjoint(attached_call)
    assert record["response"] == {"harness": "A"} and record["reward"] == 1.0
    [joined_ref] = record["ng_agent_observations"]["records"][0]["model_calls"]
    assert joined_ref["model_call_id"] == "call-A"
    assert record["ng_agent_observations"]["gaps"] == []

    with_payloads = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    merge_model_call_capture_into_record(with_payloads, [tmp_path], include_payloads=True)
    attached_call = with_payloads["ng_model_call_capture"]["calls"][0]
    assert attached_call["request"] == exchange["request"]
    assert attached_call["response"] == exchange["response"]
    assert attached_call["request_raw"] == "malformed request"
    assert attached_call["response_raw"] == "malformed response"


def test_merge_capture_owns_opencode_calls_by_client_session(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    exchange = _capture_exchange(
        "chat",
        "A",
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        {"id": "resp-A", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
    )
    exchange["client_session_id"] = "opencode-root"
    store.record("0-0", exchange)
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_agent_observations": {
            "source": "opencode",
            "records": [{"kind": "agent_invocation", "invocation_id": "opencode-root"}],
        },
    }

    merge_model_call_capture_into_record(record, [tmp_path])

    [reference] = record["ng_agent_observations"]["records"][0]["model_calls"]
    assert reference["model_call_id"] == "call-A"
    assert record["ng_agent_observations"]["gaps"] == []


def test_merge_capture_reports_observation_join_failure(tmp_path, monkeypatch):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    exchange = _capture_exchange("chat", "A", {}, {"id": "resp-A"})
    exchange["client_session_id"] = "opencode-root"
    store.record("0-0", exchange)
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_agent_observations": {
            "source": "opencode",
            "records": [{"kind": "agent_invocation", "invocation_id": "opencode-root"}],
        },
    }

    def fail_association(*_args, **_kwargs):
        raise RuntimeError("association failed")

    monkeypatch.setattr("nemo_gym.rollouts.model_calls.join_agent_model_calls", fail_association)
    merge_model_call_capture_into_record(record, [tmp_path])

    assert [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]] == ["agent_observation_join_failed"]


def test_merge_capture_reports_missing_capture(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    rec = {
        "_ng_task_index": 9,
        "_ng_rollout_index": 9,
        "reward": 1.0,
        "ng_agent_observations": {
            "source": "test",
            "records": [
                {
                    "kind": "agent_invocation",
                    "invocation_id": "root",
                    "model_calls": [{"model_call_id": "missing"}],
                }
            ],
            "gaps": [{"code": "model_call_ownership_unavailable"}],
        },
    }
    merge_model_call_capture_into_record(rec, [tmp_path])  # no capture file for 9-9
    assert rec["ng_model_call_capture"]["calls"] == []
    assert [gap["code"] for gap in rec["ng_model_call_capture"]["gaps"]] == ["model_call_capture_no_records"]
    assert {gap["code"] for gap in rec["ng_agent_observations"]["gaps"]} == {
        "model_call_ownership_unavailable",
        "model_call_reference_unmatched",
    }

    CaptureStore(tmp_path).path_for("8-8").touch()
    empty = {"_ng_task_index": 8, "_ng_rollout_index": 8}
    merge_model_call_capture_into_record(empty, [tmp_path])
    assert [gap["code"] for gap in empty["ng_model_call_capture"]["gaps"]] == ["model_call_capture_no_records"]


def test_merge_capture_preserves_valid_records_around_malformed_data(tmp_path):
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    first = _capture_exchange("responses", "A", {}, {"id": "resp-A"})
    third = _capture_exchange("responses", "B", {}, {"id": "resp-B"})
    store.path_for("9-8").write_bytes(orjson.dumps(first) + b"\n{not-json}\n" + orjson.dumps(third) + b"\n")
    record = {"_ng_task_index": 9, "_ng_rollout_index": 8}

    merge_model_call_capture_into_record(record, [tmp_path])

    assert [call["call_index"] for call in record["ng_model_call_capture"]["calls"]] == [0, 2]
    assert [call["response_id"] for call in record["ng_model_call_capture"]["calls"]] == ["resp-A", "resp-B"]
    assert record["ng_model_call_capture"]["gaps"] == [
        {
            "code": "model_call_capture_records_unreadable",
            "detail": "count=1",
        }
    ]


def test_merge_capture_reports_partial_write_loss(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.record("4-2", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    store.mark_incomplete("4-2")
    record = {"_ng_task_index": 4, "_ng_rollout_index": 2}

    merge_model_call_capture_into_record(record, [tmp_path])

    assert len(record["ng_model_call_capture"]["calls"]) == 1
    assert [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]] == ["model_call_capture_incomplete"]


def test_clear_model_call_captures_for_rollouts_run_scoping(tmp_path, monkeypatch):
    import nemo_gym.rollouts.model_calls as obs
    from nemo_gym.rollouts.model_calls import clear_model_call_captures_for_rollouts

    store = CaptureStore(tmp_path)
    store.record("0-0", {"dialect": "chat", "request": {}, "response": {}})
    store.record("1-0", {"dialect": "chat", "request": {}, "response": {}})
    store.mark_incomplete("0-0")
    assert store.read("0-0") and store.read("1-0")

    # Clears only the rollout ids about to be (re)run; rows without indices are skipped, others stay.
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 0, "_ng_rollout_index": 0}, {"no": "id"}], [tmp_path])
    assert store.read("0-0") == [] and not store.is_incomplete("0-0") and store.read("1-0")
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [])  # no dirs -> no-op
    assert store.read("1-0")

    # A stale-capture cleanup failure must be visible rather than mixing old and new calls.
    def _boom(_directory):
        raise OSError("cannot open")

    monkeypatch.setattr(obs, "CaptureStore", _boom)
    with pytest.raises(OSError, match="cannot open"):
        clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [tmp_path])


@pytest.mark.asyncio
async def test_await_quiescence_waits_for_a_detached_write_to_land(tmp_path):
    import asyncio

    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    store = CaptureStore(tmp_path)
    store.mark_pending("2-0", "call-1")

    async def late_write():
        await asyncio.sleep(0.2)
        store.record("2-0", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
        store.clear_pending("2-0", "call-1")

    task = asyncio.create_task(late_write())
    # The race is real at this point: snapshotting now would miss the row.
    assert store.read("2-0") == []

    quiesced = await await_model_call_capture_quiescence("2-0", [tmp_path])
    await task

    assert quiesced is True
    assert len(store.read("2-0")) == 1


@pytest.mark.asyncio
async def test_await_quiescence_gives_up_on_a_stranded_claim(tmp_path):
    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    CaptureStore(tmp_path).mark_pending("3-0", "call-1")

    assert await await_model_call_capture_quiescence("3-0", [tmp_path], timeout_s=0.05, poll_s=0.01) is False


def test_merge_capture_flags_a_claim_that_never_landed(tmp_path):
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.record("4-0", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    store.mark_pending("4-0", "call-2")

    record = {"_ng_task_index": 4, "_ng_rollout_index": 0}
    merge_model_call_capture_into_record(record, [tmp_path])

    assert "model_call_capture_pending" in [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]]


def test_clear_captures_for_rollouts_drops_stale_pending_claims(tmp_path):
    from nemo_gym.rollouts.model_calls import clear_model_call_captures_for_rollouts

    store = CaptureStore(tmp_path)
    store.mark_pending("5-0", "call-1")

    clear_model_call_captures_for_rollouts([{"_ng_task_index": 5, "_ng_rollout_index": 0}], [tmp_path])

    assert store.has_pending("5-0") is False


@pytest.mark.asyncio
async def test_has_pending_reads_only_the_claims_file(tmp_path, monkeypatch):
    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    store = CaptureStore(tmp_path)
    store.mark_pending("6-0", "call-1")
    store.record("6-0", {**_capture_exchange("responses", "A", {}, {"id": "resp-A"}), "model_call_id": "call-1"})

    def capture_file(*_args):
        raise AssertionError("read the capture file")

    monkeypatch.setattr(CaptureStore, "path_for", capture_file)
    # A row on disk does not resolve its claim; only the release does, so the capture is never read.
    assert store.has_pending("6-0") is True
    store.clear_pending("6-0", "call-1")
    assert store.has_pending("6-0") is False
    assert await await_model_call_capture_quiescence("6-0", [tmp_path]) is True


def test_a_torn_claims_line_is_skipped(tmp_path):
    store = CaptureStore(tmp_path)
    store.claims_path_for("8-0").write_bytes(b'{"claim":"call-1"}\n{"cla\n[]\n')

    assert store.has_pending("8-0") is True
    store.clear_pending("8-0", "call-1")
    assert store.has_pending("8-0") is False


def test_merge_drops_the_claims_file_only_when_nothing_is_outstanding(tmp_path):
    from nemo_gym.models.model_call_capture import _record
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.mark_pending("9-0", "call-1")
    store.record("9-0", {**_capture_exchange("responses", "A", {}, {"id": "resp-A"}), "model_call_id": "call-1"})
    store.clear_pending("9-0", "call-1")
    store.mark_pending("9-1", "call-2")

    landed = merge_model_call_capture_into_record({"_ng_task_index": 9, "_ng_rollout_index": 0}, [tmp_path])
    stranded = merge_model_call_capture_into_record({"_ng_task_index": 9, "_ng_rollout_index": 1}, [tmp_path])

    assert "gaps" not in landed["ng_model_call_capture"]
    assert not store.claims_path_for("9-0").exists()
    # A stranded claim stays on disk, so a later reader still sees the write that never landed.
    assert "model_call_capture_pending" in [gap["code"] for gap in stranded["ng_model_call_capture"]["gaps"]]
    assert store.claims_path_for("9-1").exists()

    # A call after the snapshot recreates the claims file, and it reads as settled once its write is done.
    store.mark_pending("9-0", "late")
    _record(
        store,
        "responses",
        "srv",
        b"{}",
        rollout_id="9-0",
        model_call_id="late",
        started_at=100.0,
        completed_at=100.01,
        response_body={},
        status_code=200,
        error_category=None,
        latency_ms=1.0,
    )
    assert store.claims_path_for("9-0").exists()
    assert store.has_pending("9-0") is False


@pytest.mark.asyncio
async def test_capture_reads_never_list_the_capture_directory(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    from nemo_gym.rollouts.model_calls import (
        await_model_call_capture_quiescence,
        clear_model_call_capture_after_snapshot,
        clear_model_call_captures_for_rollouts,
        merge_model_call_capture_into_record,
        snapshot_model_call_capture,
    )

    store = CaptureStore(tmp_path)
    store.mark_pending("10-0", "call-1")
    store.record("10-0", {**_capture_exchange("responses", "A", {}, {"id": "resp-A"}), "model_call_id": "call-1"})
    store.clear_pending("10-0", "call-1")

    def listed(*_args, **_kwargs):
        raise AssertionError("listed the capture directory")

    for owner, name in ((Path, "glob"), (Path, "iterdir"), (os, "listdir")):
        monkeypatch.setattr(owner, name, listed)
    scans = []
    scandir = os.scandir
    monkeypatch.setattr(os, "scandir", lambda path: scans.append(path) or scandir(path))
    record = {"_ng_task_index": 10, "_ng_rollout_index": 0}

    assert await await_model_call_capture_quiescence("10-0", [tmp_path]) is True
    snapshot = snapshot_model_call_capture("10-0", [tmp_path])
    merge_model_call_capture_into_record(record, [tmp_path])
    clear_model_call_capture_after_snapshot(snapshot)
    assert scans == [] and not store.path_for("10-0").exists()

    # Before dispatch, each directory is listed once per call for earlier Gym versions' markers, however many rows.
    clear_model_call_captures_for_rollouts([record, {"_ng_task_index": 10, "_ng_rollout_index": 1}], [tmp_path])
    assert scans == [tmp_path]
    assert len(record["ng_model_call_capture"]["calls"]) == 1
    assert not store.path_for("10-0").exists()


@pytest.mark.asyncio
async def test_a_run_directory_of_an_earlier_version_reads_without_waiting_on_its_marker_files(tmp_path):
    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence, merge_model_call_capture_into_record

    # Earlier Gym versions left a stranded `<rollout id>.<call token>.capture.pending` marker per claimed call.
    store = CaptureStore(tmp_path)
    store.record("11-0", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    (tmp_path / "11-0.c0ffee.capture.pending").touch()

    assert await await_model_call_capture_quiescence("11-0", [tmp_path], timeout_s=5.0) is True
    record = merge_model_call_capture_into_record({"_ng_task_index": 11, "_ng_rollout_index": 0}, [tmp_path])

    assert len(record["ng_model_call_capture"]["calls"]) == 1
    assert "gaps" not in record["ng_model_call_capture"]


@pytest.mark.asyncio
async def test_await_quiescence_backs_off_while_a_claim_is_outstanding(tmp_path, monkeypatch):
    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    store = CaptureStore(tmp_path)
    store.mark_pending("12-0", "call-1")
    polls = []
    has_pending = CaptureStore.has_pending
    monkeypatch.setattr(
        CaptureStore, "has_pending", lambda self, rid, **kwargs: polls.append(rid) or has_pending(self, rid, **kwargs)
    )

    assert await await_model_call_capture_quiescence("12-0", [tmp_path], timeout_s=0.5, poll_s=0.01) is False
    # A fixed 10 ms poll would read the claims about 50 times.
    assert len(polls) <= 10


def test_a_torn_tail_cannot_swallow_the_next_claim(tmp_path):
    store = CaptureStore(tmp_path)
    store.claims_path_for("15-0").write_bytes(b'{"cla')

    store.mark_pending("15-0", "next")
    assert store.has_pending("15-0") is True
    store.clear_pending("15-0", "next")
    assert store.has_pending("15-0") is False


def test_a_torn_tail_cannot_swallow_a_release(tmp_path):
    store = CaptureStore(tmp_path)
    store.claims_path_for("15-1").write_bytes(b'{"claim":"open"}\n{"rel')

    assert store.has_pending("15-1") is True
    store.clear_pending("15-1", "open")
    assert store.has_pending("15-1") is False


def test_deleting_settled_claims_never_hides_a_call_admitted_meanwhile(tmp_path, monkeypatch):
    import asyncio
    import fcntl
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    import nemo_gym.models.model_call_capture as capture
    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence, merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.mark_pending("13-0", "call-1")
    store.record("13-0", {**_capture_exchange("responses", "A", {}, {"id": "resp-A"}), "model_call_id": "call-1"})
    store.clear_pending("13-0", "call-1")

    admitting = threading.Event()
    admission = threading.Thread(target=store.mark_pending, args=("13-0", "call-2"))

    def flock(fd, operation):
        if threading.current_thread() is admission and operation == fcntl.LOCK_EX:
            admitting.set()  # The admission holds the file the merge is about to delete.
        fcntl.flock(fd, operation)

    monkeypatch.setattr(
        capture,
        "fcntl",
        SimpleNamespace(
            flock=flock, LOCK_EX=fcntl.LOCK_EX, LOCK_SH=fcntl.LOCK_SH, LOCK_UN=fcntl.LOCK_UN, LOCK_NB=fcntl.LOCK_NB
        ),
    )
    unlink = Path.unlink

    def unlink_after_an_admission(path, *args, **kwargs):
        if path == store.claims_path_for("13-0") and not admitting.is_set():
            # The merge has read the claims as settled; a second call opens the file before the unlink.
            admission.start()
            assert admitting.wait(5)
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_after_an_admission)

    snapshot = merge_model_call_capture_into_record({"_ng_task_index": 13, "_ng_rollout_index": 0}, [tmp_path])
    admission.join(5)

    assert "gaps" not in snapshot["ng_model_call_capture"]
    fresh = CaptureStore(tmp_path)
    assert fresh.has_pending("13-0") is True
    later = merge_model_call_capture_into_record({"_ng_task_index": 13, "_ng_rollout_index": 0}, [tmp_path])
    assert "model_call_capture_pending" in [gap["code"] for gap in later["ng_model_call_capture"]["gaps"]]
    drained = asyncio.run(await_model_call_capture_quiescence("13-0", [tmp_path], timeout_s=0.05, poll_s=0.01))
    assert drained is False

    fresh.record("13-0", {**_capture_exchange("responses", "B", {}, {"id": "resp-B"}), "model_call_id": "call-2"})
    fresh.clear_pending("13-0", "call-2")
    assert asyncio.run(await_model_call_capture_quiescence("13-0", [tmp_path], timeout_s=1.0)) is True
    assert [exchange["model_call_id"] for exchange in fresh.read("13-0")] == ["call-1", "call-2"]


def test_deleting_follows_a_claims_file_replaced_before_its_lock(tmp_path, monkeypatch):
    import nemo_gym.models.model_call_capture as capture

    store = CaptureStore(tmp_path)
    store.mark_pending("16-0", "call-1")
    is_live = capture._is_live
    checks = []

    def replaced_once(handle, path):
        checks.append(path)
        return len(checks) > 1 and is_live(handle, path)

    monkeypatch.setattr(capture, "_is_live", replaced_once)

    assert store.delete_claims_if_settled("16-0") is False
    assert len(checks) == 2 and store.claims_path_for("16-0").exists()


def test_a_claims_file_is_live_only_while_its_path_still_names_it(tmp_path):
    from nemo_gym.models.model_call_capture import _is_live

    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"")
    second.write_bytes(b"")
    with first.open("rb") as handle:
        assert _is_live(handle, first) is True
        assert _is_live(handle, second) is False
        assert _is_live(handle, tmp_path / "missing") is False


def test_claims_that_keep_being_deleted_give_up_without_losing_state(tmp_path, monkeypatch, caplog):
    import nemo_gym.models.model_call_capture as capture

    store = CaptureStore(tmp_path)
    store.mark_pending("16-1", "call-1")
    monkeypatch.setattr(capture, "_is_live", lambda *_: False)

    assert store.delete_claims_if_settled("16-1") is False
    store.mark_pending("16-1", "call-2")
    assert "Could not take the capture claim" in caplog.text


def test_settled_claims_that_cannot_be_removed_still_read_as_settled(tmp_path, monkeypatch, caplog):
    from pathlib import Path

    store = CaptureStore(tmp_path)
    store.mark_pending("16-2", "call-1")
    store.clear_pending("16-2", "call-1")

    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "unlink", refuse)

    assert store.delete_claims_if_settled("16-2") is True
    assert "Could not delete the capture claims" in caplog.text


def _snapshot(store, rollout_id):
    from nemo_gym.rollouts.model_calls import snapshot_model_call_capture

    return snapshot_model_call_capture(rollout_id, [store.root])


def _call(store, rollout_id, model_call_id):
    store.mark_pending(rollout_id, model_call_id)
    store.record(rollout_id, {"model_call_id": model_call_id, "dialect": "responses", "request": {}, "response": {}})
    store.clear_pending(rollout_id, model_call_id)


def test_deleting_a_merged_capture_removes_every_file_of_a_settled_rollout(tmp_path):
    from nemo_gym.rollouts.model_calls import clear_model_call_capture_after_snapshot

    store = CaptureStore(tmp_path)
    _call(store, "14-0", "call-1")
    snapshot = _snapshot(store, "14-0")
    # The merge already deleted the settled claims file; the deletion locks a fresh one and removes it too.
    assert store.delete_claims_if_settled("14-0") is True

    clear_model_call_capture_after_snapshot(snapshot)

    assert list(tmp_path.iterdir()) == []


def test_deleting_a_merged_capture_keeps_an_outstanding_claim(tmp_path):
    from nemo_gym.rollouts.model_calls import clear_model_call_capture_after_snapshot

    store = CaptureStore(tmp_path)
    store.mark_pending("14-1", "call-1")
    store.record("14-1", {"model_call_id": "call-1"})

    clear_model_call_capture_after_snapshot(_snapshot(store, "14-1"))

    assert store.path_for("14-1").exists() and store.has_pending("14-1") is True


def test_deleting_a_merged_capture_keeps_a_row_that_landed_after_the_merge(tmp_path, monkeypatch):
    import asyncio

    import nemo_gym.rollouts.rollout_collection as collection

    store = CaptureStore(tmp_path)
    _call(store, "22-0", "call-1")
    merge = collection.merge_model_call_capture_into_record

    def merge_then_land(record, *args, **kwargs):
        merged = merge(record, *args, **kwargs)
        _call(store, "22-0", "late")  # Read call-1 only; a later call lands and releases before the deletion.
        return merged

    monkeypatch.setattr(collection, "merge_model_call_capture_into_record", merge_then_land)
    row = {"_ng_task_index": 22, "_ng_rollout_index": 0}
    result = asyncio.run(
        collection.enrich_run_result(
            row, dict(row), capture_dirs=[tmp_path], observability_enabled=True, delete_capture=True
        )
    )

    assert [call["model_call_id"] for call in result["ng_model_call_capture"]["calls"]] == ["call-1"]
    assert [exchange["model_call_id"] for exchange in store.read("22-0")] == ["call-1", "late"]


def test_deleting_a_merged_capture_keeps_a_claim_admitted_after_the_merge(tmp_path, monkeypatch):
    import asyncio

    import nemo_gym.rollouts.rollout_collection as collection

    store = CaptureStore(tmp_path)
    _call(store, "22-1", "call-1")
    merge = collection.merge_model_call_capture_into_record

    def merge_then_admit(record, *args, **kwargs):
        merged = merge(record, *args, **kwargs)
        store.mark_pending("22-1", "late")
        return merged

    monkeypatch.setattr(collection, "merge_model_call_capture_into_record", merge_then_admit)
    row = {"_ng_task_index": 22, "_ng_rollout_index": 1}
    asyncio.run(
        collection.enrich_run_result(
            row, dict(row), capture_dirs=[tmp_path], observability_enabled=True, delete_capture=True
        )
    )

    # The late call may already hold the capture open, so nothing it could append to is deleted.
    assert store.path_for("22-1").exists()
    assert store.has_pending("22-1") is True


def test_deleting_a_merged_capture_waits_only_a_bounded_time_on_the_claims_lock(tmp_path):
    import fcntl

    store = CaptureStore(tmp_path)
    _call(store, "22-2", "call-1")
    snapshot = _snapshot(store, "22-2")

    with store.claims_path_for("22-2").open("rb") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_SH)
        assert store.delete_capture_if_unchanged("22-2", snapshot.identity, lock_wait_s=0.05) is False

    assert store.path_for("22-2").exists()


def test_deleting_a_merged_capture_logs_a_failed_unlink(tmp_path, monkeypatch, caplog):
    from nemo_gym.rollouts.model_calls import clear_model_call_capture_after_snapshot

    store = CaptureStore(tmp_path)
    _call(store, "22-3", "call-1")

    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(CaptureStore, "delete_capture_if_unchanged", refuse)
    clear_model_call_capture_after_snapshot(_snapshot(store, "22-3"))

    assert "Could not delete the model-call capture of rollout 22-3" in caplog.text


def test_a_held_claims_lock_counts_as_pending(tmp_path):
    import fcntl

    store = CaptureStore(tmp_path)
    store.mark_pending("24-0", "call-1")
    store.clear_pending("24-0", "call-1")

    with store.claims_path_for("24-0").open("rb") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        assert store.has_pending("24-0", lock_wait_s=0.05) is True
        assert store.delete_claims_if_settled("24-0", lock_wait_s=0.05) is False
    assert store.has_pending("24-0") is False


@pytest.mark.asyncio
async def test_await_quiescence_keeps_the_event_loop_free_while_a_claims_lock_is_held(tmp_path):
    import asyncio
    import fcntl

    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    store = CaptureStore(tmp_path)
    store.mark_pending("24-1", "call-1")
    store.clear_pending("24-1", "call-1")
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(tick())
    with store.claims_path_for("24-1").open("rb") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        drained = await await_model_call_capture_quiescence("24-1", [tmp_path], timeout_s=0.3)
    ticker.cancel()

    assert drained is False
    assert ticks >= 10


def test_has_pending_follows_a_claims_file_replaced_before_its_lock(tmp_path, monkeypatch):
    import nemo_gym.models.model_call_capture as capture

    store = CaptureStore(tmp_path)
    store.mark_pending("24-2", "call-1")
    store.clear_pending("24-2", "call-1")
    is_live = capture._is_live
    replaced = []

    def replaced_once(handle, path):
        if not replaced:
            replaced.append(path)
            # After this reader opened the settled file, a merge deleted it and a new call claimed in a replacement.
            path.unlink()
            store.mark_pending("24-2", "call-2")
        return is_live(handle, path)

    monkeypatch.setattr(capture, "_is_live", replaced_once)

    assert store.has_pending("24-2") is True


def test_clear_all_pending_still_drops_every_claim(tmp_path):
    store = CaptureStore(tmp_path)
    store.mark_pending("24-3", "call-1")
    (tmp_path / "24-3.old.capture.pending").touch()

    with pytest.warns(DeprecationWarning, match="delete_claims"):
        store.clear_all_pending("24-3")

    assert list(tmp_path.iterdir()) == []


def test_clear_all_pending_of_a_missing_directory_finds_nothing(tmp_path):
    store = CaptureStore(tmp_path / "missing", create=False)

    with pytest.warns(DeprecationWarning, match="delete_claims"):
        store.clear_all_pending("24-4")

    assert not (tmp_path / "missing").exists()


def test_no_call_can_claim_or_release_while_the_merge_reads_the_capture(tmp_path, monkeypatch):
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.mark_pending("25-0", "call-1")
    store.record("25-0", {**_capture_exchange("responses", "A", {}, {"id": "resp-A"}), "model_call_id": "call-1"})
    store.clear_pending("25-0", "call-1")
    read_available = CaptureStore.read_available
    late = []

    def read_then_a_late_call(self, rollout_id):
        read = read_available(self, rollout_id)
        # A call that claims, writes and releases right after the read; a claims lock held by the merge stops it.
        if not store.has_pending(rollout_id, lock_wait_s=0):
            store.mark_pending(rollout_id, "call-2")
            store.record(
                rollout_id, {**_capture_exchange("responses", "B", {}, {"id": "resp-B"}), "model_call_id": "call-2"}
            )
            store.clear_pending(rollout_id, "call-2")
            late.append("call-2")
        return read

    monkeypatch.setattr(CaptureStore, "read_available", read_then_a_late_call)
    record = merge_model_call_capture_into_record({"_ng_task_index": 25, "_ng_rollout_index": 0}, [tmp_path])

    assert late == []
    assert [call["model_call_id"] for call in record["ng_model_call_capture"]["calls"]] == ["call-1"]
    assert "gaps" not in record["ng_model_call_capture"]
    assert not store.claims_path_for("25-0").exists()


@pytest.mark.asyncio
async def test_await_quiescence_returns_at_its_deadline_behind_a_busy_executor(tmp_path):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    CaptureStore(tmp_path).mark_pending("26-0", "call-1")
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    busy, gate = threading.Event(), threading.Event()
    blocker = loop.run_in_executor(None, lambda: busy.set() or gate.wait(5))
    assert busy.wait(5)

    drained = await await_model_call_capture_quiescence("26-0", [tmp_path], timeout_s=0.2)
    held = not blocker.done()
    gate.set()
    await blocker

    assert drained is False
    assert held, "the drain waited for the busy worker past its deadline"


def test_merge_keeps_the_calls_it_read_when_the_claims_cannot_be_checked(tmp_path, monkeypatch, caplog):
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.mark_pending("17-0", "call-1")
    store.record("17-0", {**_capture_exchange("responses", "A", {}, {"id": "resp-A"}), "model_call_id": "call-1"})
    store.clear_pending("17-0", "call-1")

    def read_only(*_args, **_kwargs):
        raise PermissionError("read-only claims file")

    monkeypatch.setattr(CaptureStore, "delete_claims_if_settled", read_only)
    record = merge_model_call_capture_into_record({"_ng_task_index": 17, "_ng_rollout_index": 0}, [tmp_path])

    assert len(record["ng_model_call_capture"]["calls"]) == 1
    assert [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]] == ["model_call_capture_pending"]
    assert "Could not check the capture claims" in caplog.text


def test_clearing_before_dispatch_removes_only_the_cleared_keys_pending_markers(tmp_path):
    from nemo_gym.rollouts.model_calls import clear_model_call_captures_for_rollouts

    store = CaptureStore(tmp_path)
    for name in (
        "0-0.aa.capture.pending",
        "0-0.bb.capture.pending",
        "0-0-a1.cc.capture.pending",
        "1-0.dd.capture.pending",
    ):
        (tmp_path / name).touch()
    store.record("0-0", {"model_call_id": "stale"})
    store.mark_incomplete("0-0")
    store.mark_pending("0-0", "stale")

    clear_model_call_captures_for_rollouts([{"_ng_task_index": 0, "_ng_rollout_index": 0}], [tmp_path])

    assert sorted(path.name for path in tmp_path.iterdir()) == ["0-0-a1.cc.capture.pending", "1-0.dd.capture.pending"]
