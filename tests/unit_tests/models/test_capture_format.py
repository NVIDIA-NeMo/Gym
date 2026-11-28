# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The delta capture format: what it writes, and that every reader gets back the exchanges it was given."""

import base64
import gc
import hashlib
import threading
import tracemalloc
from pathlib import Path

import orjson
import pytest

from nemo_gym.models.capture_format import BLOB_MIN_CHARS, CAPTURE_FORMAT_KEY
from nemo_gym.models.model_call_capture import CaptureStore, ModelCallCaptureConfig, make_capture_store


def _image(seed: int, size: int = 6000) -> str:
    payload = hashlib.sha256(str(seed).encode()).digest() * (size // 32)
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _tool_output(index: int, image: str | None = None) -> dict:
    output: list[dict] = [{"type": "input_text", "text": f"tool output {index} " + "x" * 200}]
    if image is not None:
        output.append({"type": "input_image", "image_url": image, "detail": "auto"})
    return {"type": "function_call_output", "call_id": f"call_{index}", "output": output}


def _responses_exchange(call_id: str, history: list, *, text: str = "ok") -> dict:
    return {
        "model_call_id": call_id,
        "dialect": "responses",
        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
        "started_at": 1.0,
        "completed_at": 2.0,
        "latency_ms": 1000.0,
        "status_code": 200,
        "error_category": None,
        "request": {
            "model": "policy",
            "instructions": "You are a coding agent. " * 40,
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            "input": list(history),
            "stream": True,
        },
        "response": {"id": f"resp-{call_id}", "output": [{"type": "message", "content": text}]},
        "response_raw": f"event: response.completed\ndata: {call_id}\n\n",
    }


def _chat_exchange(call_id: str, messages: list) -> dict:
    return {
        "model_call_id": call_id,
        "dialect": "chat",
        "status_code": 200,
        "request": {"model": "policy", "messages": list(messages), "tools": [{"type": "function"}]},
        "response": {"choices": [{"message": {"role": "assistant", "content": call_id}}]},
    }


def _session(*, calls: int = 12, compact_at: int | None = 7) -> list[dict]:
    """A Codex-like session: every call re-sends the whole history, earlier images included; one compaction."""
    history: list = [{"role": "user", "content": "Solve the task."}]
    exchanges = []
    for index in range(calls):
        if index == compact_at:
            history = [{"role": "user", "content": "Summary of the work so far."}, history[-1]]
        history = history + [
            {"type": "function_call", "call_id": f"call_{index}", "name": "shell", "arguments": "{}"},
            _tool_output(index, _image(index // 2 % 4) if index % 2 == 0 else None),
        ]
        exchanges.append(_responses_exchange(f"c{index}", history, text=f"step {index}"))
    return exchanges


def _write(store: CaptureStore, rollout_id: str, exchanges: list[dict]) -> list[dict]:
    """Record each exchange; return what a reader must get back (the exchange as written, JSON-normalized)."""
    expected = []
    for exchange in exchanges:
        expected.append(orjson.loads(orjson.dumps(exchange)))
        store.record(rollout_id, exchange)
    return expected


def _stored_lines(store: CaptureStore, rollout_id: str) -> list[dict]:
    return [orjson.loads(line) for line in store.path_for(rollout_id).read_bytes().splitlines() if line.strip()]


def test_delta_round_trip_with_images_and_compaction(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    expected = _write(store, "0-0", _session())

    assert store.read("0-0") == expected
    # Byte-identical once rebuilt: the key order of every exchange is kept.
    assert [orjson.dumps(exchange) for exchange in store.read("0-0")] == [orjson.dumps(item) for item in expected]
    lines = _stored_lines(store, "0-0")
    specs = [line[CAPTURE_FORMAT_KEY] for line in lines]
    assert all(spec["version"] == 2 for spec in specs)
    # The first call and the call after the compaction store their history in full; every other call extends the
    # previous one by the two items it added, and inherits the unchanged instructions and tools.
    assert [spec.get("base_model_call_id") for spec in specs] == [
        None,
        *[f"c{i}" for i in range(6)],
        None,
        *[f"c{i}" for i in range(7, 11)],
    ]
    assert all(len(line["request"]["input"]) == 2 for line, spec in zip(lines, specs) if "base_model_call_id" in spec)
    assert {tuple(spec.get("inherited_keys", ())) for spec in specs if "base_model_call_id" in spec} == {
        ("instructions", "tools")
    }
    # Four distinct screenshots, stored once each however often the history repeats them.
    assert len(list(store.blob_dir_for("0-0").iterdir())) == 4
    assert b"data:image" not in store.path_for("0-0").read_bytes()


def test_delta_is_a_fraction_of_the_full_format(tmp_path: Path) -> None:
    full, delta = CaptureStore(tmp_path / "full"), CaptureStore(tmp_path / "delta", capture_format="delta")
    exchanges = _session(calls=30, compact_at=None)
    _write(full, "0-0", exchanges)
    _write(delta, "0-0", exchanges)

    full_bytes = full.path_for("0-0").stat().st_size
    delta_bytes = delta.path_for("0-0").stat().st_size + sum(
        path.stat().st_size for path in delta.blob_dir_for("0-0").iterdir()
    )
    assert delta_bytes * 10 < full_bytes
    assert delta.read("0-0") == full.read("0-0")


def test_interleaved_conversations_each_extend_their_own_history(tmp_path: Path) -> None:
    """Calls of a subagent and a judge land between the policy's calls; each line extends its own conversation."""
    store = CaptureStore(tmp_path, capture_format="delta")
    policy: list = [{"role": "user", "content": "policy task"}]
    judge: list = [{"role": "system", "content": "judge rubric"}]
    exchanges = []
    for index in range(6):
        policy = policy + [_tool_output(index)]
        exchanges.append(_responses_exchange(f"p{index}", policy))
        if index % 2:
            judge = judge + [{"role": "user", "content": f"grade {index}"}]
            exchanges.append(_chat_exchange(f"j{index}", judge))
    expected = _write(store, "0-0", exchanges)

    assert store.read("0-0") == expected
    bases = {
        line["model_call_id"]: line[CAPTURE_FORMAT_KEY].get("base_model_call_id")
        for line in _stored_lines(store, "0-0")
    }
    assert bases == {
        "p0": None,
        "p1": "p0",
        "j1": None,
        "p2": "p1",
        "p3": "p2",
        "j3": "j1",
        "p4": "p3",
        "p5": "p4",
        "j5": "j3",
    }


def test_chat_dialect_round_trip(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    messages: list = [{"role": "system", "content": "s"}]
    exchanges = []
    for index in range(5):
        messages = messages + [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _image(index)}}]},
            {"role": "assistant", "content": f"a{index}"},
        ]
        exchanges.append(_chat_exchange(f"c{index}", messages))
    expected = _write(store, "0-0", exchanges)

    assert store.read("0-0") == expected
    assert _stored_lines(store, "0-0")[3][CAPTURE_FORMAT_KEY]["history_key"] == "messages"


@pytest.mark.parametrize(
    "request_fields",
    [
        {"request": {"model": "m", "input": "a plain string prompt"}},
        {"request": {"model": "m", "input": []}},
        {"request": None, "request_raw": "not json"},
        {"request": {"model": "m"}},
    ],
    ids=["string input", "empty history", "raw request", "no history"],
)
def test_requests_without_a_history_list_are_stored_as_they_are(tmp_path: Path, request_fields: dict) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    exchanges = [{"model_call_id": f"c{index}", "status_code": 200, **request_fields} for index in range(3)]
    expected = _write(store, "0-0", exchanges)

    assert store.read("0-0") == expected
    assert all("base_model_call_id" not in line[CAPTURE_FORMAT_KEY] for line in _stored_lines(store, "0-0"))


def test_short_and_lookalike_strings_stay_inline(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    small = "data:image/png;base64," + "A" * 16
    sse = "data: " + "y" * BLOB_MIN_CHARS
    exchange = _responses_exchange("c0", [{"role": "user", "content": [small, sse, '"capture_format": 2']}])
    expected = _write(store, "0-0", [exchange])

    assert store.read("0-0") == expected
    assert not store.blob_dir_for("0-0").exists()


def test_full_and_delta_lines_mix_in_one_file(tmp_path: Path) -> None:
    """A full-format writer (another model server, an older Gym) and a delta writer can share a rollout's file."""
    full, delta = CaptureStore(tmp_path), CaptureStore(tmp_path, capture_format="delta")
    exchanges = _session(calls=8, compact_at=None)
    expected = []
    for index, exchange in enumerate(exchanges):
        expected += _write(full if index in (0, 3) else delta, "0-0", [exchange])

    assert full.read("0-0") == expected
    lines = _stored_lines(full, "0-0")
    assert [CAPTURE_FORMAT_KEY in line for line in lines] == [False, True, True, False, True, True, True, True]


def test_two_writers_of_one_file_only_extend_lines_they_wrote(tmp_path: Path) -> None:
    """Writers in different processes share no state; each names only bases it wrote and checked under the lock."""
    first, second = (CaptureStore(tmp_path, capture_format="delta") for _ in range(2))
    exchanges = _session(calls=8, compact_at=None)
    expected = []
    for index, exchange in enumerate(exchanges):
        expected += _write(first if index % 2 == 0 else second, "0-0", [exchange])

    assert first.read("0-0") == expected
    bases = [line[CAPTURE_FORMAT_KEY].get("base_model_call_id") for line in _stored_lines(first, "0-0")]
    assert bases == [None, None, "c0", "c1", "c2", "c3", "c4", "c5"]


def test_a_recreated_capture_never_extends_a_line_of_the_deleted_one(tmp_path: Path) -> None:
    """A retry deletes the rollout's capture before dispatch; the writer must notice its remembered base is gone."""
    store = CaptureStore(tmp_path, capture_format="delta")
    exchanges = _session(calls=4, compact_at=None)
    _write(store, "0-0", exchanges[:3])
    CaptureStore(tmp_path).delete("0-0")
    assert not store.blob_dir_for("0-0").exists()

    expected = _write(store, "0-0", exchanges[3:])

    assert store.read("0-0") == expected
    assert "base_model_call_id" not in _stored_lines(store, "0-0")[0][CAPTURE_FORMAT_KEY]


def test_concurrent_threads_write_readable_lines(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    exchanges = _session(calls=16, compact_at=None)
    threads = [threading.Thread(target=store.record, args=("0-0", exchange)) for exchange in exchanges]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = {exchange["model_call_id"]: orjson.loads(orjson.dumps(exchange)) for exchange in exchanges}
    read = store.read("0-0")
    assert {exchange["model_call_id"]: exchange for exchange in read} == expected


def test_a_crash_between_blob_and_line_leaves_a_readable_capture(tmp_path: Path, monkeypatch) -> None:
    """Blobs are durable before their line; a write that dies in between leaves only an unreferenced blob."""
    store = CaptureStore(tmp_path, capture_format="delta")
    exchanges = _session(calls=4, compact_at=None)
    expected = _write(store, "0-0", exchanges[:2])

    real_fsync = __import__("os").fsync
    calls = {"count": 0}

    def fail_on_line_fsync(fd: int) -> None:
        # Blob file, blob directory, then the line: fail the line's fsync, after its blob is durable.
        calls["count"] += 1
        if Path(f"/proc/self/fd/{fd}").resolve() == store.path_for("0-0"):
            raise OSError("killed")
        real_fsync(fd)

    monkeypatch.setattr("nemo_gym.models.model_call_capture.os.fsync", fail_on_line_fsync)
    with pytest.raises(OSError, match="killed"):
        store.record("0-0", exchanges[2])
    monkeypatch.undo()
    # The killed write's line may or may not have reached the file; drop it, as a torn tail would be.
    path = store.path_for("0-0")
    path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:2]))

    expected += _write(store, "0-0", exchanges[3:])
    assert store.read("0-0") == expected
    assert len(list(store.blob_dir_for("0-0").iterdir())) == 2  # the dead write's blob is kept, harmlessly


@pytest.mark.parametrize("capture_format", ["full", "delta"])
def test_a_partial_line_left_by_a_killed_writer_costs_only_itself(tmp_path: Path, capture_format: str) -> None:
    """Another process sharing the file dies mid-append. The next line must not be glued to its partial line:
    glued, that line is unreadable, and in the delta format so is every later line that extends it."""
    store = CaptureStore(tmp_path, capture_format=capture_format)
    exchanges = _session(calls=6, compact_at=None)
    expected = _write(store, "0-0", exchanges[:2])
    with store.path_for("0-0").open("ab") as handle:
        handle.write(b'{"model_call_id": "killed", "request": {"input": [')

    expected += _write(store, "0-0", exchanges[2:])

    read = list(store.iter_available("0-0"))
    assert [index for index, exchange in read if exchange is None] == [2]
    assert [exchange for _, exchange in read if exchange is not None] == expected
    if capture_format == "delta":
        # The writer still extends its own lines across the damaged one.
        lines = store.path_for("0-0").read_bytes().splitlines()
        del lines[2]
        bases = [orjson.loads(line)[CAPTURE_FORMAT_KEY].get("base_model_call_id") for line in lines]
        assert bases == [None, "c0", "c1", "c2", "c3", "c4"]


def test_unresolvable_lines_are_reported_without_hiding_the_rest(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    expected = _write(store, "0-0", _session(calls=5, compact_at=None))
    path = store.path_for("0-0")
    lines = path.read_bytes().splitlines(keepends=True)
    # Damage the second line: lines extending it can no longer be rebuilt, but their metadata still reads.
    lines[1] = b'{"broken":\n'
    path.write_bytes(b"".join(lines))

    exchanges, invalid = store.read_available("0-0")
    assert [index for index, _ in exchanges] == [0] and invalid == 4
    metadata = list(store.iter_available("0-0", payloads=False))
    assert [index for index, exchange in metadata if exchange is not None] == [0, 2, 3, 4]
    assert metadata[2][1]["response"] == expected[2]["response"]


def test_a_missing_blob_makes_only_its_line_unreadable(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    _write(store, "0-0", _session(calls=4, compact_at=None))
    for blob in store.blob_dir_for("0-0").iterdir():
        blob.unlink()

    exchanges, invalid = store.read_available("0-0")
    assert exchanges == [] and invalid == 4
    # Metadata never needed the screenshots.
    assert all(exchange is not None for _, exchange in store.iter_available("0-0", payloads=False))


def test_a_blob_reference_cannot_leave_the_blob_directory(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path)
    (tmp_path / "secret").write_text("x")
    line = {
        "model_call_id": "c0",
        "request": {"input": [{"image_url": None}]},
        CAPTURE_FORMAT_KEY: {"version": 2, "blobs": [[["request", "input", 0, "image_url"], "../../secret"]]},
    }
    store.path_for("0-0").write_bytes(orjson.dumps(line) + b"\n")

    assert store.read_available("0-0") == ([], 1)


def test_a_metadata_view_drops_the_history_but_keeps_the_answer(tmp_path: Path) -> None:
    for capture_format in ("full", "delta"):
        store = CaptureStore(tmp_path / capture_format, capture_format=capture_format)
        expected = _write(store, "0-0", _session(calls=4, compact_at=None))

        for (index, exchange), original in zip(store.iter_available("0-0", payloads=False), expected, strict=True):
            assert exchange["response"] == original["response"]
            assert exchange["response_raw"] == original["response_raw"]
            assert exchange["request"] == {"model": "policy", "stream": True} or index == 0
            assert "input" not in exchange["request"]


def test_copy_full_writes_a_self_contained_full_format_file(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "captures", capture_format="delta")
    expected = _write(store, "0-0", _session(calls=5))
    destination = tmp_path / "copy.capture.jsonl"

    store.copy_full("0-0", destination)

    assert [orjson.loads(line) for line in destination.read_bytes().splitlines()] == expected


def test_reading_shares_history_items_so_memory_follows_unique_content(tmp_path: Path) -> None:
    """Reading N calls of a growing history costs about the history once, not N copies of it."""
    full, delta = CaptureStore(tmp_path / "full"), CaptureStore(tmp_path / "delta", capture_format="delta")
    history: list = [{"role": "user", "content": "task"}]
    for index in range(80):
        history = history + [{"type": "function_call_output", "call_id": f"c{index}", "output": "z" * 4000}]
        exchange = {"model_call_id": f"c{index}", "request": {"model": "m", "input": list(history)}}
        full.record("0-0", exchange)
        delta.record("0-0", exchange)

    def peak(read) -> int:
        gc.collect()
        tracemalloc.start()
        try:
            result = read()
            return tracemalloc.get_traced_memory()[1], result
        finally:
            tracemalloc.stop()

    unshared_peak, unshared = peak(
        lambda: [orjson.loads(line) for line in full.path_for("0-0").read_bytes().splitlines()]
    )
    delta_peak, from_delta = peak(lambda: delta.read("0-0"))
    full_peak, from_full = peak(lambda: full.read("0-0"))

    assert from_delta == unshared == from_full
    assert from_delta[10]["request"]["input"][3] is from_delta[50]["request"]["input"][3]
    assert from_full[10]["request"]["input"][3] is from_full[50]["request"]["input"][3]
    # The unshared parse holds ~80 x 40 x 4 kB = 13 MB; the shared reads hold the 320 kB history once.
    assert delta_peak * 8 < unshared_peak
    assert full_peak * 4 < unshared_peak


def test_delete_removes_the_capture_its_markers_and_its_blobs(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    _write(store, "0-0", _session(calls=3, compact_at=None))
    _write(store, "0-1", _session(calls=3, compact_at=None))
    store.mark_incomplete("0-0")
    store.mark_pending("0-0", "call")

    store.delete("0-0")

    assert not store.path_for("0-0").exists() and not store.is_incomplete("0-0") and not store.has_pending("0-0")
    assert not store.blob_dir_for("0-0").exists()
    assert store.blob_dir_for("0-1").is_dir() and len(store.read("0-1")) == 3


def test_the_capture_format_is_read_from_the_global_config(tmp_path: Path) -> None:
    config = ModelCallCaptureConfig.model_validate(
        {"observability_enabled": True, "model_call_capture_dir": str(tmp_path), "model_call_capture_format": "delta"}
    )
    store = make_capture_store(config)
    assert store is not None
    store.record("0-0", _responses_exchange("c0", [{"role": "user", "content": "hi"}]))
    assert CAPTURE_FORMAT_KEY in _stored_lines(store, "0-0")[0]

    assert ModelCallCaptureConfig.model_validate({}).model_call_capture_format == "full"
    with pytest.raises(ValueError):
        ModelCallCaptureConfig.model_validate({"model_call_capture_format": "zstd"})
