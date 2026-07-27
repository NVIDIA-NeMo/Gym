import fcntl
import multiprocessing as mp
import threading
from pathlib import Path

import orjson
import pytest

from nemo_gym.base_responses_api_model import (
    maybe_rollout_id_from_run_body,
)
from nemo_gym.capture_records import CaptureStore
from nemo_gym.openai_utils import (
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from tests.unit_tests.test_base_responses_api_model import TEST_ROLLOUT_ID, _create_test_model_call_record


class TestCaptureStore:
    def test_sanity(self, tmp_path: Path):
        store = CaptureStore(tmp_path)
        assert store.path_for(TEST_ROLLOUT_ID).name == "my-test-rollout-id.capture.jsonl"
        store.path_for(TEST_ROLLOUT_ID).write_bytes(b"\n")

        record = _create_test_model_call_record()
        record.request.input[0].content[0]["text"] = "Unicode payload: café 東京"

        store.record(record)

        assert store.read(TEST_ROLLOUT_ID) == [record]

        record2 = record.model_copy(deep=True)
        record2.request.input[0].content[0]["text"] = "second"

        store.record(record2)
        assert store.read(TEST_ROLLOUT_ID) == [record, record2]

    def test_read_raises_on_malformed_nonblank_json(self, tmp_path: Path):
        store = CaptureStore(tmp_path)
        store.path_for("rollout-1").write_bytes(b'{"request": {')

        with pytest.raises(orjson.JSONDecodeError):
            store.read("rollout-1")

    def test_aggregate_sanity(self, tmp_path: Path):
        store = CaptureStore(tmp_path)

        record1 = _create_test_model_call_record()
        record1.rollout_id = "0-0"
        record1.response.usage = NeMoGymResponseUsage(
            input_tokens=1,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=2),
            output_tokens=3,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=4),
            total_tokens=4,
        )

        store.record(record1)

        aggregate_record = store.aggregate(
            rollout_id=maybe_rollout_id_from_run_body({"_ng_task_index": 0, "_ng_rollout_index": 0})
        )

        assert len(aggregate_record.records) == 1
        assert aggregate_record.records[0].rollout_id == "0-0"
        assert aggregate_record.records[0].response.usage.output_tokens == 3

    def test_clear(self, tmp_path: Path):
        store = CaptureStore(tmp_path)

        record1 = _create_test_model_call_record()
        record1.rollout_id = "0-0"
        store.record(record1)
        record2 = _create_test_model_call_record()
        record2.rollout_id = "1-0"
        store.record(record2)

        assert store.read("0-0") and store.read("1-0")

        store.clear()
        assert store.read("0-0") == [] and store.read("1-0") == []

    def test_read_waits_for_in_progress_append(self, tmp_path: Path):
        store = CaptureStore(tmp_path)
        path = store.path_for("0-0")
        writer_ready = threading.Event()
        finish_write = threading.Event()
        reader_done = threading.Event()
        rows = []

        record = _create_test_model_call_record()
        record.rollout_id = "0-0"

        def _write() -> None:
            with path.open("ab") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

                record_str = record.model_dump_json().encode() + b"\n"
                try:
                    handle.write(record_str[: len(record_str) // 2])
                    handle.flush()
                    writer_ready.set()
                    assert finish_write.wait(timeout=5)
                    handle.write(record_str[len(record_str) // 2 :])
                    handle.flush()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        def _read() -> None:
            rows.extend(store.read("0-0"))
            reader_done.set()

        writer = threading.Thread(target=_write)
        reader = threading.Thread(target=_read)
        writer.start()
        assert writer_ready.wait(timeout=5)
        reader.start()
        try:
            assert not reader_done.wait(timeout=0.1)
        finally:
            finish_write.set()
        writer.join(timeout=5)
        reader.join(timeout=5)

        assert not writer.is_alive()
        assert not reader.is_alive()
        assert rows == [record]

    def test_parallel_thread_append_no_loss(self, tmp_path: Path):
        store = CaptureStore(tmp_path)

        def _write(i: int) -> None:
            store.record(_create_test_model_call_record())

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rows = store.read(TEST_ROLLOUT_ID)
        assert len(rows) == 20

    def test_cross_process_append_no_loss(self, tmp_path: Path):
        # The threads-only test above exercises the in-process lock; this exercises fcntl.flock across
        # *processes* -- the num_workers>1 case the in-process lock cannot coordinate.

        def _cross_process_writer(root: str, base: int) -> None:
            # Module-level so it is picklable under the "spawn" start method too.
            store = CaptureStore(root)
            record = _create_test_model_call_record()
            record.rollout_id = "0-0"
            for _ in range(base, base + 100):
                store.record(record)

        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=_cross_process_writer, args=(str(tmp_path), b * 100)) for b in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            assert p.exitcode == 0
        rows = CaptureStore(tmp_path).read("0-0")
        assert len(rows) == 400
