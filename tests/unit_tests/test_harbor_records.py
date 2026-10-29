# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the record files.

The sidecar lock, the atomic replace and the read that retries once against a torn write.
"""

import pytest

from nemo_gym.data import HarborTaskError
from nemo_gym.data.tasks.harbor.records import read_json_record, record_lock, write_json_record


class TestRecordFiles:
    """D2: the primitives built-images.json and compose-images.json are read and written through."""

    def test_write_replaces_the_file_atomically_sorted_and_indented(self, tmp_path):
        path = tmp_path / "built-images.json"
        write_json_record(path, {"b": 1, "a": {"y": 2, "x": 1}})
        assert path.read_text() == '{\n  "a": {\n    "x": 1,\n    "y": 2\n  },\n  "b": 1\n}\n'
        assert read_json_record(path) == {"a": {"x": 1, "y": 2}, "b": 1}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["built-images.json"]

    def test_a_failed_replace_keeps_the_previous_file_and_no_temporary(self, tmp_path, monkeypatch):
        path = tmp_path / "built-images.json"
        write_json_record(path, {"kept": True})

        def failing_replace(*_):
            raise OSError("disk full")

        monkeypatch.setattr("nemo_gym.data.tasks.harbor.records.os.replace", failing_replace)
        with pytest.raises(OSError, match="disk full"):
            write_json_record(path, {"lost": True})
        assert read_json_record(path) == {"kept": True}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["built-images.json"]

    def test_read_returns_empty_for_a_missing_file_or_a_non_object(self, tmp_path):
        path = tmp_path / "built-images.json"
        assert read_json_record(path) == {}
        path.write_text("[1, 2]\n")
        assert read_json_record(path) == {}

    def test_read_retries_once_and_then_names_the_file(self, tmp_path, monkeypatch):
        import nemo_gym.data.tasks.harbor.records as records_module

        path = tmp_path / "compose-images.json"
        path.write_text('{"torn": ')
        pauses: list[float] = []
        monkeypatch.setattr(records_module.time, "sleep", pauses.append)
        with pytest.raises(HarborTaskError) as excinfo:
            read_json_record(path)
        assert str(path) in str(excinfo.value) and "read twice" in str(excinfo.value)
        assert pauses == [records_module.RECORD_READ_RETRY_S]
        # A file that is whole again on the second read is returned.
        pauses.clear()
        monkeypatch.setattr(records_module.time, "sleep", lambda pause: path.write_text('{"whole": 1}\n'))
        assert read_json_record(path) == {"whole": 1}

    def test_the_lock_serialises_a_second_holder(self, tmp_path):
        import threading

        path = tmp_path / "built-images.json"
        entered = threading.Event()
        released = threading.Event()

        def second():
            with record_lock(path):
                entered.set()
                released.wait(timeout=5)

        with record_lock(path):
            assert (tmp_path / "built-images.json.lock").is_file()
            worker = threading.Thread(target=second)
            worker.start()
            # The second holder waits while the first holds the lock (flock is per open file description).
            assert not entered.wait(timeout=0.3)
        assert entered.wait(timeout=5)
        released.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
