# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `check_correctness` worker + Manager lifecycle."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from lcb_integration import compute_code_generation_metrics
from lcb_integration.compute_code_generation_metrics import check_correctness


_SAMPLE: dict[str, str] = {"input_output": json.dumps({"inputs": ["1", "2"], "outputs": ["1", "4"]})}


@pytest.fixture
def patched_mp(monkeypatch):
    """Replace multiprocessing.Manager + Process so we can assert lifecycle ordering."""
    manager: MagicMock = MagicMock(name="Manager")
    manager.list.side_effect = lambda: []
    manager_factory: MagicMock = MagicMock(return_value=manager)
    process_instance: MagicMock = MagicMock(name="Process")
    process_factory: MagicMock = MagicMock(return_value=process_instance)

    monkeypatch.setattr(compute_code_generation_metrics.multiprocessing, "Manager", manager_factory)
    monkeypatch.setattr(compute_code_generation_metrics.multiprocessing, "Process", process_factory)

    return manager, manager_factory, process_instance, process_factory


class TestCheckCorrectnessReap:
    """check_correctness must reap its worker and shut down its Manager on every exit path."""

    def test_kill_is_followed_by_reap_join(self, patched_mp):
        manager, _manager_factory, process, _process_factory = patched_mp
        process.is_alive.side_effect = [True, False]

        result, metadata = check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        process.start.assert_called_once()
        assert process.join.call_count == 2
        process.kill.assert_called_once()
        first_join_timeout = process.join.call_args_list[0].kwargs.get("timeout")
        second_join_timeout = process.join.call_args_list[1].kwargs.get("timeout")
        assert first_join_timeout is not None
        assert second_join_timeout is not None
        assert result == [-1, -1]
        assert metadata is None
        manager.shutdown.assert_called_once()

    def test_manager_shutdown_runs_when_process_start_raises(self, patched_mp):
        manager, _manager_factory, process, _process_factory = patched_mp
        process.start.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        manager.shutdown.assert_called_once()

    def test_happy_path_drains_results_before_manager_shutdown(self, patched_mp):
        manager, _manager_factory, process, _process_factory = patched_mp
        process.is_alive.return_value = False

        captured_lists: list[list] = []

        def _make_list() -> list:
            new_list: list = []
            captured_lists.append(new_list)
            return new_list

        manager.list.side_effect = _make_list

        def _start_side_effect() -> None:
            assert len(captured_lists) >= 2
            captured_lists[0].append([1, 1])
            captured_lists[1].append({"ok": True})

        process.start.side_effect = _start_side_effect

        result, metadata = check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        assert result == [1, 1]
        assert metadata == {"ok": True}
        manager.shutdown.assert_called_once()

    def test_invalid_input_output_short_circuits(self, patched_mp):
        _manager, manager_factory, process, _process_factory = patched_mp

        result, metadata = check_correctness({"input_output": "not-json"}, generation="g", timeout=1, debug=False)

        assert result == [-1]
        assert metadata is None
        manager_factory.assert_not_called()
        process.start.assert_not_called()
