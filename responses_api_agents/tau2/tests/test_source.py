# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from responses_api_agents.tau2 import source


def _write_tau2_source_data(data_dir: Path) -> None:
    banking_dir = data_dir / "tau2" / "domains" / "banking_knowledge"
    banking_dir.mkdir(parents=True)
    (banking_dir / "db.json").write_text("{}")
    for directory in ("documents", "prompts", "tasks"):
        (banking_dir / directory).mkdir()


def test_concurrent_initialization_fetches_tau2_data_once(tmp_path: Path) -> None:
    data_dir = tmp_path / "tau2_data"
    partial_banking_dir = data_dir / "tau2" / "domains" / "banking_knowledge"
    (partial_banking_dir / "documents").mkdir(parents=True)
    (partial_banking_dir / "tasks").mkdir()

    start_barrier = threading.Barrier(2)
    clone_count = 0
    clone_count_lock = threading.Lock()

    def fake_run(command: list[str], *, check: bool) -> None:
        nonlocal clone_count
        assert check
        if command[1] != "clone":
            return

        with clone_count_lock:
            clone_count += 1
        time.sleep(0.1)
        _write_tau2_source_data(Path(command[-1]) / "data")

    def initialize() -> Path:
        start_barrier.wait()
        return source.ensure_tau2_data_dir(data_dir)

    with (
        patch.object(source, "run", side_effect=fake_run),
        patch.object(source, "TAU2_DATA_LOCK_POLL_INTERVAL_SECONDS", 0.01),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        futures = [executor.submit(initialize) for _ in range(2)]
        results = [future.result() for future in futures]

    assert results == [data_dir, data_dir]
    assert clone_count == 1
    assert source._has_banking_knowledge_data(data_dir)
    assert not (tmp_path / ".tau2_data.lockdir").exists()
