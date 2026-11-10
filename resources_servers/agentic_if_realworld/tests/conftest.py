# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY DIAGNOSTIC (Gym #5839 CI): report visible .py files that change during the driver tests."""

import os
import threading
import time
from pathlib import Path

import pytest


_COMPONENT_ROOT = Path(__file__).resolve().parents[3]
_BOUNDARIES = ["benchmarks", "resources_servers", "responses_api_models", "responses_api_agents", "nemo_gym"]
_START = time.monotonic()
_CHANGES: list[str] = []


def _snapshot() -> dict[str, tuple[int, int]]:
    """Map each visible .py file in the scanned trees to (mtime_ns, size), as runtime_identity() scans them."""
    files = {}
    for boundary in _BOUNDARIES:
        for directory, subdirectories, names in os.walk(_COMPONENT_ROOT / boundary):
            subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
            for name in names:
                if name.endswith(".py") and not name.startswith("."):
                    path = os.path.join(directory, name)
                    try:
                        status = os.stat(path)
                    except FileNotFoundError:
                        continue
                    files[os.path.relpath(path, _COMPONENT_ROOT)] = (status.st_mtime_ns, status.st_size)
    return files


_POLL_LOG: list[str] = []


def _poll_sources() -> None:
    """Log every visible .py change in the scanned trees, once a second, for the whole session."""
    previous = _snapshot()
    while True:
        time.sleep(1.0)
        current = _snapshot()
        added = sorted(set(current) - set(previous))
        removed = sorted(set(previous) - set(current))
        changed = sorted(path for path in set(previous) & set(current) if previous[path] != current[path])
        if (added or removed or changed) and len(_POLL_LOG) < 200:
            _POLL_LOG.append(
                f"[{time.monotonic() - _START:.1f}s] added {added[:10]} removed {removed[:10]} changed {changed[:10]}"
            )
        previous = current


threading.Thread(target=_poll_sources, name="diag5839-poll", daemon=True).start()


@pytest.fixture(autouse=True)
def _diag_watch_sources(request):
    if request.node.fspath.basename not in {"test_driver.py", "test_observed_driver.py"}:
        yield
        return
    started = time.monotonic() - _START
    before = _snapshot()
    yield
    after = _snapshot()
    ended = time.monotonic() - _START
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    if added or removed or changed:
        _CHANGES.append(
            f"{request.node.nodeid} [{started:.1f}s-{ended:.1f}s after session start]: "
            f"added {added[:20]} removed {removed[:20]} changed {changed[:20]}"
        )


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"DIAG5839 tests with source changes: {len(_CHANGES)}")
    for line in _CHANGES:
        terminalreporter.write_line(f"DIAG5839 {line}")
    terminalreporter.write_line(f"DIAG5839 background poll found {len(_POLL_LOG)} changes:")
    for line in _POLL_LOG:
        terminalreporter.write_line(f"DIAG5839 poll {line}")
