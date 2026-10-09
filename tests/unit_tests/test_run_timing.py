# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run timing summary: start coverage, wall-clock-limit hits and long rollouts."""

import asyncio
import signal

from nemo_gym.rollout_collection import _NO_SIGTERM_HOOK, RunTimingReport, _print_on_sigterm, _restore_sigterm


def _report(total, **kwargs):
    defaults = dict(
        start_within_s=1800,
        start_min_fraction=0.95,
        all_started_within_s=2700,
        long_rollout_s=9000,
        long_rollout_max_fraction=0.10,
        timed_out_max_fraction=0.05,
    )
    return RunTimingReport(total=total, **(defaults | kwargs))


def _row(table, label):
    return next(line for line in table.splitlines() if label in line)


def test_healthy_run_has_no_warnings():
    r = _report(2)
    for offset in (10, 20):
        r.record_start(r._t0 + offset)
        r.record_result({"reward": 1.0, "timed_out": 0}, 100)
    assert "WARNING" not in r.table()


def test_all_started_rows_and_late_start():
    r = _report(4)
    for offset in (60, 600, 1500, 3000):  # last start at 50 min
        r.record_start(r._t0 + offset)
    for result, seconds in (({"timed_out": 0}, 1200), ({"timed_out": 1}, 10800), ({}, 3000), ({}, 9500)):
        r.record_result(result, seconds)
    table = r.table()
    assert "50.0 min" in _row(table, "all rollouts started by") and "WARNING" in _row(table, "all rollouts started by")
    assert "75.0%" in _row(table, "started within 30 min") and "WARNING" in _row(table, "started within 30 min")
    assert "25.0% (1)" in _row(table, "ended by wall-clock limit")
    assert "50.0%" in _row(table, "longer than 150 min")


def test_partial_start_is_not_judged_while_the_window_is_open():
    r = _report(100)
    for offset in (10, 20, 30):
        r.record_start(r._t0 + offset)
    table = r.table()
    assert "3/100" in _row(table, "rollouts started so far")
    assert "(window open)" in table
    assert "WARNING" not in table


def test_partial_start_warns_after_the_deadlines():
    r = _report(100)
    r._t0 -= 3600
    for offset in (10, 20, 30):
        r.record_start(r._t0 + offset)
    table = r.table()
    assert "all by 45 min" in _row(table, "rollouts started so far")
    assert table.count("WARNING") == 2


def test_timeout_failure_class_counts_as_timed_out():
    r = _report(1)
    r.record_start(r._t0)
    r.record_result({"_ng_failure_class": "timeout_exceeded"}, 10)
    assert "100.0% (1)" in _row(r.table(), "ended by wall-clock limit")


async def test_sigterm_hook_installs_and_restores():
    before = signal.getsignal(signal.SIGTERM)
    previous = _print_on_sigterm(lambda: "table")
    assert previous is not _NO_SIGTERM_HOOK
    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGTERM) is True
    loop.add_signal_handler(signal.SIGTERM, lambda: None)
    _restore_sigterm(previous)
    assert signal.getsignal(signal.SIGTERM) == before
