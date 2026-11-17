# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[CUSTOM-VM-WATCHDOG-TOLERANCE] / [CUSTOM-VM-WATCHDOG-ONCE]: the VM sidecar must not be killed
over a transient SSH stall of a live qemu, must be restarted once it has been unreachable for
_UNREACHABLE_RESTART_S, and ensure_vm() must not pile up watchdog threads."""

import importlib.util
from pathlib import Path

import pytest


_VM_LAUNCHER = Path(__file__).resolve().parents[2] / "nemo_gym" / "vm_launcher.py"


@pytest.fixture
def vm(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("vm_launcher_under_test", _VM_LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "id").write_text("key")
    (tmp_path / "ssh_port").write_text("2222")
    # health: list of planned answers, else `default_health`; each check costs `check_cost` seconds
    state = {"health": [], "default_health": False, "qemu_alive": True, "spawns": 0, "spawn_error": None}
    state.update(now=1000.0, sleeps=[], check_cost=0.0)

    def check_health(info, timeout_s=mod._SSH_CONNECT_TIMEOUT_S):
        state["now"] += state["check_cost"]
        return state["health"].pop(0) if state["health"] else state["default_health"]

    def spawn(*_args):
        state["spawns"] += 1
        state["spawned_at"] = state["now"]
        if state["spawn_error"]:
            raise state["spawn_error"]

    def sleep(s):
        state["sleeps"].append(s)
        state["now"] += s

    monkeypatch.setattr(mod, "_work_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mod, "_check_health", check_health)
    monkeypatch.setattr(mod, "_qemu_alive", lambda: state["qemu_alive"])
    monkeypatch.setattr(mod, "_spawn", spawn)
    monkeypatch.setattr(mod.time, "sleep", sleep)
    monkeypatch.setattr(mod.time, "monotonic", lambda: state["now"])
    mod.state = state
    return mod


def _step(vm):
    return vm._watchdog_step("/golden", [], [])


def _run_watchdog(vm, seconds):
    """Drive the watchdog loop (sleep, step, ...) for `seconds` of fake time."""
    end = vm.state["now"] + seconds
    delay = vm._WATCHDOG_INTERVAL
    while vm.state["now"] < end:
        vm.time.sleep(delay)
        delay = _step(vm)


def test_healthy_tick_polls_at_interval(vm):
    vm.state["health"] = [True]
    assert _step(vm) == vm._WATCHDOG_INTERVAL
    assert vm.state["spawns"] == 0
    assert vm._last_healthy_at == vm.state["now"]


def test_stall_shorter_than_limit_never_restarts(vm):
    vm.state["health"] = [True]
    _step(vm)
    vm.state["health"] = [False] * 4 + [True] * 10  # ~2 min of failed checks, then back
    _run_watchdog(vm, 400)
    assert vm.state["spawns"] == 0
    assert vm._first_failure_at is None


@pytest.mark.parametrize("check_cost", [0.0, 20.0])  # instant refusals vs checks hitting their full timeout
def test_live_qemu_restarted_once_unreachable_for_limit(vm, check_cost):
    vm.state["health"] = [True]
    _step(vm)
    last_ok = vm.state["now"]
    vm.state["check_cost"] = check_cost
    delay = vm._WATCHDOG_INTERVAL
    while vm.state["spawns"] == 0:
        vm.time.sleep(delay)
        delay = _step(vm)
    down = vm.state["spawned_at"] - last_ok
    assert vm._UNREACHABLE_RESTART_S <= down <= vm._UNREACHABLE_RESTART_S + 2 * check_cost + 1


@pytest.mark.parametrize("why", ["qemu_dead", "never_booted"])
def test_restart_immediately_when_vm_is_really_gone(vm, tmp_path, why):
    vm.state["health"] = [True]
    _step(vm)
    if why == "qemu_dead":
        vm.state["qemu_alive"] = False
    else:
        (tmp_path / "ssh_port").unlink()
    assert _step(vm) == vm._WATCHDOG_INTERVAL
    assert vm.state["spawns"] == 1


def test_successful_restart_resets_the_clock(vm):
    vm.state["qemu_alive"] = False
    _step(vm)
    assert vm.state["spawns"] == 1
    assert vm._unreachable_for() == 0.0 and vm._first_failure_at is None


def test_failed_restart_retries_next_tick(vm):
    vm.state["qemu_alive"] = False
    vm.state["spawn_error"] = RuntimeError("VM boot failed")
    _step(vm)
    _step(vm)
    assert vm.state["spawns"] == 2


def test_lock_recheck_skips_spawn_if_someone_else_recovered(vm):
    vm.state["qemu_alive"] = False
    vm.state["health"] = [False, True]  # tick check fails, re-check under the spawn lock passes
    _step(vm)
    assert vm.state["spawns"] == 0


def test_patient_check_waits_out_a_stall(vm):
    vm.state["health"] = [True, False, False, True]
    assert vm._probe(vm._current_connection_info())
    assert vm._healthy_or_patient(vm._current_connection_info()) is True
    assert vm.state["sleeps"] == [vm._PATIENT_RETRY_GAP_S] * 2


def test_patient_check_gives_up_at_the_limit(vm):
    vm.state["health"] = [True]
    vm._probe(vm._current_connection_info())
    t0 = vm.state["now"]
    assert vm._healthy_or_patient(vm._current_connection_info()) is False
    assert vm._UNREACHABLE_RESTART_S <= vm.state["now"] - t0 < vm._UNREACHABLE_RESTART_S + vm._PATIENT_RETRY_GAP_S


def test_patient_check_fails_fast_when_qemu_dead(vm):
    vm.state["qemu_alive"] = False
    assert vm._healthy_or_patient(vm._current_connection_info()) is False
    assert vm.state["sleeps"] == []


def test_patient_check_none_info(vm):
    assert vm._healthy_or_patient(None) is False


def test_ensure_vm_starts_a_single_watchdog(vm, monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs["name"])

    monkeypatch.setattr(vm.threading, "Thread", FakeThread)
    for _ in range(5):
        vm.state["health"] = [True]
        vm.ensure_vm(golden_image_dir="/golden")
    assert started == ["vm-watchdog"]
    assert vm.state["spawns"] == 0
