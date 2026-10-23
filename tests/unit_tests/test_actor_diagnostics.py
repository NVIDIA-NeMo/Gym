# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only diagnostics tests; no Ray actors, serving processes, or network."""

import ast
import asyncio
import gc
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo_gym.telemetry.actor_diagnostics import ActorDiagnostics, startup_phase


def _events(caplog):
    return [
        json.loads(record.message.split("gym_actor_probe ", 1)[1])
        for record in caplog.records
        if "gym_actor_probe " in record.message
    ]


def test_disabled_probe_is_noop(caplog):
    probe = ActorDiagnostics(enabled=False)
    original_callbacks = list(gc.callbacks)
    with probe.startup(), startup_phase("not_recorded"), probe.rollout():
        probe.event("not_logged")
    probe.close()
    assert not probe._timings
    assert probe._heartbeat is None
    assert gc.callbacks == original_callbacks
    assert not _events(caplog)


def test_nested_timings_are_scoped_and_exception_safe(caplog):
    caplog.set_level(logging.WARNING)
    probe = ActorDiagnostics(enabled=True)
    with pytest.raises(ValueError, match="sentinel"):
        with probe.startup(), startup_phase("parse"):
            raise ValueError("sentinel")
    with startup_phase("outside_scope"):
        pass
    probe.event("test")
    event = _events(caplog)[0]
    assert set(event["phases"]) == {"parse", "rollout_start_total"}
    assert event["phases"]["parse"]["calls"] == 1
    assert event["phases"]["parse"]["errors"] == 1
    assert event["phases"]["parse"]["thread_cpu_s"] >= 0
    assert event["active_sync_phases"] == []


def test_probe_gc_lifecycle_and_rollout_cancel_cleanup(caplog):
    async def exercise():
        probe = ActorDiagnostics(enabled=True)
        baseline = len(gc.callbacks)
        entered = asyncio.Event()

        async def run():
            with probe.rollout():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(run())
        await entered.wait()
        probe.start()
        assert len(gc.callbacks) == baseline + 1
        assert probe._rollouts == 1
        probe._gc("start", {"generation": 2})
        probe._gc("stop", {"generation": 2})
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert probe._rollouts == 0
        probe.close()
        await asyncio.sleep(0)
        assert len(gc.callbacks) == baseline
        assert probe._heartbeat.done()
        assert probe._gc_count[2] >= 1

    asyncio.run(exercise())


def test_watchdog_reports_actual_blocking_phase_without_payload(caplog):
    caplog.set_level(logging.WARNING)

    async def exercise():
        probe = ActorDiagnostics(enabled=True, report_s=0.1, stall_s=0.05)
        probe.start()
        request_secret = "DO_NOT_LOG_THIS_REQUEST_VALUE"
        try:
            with probe.startup(), startup_phase("fake_config_http_get"):
                # Deliberately block the loop while releasing the GIL, so the
                # independent watchdog can sample the blocked thread.
                time.sleep(1.2)
                assert request_secret
        finally:
            probe.close()
            await asyncio.sleep(0)

    asyncio.run(exercise())
    stalled = [event for event in _events(caplog) if event["event"] == "loop_stall"]
    assert stalled
    assert any(phase["phase"] == "fake_config_http_get" for phase in stalled[0]["active_sync_phases"])
    assert any("exercise" in frame for frame in stalled[0]["stack"])
    assert "DO_NOT_LOG_THIS_REQUEST_VALUE" not in caplog.text
    assert stalled[0]["heartbeat_age_s"] > 0.05


@pytest.mark.parametrize("injected", [False, True])
def test_actual_client_load_records_expected_phases_without_changing_result(injected):
    # Load the production method, not the serving/Ray import graph. Only its
    # network boundary is faked; JSON and OmegaConf parsing are real.
    root = Path(__file__).resolve().parents[2]
    source = ast.parse((root / "nemo_gym/server_utils.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ServerClient")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_from_global_config"
    )
    method.decorator_list = []
    values = {"head_server": {"host": "localhost", "port": 1234}, "policy_model": {"name": "policy"}}
    gets = []

    def get(url):
        gets.append(url)
        return SimpleNamespace(content=json.dumps(values).encode())

    namespace = {
        "__name__": __name__,
        "startup_phase": startup_phase,
        "json": json,
        "OmegaConf": OmegaConf,
        "requests": SimpleNamespace(get=get),
        "_has_injected_global_config_env": lambda: injected,
        "get_global_config_dict": lambda: OmegaConf.create(values),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "production_client_load", "exec"), namespace)
    probe = ActorDiagnostics(enabled=True)
    with probe.startup():
        client = namespace["load_from_global_config"](SimpleNamespace, SimpleNamespace(host="localhost", port=1234))
    assert OmegaConf.to_container(client.global_config_dict) == values
    if injected:
        assert not gets
        assert "config_injected_client_construct" in probe._timings
    else:
        assert gets == ["http://localhost:1234/global_config_dict_yaml"]
        assert {
            "config_http_get",
            "config_json_decode",
            "config_omegaconf_create",
            "config_client_construct",
        } <= probe._timings.keys()


@pytest.mark.parametrize("fails", [False, True])
def test_actual_prepare_method_records_submit_entry_and_return(caplog, fails):
    caplog.set_level(logging.WARNING)
    root = Path(__file__).resolve().parents[2]
    rl_root = root.parents[2]
    source = ast.parse((rl_root / "nemo_rl/environments/nemo_gym.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "NemoGym")
    method = next(
        node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "checkpoint_prepare"
    )
    method.decorator_list = []
    namespace = {
        "monotonic": time.monotonic,
        "time": time,
        "GymCheckpointPrepareSummary": SimpleNamespace(from_result=lambda result: result),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "production_prepare", "exec"), namespace)

    async def exercise():
        async def prepare(checkpoint_id, *, deadline_ts):
            assert checkpoint_id == "test-checkpoint"
            if fails:
                raise ValueError("prepare failed")
            return "prepared"

        actor = SimpleNamespace(
            _actor_diagnostics=ActorDiagnostics(enabled=True),
            _require_checkpoint_adapter=lambda: SimpleNamespace(prepare=prepare),
        )
        call = namespace["checkpoint_prepare"](
            actor, "test-checkpoint", deadline_ts=time.time() + 5, submitted_unix_s=time.time() - 3
        )
        if fails:
            with pytest.raises(ValueError, match="prepare failed"):
                await call
        else:
            assert await call == "prepared"

    asyncio.run(exercise())
    events = _events(caplog)
    assert [event["event"] for event in events] == ["prepare_enter", "prepare_return"]
    assert events[0]["submit_to_entry_s"] >= 3
    assert events[1]["outcome"] == ("failed" if fails else "completed")


def test_actual_rollout_wrapper_preserves_stream_and_cancellation_cleanup():
    root = Path(__file__).resolve().parents[2]
    source = ast.parse((root.parents[2] / "nemo_rl/environments/nemo_gym.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "NemoGym")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_rollouts")
    method.decorator_list = []
    namespace = {
        "nullcontext": nullcontext,
        "streaming_umbrella_span": lambda *args, **kwargs: nullcontext(nullcontext),
        "RLSpanGroup": SimpleNamespace(U_PER_PROMPT="per_prompt", U_ROLLOUT="rollout"),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "production_rollout", "exec"), namespace)

    async def exercise():
        closed = asyncio.Event()

        async def stream(*args):
            try:
                yield (0, {}, {}, None)
                await asyncio.Event().wait()
            finally:
                closed.set()

        probe = ActorDiagnostics(enabled=True)
        actor = SimpleNamespace(_actor_diagnostics=probe, _stream_rollouts=stream)
        iterator = namespace["run_rollouts"](actor, [{}], "timing")
        try:
            assert await iterator.__anext__() == (0, {}, {}, None)
            assert probe._rollouts == 1
            await iterator.aclose()
            assert closed.is_set()
            assert probe._rollouts == 0
        finally:
            probe.close()
            await asyncio.sleep(0)

    asyncio.run(exercise())
