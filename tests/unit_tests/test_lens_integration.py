# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for NeMo Lens integration with NeMo Gym (sandbox observability)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Import guard pattern — nemo.lens is optional in Gym
# ---------------------------------------------------------------------------


def test_lens_import_guard_no_error_when_missing():
    """Code that guards on ImportError must not raise when nemo-lens is absent."""
    with patch.dict(sys.modules, {"nemo.lens": None, "nemo.lens.sandbox": None}):
        try:
            from nemo.lens.sandbox.observability import SandboxEventRecorder  # type: ignore[import]
        except ImportError:
            SandboxEventRecorder = None
    # The important thing is no *unexpected* exception propagated.


def test_lens_sandbox_observability_importable():
    """nemo.lens.sandbox.observability must be importable when nemo-lens is installed."""
    pytest.importorskip("nemo.lens")
    from nemo.lens.sandbox.observability import (  # noqa: F401
        SandboxEventRecorder,
        build_recorder_from_env,
        observability_span,
        observability_sync_span,
        suppress_observability_events,
    )


def test_build_recorder_from_env_returns_none_without_dir(monkeypatch):
    """build_recorder_from_env() returns None when NEMO_RL_SANDBOX_OBSERVABILITY_DIR is unset."""
    pytest.importorskip("nemo.lens")
    monkeypatch.delenv("NEMO_RL_SANDBOX_OBSERVABILITY_DIR", raising=False)
    from nemo.lens.sandbox.observability import build_recorder_from_env

    assert build_recorder_from_env() is None


def test_build_recorder_from_env_creates_recorder(tmp_path, monkeypatch):
    """build_recorder_from_env() creates a working recorder when env var is set."""
    pytest.importorskip("nemo.lens")
    obs_dir = tmp_path / "gym_obs"
    monkeypatch.setenv("NEMO_RL_SANDBOX_OBSERVABILITY_DIR", str(obs_dir))
    from nemo.lens.sandbox.observability import build_recorder_from_env

    rec = build_recorder_from_env()
    assert rec is not None
    assert rec.output_dir == obs_dir
    rec.finalize()
    assert (obs_dir / "events.jsonl").exists()


# ---------------------------------------------------------------------------
# Sandbox recorder records events correctly
# ---------------------------------------------------------------------------


def test_recorder_records_span_events(tmp_path):
    """SandboxEventRecorder produces correctly structured span events."""
    pytest.importorskip("nemo.lens")
    from nemo.lens.sandbox.observability import SandboxEventRecorder

    rec = SandboxEventRecorder(
        output_dir=tmp_path / "obs",
        resource_sample_interval_s=0,
        max_rendered_trajectories=5,
        artifacts={"enabled": False, "render_html": False, "render_png": False, "export_otlp_json": False},
        otel={"service_name": "gym-test", "export_logs": False},
        wandb={"enabled": False},
        process_trace={"enabled": False},
        privacy={"include_command_text": False},
        run_id="gym-test-run",
    )
    with rec.sync_span("sandbox.exec", phase="execution", attributes={"trajectory_id": "t-gym-1"}):
        pass
    rec.finalize()

    events_path = tmp_path / "obs" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    exec_events = [e for e in events if e.get("name") == "sandbox.exec"]
    assert exec_events
    assert exec_events[0]["attributes"]["phase"] == "execution"
    assert exec_events[0]["attributes"]["trajectory_id"] == "t-gym-1"


def test_recorder_exports_otlp_json(tmp_path):
    """finalize() with export_otlp_json=True writes a valid OTLP JSON file."""
    pytest.importorskip("nemo.lens")
    from nemo.lens.sandbox.observability import SandboxEventRecorder

    rec = SandboxEventRecorder(
        output_dir=tmp_path / "obs",
        resource_sample_interval_s=0,
        max_rendered_trajectories=5,
        artifacts={"enabled": False, "render_html": False, "render_png": False, "export_otlp_json": True},
        otel={"service_name": "gym-test", "export_logs": False},
        wandb={"enabled": False},
        process_trace={"enabled": False},
        privacy={"include_command_text": False},
        run_id="gym-test-run",
    )
    with rec.sync_span("sandbox.exec", phase="execution", attributes={"trajectory_id": "t-gym-2"}):
        pass
    rec.finalize()

    otlp_path = tmp_path / "obs" / "traces" / "otel_traces.json"
    assert otlp_path.exists()
    otlp = json.loads(otlp_path.read_text())
    assert "resourceSpans" in otlp


# ---------------------------------------------------------------------------
# Observability span context managers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observability_span_is_noop_without_recorder(tmp_path):
    """observability_span() is a no-op when no recorder is active."""
    pytest.importorskip("nemo.lens")
    from nemo.lens.sandbox.observability import observability_span

    async with observability_span("test.op"):
        pass  # should not raise


def test_observability_sync_span_is_noop_without_recorder(tmp_path):
    """observability_sync_span() is a no-op when no recorder is active."""
    pytest.importorskip("nemo.lens")
    from nemo.lens.sandbox.observability import observability_sync_span

    with observability_sync_span("test.op"):
        pass  # should not raise
