# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for apply_otel_config_to_env() and _try_init_nemo_lens() helpers."""

import os

import pytest

from nemo_gym.observability.recorder import apply_otel_config_to_env


_OTEL_KEYS = [
    "NEMO_LENS_ENABLED",
    "OTEL_SERVICE_NAME",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
]


@pytest.fixture(autouse=True)
def clean_otel_env(monkeypatch):
    """Remove OTel env vars before each test so tests are isolated."""
    for key in _OTEL_KEYS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# apply_otel_config_to_env
# ---------------------------------------------------------------------------


def test_disabled_config_sets_nothing():
    apply_otel_config_to_env({"enabled": False, "endpoint": "https://api.honeycomb.io"})
    assert "NEMO_LENS_ENABLED" not in os.environ


def test_none_config_sets_nothing():
    apply_otel_config_to_env(None)
    assert "NEMO_LENS_ENABLED" not in os.environ


def test_empty_config_sets_nothing():
    apply_otel_config_to_env({})
    assert "NEMO_LENS_ENABLED" not in os.environ


def test_enabled_sets_nemo_lens_flag():
    apply_otel_config_to_env({"enabled": True})
    assert os.environ["NEMO_LENS_ENABLED"] == "1"


def test_service_name_is_set():
    apply_otel_config_to_env({"enabled": True, "service_name": "my-eval"})
    assert os.environ["OTEL_SERVICE_NAME"] == "my-eval"


def test_endpoint_is_set():
    apply_otel_config_to_env({"enabled": True, "endpoint": "https://api.honeycomb.io"})
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://api.honeycomb.io"


def test_honeycomb_api_key_becomes_header():
    apply_otel_config_to_env({"enabled": True, "api_key": "mykey"})
    assert "x-honeycomb-team=mykey" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]


def test_honeycomb_dataset_becomes_header():
    apply_otel_config_to_env({"enabled": True, "dataset": "nemo-gym"})
    assert "x-honeycomb-dataset=nemo-gym" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]


def test_api_key_and_dataset_combined_in_headers():
    apply_otel_config_to_env({"enabled": True, "api_key": "k1", "dataset": "ds1"})
    headers = os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "x-honeycomb-team=k1" in headers
    assert "x-honeycomb-dataset=ds1" in headers


def test_extra_headers_are_included():
    apply_otel_config_to_env({"enabled": True, "headers": {"x-custom": "val"}})
    assert "x-custom=val" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]


def test_no_headers_key_not_set_when_empty():
    apply_otel_config_to_env({"enabled": True, "headers": {}})
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ


def test_null_api_key_not_included_in_headers():
    apply_otel_config_to_env({"enabled": True, "api_key": None, "dataset": "ds"})
    headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
    assert "x-honeycomb-team" not in headers
    assert "x-honeycomb-dataset=ds" in headers


def test_full_honeycomb_config():
    apply_otel_config_to_env(
        {
            "enabled": True,
            "service_name": "nemo-gym-swebench",
            "endpoint": "https://api.honeycomb.io",
            "api_key": "testkey",
            "dataset": "nemo-gym",
        }
    )
    assert os.environ["NEMO_LENS_ENABLED"] == "1"
    assert os.environ["OTEL_SERVICE_NAME"] == "nemo-gym-swebench"
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://api.honeycomb.io"
    headers = os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "x-honeycomb-team=testkey" in headers
    assert "x-honeycomb-dataset=nemo-gym" in headers


# ---------------------------------------------------------------------------
# Shell exports take precedence (_setenv does not overwrite)
# ---------------------------------------------------------------------------


def test_existing_env_var_not_overwritten(monkeypatch):
    monkeypatch.setenv("NEMO_LENS_ENABLED", "0")
    apply_otel_config_to_env({"enabled": True})
    assert os.environ["NEMO_LENS_ENABLED"] == "0"


def test_existing_service_name_not_overwritten(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-shell-value")
    apply_otel_config_to_env({"enabled": True, "service_name": "yaml-value"})
    assert os.environ["OTEL_SERVICE_NAME"] == "my-shell-value"


def test_existing_endpoint_not_overwritten(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://shell.endpoint")
    apply_otel_config_to_env({"enabled": True, "endpoint": "https://yaml.endpoint"})
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://shell.endpoint"


def test_existing_headers_not_overwritten(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-shell=1")
    apply_otel_config_to_env({"enabled": True, "api_key": "yamlkey", "dataset": "yamldb"})
    assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == "x-shell=1"


# ---------------------------------------------------------------------------
# command classification (events.py)
# ---------------------------------------------------------------------------


from nemo_gym.observability.events import classify_command, command_attributes, safe_attributes, stable_hash


@pytest.mark.parametrize(
    "command,expected",
    [
        ("sleep 5", "sleep_poll"),
        ("while true; do sleep 1; done", "sleep_poll"),
        ("make all", "build"),
        ("npm install", "build"),
        ("pip install requests", "build"),
        ("cat README.md", "read_write_edit"),
        ("git diff HEAD", "read_write_edit"),
        ("sed -i 's/foo/bar/' file.py", "read_write_edit"),
        ("python train.py", "foreground_compute"),
        ("python3 -c 'print(1)'", "foreground_compute"),
        ("node server.js", "foreground_compute"),
        ("echo hello", "other_bash"),
        ("", "other_bash"),
        (None, "other_bash"),
    ],
)
def test_classify_command(command, expected):
    assert classify_command(command) == expected


def test_command_attributes_with_text():
    attrs = command_attributes("python train.py", include_command_text=True)
    assert attrs["command_class"] == "foreground_compute"
    assert attrs["command"] == "python train.py"
    assert "command_redacted" not in attrs
    assert "command_hash" in attrs


def test_command_attributes_without_text():
    attrs = command_attributes("python train.py", include_command_text=False)
    assert attrs["command_class"] == "foreground_compute"
    assert attrs["command_redacted"] is True
    assert "command" not in attrs
    assert "command_hash" in attrs


def test_command_attributes_none_command():
    attrs = command_attributes(None, include_command_text=True)
    assert attrs["command_class"] == "other_bash"
    assert "command_hash" not in attrs


# ---------------------------------------------------------------------------
# stable_hash
# ---------------------------------------------------------------------------


def test_stable_hash_deterministic():
    assert stable_hash("hello") == stable_hash("hello")


def test_stable_hash_different_inputs():
    assert stable_hash("hello") != stable_hash("world")


def test_stable_hash_default_length():
    assert len(stable_hash("hello")) == 12


def test_stable_hash_custom_length():
    assert len(stable_hash("hello", length=8)) == 8


def test_stable_hash_empty_string():
    h = stable_hash("")
    assert isinstance(h, str)
    assert len(h) == 12


# ---------------------------------------------------------------------------
# safe_attributes
# ---------------------------------------------------------------------------


def test_safe_attributes_scalar_types():
    attrs = safe_attributes({"s": "str", "i": 1, "f": 1.5, "b": True})
    assert attrs == {"s": "str", "i": 1, "f": 1.5, "b": True}


def test_safe_attributes_drops_none():
    attrs = safe_attributes({"a": None, "b": "keep"})
    assert "a" not in attrs
    assert attrs["b"] == "keep"


def test_safe_attributes_converts_unknown_type():
    class Foo:
        def __str__(self):
            return "foo_str"

    attrs = safe_attributes({"x": Foo()})
    assert attrs["x"] == "foo_str"


def test_safe_attributes_handles_list():
    attrs = safe_attributes({"items": [1, "two", None]})
    assert attrs["items"] == [1, "two", None]


def test_safe_attributes_empty():
    assert safe_attributes({}) == {}
    assert safe_attributes(None) == {}


# ---------------------------------------------------------------------------
# SandboxEventRecorder basics
# ---------------------------------------------------------------------------


from pathlib import Path

from nemo_gym.observability.recorder import (
    SandboxEventRecorder,
    current_recorder,
    event_context,
    observability_suppressed,
    record_event,
    suppress_observability_events,
    use_recorder,
)


def _make_recorder(tmp_path: Path) -> SandboxEventRecorder:
    return SandboxEventRecorder(
        output_dir=tmp_path / "obs",
        resource_sample_interval_s=10,
        max_rendered_trajectories=5,
        artifacts={"enabled": False, "render_html": False, "render_png": False, "export_otlp_json": False},
        otel={"enabled": False, "endpoint": None, "service_name": "test"},
        wandb={"enabled": False},
        process_trace={
            "enabled": False,
            "sample_interval_s": 1.0,
            "max_processes_per_sample": 128,
            "include_cmdline": False,
        },
        privacy={"include_command_text": False},
    )


def test_recorder_writes_events_jsonl(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_event("span_end", "sandbox.exec", attributes={"phase": "execution", "duration_s": 1.0})
    rec.finalize()
    lines = (tmp_path / "obs" / "events.jsonl").read_text().splitlines()
    event_names = [__import__("json").loads(l)["name"] for l in lines]
    assert "run.start" in event_names
    assert "sandbox.exec" in event_names
    assert "run.end" in event_names


def test_recorder_summary_json_written(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_event("span_end", "sandbox.exec", attributes={"phase": "execution", "duration_s": 2.5, "status": "ok"})
    rec.finalize()
    import json

    summary = json.loads((tmp_path / "obs" / "summary.json").read_text())
    assert summary["events_count"] >= 1
    assert "sandbox.exec" in summary["durations_by_name"]


def test_recorder_finalize_idempotent(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.finalize()
    rec.finalize()  # second call must not raise
    lines = (tmp_path / "obs" / "events.jsonl").read_text().splitlines()
    run_ends = [l for l in lines if '"run.end"' in l]
    assert len(run_ends) == 1


# ---------------------------------------------------------------------------
# use_recorder / current_recorder context var
# ---------------------------------------------------------------------------


def test_use_recorder_sets_and_clears(tmp_path):
    assert current_recorder() is None
    rec = _make_recorder(tmp_path)
    with use_recorder(rec):
        assert current_recorder() is rec
    assert current_recorder() is None


def test_use_recorder_none_is_noop():
    with use_recorder(None):
        assert current_recorder() is None


def test_record_event_routes_to_active_recorder(tmp_path):
    rec = _make_recorder(tmp_path)
    with use_recorder(rec):
        record_event("test_type", "my.event", attributes={"x": 1})
    rec.finalize()
    import json

    events = [json.loads(l) for l in (tmp_path / "obs" / "events.jsonl").read_text().splitlines()]
    assert any(e["name"] == "my.event" for e in events)


def test_record_event_noop_without_recorder():
    # Should not raise even with no active recorder
    record_event("test_type", "my.event", attributes={"x": 1})


# ---------------------------------------------------------------------------
# event_context propagates attributes
# ---------------------------------------------------------------------------


def test_event_context_injects_attributes(tmp_path):
    rec = _make_recorder(tmp_path)
    with use_recorder(rec):
        with event_context(trajectory_id="task-0-rollout-0", agent="my_agent"):
            record_event("span_end", "sandbox.exec", attributes={"duration_s": 1.0})
    rec.finalize()
    import json

    events = [json.loads(l) for l in (tmp_path / "obs" / "events.jsonl").read_text().splitlines()]
    exec_event = next(e for e in events if e["name"] == "sandbox.exec")
    assert exec_event["attributes"]["trajectory_id"] == "task-0-rollout-0"
    assert exec_event["attributes"]["agent"] == "my_agent"


def test_event_context_clears_after_exit(tmp_path):
    rec = _make_recorder(tmp_path)
    with use_recorder(rec):
        with event_context(trajectory_id="t0"):
            pass
        record_event("test", "after.context")
    rec.finalize()
    import json

    events = [json.loads(l) for l in (tmp_path / "obs" / "events.jsonl").read_text().splitlines()]
    after = next(e for e in events if e["name"] == "after.context")
    assert "trajectory_id" not in (after.get("attributes") or {})


# ---------------------------------------------------------------------------
# suppress_observability_events
# ---------------------------------------------------------------------------


def test_suppress_blocks_record_event(tmp_path):
    rec = _make_recorder(tmp_path)
    with use_recorder(rec):
        with suppress_observability_events():
            assert observability_suppressed()
            record_event("suppressed", "should.not.appear")
    rec.finalize()
    import json

    events = [json.loads(l) for l in (tmp_path / "obs" / "events.jsonl").read_text().splitlines()]
    assert not any(e["name"] == "should.not.appear" for e in events)


def test_suppress_is_not_suppressed_outside():
    assert not observability_suppressed()
    with suppress_observability_events():
        assert observability_suppressed()
    assert not observability_suppressed()


# ---------------------------------------------------------------------------
# build_recorder_from_config
# ---------------------------------------------------------------------------


from nemo_gym.observability.recorder import build_recorder_from_config


def test_build_recorder_returns_none_when_disabled(tmp_path):
    assert build_recorder_from_config({"enabled": False, "output_dir": str(tmp_path)}) is None


def test_build_recorder_returns_none_when_no_output_dir():
    assert build_recorder_from_config({"output_dir": None}) is None


def test_build_recorder_returns_none_for_none_config():
    assert build_recorder_from_config(None) is None


def test_build_recorder_creates_recorder_with_output_dir(tmp_path):
    rec = build_recorder_from_config({"output_dir": str(tmp_path / "obs")})
    assert rec is not None
    assert rec.output_dir == tmp_path / "obs"
    rec.finalize()
