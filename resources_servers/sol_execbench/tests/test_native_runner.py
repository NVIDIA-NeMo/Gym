# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual sandbox entrypoint with mocked native package and GPU CLI."""

import hashlib
import json
import subprocess
import sys
from types import ModuleType

import pytest
from pydantic import BaseModel, Field, JsonValue

from resources_servers.sol_execbench import native_runner
from resources_servers.sol_execbench.problem_store import NATIVE_REVISION


class NativeSchemaStub(BaseModel):
    name: str = "synthetic"
    definition: str = "synthetic"


class NativeToleranceStub(BaseModel):
    max_atol: float = 0.01


class NativeWorkloadStub(BaseModel):
    uuid: str = "synthetic"
    axes: dict[str, int] = Field(default_factory=dict)
    inputs: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    tolerance: NativeToleranceStub = Field(default_factory=NativeToleranceStub)


class NativeTraceStub(BaseModel):
    workload: NativeWorkloadStub = Field(default_factory=NativeWorkloadStub)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("# original synthetic package\n")
    module = ModuleType("sol_execbench")
    module.__file__ = str(source)
    core = ModuleType("sol_execbench.core")
    for name in ("Definition", "Solution"):
        setattr(core, name, NativeSchemaStub)
    core.Trace = NativeTraceStub
    core.Workload = NativeWorkloadStub
    monkeypatch.setitem(sys.modules, "sol_execbench", module)
    monkeypatch.setitem(sys.modules, "sol_execbench.core", core)
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "problem").mkdir()
    (root / "problem/definition.json").write_text('{"name":"synthetic"}')
    (root / "problem/workload.jsonl").write_text("{}\n")
    (root / "solution.json").write_text('{"definition":"synthetic"}')
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "native_revision": NATIVE_REVISION,
                "target_hardware": "B200",
                "compile_timeout_s": 120,
                "evaluation_timeout_s": 600,
            }
        )
    )
    (root / "native_source_hashes.json").write_text(
        json.dumps({"__init__.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    )
    marker = tmp_path / "revision"
    marker.write_text(NATIVE_REVISION)
    monkeypatch.setattr(native_runner, "ROOT", root)
    monkeypatch.setattr(native_runner, "REVISION_MARKER", marker)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="NVIDIA B200, GPU-synthetic, synthetic" if "--query-gpu=" in command[1] else "",
                stderr="",
            )
        (root / "trace.jsonl").write_text("{}\n")
        kwargs["stdout"].write("native stdout")
        kwargs["stderr"].write("native stderr")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(native_runner.subprocess, "run", run)
    return root, marker, source, calls


def test_native_cli_invoked_once_with_pinned_timeouts_and_validated_exit_one(runner):
    root, _, _, calls = runner
    native_runner.main()
    command, kwargs = calls[-1]
    assert len(calls) == 3  # hardware, process observation, native evaluator
    assert command == [
        "/venv/bin/sol-execbench",
        str(root / "problem"),
        "--solution",
        str(root / "solution.json"),
        "--config",
        str(root / "config.json"),
        "--output",
        str(root / "trace.jsonl"),
        "--compile-timeout",
        "120",
        "--timeout",
        "600",
        "--verbose",
    ]
    assert kwargs["env"]["FLASHINFER_TRACE_DIR"] == str(root / "assets")
    assert json.loads((root / "execution.json").read_text())["return_code"] == 1
    assert json.loads((root / "execution.json").read_text())["native_schema_validated"] is True
    assert json.loads((root / "execution.json").read_text())["native_workloads_validated"] is True
    assert (root / "native.stderr").read_text() == "native stderr"
    assert json.loads((root / "hardware.json").read_text())["native_revision"] == NATIVE_REVISION


@pytest.mark.parametrize("change", ["marker", "source"])
def test_unpinned_native_source_never_runs_candidate(runner, change):
    _, marker, source, calls = runner
    (marker if change == "marker" else source).write_text("modified")
    with pytest.raises(RuntimeError, match="mismatch|differ"):
        native_runner.main()
    assert not calls


def test_native_solution_validation_failure_is_explicit_and_skips_cli(runner):
    root, _, _, calls = runner
    (root / "solution.json").write_text('{"definition":"wrong"}')
    native_runner.main()
    assert len(calls) == 2 and all(call[0][0] == "nvidia-smi" for call in calls)
    assert json.loads((root / "validation.json").read_text())["valid"] is False
    assert not (root / "execution.json").exists()


@pytest.mark.parametrize(
    "hardware_name,accepted",
    [
        ("NVIDIA B200", True),
        (" NVIDIA B200 ", True),
        ("NVIDIA GB200", False),
        ("MIG NVIDIA B200", False),
        ("NVIDIA B200 MIG 1g", False),
    ],
)
def test_b200_requires_exact_product_name(runner, monkeypatch, hardware_name, accepted):
    root, _, _, calls = runner
    original_run = native_runner.subprocess.run

    def run(command, **kwargs):
        if command[0] == "nvidia-smi" and "--query-gpu=" in command[1]:
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{hardware_name}, GPU-synthetic, synthetic", stderr=""
            )
        return original_run(command, **kwargs)

    monkeypatch.setattr(native_runner.subprocess, "run", run)
    if accepted:
        native_runner.main()
        assert (root / "execution.json").exists()
    else:
        with pytest.raises(RuntimeError, match="requires an NVIDIA B200"):
            native_runner.main()
        assert len(calls) == 1
        assert not (root / "execution.json").exists()


@pytest.mark.parametrize(
    "workload",
    [
        {"uuid": "different"},
        {"axes": {"N": 2}},
        {"inputs": {"x": {"type": "scalar", "value": 1}}},
        {"tolerance": {"max_atol": 1.0}},
    ],
)
def test_native_trace_payload_mismatch_never_attests_success(runner, monkeypatch, workload):
    root, _, _, _ = runner
    original_run = native_runner.subprocess.run

    def run(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[0] != "nvidia-smi":
            (root / "trace.jsonl").write_text(json.dumps({"workload": workload}) + "\n")
        return result

    monkeypatch.setattr(native_runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="workload payload differs"):
        native_runner.main()
    assert not (root / "execution.json").exists()


@pytest.mark.parametrize("trusted", [{}, {"tolerance": {"required_match_ratio": 1.0}}])
def test_native_workload_defaults_are_compared_after_schema_normalization(runner, monkeypatch, trusted):
    root, _, _, _ = runner
    (root / "problem/workload.jsonl").write_text(json.dumps(trusted) + "\n")
    original_run = native_runner.subprocess.run

    def run(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[0] != "nvidia-smi":
            payload = {"uuid": "synthetic", "axes": {}, "inputs": {}, "tolerance": {"max_atol": 0.01}}
            (root / "trace.jsonl").write_text(json.dumps({"workload": payload}) + "\n")
        return result

    monkeypatch.setattr(native_runner.subprocess, "run", run)
    native_runner.main()
    assert json.loads((root / "execution.json").read_text())["native_workloads_validated"] is True
