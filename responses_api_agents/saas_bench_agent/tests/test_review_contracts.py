# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reviewer failure cases exercise parser, live readiness, retry and bundle contracts."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from responses_api_agents.saas_bench_agent import runtime


@pytest.fixture(autouse=True)
def native_test_environment(monkeypatch):
    monkeypatch.syspath_prepend(str(runtime.BENCHMARK_ROOT))
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")


def runner():
    spec = importlib.util.spec_from_file_location(
        "review_runner", runtime.BENCHMARK_ROOT / "saas_bench/verify_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_checks_without_score_do_not_become_a_grade(tmp_path):
    r = runner()
    script = tmp_path / "verify.py"
    script.write_text(
        "import sys\nprint('[PASS] (1pt) first',file=sys.stderr)\nraise TimeoutError('state unavailable')\n"
    )
    result = r.run_verify({"task_id": "fixture", "verify_py_path": str(script)}, 0, {}, "localhost", str(tmp_path))
    assert result["status"] == "ERROR" and result["score"] == 0
    assert "SCORE line" in result["error"]


def test_live_state_readiness_invalidates_otherwise_valid_grade(tmp_path, monkeypatch):
    r = runner()
    slots = types.ModuleType("saas_bench.slot")

    def unavailable(*args):
        raise RuntimeError("service stopped")

    slots._wait_ready = unavailable
    monkeypatch.setitem(sys.modules, "saas_bench.slot", slots)
    script = tmp_path / "verify.py"
    script.write_text("import sys\nprint('[PASS] (1pt) first\\nSCORE: 1.000 PASS: True (1/1)',file=sys.stderr)\n")
    task = {"task_id": "fixture", "verify_py_path": str(script), "meta": {"meta_data": {"sites": ["siyuan"]}}}
    result = r.run_verify(task, 0, {"siyuan": 12345}, "localhost", str(tmp_path))
    assert result["status"] == "ERROR" and "stopped answering" in result["error"]
    slots._wait_ready = lambda *args: None
    assert r.run_verify(task, 0, {"siyuan": 12345}, "localhost", str(tmp_path))["score"] == 1


@pytest.mark.parametrize("recover", [False, True])
def test_inconclusive_verification_retry_is_bounded_and_does_not_rerun_actor(tmp_path, recover):
    r = runner()
    script = tmp_path / "verify.py"
    marker = tmp_path / "attempt"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"p=Path({str(marker)!r});n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))\n"
        f"ok={recover!r} and n>1\n"
        "print('[PASS] (1pt) state\\nSCORE: 1.000 PASS: True (1/1)' if ok else 'FATAL: judge unavailable',file=sys.stderr)\n"
    )
    result = r.run_verify(
        {"task_id": "fixture", "verify_py_path": str(script)},
        0,
        {},
        "localhost",
        str(tmp_path),
        inconclusive_verification_retries=1,
    )
    assert marker.read_text() == "2" and len(result["verification_attempts"]) == 2
    assert result["status"] == ("PASS" if recover else "ERROR")


def test_bundle_start_checks_runtime_and_external_grader_bytes(tmp_path, monkeypatch):
    shipped = tmp_path / "shipped"
    bundle = tmp_path / "bundle"
    shipped.mkdir()
    bundle.mkdir()
    (shipped / "runtime.py").write_text("current")
    (bundle / "runtime.py").write_text("old")
    (shipped / "provenance.json").write_text(json.dumps({"files": [{"source_path": "runtime.py"}]}))
    (shipped / "task-data-manifest.json").write_text(
        json.dumps({"files": {"verify.py": hashlib.sha256(b"grader").hexdigest()}})
    )
    monkeypatch.setattr(runtime, "BENCHMARK_ROOT", shipped)
    with pytest.raises(ValueError, match="code differs"):
        runtime.check_bundle_code(bundle)
    (bundle / "runtime.py").write_text("current")
    with pytest.raises(ValueError, match="data differs"):
        runtime.check_bundle_code(bundle)
    (bundle / "verify.py").write_text("grader")
    runtime.check_bundle_code(bundle)


@pytest.mark.parametrize("harness", ["opencode", "browser_use"])
def test_catalog_manifest_and_explicit_run_root(harness, tmp_path, monkeypatch):
    from nemo_gym.environment.manifest import load_manifest

    root = Path(__file__).resolve().parents[3]
    monkeypatch.delenv("SAAS_BENCH_RUN_ROOT", raising=False)
    manifest = load_manifest(root / f"benchmarks/saas_bench_{harness}/manifest.yaml")
    assert manifest.name == f"saas_bench_{harness}"
    config = OmegaConf.load(root / f"benchmarks/saas_bench_{harness}/config.yaml")
    with pytest.raises(InterpolationResolutionError, match="SAAS_BENCH_RUN_ROOT"):
        _ = config.saas_bench_output_root
    monkeypatch.setenv("SAAS_BENCH_RUN_ROOT", str(tmp_path / "env-run"))
    assert config.saas_bench_output_root == str(tmp_path / "env-run/episodes")
    config.saas_bench_run_root = str(tmp_path / "run")
    config.saas_bench_concurrency = 3
    assert config.saas_bench_output_root == str(tmp_path / "run/episodes")
    assert config.num_samples_in_parallel == 3


def test_exec_does_not_inherit_host_secrets(tmp_path, monkeypatch):
    path = runtime.BENCHMARK_ROOT / "scripts/nrt/docker_enroot_shim.py"
    spec = importlib.util.spec_from_file_location("exec_shim", path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    monkeypatch.setattr(shim, "resolve_exec_pids", lambda name: (os.getpid(), os.getpid()))
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setenv("MINDRA_API_KEY", "test-secret")
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "private-config")
    seen = []

    def call(cmd, **kw):
        seen.append((cmd, kw))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", call)
    assert shim.docker_exec(["-i", "-e", "VISIBLE=value", "fixture", "cat", "/file"]) == 0
    cmd, kw = seen[0]
    assert kw["env"] == {"PATH": os.environ.get("PATH", "")}
    assert "VISIBLE=value" in " ".join(cmd)


def test_both_container_network_paths_block_host_loopback():
    tree = ast.parse((runtime.BENCHMARK_ROOT / "saas_bench/enroot_slot.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and any(isinstance(x, ast.Constant) and x.value == "--mtu=65520" for x in node.elts)
    ]
    assert len(calls) == 2
    assert all(
        any(isinstance(x, ast.Constant) and x.value == "--disable-host-loopback" for x in node.elts) for node in calls
    )
