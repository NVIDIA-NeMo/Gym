# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission of external task data and native lifecycle ownership."""

import hashlib
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from responses_api_agents.saas_bench_agent import runtime


def test_materialization_validates_data_and_reused_code(tmp_path, monkeypatch):
    template = tmp_path / "template"
    template.mkdir()
    (template / "verify.py").write_text("assert True\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "description.md").write_text("Native task\n")
    (template / "task-data-manifest.json").write_text(
        json.dumps({"files": {"description.md": hashlib.sha256((data / "description.md").read_bytes()).hexdigest()}})
    )
    (template / "provenance.json").write_text(json.dumps({"files": [{"source_path": "verify.py"}]}))
    monkeypatch.setattr(runtime, "BENCHMARK_ROOT", template)
    output = tmp_path / "bundle"
    assert runtime.materialize_bundle(str(data), str(output)) == output
    assert runtime.materialize_bundle(str(data), str(output)) == output
    (output / "verify.py").write_text("assert False\n")
    with pytest.raises(ValueError, match="code differs"):
        runtime.materialize_bundle(str(data), str(output))
    (data / "description.md").write_text("Wrong task\n")
    with pytest.raises(ValueError, match="pinned release"):
        runtime.materialize_bundle(str(data), str(tmp_path / "other"))


@pytest.mark.parametrize("root", ["/root/cache", "{home}/cache"])
def test_forbidden_cache_roots(tmp_path, monkeypatch, root):
    import pwd
    from types import SimpleNamespace

    home = tmp_path / "home"
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(home)))
    with pytest.raises(ValueError, match="forbidden"):
        runtime.configure_cache(root.format(home=home))


@pytest.mark.parametrize(
    "status,unavailable,actor_status,masked",
    [
        ("PASS", False, "completed", False),
        ("FAIL", False, "completed", False),
        ("ERROR", True, "completed", True),
        ("FAIL", False, "error", True),
        ("FAIL", False, "model_error", False),
        ("FAIL", False, "context_limit", False),
        ("FAIL", False, "runtime_after_turn", True),
        ("FAIL", False, "timeout", True),
        ("FAIL", False, "exception", True),
    ],
)
def test_browser_worker_grades_live_state_and_always_stops(
    tmp_path, monkeypatch, status, unavailable, actor_status, masked
):
    monkeypatch.syspath_prepend(str(runtime.BENCHMARK_ROOT))
    from responses_api_agents.saas_bench_agent import browser_worker as worker

    root = tmp_path / "bundle"
    (root / "saas_bench").mkdir(parents=True)
    (root / "saas_bench/apps.yaml").write_text("apps:\n  fixture: {}\n")
    task = {"task_id": "fixture", "source_sha256": "sha", "meta": {"meta_data": {"sites": ["fixture"]}}}
    monkeypatch.setattr(worker, "inventory", lambda *args: {"fixture": task})
    monkeypatch.setattr(worker, "measured_score", lambda result: result["score"])
    monkeypatch.setattr(worker, "verifier_diagnostics", lambda result: {"unavailable": unavailable})

    @contextmanager
    def claim(*args):
        yield 0

    monkeypatch.setattr(worker, "claim_slot", claim)
    slot = MagicMock()
    slot.get_port_map.return_value = {"fixture": 8001}
    loader = types.ModuleType("saas_bench.loader")
    loader.build_prompt = lambda *args, **kw: ("Exact native prompt", "todo", [])
    slots = types.ModuleType("saas_bench.slot")
    slots.SlotManager = lambda *args: slot
    verifier = types.ModuleType("saas_bench.verify_runner")

    def grade(*args, **kw):
        slot.stop_apps.assert_not_called()
        assert __import__("os").environ["LLM_API_KEY"] == "judge-fixture"
        return {"status": status, "score": 0.25 if status in {"PASS", "FAIL"} else 0.0}

    verifier.run_verify = MagicMock(side_effect=grade)
    actor = types.ModuleType("saas_bench.agent")

    async def run_task(*args, **kw):
        if actor_status == "timeout":
            raise TimeoutError("actor deadline")
        if actor_status == "exception":
            raise RuntimeError("actor exception")
        assert __import__("os").environ["LLM_API_KEY"] == "EMPTY"
        return {
            "status": "completed" if actor_status == "completed" else "error",
            **(
                {
                    "failure_origin": "model",
                    "failure_kind": "context_limit" if actor_status == "context_limit" else "malformed_tool_call",
                }
                if actor_status in {"model_error", "context_limit"}
                else {}
            ),
            "trajectory": [
                {
                    "actions": [{"click": {"index": 1}}, {"input_text": {"text": "fixture"}}],
                    "results": [{"extracted_content": "clicked"}, {"error": "field unavailable"}],
                    "screenshot": "original.png",
                }
            ],
        }

    actor.run_task = run_task
    for name, module in [("loader", loader), ("slot", slots), ("verify_runner", verifier), ("agent", actor)]:
        monkeypatch.setitem(sys.modules, "saas_bench." + name, module)
    monkeypatch.setenv("LLM_API_KEY", "judge-fixture")
    monkeypatch.chdir(tmp_path)
    result = worker.execute(
        {
            "source_root": str(root),
            "output": str(tmp_path / "episode"),
            "cache_root": str(tmp_path / "cache"),
            "task_id": "fixture",
            "source_sha256": "sha",
            "slot_capacity": 1,
            "slot_lock_root": str(tmp_path / "locks"),
            "endpoint": "http://proxy/v1",
            "model_name": "policy",
            "output_tokens": 100,
            "context_tokens": 1000,
            "max_steps": 5,
            "agent_timeout": 10,
        }
    )
    assert result["mask_sample"] is masked, result
    if actor_status in {"timeout", "exception", "error", "runtime_after_turn"}:
        verifier.run_verify.assert_not_called()
        assert result["status"] == "error" and result["failure_kind"] == "saas_bench:infrastructure"
        slot.stop_apps.assert_called_once_with(["fixture"])
        return
    assert result["verify"]["status"] == status
    if actor_status in {"model_error", "context_limit"}:
        assert result["reward"] == 0.25 and not result["mask_sample"]
        assert result["execution_health"]["termination"] == (
            "context_limit" if actor_status == "context_limit" else "malformed_tool_call"
        )
    assert result["native_trajectory"][0]["screenshot"] == "original.png"
    assert result["execution_health"]["tool_calls"] == 2
    assert result["execution_health"]["tool_errors"] == 1
    assert result["native_trajectory"][0]["results"][1]["error"] == "field unavailable"
    slot.stop_apps.assert_called_once_with(["fixture"])


@pytest.mark.parametrize("harness", ["opencode", "browser_use"])
def test_gym_preparation_and_collection_use_same_dataset(tmp_path, monkeypatch, harness):
    from omegaconf import OmegaConf

    from nemo_gym.config import GlobalConfigDictParser
    from nemo_gym.environment.benchmark_preparation import (
        _multiprocess_benchmark_prepare_fn,
        datasets_to_prepare,
    )

    root = tmp_path / "fixture"
    task = root / "tasks/Fixture/fixture"
    task.mkdir(parents=True)
    (root / "saas_bench").mkdir()
    (root / "saas_bench/apps.yaml").write_text("apps:\n  siyuan: {}\n")
    (task / "meta.json").write_text(
        json.dumps({"task_id": "fixture", "category_id": "Fixture", "meta_data": {"sites": ["siyuan"]}})
    )
    (task / "description.md").write_text("Create the requested fixture document.")
    (task / "verify.py").write_text("print('PASS')\n")
    monkeypatch.setattr(runtime, "materialize_bundle", lambda *a, **kw: root)
    monkeypatch.syspath_prepend(str(runtime.BENCHMARK_ROOT))
    monkeypatch.setenv("SAAS_BENCH_DATA_DIR", str(root))
    monkeypatch.setenv("SAAS_BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    gym = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(gym / f"benchmarks/saas_bench_{harness}/config.yaml")
    config.saas_bench_run_root = str(tmp_path / "run")
    config.saas_bench_tasks = "fixture"
    selected = datasets_to_prepare(config)
    assert len(selected) == 1
    args = OmegaConf.to_container(config.prepare_script_args, resolve=True)
    _multiprocess_benchmark_prepare_fn((selected[0], str(gym / "benchmarks/saas_bench/prepare.py"), args, None))
    servers = GlobalConfigDictParser().filter_for_server_instance_configs(config)
    datasets = [dataset for server in servers for dataset in (server.datasets or [])]
    assert len(datasets) == 1
    rows = [json.loads(line) for line in Path(datasets[0].jsonl_fpath).read_text().splitlines()]
    assert rows[0]["verifier_metadata"]["harness"] == harness
    assert rows[0]["responses_create_params"]["input"][0]["content"] == "Create the requested fixture document."


def test_long_workspace_caches_keep_native_socket_paths_short(tmp_path, monkeypatch):
    import os
    import socket
    import tempfile

    root = tmp_path / ("long-workspace-" * 12)
    with monkeypatch.context() as context:
        for name in list(os.environ):
            context.setenv(name, os.environ[name])
        runtime.configure_cache(str(root))
        assert os.environ["XDG_CACHE_HOME"].startswith(str(root))
        assert Path(os.environ["TMPDIR"]).resolve().is_relative_to(root)
        with tempfile.TemporaryDirectory(prefix="saas-visual-", dir=os.environ["TMPDIR"]) as directory:
            with socket.socket(socket.AF_UNIX) as sock:
                sock.bind(str(Path(directory) / "browser.sock"))


def test_staged_runtime_tools_are_available_to_native_workers(tmp_path, monkeypatch):
    import os

    helper = tmp_path / "helpers"
    monkeypatch.setenv("SAAS_ENROOT_RUNTIME_DIR", str(helper))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing-libs")
    runtime.configure_cache(str(tmp_path / "cache"))
    assert str(helper / "usr/bin") in os.environ["PATH"].split(os.pathsep)
    assert str(helper / "usr/sbin") in os.environ["PATH"].split(os.pathsep)
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(helper / "usr/lib/x86_64-linux-gnu"),
        "/existing-libs",
    ]


def test_cpu_app_enroot_config_retains_mounts_without_host_gpu_hooks(tmp_path, monkeypatch):
    source = tmp_path / "host-enroot"
    files = {
        "enroot.conf": "ENROOT_ROOTFS_WRITABLE=yes\n",
        "mounts.d/10-runtime.fstab": "/etc/resolv.conf /etc/resolv.conf none bind,ro\n",
        "hooks.d/10-runtime.sh": "#!/bin/sh\nexit 0\n",
        "hooks.d/50-mig-config.sh": "#!/bin/sh\nexit 90\n",
        "hooks.d/98-nvidia.sh": "#!/bin/sh\nexit 91\n",
    }
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o700)
    monkeypatch.setattr(runtime, "_SYSTEM_ENROOT_CONFIG", source)
    cache = tmp_path / "cache"
    runtime.configure_cache(str(cache))
    target = Path(__import__("os").environ["ENROOT_SYSCONF_PATH"])
    for name in ("enroot.conf", "mounts.d/10-runtime.fstab", "hooks.d/10-runtime.sh"):
        assert (target / name).read_text() == files[name]
    assert not (target / "hooks.d/50-mig-config.sh").exists()
    assert not (target / "hooks.d/98-nvidia.sh").exists()
    before = (target / "hooks.d/10-runtime.sh").stat().st_mtime_ns
    runtime.configure_cache(str(cache))
    assert Path(__import__("os").environ["ENROOT_SYSCONF_PATH"]) == target
    assert (target / "hooks.d/10-runtime.sh").stat().st_mtime_ns == before
    (source / "hooks.d/10-runtime.sh").write_text("#!/bin/sh\nexit 2\n")
    runtime.configure_cache(str(cache))
    updated = Path(__import__("os").environ["ENROOT_SYSCONF_PATH"])
    assert updated != target
    assert (target / "hooks.d/10-runtime.sh").read_text() == files["hooks.d/10-runtime.sh"]


def test_svg_rootfs_uses_host_slurm_uid_but_caches_stay_in_workspace(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("SAAS_ENROOT_PLATFORM_ROOT", "/raid/enroot")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_UID", "99353")
    monkeypatch.setattr(os, "getuid", lambda: 0)
    monkeypatch.delenv("SAAS_NRT_ENROOT", raising=False)
    real_mkdir = Path.mkdir
    created = []

    def mkdir(path, *args, **kwargs):
        if str(path).startswith("/raid/enroot/"):
            created.append(str(path))
        else:
            real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    runtime.configure_cache(str(tmp_path / "cache"))
    assert len(created) == 2
    assert all("/user-99353/saas-123-" in path for path in created)
    assert os.environ["ENROOT_DATA_PATH"] in created
    assert Path(os.environ["ENROOT_CACHE_PATH"]).is_relative_to(tmp_path)
    assert Path(os.environ["UV_CACHE_DIR"]).is_relative_to(tmp_path)


def test_enroot_dependency_failure_precedes_episode_work(tmp_path, monkeypatch):
    monkeypatch.delenv("SAAS_ENROOT_PLATFORM_ROOT", raising=False)
    monkeypatch.setenv("SAAS_NRT_ENROOT", "1")
    monkeypatch.setenv("SAAS_ENROOT_RUNTIME_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="before episode admission: enroot"):
        runtime.configure_cache(str(tmp_path / "cache"))
