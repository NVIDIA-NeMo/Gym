# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for executable plans, router CLI compatibility and replica metrics."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from responses_api_models.local_vllm_model import plan
from responses_api_models.local_vllm_model.router_launcher import VLLMRouterConfig, VLLMRouterLauncher


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "launcher_comparison", ROOT / "benchmarks/nemotron_3.5_super/launcher_comparison.py"
)
launcher_comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher_comparison)


@pytest.fixture(autouse=True)
def checkout_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.parametrize("gpus", [4, 8])
def test_generated_metrics_cover_every_planned_replica(tmp_path, monkeypatch, gpus):
    image = tmp_path / "image.sqsh"
    router = tmp_path / "router"
    image.touch()
    router.write_text("test router")
    output = tmp_path / "run"
    for key, value in {
        "MODEL": "test-model",
        "CONTAINER": str(image),
        "VLLM_CONFIG": "benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super.sh",
        "NUM_PREFILL_NODES": "2",
        "NUM_DECODE_NODES": "2",
        "GPUS_PER_NODE": str(gpus),
    }.items():
        monkeypatch.setenv(key, value)
    hosts = [(f"node{i}", f"192.0.2.{i + 1}") for i in range(4)]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher_comparison",
            "build",
            "--output",
            str(output),
            "--router",
            str(router),
            *[arg for n, ip in hosts for arg in ("--node", f"{n}={ip}")],
        ],
    )
    launcher_comparison.main()
    config = launcher_comparison.ClusterConfig.model_validate_json((output / "cluster-config.json").read_text())
    deployment = launcher_comparison.deployment_plan(config, hosts, output, "test")
    metrics = json.loads((output / "inference-metrics.json").read_text())["inference_metrics"]
    for role, urls in deployment["urls"].items():
        actual = {metrics["endpoints"][name] for name in metrics["endpoint_groups"][role]}
        assert actual == {url + "/metrics" for url in urls}
    assert len(metrics["endpoints"]) == gpus


def test_plan_composes_shipped_config_without_masking_values(tmp_path):
    config = plan.load_config(
        Path("responses_api_models/local_vllm_model/configs/subprocess.yaml"),
        "policy_model",
        ["policy_model_name=test-model"],
    )
    assert config.name == "policy_model"
    assert config.model == "test-model"
    assert config.return_token_id_information is False
    masked = tmp_path / "resolved.yaml"
    masked.write_text(
        "policy_model:\n  responses_api_models:\n    local_vllm_model:\n      return_token_id_information: '****'\n"
    )
    with pytest.raises(ValueError, match="redacted"):
        plan.load_config(masked, "policy_model", [])


def test_documented_plan_probes_executable_and_writes_manifest(tmp_path, monkeypatch, capsys):
    executable = tmp_path / "vllm"
    flags = (
        "--host --port --distributed-executor-backend --data-parallel-backend --data-parallel-size "
        "--tensor-parallel-size --pipeline-parallel-size --served-model-name --download-dir "
        "--gpu-memory-utilization --disable-log-stats"
    )
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nprint('test-vllm' if '--version' in sys.argv else {flags!r})\n"
    )
    executable.chmod(0o700)
    output = tmp_path / "plans"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan",
            "--config",
            "responses_api_models/local_vllm_model/configs/subprocess.yaml",
            "--name",
            "policy_model",
            "--override",
            "policy_model_name=test-model",
            "--override",
            f"policy_model.responses_api_models.local_vllm_model.subprocess.executable={executable}",
            "--override",
            f"policy_model.responses_api_models.local_vllm_model.subprocess.log_dir={output}",
        ],
    )
    plan.main()
    manifests = list(output.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["dry_run"] is True
    assert manifest["argv_redacted"][2] == "test-model"
    assert str(manifests[0]) in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["--eviction-interval", "--eviction-interval-secs"])
async def test_router_eviction_flag_matches_selected_cli(tmp_path, flag):
    executable = tmp_path / "router"
    help_text = (
        "--host --port --intra-node-data-parallel-size --request-timeout-secs "
        "--worker-startup-timeout-secs --prometheus-host --prometheus-port --log-level "
        "--policy --worker-urls " + flag
    )
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nprint('vllm-router 0.1.15' if '--version' in sys.argv else {help_text!r})\n"
    )
    executable.chmod(0o700)
    router = VLLMRouterLauncher(
        config=VLLMRouterConfig(
            executable=str(executable),
            executable_type="binary",
            profile="external_benchmark",
            expected_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
            log_dir=tmp_path / "logs",
            eviction_interval=120,
        ),
        model="test-model",
        api_key="",
    )
    await router.start(8000, worker_urls=["http://127.0.0.1:8001"], dry_run=True)
    manifest = json.loads((router.run_dir / "manifest.json").read_text())
    assert f"{flag}=120" in manifest["argv_redacted"]
