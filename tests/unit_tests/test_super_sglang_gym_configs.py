# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Gym evaluation settings consistent across SGLang topology recipes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nemo_gym.inference_metrics import InferenceMetricsConfig


CONFIG_DIR = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/sglang_configs"


def _benchmark_without_experiment_name(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text())
    assert isinstance(config, dict), f"{path.name}: expected a YAML mapping"
    benchmark = config.get("benchmark")
    assert isinstance(benchmark, dict), f"{path.name}: missing benchmark mapping"
    assert benchmark.get("type") == "custom", f"{path.name}: expected a custom Gym benchmark"
    command = benchmark.get("command")
    assert isinstance(command, str) and command.strip(), f"{path.name}: missing benchmark command"
    env = benchmark.get("env")
    assert isinstance(env, dict), f"{path.name}: missing benchmark environment"
    experiment_name = env.pop("EXPERIMENT_NAME", None)
    assert isinstance(experiment_name, str) and experiment_name.strip(), f"{path.name}: missing EXPERIMENT_NAME"
    return benchmark


def test_gym_benchmark_sections_match() -> None:
    benchmarks = {}
    # Discover topology recipes automatically, excluding explicit serving-only recipes.
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and config.get("benchmark") == {"type": "manual"}:
            continue
        benchmarks[path] = _benchmark_without_experiment_name(path)

    assert len(benchmarks) >= 2, "Expected at least two Gym recipes to check benchmark consistency"
    # Any existing recipe can be the reference; topology names may change or disappear.
    (reference_path, reference), *others = benchmarks.items()
    for path, benchmark in others:
        assert benchmark == reference, (
            f"{path.name}: benchmark differs from {reference_path.name}; only benchmark.env.EXPERIMENT_NAME may differ"
        )


@pytest.mark.parametrize("missing_role", [None, "prefill", "decode"])
def test_srt_worker_metrics_config(tmp_path: Path, missing_role: str | None) -> None:
    config = yaml.safe_load((CONFIG_DIR / "2P2D.yaml").read_text())
    command = config["benchmark"]["command"]
    syntax = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    # Execute the recipe's actual config generation with SRT's documented endpoint format.
    setup = (
        "inference_metrics_config=" + command.split("inference_metrics_config=", 1)[1].split("gym eval prepare", 1)[0]
    )
    env = os.environ | {
        # The recipe activates Gym's venv before this block.
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "output_dir": str(tmp_path),
        "SRT_PREFILL_ENDPOINTS": "10.0.0.1:6100,10.0.0.2:6100",
        "SRT_DECODE_ENDPOINTS": "10.0.0.3:6200,10.0.0.3:6201",
    }
    if missing_role:
        env.pop(f"SRT_{missing_role.upper()}_ENDPOINTS")
    result = subprocess.run(["bash", "-euc", setup], env=env, text=True, capture_output=True)
    output = tmp_path / "inference-metrics.yaml"
    if missing_role:
        assert result.returncode != 0
        assert f"SRT must provide SRT_{missing_role.upper()}_ENDPOINTS" in result.stderr
        assert not output.exists()
        return
    assert result.returncode == 0, result.stderr
    metrics = InferenceMetricsConfig.model_validate(yaml.safe_load(output.read_text())["inference_metrics"])
    assert metrics.enabled
    assert {name: str(url) for name, url in metrics.endpoints.items()} == {
        "prefill0": "http://10.0.0.1:6100/metrics",
        "prefill1": "http://10.0.0.2:6100/metrics",
        "decode0": "http://10.0.0.3:6200/metrics",
        "decode1": "http://10.0.0.3:6201/metrics",
    }
    assert '--config "$inference_metrics_config"' in command.split("gym eval run", 1)[1]
    assert all(role["args"]["enable-metrics"] for role in config["roles"].values())


def test_hicache_mooncake_prefill_config(tmp_path: Path) -> None:
    base = yaml.safe_load((CONFIG_DIR / "2P2D.yaml").read_text())
    config = yaml.safe_load((CONFIG_DIR / "2P2D_hicachemooncake.yaml").read_text())
    assert config["roles"]["decode"] == base["roles"]["decode"]
    assert config["frontend"] == base["frontend"]
    prefill = config["roles"]["prefill"]
    assert prefill["args"]["enable-hierarchical-cache"]
    assert prefill["args"]["hicache-storage-backend"] == "mooncake"
    assert prefill["args"]["disaggregation-transfer-backend"] == "nixl"
    master, store = config["services"]
    assert master["placement"]["node"] == "head"
    assert store["placement"]["node"] == "prefill"
    assert master["readiness"]["port"] == 50051
    assert "--port=50051" in master["args"]
    for service in (master, store):
        assert service["start"] == "before_workers"
        assert service["critical"]
    for index in range(2):
        substitutions = {"node": f"prefill-{index}", "node_ip": f"10.0.0.{index + 1}", "head_ip": "10.0.0.1"}
        service_env = {key: str(value).format_map(substitutions) for key, value in store["env"].items()}
        worker_path = prefill["env"]["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH"].format_map(substitutions)
        assert worker_path == service_env["HICACHE_CONFIG_PATH"]
        output = tmp_path / Path(worker_path).name
        env = (
            os.environ
            | service_env
            | {
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
                "HICACHE_CONFIG_PATH": str(output),
            }
        )
        # SRT strips the preamble and joins it to the service command with &&.
        command = store["preamble"].rstrip() + " && printf service-started"
        result = subprocess.run(["bash", "-euc", command], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "service-started"
        assert json.loads(output.read_text()) == {
            "local_hostname": substitutions["node_ip"],
            "master_server_address": "10.0.0.1:50051",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "rdma",
            "device_name": "",
            "global_segment_size": 0,
        }
