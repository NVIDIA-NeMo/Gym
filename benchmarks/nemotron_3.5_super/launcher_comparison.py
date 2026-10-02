# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Translate the external recipe to cluster settings and compare saved rollouts."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from aiohttp import ClientTimeout

from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, request, set_global_aiohttp_client
from responses_api_models.local_vllm_model.cluster import ClusterConfig, deployment_plan


def parse_flags(argv: list[str]) -> dict:
    """Decode the scalar/JSON CLI options used by the shared serving recipe."""
    result = {}
    index = 0
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--"):
            raise ValueError(f"Unexpected positional argument in serving recipe: {flag!r}")
        index += 1
        if "=" in flag:
            flag, value = flag.split("=", 1)
        elif index < len(argv) and not argv[index].startswith("--"):
            value = argv[index]
            index += 1
        else:
            value = not flag.startswith("--no-")
            if not value:
                flag = "--" + flag[5:]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        key = flag[2:].replace("-", "_")
        if key in result:
            raise ValueError(f"Repeated recipe option: {key}")
        result[key] = value
    return result


def read_recipe(path: Path) -> dict:
    """Source the same trusted shell config as sbatch_external_vllm.sh."""
    capture = """import json,os,sys
args=sys.argv[1:];p=args.index("::PREFILL::");d=args.index("::DECODE::");g=args.index("::GYM::")
print(json.dumps({"common":args[:p],"prefill":args[p+1:d],"decode":args[d+1:g],"gym":args[g+1:],
"env":{k:v for k,v in os.environ.items() if k.startswith(("VLLM_","UCX_","NCCL_")) and k not in ("VLLM_API_KEY",)}}))"""
    script = """set -euo pipefail
source "$1"
"$2" -c "$3" "${VLLM_COMMON_ARGS[@]}" ::PREFILL:: "${VLLM_PREFILL_ARGS[@]}" ::DECODE:: "${VLLM_DECODE_ARGS[@]}" ::GYM:: "${GYM_MODEL_PARAMS[@]}"
"""
    return json.loads(
        subprocess.check_output(["bash", "-c", script, "bash", str(path), sys.executable, capture], text=True)
    )


def make_config(
    recipe: dict,
    *,
    model: str,
    model_name: str,
    image: Path,
    router: Path,
    router_host: str,
    prefill_nodes: int = 2,
    decode_nodes: int = 2,
    gpus_per_node: int = 4,
    cpus_per_node: int = 32,
) -> ClusterConfig:
    """Require independent DP1 replicas and retain each role's scheduler tuning."""
    common = parse_flags(recipe["common"])
    roles = {role: parse_flags(recipe[role]) for role in ("prefill", "decode")}
    expected = {role: common | knobs for role, knobs in roles.items()}
    tp = expected["prefill"].get("tensor_parallel_size", 1)
    if any(knobs.get("tensor_parallel_size", 1) != tp for knobs in expected.values()):
        raise ValueError("Both tiers must use the same TP size")
    for role, knobs in expected.items():
        for key in ("data_parallel_size", "data_parallel_size_local", "api_server_count"):
            if knobs.get(key, 1) != 1:
                raise ValueError(f"The independent replica comparison requires {key}=1")
        connector = knobs.get("kv_transfer_config")
        required = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "kv_load_failure_policy": "fail",
        }
        if isinstance(connector, dict) and "kv_connector_extra_config" in connector:
            extra = connector["kv_connector_extra_config"]
            if not isinstance(extra, dict) or set(extra) != {"kv_lease_duration"}:
                raise ValueError(f"Unsupported {role} connector extras: {extra}")
            required["kv_connector_extra_config"] = extra
        if connector != required:
            raise ValueError(f"Unsupported {role} connector recipe: {connector}")
    managed = {
        "data_parallel_size",
        "data_parallel_size_local",
        "api_server_count",
        "kv_transfer_config",
        "tensor_parallel_size",
    }
    common = {k: v for k, v in common.items() if k not in managed}
    common.update(model=model, tensor_parallel_size=tp)
    groups = {}
    for role, count, port in (("prefill", prefill_nodes, 5600), ("decode", decode_nodes, 5700)):
        groups[role] = {
            "nodes": count,
            "gpus_per_node": gpus_per_node,
            "api_port": 8001,
            "side_channel_port": port,
            "kv_lease_duration": expected[role]["kv_transfer_config"]
            .get("kv_connector_extra_config", {})
            .get("kv_lease_duration"),
            "serve_kwargs": {k: v for k, v in roles[role].items() if k not in managed},
        }
    env = {
        "VLLM_USE_FASTOKENS": "1",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "180",
        "UCX_TLS": "rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "UCX_RNDV_THRESH": "0",
        "UCX_NET_DEVICES": "all",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_MNNVL_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "1",
        **recipe["env"],
    }
    policy = os.environ.get("ROUTER_PREFILL_POLICY", "cache_aware")
    if policy != os.environ.get("ROUTER_DECODE_POLICY", "cache_aware"):
        raise ValueError("This comparison requires identical prefill/decode routing policies")
    return ClusterConfig.model_validate(
        {
            "model": model_name,
            "image": str(image),
            "api_key": "",
            "deployment_mode": "replicas",
            "expected_vllm_version": os.environ.get("EXPECTED_VLLM_VERSION", "0.29.0+precompiled"),
            "groups": groups,
            "serve_kwargs": common,
            "env": env,
            "cpus_per_node": cpus_per_node,
            "probe_timeout_seconds": 180,
            "startup_timeout_seconds": 1800,
            "container_mounts": os.environ.get("MOUNTS", "/lustre:/lustre").split(","),
            "router": {
                "executable": str(router),
                "executable_type": "binary",
                "expected_sha256": hashlib.sha256(router.read_bytes()).hexdigest(),
                "profile": "external_benchmark",
                "host": router_host,
                "port": 8000,
                "metrics_port": int(os.environ.get("ROUTER_METRICS_PORT", 29000)),
                **{
                    name: (
                        float(os.environ[f"ROUTER_{name.upper()}"])
                        if name in {"cache_threshold", "balance_rel_threshold"}
                        else int(os.environ[f"ROUTER_{name.upper()}"])
                    )
                    for name in (
                        "cache_threshold",
                        "balance_abs_threshold",
                        "balance_rel_threshold",
                        "eviction_interval",
                        "max_tree_size",
                    )
                    if f"ROUTER_{name.upper()}" in os.environ
                },
                "policy": policy,
                "inference_timeout_seconds": 86400,
                "startup_timeout_seconds": 1200,
                "probe_timeout_seconds": 180,
                "shutdown_timeout_seconds": 30,
            },
        }
    )


def check_parity(config: ClusterConfig, recipe: dict, hosts: list[tuple[str, str]], output: Path) -> dict:
    """Fail before launch if translation drops or changes a serving option."""
    plan = deployment_plan(config, hosts, output, "parity-check")
    checks = []
    for node in plan["nodes"]:
        expected = parse_flags(recipe["common"]) | parse_flags(recipe[node["role"]])
        for rank in node["ranks"]:
            actual = parse_flags(rank["argv"][2:])
            differences = {
                key: {"expected": value, "actual": actual.get(key)}
                for key, value in expected.items()
                if actual.get(key) != value
            }
            if differences:
                raise ValueError(f"Serving settings changed on {node['node']}: {differences}")
            checks.append({"node": node["node"], "role": node["role"], "rank": rank["rank"], "matched": True})
    return {
        "workers": checks,
        "router_profile": config.router.profile,
        "note": "Host/port/GPU placement and explicit PP1 are managed; worker authentication is disabled to match the external script.",
    }


def read_results(path: Path) -> dict:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        key = (row["_ng_task_index"], row["_ng_rollout_index"])
        if key in rows:
            raise ValueError(f"Duplicate rollout identifier: {key} in {path}")
        rows[key] = {
            k: row.get(k)
            for k in ("reward", "mask_sample", "opencode_finished", "evaluation_completed", "instance_id")
        }
    return rows


def summarize(path: Path, *, expected: int) -> tuple[dict, dict]:
    rows = read_results(path)
    scored = {
        k: r
        for k, r in rows.items()
        if not r["mask_sample"] and type(r["reward"]) in (int, float) and math.isfinite(r["reward"])
    }
    timings = {}
    timing_path = path.with_suffix(".timings.jsonl")
    for line in timing_path.open():
        row = json.loads(line)
        key = (row["task_index"], row["rollout_index"])
        if key in timings:
            raise ValueError(f"Duplicate timing identifier: {key}")
        timings[key] = row
    if any(key not in timings or timings[key]["reward"] != row["reward"] for key, row in scored.items()):
        raise ValueError("Persisted rewards and timing sidecar disagree")
    elapsed = sorted(timings[key]["elapsed_seconds"] for key in scored)
    reward = sum(r["reward"] for r in scored.values())

    def quantile(q: float) -> float | None:
        index = math.ceil(q * expected) - 1
        return elapsed[index] / 60 if len(elapsed) > index else None

    return {
        "path": str(path),
        "expected": expected,
        "persisted": len(rows),
        "scored": len(scored),
        "reward_sum": reward,
        "accuracy_scored_percent": 100 * reward / len(scored) if scored else None,
        "accuracy_full_percent": 100 * reward / expected if len(scored) == expected else None,
        "accuracy_lower_bound_percent": 100 * reward / expected,
        "opencode_finished_count": sum(r["opencode_finished"] is True for r in rows.values()),
        "evaluation_completed_count": sum(r["evaluation_completed"] is True for r in rows.values()),
        "p50_completion_minutes": quantile(0.5),
        "p90_completion_minutes": quantile(0.9),
        "p100_completion_minutes": quantile(1),
        "last_scored_completion_minutes": elapsed[-1] / 60 if elapsed else None,
    }, scored


def compare(baseline: Path, candidate: Path, *, expected: int = 1500) -> dict:
    old, old_rows = summarize(baseline, expected=expected)
    new, new_rows = summarize(candidate, expected=expected)
    shared = old_rows.keys() & new_rows.keys()
    if any(old_rows[k]["instance_id"] != new_rows[k]["instance_id"] for k in shared):
        raise ValueError("Task identifiers refer to different benchmark instances")
    old_accuracy = 100 * sum(old_rows[k]["reward"] for k in shared) / len(shared) if shared else None
    new_accuracy = 100 * sum(new_rows[k]["reward"] for k in shared) / len(shared) if shared else None
    speed = {}
    for metric in ("p50_completion_minutes", "p90_completion_minutes"):
        if old[metric] and new[metric]:
            speed[metric] = {
                "difference_minutes": new[metric] - old[metric],
                "candidate_over_baseline": new[metric] / old[metric],
            }
    return {
        "baseline": old,
        "candidate": new,
        "common_scored": len(shared),
        "common_baseline_accuracy_percent": old_accuracy,
        "common_candidate_accuracy_percent": new_accuracy,
        "common_accuracy_delta_percentage_points": new_accuracy - old_accuracy if shared else None,
        "speed": speed,
        "caveats": [
            "Timing starts at rollout dispatch and excludes service startup.",
            "Missing or masked samples are not treated as failures.",
            "Stochastic sampling and different allocations limit causal attribution.",
        ],
    }


async def smoke(run_dir: Path) -> None:
    """Prove actual routed inference and KV transfer before creating sandboxes."""
    connection = json.loads((run_dir / "cluster/gym-connection.private.json").read_text())
    manifest = json.loads((run_dir / "cluster/manifest.json").read_text())
    client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    headers = {"Authorization": f"Bearer {connection['api_key']}"} if connection["api_key"] else {}
    try:
        # More than one token forces a decode stage after the prefill token.
        response = await request(
            "POST",
            connection["base_url"] + "/chat/completions",
            _retry=False,
            headers=headers,
            json={
                "model": connection["model"],
                "messages": [{"role": "user", "content": f"Connectivity probe {uuid4().hex}"}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": False,
            },
            timeout=ClientTimeout(total=120),
        )
        async with response:
            if response.status != 200:
                raise RuntimeError(f"Routed inference smoke failed: HTTP {response.status}")
            body = await response.json()
        if not body.get("choices") or body.get("usage", {}).get("completion_tokens", 0) < 1:
            raise RuntimeError("Routed inference smoke returned no generated token")
        transferred = 0.0
        try:
            # Engine counters can be published after the HTTP response returns.
            async with asyncio.timeout(15):
                while transferred <= 0:
                    transferred = 0.0
                    for url in manifest["urls"]["decode"]:
                        response = await request("GET", url + "/metrics", _retry=False, timeout=ClientTimeout(total=5))
                        async with response:
                            response.raise_for_status()
                            raw = await response.text()
                        transferred += sum(
                            float(line.rsplit(" ", 1)[1])
                            for line in raw.splitlines()
                            if line.startswith("vllm:nixl_bytes_transferred_sum{")
                        )
                    if transferred <= 0:
                        await asyncio.sleep(0.5)
        except TimeoutError as exc:
            raise RuntimeError("No actual NIXL KV bytes observed during routed inference smoke") from exc
        (run_dir / "inference-smoke.json").write_text(
            json.dumps(
                {
                    "http_status": 200,
                    "completion_tokens": body["usage"]["completion_tokens"],
                    "nixl_bytes": transferred,
                    "note": "One deterministic four-token probe before rollout dispatch; excluded from score and timing.",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--router", type=Path, required=True)
    build.add_argument("--node", action="append", required=True, help="hostname=IPv4")
    check = sub.add_parser("compare")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--expected", type=int, default=1500)
    probe = sub.add_parser("smoke")
    probe.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        asyncio.run(smoke(args.run_dir))
        return
    if args.command == "compare":
        result = compare(args.baseline, args.candidate, expected=args.expected)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    hosts = [tuple(item.split("=", 1)) for item in args.node]
    recipe_path = Path(os.environ["VLLM_CONFIG"])
    recipe = read_recipe(recipe_path)
    config = make_config(
        recipe,
        model=os.environ["MODEL"],
        model_name=os.environ.get("MODEL_NAME", os.environ["MODEL"]),
        image=Path(os.environ["CONTAINER"]),
        router=args.router,
        router_host=hosts[0][1],
        prefill_nodes=int(os.environ.get("NUM_PREFILL_NODES", 2)),
        decode_nodes=int(os.environ.get("NUM_DECODE_NODES", 2)),
        gpus_per_node=int(os.environ.get("GPUS_PER_NODE", 4)),
        cpus_per_node=int(os.environ.get("SLURM_CPUS_ON_NODE", 32)),
    )
    parity = check_parity(config, recipe, hosts, args.output)
    plan = deployment_plan(config, hosts, args.output, "metrics")
    endpoints = {}
    endpoint_groups = {}
    for i, node in enumerate(plan["nodes"]):
        for rank in node["ranks"]:
            # Keep original names for single-replica nodes/report compatibility.
            name = f"node{i}" if len(node["ranks"]) == 1 else f"node{i}_replica{rank['local_rank']}"
            port = parse_flags(rank["argv"][2:])["port"]
            endpoints[name] = f"http://{node['address']}:{port}/metrics"
            endpoint_groups.setdefault(node["role"], []).append(name)
    metrics_config = {
        "inference_metrics": {
            "enabled": True,
            "endpoints": endpoints,
            "router_endpoints": {"main": f"http://{hosts[0][1]}:{config.router.metrics_port}/metrics"},
            "endpoint_groups": endpoint_groups,
            "require_wandb": os.environ.get("WANDB_MODE") != "disabled",
        }
    }
    if os.environ.get("WANDB_API_KEY"):
        # Resolve inside Gym; never embed credentials in generated configs or process arguments.
        metrics_config["wandb_api_key"] = "${oc.env:WANDB_API_KEY}"
    for name, value in (
        ("inference-metrics.json", metrics_config),
        ("cluster-config.json", config.model_dump(mode="json")),
        ("setting-parity.json", parity),
        ("reference-recipe.json", recipe),
    ):
        path = args.output / name
        path.touch(mode=0o600, exist_ok=False)
        path.write_text(json.dumps(value, indent=2) + "\n")
    files = [
        recipe_path,
        Path(__file__),
        Path("benchmarks/inference_metrics.py"),
        Path("benchmarks/rollout_timing.py"),
        Path("nemo_gym/exporters/wandb.py"),
        Path(__file__).with_name("sbatch_cluster_vllm.sh"),
        *Path("responses_api_models/local_vllm_model").glob("*.py"),
    ]
    manifest = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "image": str(config.image),
        "image_size": config.image.stat().st_size,
        "image_mtime_ns": config.image.stat().st_mtime_ns,
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "launcher": "local_vllm_model.cluster",
        "router_profile": config.router.profile,
    }
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
