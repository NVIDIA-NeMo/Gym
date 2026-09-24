# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run real agent/resources smoke checks in CPU, Compose, GPU order.

Uses loopback HTTP between Gym services. mini-SWE routes model calls through
the local Gym model server.
"""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import uvicorn
from dotenv import dotenv_values
from omegaconf import OmegaConf

from nemo_gym import global_config, server_utils
from nemo_gym.base_responses_api_model import merge_model_call_capture_into_record
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.server_utils import BaseServerConfig, GlobalAIOHTTPAsyncClientConfig, ServerClient
from resources_servers.terminal_bench_4.app import TerminalBench4Config, TerminalBench4ResourcesServer
from responses_api_agents.miniswe_sandboxed_agent.app import MiniSWESandboxedAgent, MiniSWESandboxedConfig
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def load_baseline(paths):
    """Later health files replace earlier observations for the same harness/task."""
    baseline = {}
    for path in paths:
        for row in json.loads(path.read_text()):
            baseline[row["harness"], row["task"]] = bool(row["healthy"])
    return baseline


async def main(args):
    baseline = load_baseline(args.baseline_health)
    if args.env_file:
        for key, value in dotenv_values(args.env_file).items():
            if value and (key.startswith("OPENSANDBOX_") or key == "OPENAI_API_KEY"):
                os.environ[key] = value
    root = Path(__file__).parent
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = root.parents[1]
    source_paths = [
        source_root / path
        for path in [
            "nemo_gym/sandbox/api.py",
            "resources_servers/terminal_bench_4/models.py",
            "nemo_gym/sandbox/adapters/docker_compose.py",
            "nemo_gym/sandbox/providers/opensandbox/provider.py",
            "benchmarks/terminal_bench_4/resources.yaml",
            "benchmarks/terminal_bench_4/manifest.json",
            "benchmarks/terminal_bench_4/compose-images.json",
            "resources_servers/terminal_bench_4/compose_config.py",
            "resources_servers/terminal_bench_4/app.py",
            "resources_servers/terminal_bench_4/lifecycle.py",
            "resources_servers/terminal_bench_4/task.py",
            "resources_servers/terminal_bench_4/environment.py",
            "resources_servers/terminal_bench_4/shared_logs.py",
            "resources_servers/terminal_bench_4/transfers.py",
            "resources_servers/terminal_bench_4/collection.py",
            "resources_servers/terminal_bench_4/verifier.py",
            "responses_api_agents/miniswe_sandboxed_agent/app.py",
            "responses_api_agents/miniswe_sandboxed_agent/harness.py",
            "responses_api_agents/miniswe_sandboxed_agent/sandbox_runner.py",
            "responses_api_agents/miniswe_sandboxed_agent/mcp_client.py",
        ]
    ]
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "harness": args.harness,
                "model": args.model,
                "steps": args.steps,
                "agent_timeout_sec": args.agent_timeout,
                "category": args.category,
                "tasks": args.tasks,
                "excluded_tasks": args.exclude_tasks,
                "baseline_health": [str(path) for path in args.baseline_health],
                "source_sha256": {
                    str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in source_paths
                },
            },
            indent=2,
        )
    )
    config = OmegaConf.load(root / "resources.yaml")
    config.observability_enabled = True
    capture_dir = args.output / "model_calls"
    config.model_call_capture_dir = str(capture_dir)
    config.tb4_split_sandbox_endpoints = True
    config.tb4_agent_max_timeout_sec = args.agent_timeout
    config.tb4_jobs_dir = str(args.output / "resources")
    config.tb4_agent_artifacts_dir = str(args.output / "agent")
    config.tb4_max_steps = args.steps
    agent_name = "terminal_bench_4_" + args.harness
    profile = OmegaConf.load(root / f"{args.harness}.yaml")
    del profile["config_paths"]
    config = OmegaConf.merge(config, profile)
    ports = {name: free_port() for name in ["terminal_bench_4", agent_name, "policy_model"]}
    resource_config = config.terminal_bench_4.resources_servers.terminal_bench_4
    resource_config.task_download_dir = str(args.task_cache)
    resource_config.environment.sandbox_metadata["nemo-gym.nvidia.com/run"] = args.output.name
    agent_config = next(iter(config[agent_name].responses_api_agents.values()))
    model_config = {
        "host": "127.0.0.1",
        "port": ports["policy_model"],
        "name": "policy_model",
        "entrypoint": "app.py",
        "openai_base_url": args.model_url,
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "openai_model": args.model,
    }
    config.policy_model = {"responses_api_models": {"openai_model": model_config}}
    for name, values in [("terminal_bench_4", resource_config), (agent_name, agent_config)]:
        values.update({"name": name, "host": "127.0.0.1", "port": ports[name]})
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    global_config._GLOBAL_CONFIG_DICT = config
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)
    http = server_utils.set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    resource = TerminalBench4ResourcesServer(
        config=TerminalBench4Config.model_validate(OmegaConf.to_container(resource_config, resolve=True)),
        server_client=client,
    )
    agent = MiniSWESandboxedAgent(
        config=MiniSWESandboxedConfig.model_validate(OmegaConf.to_container(agent_config, resolve=True)),
        server_client=client,
    )
    model = SimpleModelServer(config=SimpleModelServerConfig.model_validate(model_config), server_client=client)
    servers = [
        uvicorn.Server(
            uvicorn.Config(instance.setup_webserver(), host="127.0.0.1", port=ports[name], log_level="warning")
        )
        for name, instance in [("terminal_bench_4", resource), (agent_name, agent), ("policy_model", model)]
    ]
    workers = [asyncio.create_task(s.serve()) for s in servers]
    while not all(s.started for s in servers):
        if any(t.done() for t in workers):
            raise RuntimeError("A smoke server failed to start")
        await asyncio.sleep(0.1)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    rollout_records = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def check(task):
        async with semaphore:
            started = datetime.now(timezone.utc).isoformat()
            print(f"START {args.harness} {task['category']} {task['name']}", flush=True)
            row = {
                "task": task["name"],
                "category": task["category"],
                "started_at": started,
                "harness": args.harness,
                "model": args.model,
            }
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
                    async with session.post(
                        f"http://127.0.0.1:{ports[agent_name]}/run",
                        json={
                            "task_id": f"terminal-bench/{task['name']}",
                            "task_name": "terminal-bench/" + task["name"],
                            "task_ref": task["ref"],
                            "dataset_ref": manifest["ref"],
                            "rollout_id": f"{args.output.name}-{task['name']}",
                            "_ng_rollout_id": f"{args.output.name}-{task['name']}",
                            "_ng_task_index": manifest["tasks"].index(task),
                            "_ng_rollout_index": 0,
                            "responses_create_params": {"input": [], "max_output_tokens": 16384},
                        },
                    ) as response:
                        text = await response.text()
                        (args.output / f"{task['name']}.json").write_text(text)
                        if response.status != 200:
                            raise RuntimeError(f"Agent HTTP {response.status}; see task response")
                        result = json.loads(text)
                        result.update(
                            _ng_rollout_id=f"{args.output.name}-{task['name']}",
                            _ng_task_index=manifest["tasks"].index(task),
                            _ng_rollout_index=0,
                        )
                        merge_model_call_capture_into_record(result, [capture_dir], include_payloads=True)
                        _attach_trajectory_record(result, result)
                        rollout_records.append(result)
                        agent_observed = bool((result.get("response") or {}).get("output"))
                        row |= {
                            "reward": result.get("reward"),
                            "evaluation_completed": result.get("evaluation_completed"),
                            "termination": result.get("termination"),
                            "infrastructure_error": result.get("infrastructure_error"),
                            "agent_observed": agent_observed,
                            "healthy": bool(
                                result.get("evaluation_completed")
                                and not result.get("infrastructure_error")
                                and agent_observed
                            ),
                        }
            except Exception as exc:
                row |= {"healthy": False, "error": str(exc)}
                (args.output / f"{task['name']}.error.txt").write_text(traceback.format_exc())
            row["baseline_healthy"] = baseline.get((args.harness, task["name"]))
            row["regression"] = not row["healthy"] and row["baseline_healthy"] is not False
            rows.append(row)
            (args.output / "health.json").write_text(json.dumps(rows, indent=2))
            print(f"END {task['name']} healthy={row['healthy']} reward={row.get('reward')}", flush=True)

    try:
        for category in ["cpu", "compose", "gpu"]:
            tasks = [
                t
                for t in manifest["tasks"]
                if t["category"] == category
                and (not args.tasks or t["name"] in args.tasks)
                and t["name"] not in args.exclude_tasks
                and (not args.category or args.category == category)
            ]
            await asyncio.gather(*(check(t) for t in tasks))
            if any(row["regression"] for row in rows if row["category"] == category):
                print(
                    f"Category {category} has new or unbaselined unhealthy tasks; inspect health.json before the next stage.",
                    flush=True,
                )
                break
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*workers, return_exceptions=True)
        await http.close()
        server_utils._GLOBAL_AIOHTTP_CLIENT = None
    if rollout_records:
        rollout_path = args.output / "evaluator_rollouts.jsonl"
        with rollout_path.open("w") as handle:
            for record in rollout_records:
                handle.write(json.dumps(record) + "\n")
        run_health_checks(rollout_path, output_dir=args.output, workers=1)
    return bool(rows) and not any(row["regression"] for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=["miniswe"], default="miniswe")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--exclude-tasks", nargs="*", default=[])
    parser.add_argument("--category", choices=["cpu", "compose", "gpu"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--baseline-health", type=Path, nargs="*", default=[])
    parser.add_argument("--task-cache", type=Path, default=Path("/tmp/tb4-packages"))
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--model-url", default="https://api.openai.com/v1")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--concurrency", type=int, default=3)
    raise SystemExit(0 if asyncio.run(main(parser.parse_args())) else 1)
