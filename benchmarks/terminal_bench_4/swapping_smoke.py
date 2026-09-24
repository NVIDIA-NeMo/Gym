# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test interchangeable terminal harnesses through real episode HTTP servers."""

import argparse
import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path

import aiohttp
import uvicorn
from dotenv import dotenv_values
from omegaconf import OmegaConf

from benchmarks.terminal_bench_4.smoke import free_port
from environment_servers.single_agent.app import SingleAgentEnvironmentServer, SingleAgentEnvironmentServerConfig
from nemo_gym import global_config, server_utils
from nemo_gym.server_utils import BaseServerConfig, GlobalAIOHTTPAsyncClientConfig, ServerClient
from resources_servers.swebench_pro.app import SWEBenchProResourcesServer, SWEBenchProResourcesServerConfig
from resources_servers.terminal_bench_4.episode import (
    TerminalBench4EpisodeConfig,
    TerminalBench4EpisodeResourcesServer,
)
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig
from responses_api_agents.miniswe_sandboxed_agent.app import MiniSWESandboxedConfig
from responses_api_agents.miniswe_sandboxed_agent.episode import MiniSWEEpisodeAgent
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


async def main(args: argparse.Namespace) -> None:
    values = dotenv_values(args.env_file) if args.env_file else {}
    for key in (
        "OPENSANDBOX_DOMAIN",
        "OPENSANDBOX_API_KEY",
        "OPENAI_API_KEY",
        "OPENSANDBOX_DOMAIN_CPU",
        "OPENSANDBOX_API_KEY_CPU",
        "OPENSANDBOX_DOMAIN_GPU",
        "OPENSANDBOX_API_KEY_GPU",
    ):
        if values.get(key):
            os.environ[key] = values[key]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ports = {name: free_port() for name in ("resources", "agent", "policy_model", "environment")}
    root = Path(__file__).resolve().parents[2]
    base = OmegaConf.load(root / "benchmarks/terminal_bench_4/resources.yaml")
    env = OmegaConf.to_container(base.terminal_bench_4.resources_servers.terminal_bench_4.environment, resolve=True)
    provider = env["sandbox_provider"]
    provider["opensandbox"]["connection"]["api_key"] = os.environ["OPENSANDBOX_API_KEY"]
    provider["opensandbox"]["operations"]["background_exec"] = True
    providers = {"sandbox": provider}
    if args.split_endpoints:
        env["sandbox_split_endpoints"] = True
        for pool in ("cpu", "gpu"):
            selected = deepcopy(provider)
            selected["opensandbox"]["connection"].update(
                domain=os.environ["OPENSANDBOX_DOMAIN_" + pool.upper()],
                api_key=os.environ["OPENSANDBOX_API_KEY_" + pool.upper()],
            )
            providers["sandbox_" + pool] = selected

    def common(name: str) -> dict:
        return dict(name=name, host="127.0.0.1", port=ports[name], entrypoint="app.py")

    model_config = common("policy_model") | dict(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=args.model or ("gpt-4.1" if args.pair == "tb4-hermes" else "gpt-5.4-mini-2026-03-17"),
    )
    model_ref = dict(type="responses_api_models", name="policy_model")
    resource_ref = dict(type="resources_servers", name="resources")
    if args.pair.startswith("tb4-"):
        resource_config = common("resources") | dict(
            environment=env,
            sandbox_provider_ref="sandbox",
            sandbox_provider_ref_cpu="sandbox_cpu" if args.split_endpoints else None,
            sandbox_provider_ref_gpu="sandbox_gpu" if args.split_endpoints else None,
            artifacts_dir=str(output / "resources"),
            task_download_dir=str(args.task_cache),
        )
        resource_cls, resource_schema = TerminalBench4EpisodeResourcesServer, TerminalBench4EpisodeConfig
        agent_config = common("agent") | dict(
            resources_server=resource_ref,
            model_server=model_ref,
            model=model_config["openai_model"],
            max_turns=args.steps,
            max_tokens=4096,
            enabled_toolsets=["terminal"],
            chat_template_kwargs_enabled=False,
            temperature=1,
            compression_enabled=False,
        )
        agent_cls, agent_schema = HermesAgent, HermesAgentConfig
        if args.pair == "tb4-miniswe":
            agent_config = common("agent") | dict(
                model_server=model_ref,
                harness=dict(step_limit=args.steps, step_timeout_sec=30),
                agent_max_timeout_sec=args.agent_timeout,
                artifacts_dir=str(output / "agent"),
            )
            agent_cls, agent_schema = MiniSWEEpisodeAgent, MiniSWESandboxedConfig
        manifest = json.loads((root / "benchmarks/terminal_bench_4/manifest.json").read_text())
        task = next(t for t in manifest["tasks"] if t["name"] == args.task)
        task_id = "terminal-bench/" + task["name"]
        data = dict(task_name=task_id, task_ref=task["ref"], dataset_ref=manifest["ref"])
        params = dict(input=[], max_output_tokens=4096)
    else:
        config = OmegaConf.load(root / "resources_servers/swebench_pro/configs/swebench_pro_resources_server.yaml")
        resource_config = OmegaConf.to_container(
            config.swebench_pro_resources_server.resources_servers.swebench_pro, resolve=True
        )
        resource_config.update(common("resources"))
        resource_config.update(
            verification_total_timeout=900,
            verification_attempt_timeout=800,
            evaluation_timeout=600,
            inconclusive_verification_retries=0,
        )
        resource_cls, resource_schema = SWEBenchProResourcesServer, SWEBenchProResourcesServerConfig
        agent_config = common("agent") | dict(
            model_server=model_ref,
            harness=dict(step_limit=args.steps, step_timeout_sec=30),
            agent_max_timeout_sec=180,
            artifacts_dir=str(output / "agent"),
        )
        agent_cls, agent_schema = MiniSWEEpisodeAgent, MiniSWESandboxedConfig
        raw = json.loads(args.swe_row.read_text().splitlines()[0])
        data = raw.get("sample", raw)
        data = {key: value for key, value in data.items() if key not in {"task_source", "agent_ref"}}
        params = data.pop("responses_create_params")
        params["max_output_tokens"] = 4096
        task_id = data["instance_id"]
    config = OmegaConf.create(
        dict(
            **providers,
            observability_enabled=True,
            model_call_capture_dir=str(output / "model_calls"),
            resources={
                "resources_servers": {
                    "terminal_bench_4" if args.pair.startswith("tb4-") else "swebench_pro": resource_config
                }
            },
            agent={
                "responses_api_agents": {
                    "hermes_agent" if args.pair == "tb4-hermes" else "miniswe_sandboxed_agent": agent_config
                }
            },
            policy_model={"responses_api_models": {"openai_model": model_config}},
        )
    )
    global_config._GLOBAL_CONFIG_DICT = config
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)
    http = server_utils.set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    resources = resource_cls(config=resource_schema.model_validate(resource_config), server_client=client)
    agent = agent_cls(config=agent_schema.model_validate(agent_config), server_client=client)
    model = SimpleModelServer(config=SimpleModelServerConfig.model_validate(model_config), server_client=client)
    environment = SingleAgentEnvironmentServer(
        config=SingleAgentEnvironmentServerConfig(
            **common("environment"),
            resources_server=resource_ref,
            agent_server=dict(type="responses_api_agents", name="agent"),
            default_episode_timeout_seconds=2400,
            cleanup_timeout_seconds=180,
        ),
        server_client=client,
    )
    instances = dict(resources=resources, agent=agent, policy_model=model, environment=environment)
    servers = [
        uvicorn.Server(
            uvicorn.Config(instance.setup_webserver(), host="127.0.0.1", port=ports[name], log_level="warning")
        )
        for name, instance in instances.items()
    ]
    workers = [asyncio.create_task(server.serve()) for server in servers]
    try:
        while not all(server.started for server in servers):
            if any(worker.done() for worker in workers):
                raise RuntimeError("Smoke server failed to start")
            await asyncio.sleep(0.1)
        request = dict(
            episode_id=dict(rollout_id=output.name),
            task=dict(
                task_id=dict(taskset=args.pair, task_id=task_id),
                task_input=dict(task_data=data, responses_create_params=params),
            ),
        )
        (output / "request.json").write_text(json.dumps(request, indent=2))
        print("START", args.pair, task_id, flush=True)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2600)) as session:
            async with session.post(f"http://127.0.0.1:{ports['environment']}/run", json=request) as response:
                raw = await response.text()
                (output / "episode.json").write_text(raw)
                response.raise_for_status()
        result = json.loads(raw)
        verification = (result.get("result") or {}).get("verification")
        print(
            "RESULT",
            json.dumps(
                dict(
                    failure=result.get("failure"),
                    reward=verification.get("reward") if verification else None,
                    evaluation_completed=verification.get("evaluation_completed") if verification else None,
                )
            ),
            flush=True,
        )
        if result.get("failure") or not verification or not verification.get("evaluation_completed"):
            raise RuntimeError("Smoke did not complete official verification")
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*workers, return_exceptions=True)
        await http.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["tb4-hermes", "tb4-miniswe", "swepro-miniswe"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--swe-row", type=Path, help="Prepared public SWE Pro JSONL; runs its first row")
    parser.add_argument("--task-cache", type=Path, default=Path("cache/tb4-tasks"))
    parser.add_argument(
        "--model", help="Hosted OpenAI model; defaults to GPT-4.1 for Hermes and GPT-5.4-mini for mini-SWE"
    )
    parser.add_argument("--task", default="interleaved-vigenere", help="Pinned TB4 task name")
    parser.add_argument("--split-endpoints", action="store_true", help="Use CPU/GPU provider environment variables")
    parser.add_argument("--agent-timeout", type=float, default=180)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.pair == "swepro-miniswe" and args.swe_row is None:
        parser.error("--swe-row is required for swepro-miniswe")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
