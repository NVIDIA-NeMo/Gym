# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test interchangeable terminal harnesses through real episode HTTP servers."""

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import time
from copy import deepcopy
from functools import wraps
from pathlib import Path

import aiohttp
import uvicorn
from dotenv import dotenv_values
from omegaconf import OmegaConf

from benchmarks.terminal_bench_4.smoke import free_port
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
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
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


def audited_classes(resource_cls, agent_cls, output):
    """Persist lifecycle evidence without changing the harness or benchmark requests."""
    events = []

    def record(event, **fields):
        events.append(dict(event=event, time=time.time(), **fields))
        (output / "lifecycle.json").write_text(json.dumps(events, indent=2, default=str))

    @wraps(agent_cls.seed_agent_session)
    async def seed_agent(self, request, body):
        record("agent_seed_start", session_id=body.agent_session_id)
        result = await agent_cls.seed_agent_session(self, request, body)
        record(
            "agent_seed_complete",
            session_id=result.agent_session_id,
            user=body.agent_context.user if body.agent_context else None,
            workdir=body.sandbox_access.workdir if body.sandbox_access else None,
        )
        return result

    @wraps(agent_cls.responses)
    async def activate(self, request, body):
        record("agent_activation_start")
        result = await agent_cls.responses(self, request, body)
        record("agent_activation_complete", status=result.status, metadata=result.metadata)
        return result

    @wraps(agent_cls.close_agent_session)
    async def close_agent(self, request, body):
        states = getattr(self, "_native_sessions", None)
        if states is None:
            states = self._agent_sessions
        state = states.get(body.agent_session_id)
        result = await agent_cls.close_agent_session(self, request, body)
        evidence = {}
        if state is not None:
            if hasattr(state, "harness"):
                evidence = dict(cleanup_confirmed=state.harness.cleanup_confirmed, disposed=state.harness.disposed)
                receipt = state.harness.directory / "cleanup.json"
                if receipt.exists():
                    evidence["receipt"] = json.loads(receipt.read_text())
                assert evidence["cleanup_confirmed"] and evidence["disposed"]
            elif hasattr(state, "runner_cleanup"):
                evidence = dict(runner_cleanup=str(state.runner_cleanup), phase=str(state.phase))
            else:
                evidence = dict(closed=state.closed, runner_closed=state.runner_closed)
                assert state.closed and state.runner_closed
        record("agent_close_complete", session_id=result.agent_session_id, **evidence)
        return result

    @wraps(resource_cls.verify)
    async def verify(self, request, body):
        assert any(e["event"] == "agent_close_complete" for e in events), "Verification before agent cleanup"
        record("verification_start")
        result = await resource_cls.verify(self, request, body)
        record("verification_complete", reward=result.reward, evaluation_completed=result.evaluation_completed)
        return result

    @wraps(resource_cls.close_resources_session)
    async def close_resources(self, request, body):
        result = await resource_cls.close_resources_session(self, request, body)
        resources = [
            env
            for session in self._sessions.values()
            for env in (session.environment, session.verifier_environment, session.shared_logs)
            if env is not None
        ]
        assert all(env.closed and not env.cleanup_errors for env in resources), "Resources remain after close"
        record("resources_close_complete", closed_environments=len(resources))
        return result

    return (
        type("AuditedResources", (resource_cls,), {"verify": verify, "close_resources_session": close_resources}),
        type(
            "AuditedAgent",
            (agent_cls,),
            {"seed_agent_session": seed_agent, "responses": activate, "close_agent_session": close_agent},
        ),
    )


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
    if args.split_endpoints:
        os.environ.setdefault("OPENSANDBOX_DOMAIN", os.environ["OPENSANDBOX_DOMAIN_CPU"])
        os.environ.setdefault("OPENSANDBOX_API_KEY", os.environ["OPENSANDBOX_API_KEY_CPU"])
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
        return dict(name=name, host=args.host, port=ports[name], entrypoint="app.py")

    model_config = common("policy_model") | dict(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=args.model
        or ("gpt-4.1" if args.pair in {"tb4-hermes", "tb4-opencode"} else "gpt-5.4-mini-2026-03-17"),
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
            sandbox_runner_timeout_seconds=args.agent_timeout,
        )
        agent_cls, agent_schema = HermesAgent, HermesAgentConfig
        if args.pair == "tb4-miniswe":
            agent_config = common("agent") | dict(
                model_server=model_ref,
                resources_server=resource_ref,
                harness=dict(step_limit=args.steps, step_timeout_sec=30),
                agent_max_timeout_sec=args.agent_timeout,
                artifacts_dir=str(output / "agent"),
            )
            agent_cls, agent_schema = MiniSWEEpisodeAgent, MiniSWESandboxedConfig
        if args.pair == "tb4-opencode":
            agent_config = common("agent") | dict(
                execution_mode="sandbox",
                model_server=model_ref,
                opencode_version="1.17.11",
                timeout=args.agent_timeout,
                max_steps=args.steps or None,
                max_output_tokens=4096,
            )
            agent_cls, agent_schema = OpenCodeAgent, OpenCodeAgentConfig
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
                    (
                        "hermes_agent"
                        if args.pair == "tb4-hermes"
                        else "opencode_agent"
                        if args.pair == "tb4-opencode"
                        else "miniswe_sandboxed_agent"
                    ): agent_config
                }
            },
            policy_model={"responses_api_models": {"openai_model": model_config}},
        )
    )
    source_files = {}
    for cls in (resource_cls, agent_cls, SingleAgentTurnEnvironmentServer, SimpleModelServer):
        source = Path(inspect.getfile(cls)).resolve()
        if not source.is_relative_to(root.resolve()):
            raise RuntimeError(f"Mixed Gym installation: {cls.__name__} loaded from {source}, expected {root}")
        source_files[str(source.relative_to(root.resolve()))] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source.json").write_text(json.dumps(source_files, indent=2))
    global_config._GLOBAL_CONFIG_DICT = config
    rpc_events = []

    class SmokeClient(ServerClient):
        async def post(self, server_name, url_path, **kwargs):
            rpc_events.append(dict(server=server_name, path=url_path, event="start", time=time.time()))
            (output / "rpc.json").write_text(json.dumps(rpc_events, indent=2))
            response = await super().post(server_name=server_name, url_path=url_path, **kwargs)
            rpc_events.append(
                dict(server=server_name, path=url_path, event="response", status=response.status, time=time.time())
            )
            (output / "rpc.json").write_text(json.dumps(rpc_events, indent=2))
            return response

    client = SmokeClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)
    http = server_utils.set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    if args.pair.startswith("tb4-"):
        resource_cls, agent_cls = audited_classes(resource_cls, agent_cls, output)
    resources = resource_cls(config=resource_schema.model_validate(resource_config), server_client=client)
    agent = agent_cls(config=agent_schema.model_validate(agent_config), server_client=client)
    model = SimpleModelServer(config=SimpleModelServerConfig.model_validate(model_config), server_client=client)
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
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
            uvicorn.Config(instance.setup_webserver(), host=args.host, port=ports[name], log_level="warning")
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
            async with session.post(f"http://{args.host}:{ports['environment']}/run", json=request) as response:
                raw = await response.text()
                (output / "episode.json").write_text(raw)
                response.raise_for_status()
        result = json.loads(raw)
        if args.pair.startswith("tb4-"):
            lifecycle = json.loads((output / "lifecycle.json").read_text())
            if not any(event["event"] == "resources_close_complete" for event in lifecycle):
                raise RuntimeError("Smoke did not confirm resource cleanup")
        verification = result.get("result")
        captures = [
            json.loads(line)
            for path in (output / "model_calls").glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        if not captures or any(call.get("status_code") != 200 for call in captures):
            raise RuntimeError("Smoke did not confirm successful captured model calls")
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
        if (
            result.get("failure")
            or not verification
            or not verification.get("evaluation_completed")
            or verification.get("response", {}).get("status") == "failed"
        ):
            raise RuntimeError("Smoke did not complete official verification")
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*workers, return_exceptions=True)
        await http.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", choices=["tb4-hermes", "tb4-miniswe", "tb4-opencode", "swepro-miniswe"], required=True
    )
    parser.add_argument("--host", default="127.0.0.1", help="Gym address reachable from task sandboxes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--swe-row", type=Path, help="Prepared public SWE Pro JSONL; runs its first row")
    parser.add_argument("--task-cache", type=Path, default=Path("cache/tb4-tasks"))
    parser.add_argument(
        "--model", help="Hosted OpenAI model; defaults to GPT-4.1 for Hermes/OpenCode and GPT-5.4-mini for mini-SWE"
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
