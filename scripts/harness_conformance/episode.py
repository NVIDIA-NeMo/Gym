# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One disposable local Gym episode, driven through the normal /run collector."""

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
from contextlib import ExitStack, nullcontext
from pathlib import Path

import uvicorn
import yaml

from nemo_gym.harness_capabilities.cli import digest_file

from .provider import Probe
from .scenarios import SCENARIOS


HARNESSES = {"opencode": "OpenCodeAgent", "pi": "PiAgent", "codex": "CodexAgent", "hermes": "HermesAgent"}


class _Server(uvicorn.Server):
    # The episode process owns cancellation; the embedded servers must not
    # independently install competing signal handlers.
    def capture_signals(self) -> nullcontext[None]:
        return nullcontext()


def check_runtime(harness: str) -> dict:
    """Require preinstalled runtimes so a probe cannot trigger auto-installation."""
    if harness == "hermes":
        if importlib.util.find_spec("run_agent") is None or importlib.util.find_spec("model_tools") is None:
            raise RuntimeError("install responses_api_agents/hermes_agent/requirements.txt in this Python environment")
        source = Path(importlib.util.find_spec("run_agent").origin)
        return {"source": str(source), "sha256": digest_file(source)}
    elif not shutil.which(harness):
        raise RuntimeError(f"install the {harness} runtime and put it on PATH before running probes")
    version = subprocess.run(
        [harness, "--version"], capture_output=True, text=True, errors="replace", timeout=10, check=True
    )
    return {"executable": shutil.which(harness), "version": version.stdout.strip()}


def _config(harness: str, directory: Path, ports: list[int], timeout: float) -> dict:
    def server(port: int) -> dict:
        return {"name": "unused", "host": "127.0.0.1", "port": port, "entrypoint": "app.py", "num_workers": 1}

    agent = {
        **server(ports[2]),
        "name": "probe_agent",
        "concurrency": 1,
        "model": "conformance-model",
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "resources_server": {"type": "resources_servers", "name": "probe_resources"},
    }
    defaults_path = (
        Path(__file__).resolve().parents[2] / f"responses_api_agents/{harness}_agent/configs/{harness}_agent.yaml"
    )
    defaults = yaml.safe_load(defaults_path.read_text())[f"{harness}_agent"]["responses_api_agents"][
        f"{harness}_agent"
    ]
    agent.update({key: value for key, value in defaults.items() if key.endswith("_version")})
    if harness in ("opencode", "pi"):
        agent.update(workspace_root=str(directory / "workspace"), timeout=timeout)
    if harness == "opencode":
        # Auxiliary title/summary requests would consume the scripted policy replies.
        agent["opencode_config"] = {
            "permission": {"bash": "allow"},
            "agent": {"title": {"disable": True}, "summary": {"disable": True}},
        }
    if harness == "codex":
        workspace = directory / "workspace"
        workspace.mkdir()
        agent.update(
            cwd=str(workspace),
            timeout=timeout,
            sandbox_mode="danger-full-access",
            extra_config={"features": {"shell_snapshot": False}, "web_search": "disabled"},
        )
    if harness == "hermes":
        agent.update(
            max_turns=6,
            enabled_toolsets=["terminal"],
            terminal_backend="local",
            terminal_timeout=10,
            compression_enabled=False,
            checkpoints_enabled=False,
            api_key="conformance",
        )
    return {
        "head_server": {"host": "127.0.0.1", "port": ports[0]},
        "observability_enabled": True,
        "model_call_capture_dir": str(directory / "capture"),
        "policy_model": {"responses_api_models": {"probe": {**server(ports[0]), "name": "policy_model"}}},
        "probe_resources": {"resources_servers": {"probe": {**server(ports[1]), "name": "probe_resources"}}},
        "probe_agent": {"responses_api_agents": {f"{harness}_agent": agent}},
        "probe_environment": {
            "environment_servers": {
                "legacy_agent": {
                    **server(ports[3]),
                    "name": "probe_environment",
                    "agent_server": {"type": "responses_api_agents", "name": "probe_agent"},
                }
            }
        },
    }


async def run_episode(*, harness: str, scenario_name: str, directory: Path, timeout: float) -> None:
    """Start fresh servers and let Gym write the rollout without modifying the harness."""
    runtime = check_runtime(harness)
    runtime["adapter_sha256"] = digest_file(
        Path(__file__).resolve().parents[2] / f"responses_api_agents/{harness}_agent/app.py"
    )
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    scenario = next(s for s in SCENARIOS if s.name == scenario_name)
    probe = Probe(scenario, directory)
    probe.save()
    tasks = []
    servers = []
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(4)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        config = _config(harness, directory, [s.getsockname()[1] for s in sockets], timeout)
        os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(config)
        (directory / "launch.json").write_text(json.dumps(config, indent=2) + "\n")
        # Import after the isolated process's config has been injected, like Gym's launcher.
        from environment_servers.legacy_agent.app import (
            LegacyAgentEnvironmentServer,
            LegacyAgentEnvironmentServerConfig,
        )
        from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
        from nemo_gym.server_utils import get_global_aiohttp_client, setup_server_client

        client = setup_server_client()
        module = importlib.import_module(f"responses_api_agents.{harness}_agent.app")
        agent_cls = getattr(module, HARNESSES[harness])
        config_cls = getattr(module, HARNESSES[harness] + "Config")
        agent = agent_cls(
            config=config_cls.model_validate(config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]),
            server_client=client,
        )
        environment = LegacyAgentEnvironmentServer(
            config=LegacyAgentEnvironmentServerConfig.model_validate(
                config["probe_environment"]["environment_servers"]["legacy_agent"]
            ),
            server_client=client,
        )
        try:
            apps = (probe.model_app(), probe.resources_app(), agent.setup_webserver(), environment.setup_webserver())
            for app, sock in zip(apps, sockets, strict=True):
                server = _Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=3))
                servers.append(server)
                tasks.append(asyncio.create_task(server.serve(sockets=[sock])))
            async with asyncio.timeout(10):
                while not all(server.started for server in servers):
                    for task in tasks:
                        if task.done():
                            task.result()
                            raise RuntimeError("probe server exited during startup")
                    await asyncio.sleep(0.01)
            inputs = directory / "requests.jsonl"
            inputs.write_text(json.dumps(scenario.task()) + "\n")
            async with asyncio.timeout(timeout + 10):
                await RolloutCollectionHelper().run_from_config(
                    RolloutCollectionConfig(
                        agent_name="probe_agent",
                        input_jsonl_fpath=str(inputs),
                        output_jsonl_fpath=str(directory / "rollouts.jsonl"),
                        num_samples_in_parallel=1,
                        num_repeats=1,
                        resume_from_cache=False,
                        disable_aggregation=True,
                        disable_health_check=True,
                        route_failures_to_sidecar=True,
                    )
                )
        finally:
            for server in servers:
                server.should_exit = True
            await asyncio.gather(*tasks, return_exceptions=True)
            await get_global_aiohttp_client().close()
            probe.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=HARNESSES)
    parser.add_argument("--scenario", required=True, choices=[s.name for s in SCENARIOS])
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()
    asyncio.run(
        run_episode(
            harness=args.harness, scenario_name=args.scenario, directory=args.directory.resolve(), timeout=args.timeout
        )
    )


if __name__ == "__main__":
    main()
