# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One disposable local Gym episode, driven through the EnvironmentServer collector."""

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from contextlib import AsyncExitStack, ExitStack, nullcontext
from pathlib import Path

import uvicorn
from omegaconf import DictConfig, OmegaConf

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.core import NeMoGymResponse
from nemo_gym.rollouts.harness_capabilities import digest_file

from .local_sandbox import prepare_sandbox
from .registry import HARNESSES
from .scenarios import SCENARIOS, Scenario


class _Server(uvicorn.Server):
    # The episode process owns cancellation; the embedded servers must not
    # independently install competing signal handlers.
    def capture_signals(self) -> nullcontext[None]:
        return nullcontext()


def check_runtime(harness: str, *, use_agent_sessions: bool = False) -> dict:
    """Record session runtime pins or require a preinstalled legacy runtime."""
    definition = HARNESSES[harness]
    if definition.requires_root and os.geteuid() != 0:
        raise RuntimeError(f"{harness} conformance requires a disposable Linux runner as root")
    for command in definition.required_commands:
        if not shutil.which(command):
            raise RuntimeError(f"{harness} conformance requires {command} on PATH")
    if definition.builtin:
        return {"installation": "builtin"}
    if use_agent_sessions and not definition.runtime_modules:
        root = Path(__file__).resolve().parents[2]
        return {
            "installation": "agent_session",
            **definition.runtime_metadata,
            **{key: digest_file(root / path) for key, path in definition.runtime_files.items()},
        }
    if definition.runtime_modules:
        modules = [importlib.util.find_spec(name) for name in definition.runtime_modules]
        if any(module is None for module in modules):
            raise RuntimeError(
                f"sync responses_api_agents/{harness}_agent with uv sync --locked in this isolated Python environment"
            )
        source = Path(modules[0].origin)
        return {"source": str(source), "sha256": digest_file(source)}
    if not shutil.which(harness):
        raise RuntimeError(f"install the {harness} runtime and put it on PATH before running probes")
    version = subprocess.run(
        [harness, "--version"], capture_output=True, text=True, errors="replace", timeout=10, check=True
    )
    return {"executable": shutil.which(harness), "version": version.stdout.strip()}


def _settings(harness: str, directory: Path, timeout: float, scenario: Scenario | None = None) -> DictConfig:
    return HARNESSES[harness].settings(directory, timeout, scenario)


def _config(
    harness: str,
    directory: Path,
    ports: dict[str, int],
    timeout: float,
    *,
    use_agent_sessions: bool = False,
    scenario: Scenario | None = None,
) -> dict:
    def server(port: int) -> dict:
        return {"name": "unused", "host": "127.0.0.1", "port": port, "entrypoint": "app.py", "num_workers": 1}

    defaults_path = (
        Path(__file__).resolve().parents[2] / f"responses_api_agents/{harness}_agent/configs/{harness}_agent.yaml"
    )
    defaults = OmegaConf.load(defaults_path)[f"{harness}_agent"]["responses_api_agents"][f"{harness}_agent"]
    settings = _settings(harness, directory, timeout, scenario)
    for module_name in settings.get("required_imports", []):
        importlib.import_module(module_name)
    mode = settings["session" if use_agent_sessions else "legacy"]
    agent = OmegaConf.merge(
        defaults if settings.get("inherit_adapter_defaults") else {},
        {key: defaults[key] for key in settings.get("inherit_adapter_fields", [])},
        server(ports["probe_agent"]),
        {
            "name": "probe_agent",
            "concurrency": 1,
            "model": "conformance-model",
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        },
        {key: defaults[key] for key in defaults if key.endswith("_version")},
        settings.get("harness_overrides", {}),
    )
    for field in [*settings.get("pop_fields", []), *mode.get("pop_fields", [])]:
        agent.pop(field, None)
    agent = OmegaConf.merge(agent, mode.get("harness_overrides") or {})
    if mode.get("requires_workspace"):
        (directory / "workspace").mkdir()
    environment_type = "single_agent_turn_legacy" if use_agent_sessions else "legacy_agent"
    environment = OmegaConf.merge(
        server(ports["probe_environment"]),
        {"name": "probe_environment", "agent_server": {"type": "responses_api_agents", "name": "probe_agent"}},
    )
    if use_agent_sessions:
        environment = OmegaConf.merge(
            environment,
            {
                "resources_server": {"type": "resources_servers", "name": "probe_resources"},
                "resources_tool_transports": mode["tool_transports"],
                "default_episode_timeout_seconds": timeout,
                "cleanup_timeout_seconds": 10,
            },
        )
    config = {
        "head_server": {"host": "127.0.0.1", "port": ports["policy_model"]},
        "observability_enabled": True,
        "model_call_capture_dir": str(directory / "capture"),
        "policy_model": {"responses_api_models": {"probe": {**server(ports["policy_model"]), "name": "policy_model"}}},
        "probe_resources": {
            "resources_servers": {"probe": {**server(ports["probe_resources"]), "name": "probe_resources"}}
        },
        "probe_agent": {"responses_api_agents": {f"{harness}_agent": agent}},
        "probe_environment": {"environment_servers": {environment_type: environment}},
    }
    if "judge_model" in ports:
        config["judge_model"] = {
            "responses_api_models": {"probe": {**server(ports["judge_model"]), "name": "judge_model"}}
        }
    if use_agent_sessions and mode["sandbox_backend"]:
        config["probe_sandbox"] = {mode["sandbox_backend"]: {"concurrency": 1}}
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True, throw_on_missing=True)


async def run_episode(*, harness: str, scenario_name: str, directory: Path, timeout: float) -> None:
    """Start fresh servers and let Gym write the rollout without modifying the harness."""
    module = importlib.import_module(f"responses_api_agents.{harness}_agent.app")
    agent_cls = getattr(module, HARNESSES[harness].agent)
    config_cls = agent_cls.model_fields["config"].annotation
    use_agent_sessions = agent_cls.supports_agent_sessions()
    if use_agent_sessions and HARNESSES[harness].session_requires_linux and sys.platform != "linux":
        raise RuntimeError(
            "Conformance probes using agent sessions require Linux; run the probes on a Linux host or in a container"
        )
    runtime = check_runtime(harness, use_agent_sessions=use_agent_sessions)
    runtime["agent_interface"] = "agent_sessions" if use_agent_sessions else "legacy_run"
    runtime["adapter_sha256"] = digest_file(
        Path(__file__).resolve().parents[2] / f"responses_api_agents/{harness}_agent/app.py"
    )
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    scenario = next(s for s in SCENARIOS if s.name == scenario_name)
    settings = _settings(harness, directory, timeout, scenario)
    probe_module, probe_name = settings.probe.split(":")
    probe_cls = getattr(importlib.import_module(probe_module, package=__package__), probe_name)
    probe = probe_cls(scenario, directory, **settings.get("probe_overrides", {}))
    probe.save()
    tasks = []
    servers = []
    with ExitStack() as stack:
        names = ["policy_model", "probe_resources", "probe_agent", "probe_environment"]
        if scenario.judge_verifier:
            names.append("judge_model")
        sockets = {name: stack.enter_context(socket.socket()) for name in names}
        for sock in sockets.values():
            sock.bind(("127.0.0.1", 0))
        config = _config(
            harness,
            directory,
            {name: sock.getsockname()[1] for name, sock in sockets.items()},
            timeout,
            use_agent_sessions=use_agent_sessions,
            scenario=scenario,
        )
        os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(config)
        (directory / "launch.json").write_text(json.dumps(config, indent=2) + "\n")
        if use_agent_sessions:
            from environment_servers.single_agent_turn.app import (
                SessionHandles,
                SingleAgentTurnEnvironmentServerConfig,
            )
            from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
            from nemo_gym.server_runtime import SingleAgentTurnRequest

            class ObservedEnvironment(SingleAgentTurnLegacyEnvironmentServer):
                async def _invoke_agent(
                    self, request: SingleAgentTurnRequest, handles: SessionHandles
                ) -> NeMoGymResponse:
                    response = await super()._invoke_agent(request, handles)
                    # Witness what actually reached the caller, before verification or masking.
                    probe.agent_responses.append(
                        response.model_dump(
                            mode="json",
                            include={"status": True, "error": True, "metadata": {AGENT_OUTCOME_METADATA_KEY}},
                        )
                    )
                    probe.save()
                    return response

            environment_cls = ObservedEnvironment
            environment_config_cls = SingleAgentTurnEnvironmentServerConfig
        else:
            from environment_servers.legacy_agent.app import (
                LegacyAgentEnvironmentServer,
                LegacyAgentEnvironmentServerConfig,
            )

            environment_cls = LegacyAgentEnvironmentServer
            environment_config_cls = LegacyAgentEnvironmentServerConfig
        from nemo_gym.client import get_global_aiohttp_client, setup_server_client
        from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper

        client = setup_server_client()
        agent = agent_cls(
            config=config_cls.model_validate(config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]),
            server_client=client,
        )
        if use_agent_sessions:
            runtime["requested_versions"] = {
                key: value for key, value in agent.config.model_dump().items() if key.endswith("_version")
            }
            (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        environment = environment_cls(
            config=environment_config_cls.model_validate(
                next(iter(config["probe_environment"]["environment_servers"].values()))
            ),
            server_client=client,
        )
        sandbox_stack = AsyncExitStack()
        try:
            sandbox_access = await sandbox_stack.enter_async_context(prepare_sandbox(directory, settings))
            apps = {
                "policy_model": probe.model_app(
                    assistant_message_header=client.assistant_message_header("policy_model")
                ),
                "probe_resources": probe.resources_app(server_client=client, sandbox_access=sandbox_access),
                "probe_agent": agent.setup_webserver(),
                "probe_environment": environment.setup_webserver(),
            }
            if scenario.judge_verifier:
                apps["judge_model"] = probe.judge_app()
            for name, app in apps.items():
                server = _Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=3))
                servers.append(server)
                tasks.append(asyncio.create_task(server.serve(sockets=[sockets[name]])))
            async with asyncio.timeout(10):
                while not all(server.started for server in servers):
                    for task in tasks:
                        if task.done():
                            task.result()
                            raise RuntimeError("probe server exited during startup")
                    await asyncio.sleep(0.01)
            inputs = directory / "requests.jsonl"
            task = OmegaConf.to_container(
                OmegaConf.merge(scenario.task(), settings.get("task_overrides", {})), resolve=True
            )
            inputs.write_text(json.dumps(task) + "\n")
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
            try:
                await sandbox_stack.aclose()
            finally:
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
