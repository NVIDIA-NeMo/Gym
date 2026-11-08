# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One disposable local Gym episode, driven through the EnvironmentServer collector."""

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, Request
from omegaconf import OmegaConf

from nemo_gym import component_search_roots
from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, SimpleResponsesAPIAgent
from nemo_gym.config import MODEL_SAMPLING_KEY_NAME, GlobalConfigDictParser
from nemo_gym.core import ModelCallSampling, NeMoGymResponse
from nemo_gym.rollouts.harness_capabilities import digest_file
from nemo_gym.server_runtime import CallerSamplingMiddleware, ModelSamplingMiddleware

from .provider import OpenClawProbe, Probe
from .registry import HARNESSES as HARNESSES
from .registry import PROBE_HARNESSES
from .scenarios import SCENARIOS


# These identify the disposable local servers and workspace, not harness behavior.
_LOCAL_AGENT_FIELDS = (
    "name",
    "host",
    "port",
    "entrypoint",
    "num_workers",
    "concurrency",
    "model_server",
    "resources_server",
    "sandbox_provider",
    "cwd",
    "workspace_root",
    "api_key",
    "openai_api_key",
    "openai_base_url",
    "anthropic_api_key",
    "anthropic_base_url",
)


def _resolved_value(value: object) -> object:
    return OmegaConf.to_container(value, resolve=True, throw_on_missing=True) if OmegaConf.is_config(value) else value


def prepared_settings(
    *,
    harness: str,
    config_path: Path | None,
    agent_name: str | None,
    sampling_path: Path | None,
    policy_model_name: str | None = None,
) -> dict | None:
    """Select and freeze prepared inputs without retaining unrelated provider configuration."""
    if (config_path is None) != (agent_name is None):
        raise ValueError("--prepared-config and --agent-name must be supplied together")
    if policy_model_name is not None and (config_path is None or not policy_model_name.strip()):
        raise ValueError("--policy-model-name requires --prepared-config and a nonempty served alias")
    if config_path is None and sampling_path is None:
        return None
    result = {"sources": {}}
    if config_path is not None:
        config_path = config_path.resolve()
        source = OmegaConf.load(config_path)
        gym = OmegaConf.select(source, "env.nemo_gym", default=source)
        paths, includes = GlobalConfigDictParser().load_extra_config_paths(list(gym.get("config_paths") or []))
        # Keep outer RL fields available to selected references such as
        # ${policy.generation.max_new_tokens}, without resolving or retaining them.
        merged = OmegaConf.merge({}, source if gym is not source else {}, *includes, gym)
        if policy_model_name is not None:
            configured = OmegaConf.select(merged, "policy_model_name", throw_on_resolution_failure=False)
            if configured is not None and configured != policy_model_name:
                raise ValueError("--policy-model-name conflicts with the configured policy_model_name")
            merged["policy_model_name"] = policy_model_name
            result["policy_model_name"] = policy_model_name
        adapters = merged[agent_name]["responses_api_agents"]
        adapter = f"{harness}_agent"
        if set(adapters) != {adapter}:
            raise ValueError(f"Prepared agent {agent_name!r} must select only {adapter}")
        selected = adapters[adapter]
        original = OmegaConf.to_container(selected, resolve=False)
        result.update(
            agent_name=agent_name,
            harness=harness,
            selected_agent_sha256=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
            local_fields=[*_LOCAL_AGENT_FIELDS, "sandbox_config.workdir"],
        )
        # Resolve retained fields while their original siblings still exist. Do not
        # read discarded wiring: it can contain unrelated credential interpolations.
        result["agent"] = {}
        for name in selected:
            if name in _LOCAL_AGENT_FIELDS:
                continue
            value = selected[name]
            if name == "sandbox_config" and OmegaConf.is_dict(value):
                result["agent"][name] = {key: _resolved_value(value[key]) for key in value if key != "workdir"}
            else:
                result["agent"][name] = _resolved_value(value)
        result["referenced_files"] = {}
        for field in ("model_catalog_json", "settings", "settings_file"):
            value = result["agent"].get(field)
            if isinstance(value, str) and value and not (field == "settings" and value.lstrip().startswith("{")):
                path = Path(value).expanduser().resolve()
                result["referenced_files"][str(path)] = digest_file(path)
                result["agent"][field] = str(path)
        for path in [config_path, *map(Path, paths)]:
            resolved = (
                path
                if path.is_absolute()
                else next(root / path for root in component_search_roots() if (root / path).is_file())
            )
            result["sources"][str(resolved.resolve())] = digest_file(resolved)
    if sampling_path is not None:
        result["model_sampling"] = ModelCallSampling.model_validate(
            yaml.safe_load(sampling_path.read_text())
        ).model_dump(exclude_none=True)
        result["sources"][str(sampling_path.resolve())] = digest_file(sampling_path)
    return result


def _install_sampling(app: FastAPI, directory: Path, *, name: str, required: bool) -> None:
    app.add_middleware(ModelSamplingMiddleware, required=required)

    @app.middleware("http")
    async def witness_sampling_path(request: Request, call_next):
        response = await call_next(request)
        with (directory / "sampling_requests.jsonl").open("a") as output:
            output.write(json.dumps({"server": name, "path": request.url.path, "status": response.status_code}) + "\n")
        return response


class _Server(uvicorn.Server):
    # The episode process owns cancellation; the embedded servers must not
    # independently install competing signal handlers.
    def capture_signals(self) -> nullcontext[None]:
        return nullcontext()


def check_runtime(harness: str, *, use_agent_sessions: bool = False) -> dict:
    """Record session runtime pins or require a preinstalled legacy runtime."""
    if harness == "simple":
        return {"installation": "builtin"}
    if use_agent_sessions and harness != "hermes":
        runtime = {"installation": "agent_session"}
        if harness == "nooa":
            adapter = Path(__file__).resolve().parents[2] / "responses_api_agents/nooa_agent"
            runtime.update(
                pyproject_sha256=digest_file(adapter / "pyproject.toml"),
                lock_sha256=digest_file(adapter / "uv.lock"),
                bootstrap_sha256=digest_file(adapter / "sandbox_runtime.py"),
                sandbox_ownership="resources",
                execution_mode="sandboxed",
                observer_sha256=digest_file(Path(__file__).with_name("nooa_observer.py")),
            )
        return runtime
    if harness == "hermes":
        if importlib.util.find_spec("run_agent") is None or importlib.util.find_spec("model_tools") is None:
            raise RuntimeError(
                "sync responses_api_agents/hermes_agent with uv sync --locked in this isolated Python environment"
            )
        source = Path(importlib.util.find_spec("run_agent").origin)
        return {"source": str(source), "sha256": digest_file(source)}
    elif not shutil.which(harness):
        raise RuntimeError(f"install the {harness} runtime and put it on PATH before running probes")
    version = subprocess.run(
        [harness, "--version"], capture_output=True, text=True, errors="replace", timeout=10, check=True
    )
    return {"executable": shutil.which(harness), "version": version.stdout.strip()}


def _supports_agent_sessions(agent_cls: type[SimpleResponsesAPIAgent]) -> bool:
    # Every SimpleResponsesAPIAgent exposes session routes, including legacy agents
    # whose handlers raise NotImplementedError. Check implementations, not route presence.
    return all(
        getattr(agent_cls, handler) is not getattr(SimpleResponsesAPIAgent, handler)
        or getattr(agent_cls, hook) is not getattr(SimpleResponsesAPIAgent, hook)
        for handler, hook in (
            ("seed_agent_session", "_seed_agent_session_state"),
            ("close_agent_session", "_close_agent_session_state"),
        )
    )


def _config(
    harness: str,
    directory: Path,
    ports: dict[str, int],
    timeout: float,
    *,
    use_agent_sessions: bool = False,
    prepared_agent: dict | None = None,
) -> dict:
    def server(port: int) -> dict:
        return {"name": "unused", "host": "127.0.0.1", "port": port, "entrypoint": "app.py", "num_workers": 1}

    agent = {
        **server(ports["probe_agent"]),
        "name": "probe_agent",
        "concurrency": 1,
        "model": "conformance-model",
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
    }
    if not use_agent_sessions:
        agent["resources_server"] = {"type": "resources_servers", "name": "probe_resources"}
    defaults_path = (
        Path(__file__).resolve().parents[2] / f"responses_api_agents/{harness}_agent/configs/{harness}_agent.yaml"
    )
    defaults = yaml.safe_load(defaults_path.read_text())[f"{harness}_agent"]["responses_api_agents"][
        f"{harness}_agent"
    ]
    agent.update({key: value for key, value in defaults.items() if key.endswith("_version")})
    if harness == "nooa":
        agent.pop("model")
        agent.pop("concurrency")
        agent.update(
            nooa={
                **defaults["nooa"],
                "execution_mode": "sandboxed",
            }
        )
        agent["sandbox_runner_timeout_seconds"] = timeout
    if use_agent_sessions and harness not in ("simple", "nooa"):
        # The verifier only reads retained output, so the agent can own this local sandbox.
        agent.update(sandbox_provider="probe_sandbox", sandbox_config={"workdir": str(directory / "workspace")})
    if harness == "simple":
        agent["max_steps"] = 6
    if harness in ("opencode", "pi"):
        agent["timeout"] = timeout
        if not use_agent_sessions:
            agent["workspace_root"] = str(directory / "workspace")
    if harness == "openclaw":
        agent["timeout"] = math.ceil(timeout)
        if not use_agent_sessions:
            agent["workspace_root"] = str(directory / "workspace")
    if harness == "opencode":
        # Auxiliary title/summary requests would consume the scripted policy replies.
        agent["opencode_config"] = {"permission": {"bash": "allow"}}
        if not use_agent_sessions:
            agent["opencode_config"]["agent"] = {"title": {"disable": True}, "summary": {"disable": True}}
    if harness == "codex" and not use_agent_sessions:
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
    if harness == "claude_code":
        agent.update(anthropic_api_key="conformance", anthropic_base_url=None)
    if prepared_agent is not None:
        # Replace the preset completely: its permissions, turn limits, compaction,
        # and model settings must not leak into a prepared experiment's probe.
        local = {key: agent[key] for key in _LOCAL_AGENT_FIELDS if key in agent}
        agent = deepcopy(prepared_agent) | local
        if use_agent_sessions and harness not in ("simple", "nooa"):
            agent["sandbox_config"] = (agent.get("sandbox_config") or {}) | {"workdir": str(directory / "workspace")}
    if harness == "nooa":
        from .nooa import observer_module_name

        adapter = agent["nooa"]["invocation_adapter"]
        agent["nooa"]["invocation_adapter"] = f"{observer_module_name(adapter)}:invoke_with_witness"
    environment_type = "single_agent_turn_legacy" if use_agent_sessions else "legacy_agent"
    environment = {
        **server(ports["probe_environment"]),
        "name": "probe_environment",
        "agent_server": {"type": "responses_api_agents", "name": "probe_agent"},
    }
    if use_agent_sessions:
        environment.update(
            resources_server={"type": "resources_servers", "name": "probe_resources"},
            resources_tool_transports=["direct_http"] if harness == "simple" else [],
            default_episode_timeout_seconds=timeout,
            cleanup_timeout_seconds=10,
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
    if use_agent_sessions and harness != "simple":
        config["probe_sandbox"] = {("conformance_local" if harness == "nooa" else "local"): {"concurrency": 1}}
    return config


async def run_episode(
    *, harness: str, scenario_name: str, directory: Path, timeout: float, settings: dict | None = None
) -> None:
    """Start fresh servers and let Gym write the rollout without modifying the harness."""
    if settings is not None:
        if settings.get("harness", harness) != harness:
            raise ValueError("Prepared settings select a different harness")
        for path, expected in settings.get("referenced_files", {}).items():
            if digest_file(Path(path)) != expected:
                raise ValueError(f"Prepared behavior file changed before this episode: {path}")
        (directory / "prepared_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    module = importlib.import_module(f"responses_api_agents.{harness}_agent.app")
    agent_cls = getattr(module, PROBE_HARNESSES[harness])
    config_cls = getattr(module, PROBE_HARNESSES[harness] + "Config")
    use_agent_sessions = _supports_agent_sessions(agent_cls)
    if use_agent_sessions and harness != "simple" and sys.platform != "linux":
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
    if harness == "nooa":
        from .nooa import BENCH_ADAPTER, BorrowedSandboxProbe

        prepared = (settings or {}).get("agent", {})
        invocation = prepared.get("nooa", {})
        probe = BorrowedSandboxProbe(
            scenario,
            directory,
            invocation_adapter=invocation.get("invocation_adapter", BENCH_ADAPTER),
            atom=invocation.get("agent_class") == "nooa_atom.gym:AtomHarness",
            runtime_wheelhouse=Path(prepared["runtime_wheelhouse"]) if prepared.get("runtime_wheelhouse") else None,
        )
    else:
        probe_cls = {"openclaw": OpenClawProbe}.get(harness, Probe)
        probe = probe_cls(scenario, directory)
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
            prepared_agent=(settings or {}).get("agent"),
        )
        if scenario.compaction is not None and "agent" not in (settings or {}):
            agent_config = config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]
            if harness in ("opencode", "pi"):
                agent_config.update(context_window=32768 if scenario.compaction else 65536, max_output_tokens=8192)
            elif harness == "codex":
                agent_config.update(model_context_window=32768, model_auto_compact_token_limit=24000)
            elif harness == "stirrup":
                agent_config.update(context_window_tokens=32768)
            elif harness == "hermes":
                agent_config.update(
                    compression_enabled=True, compression_threshold=0.1, max_turns=scenario.tool_steps + 4
                )
            elif harness == "simple":
                agent_config.update(max_steps=scenario.tool_steps + 4)
        sampling = (settings or {}).get("model_sampling")
        if sampling is not None:
            config["model_call_sampling"] = {"required_model_servers": ["policy_model"]}
        if "agent" in (settings or {}):
            effective = config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]
            overrides = {key: effective[key] for key in _LOCAL_AGENT_FIELDS if key in effective}
            if "sandbox_config" in effective:
                overrides["sandbox_config.workdir"] = effective["sandbox_config"].get("workdir")
            if harness == "nooa":
                overrides["nooa.invocation_adapter"] = {
                    "configured": probe.invocation_adapter,
                    "effective": effective["nooa"]["invocation_adapter"],
                }
            (directory / "local_overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
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
        (directory / "effective_agent_config.json").write_text(
            json.dumps(agent.config.model_dump(mode="json"), indent=2) + "\n"
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
        try:
            apps = {
                "policy_model": probe.model_app(
                    assistant_message_header=client.assistant_message_header("policy_model")
                ),
                "probe_resources": probe.resources_app(server_client=client),
                "probe_agent": agent.setup_webserver(),
                "probe_environment": environment.setup_webserver(),
            }
            if scenario.judge_verifier:
                apps["judge_model"] = probe.judge_app()
            if sampling is not None:
                apps["probe_resources"].add_middleware(CallerSamplingMiddleware)
                for name in ("policy_model", "judge_model"):
                    if name in apps:
                        _install_sampling(apps[name], directory, name=name, required=name == "policy_model")
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
            task = scenario.task()
            if sampling is not None:
                task[MODEL_SAMPLING_KEY_NAME] = sampling
            if harness == "simple":
                task["responses_create_params"]["model"] = "conformance-model"
                task["responses_create_params"]["tools"] = [
                    {
                        "type": "function",
                        "name": "bash",
                        "strict": False,
                        "description": "Run a shell command.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    }
                ]
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
            await get_global_aiohttp_client().close()
            probe.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=PROBE_HARNESSES)
    parser.add_argument("--scenario", required=True, choices=[s.name for s in SCENARIOS])
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--prepared-settings", type=Path, help="frozen input written by the suite runner")
    args = parser.parse_args()
    asyncio.run(
        run_episode(
            harness=args.harness,
            scenario_name=args.scenario,
            directory=args.directory.resolve(),
            timeout=args.timeout,
            settings=json.loads(args.prepared_settings.read_text()) if args.prepared_settings else None,
        )
    )


if __name__ == "__main__":
    main()
