# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One disposable local Gym episode, driven through the EnvironmentServer collector."""

import argparse
import asyncio
import importlib
import importlib.util
import json
import math
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
    if harness == "terminus_2_sandboxed" and (os.geteuid() != 0 or not shutil.which("tmux")):
        raise RuntimeError("Terminus conformance requires a disposable Linux runner as root with tmux installed")
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


def _base_settings(harness: str, directory: Path, timeout: float) -> DictConfig:
    """Describe harness settings, probe settings and setup requirements."""
    workspace = str(directory / "workspace")
    opencode = {"permission": {"bash": "allow"}}
    settings = {
        "simple": {
            "harness_overrides": {"max_steps": 6},
            "session": {"harness_overrides": None, "tool_transports": ["direct_http"], "sandbox_backend": None},
        },
        "opencode": {
            "harness_overrides": {"timeout": timeout, "opencode_config": opencode},
            "legacy": {
                "harness_overrides": {
                    "workspace_root": workspace,
                    # Auxiliary requests would consume the scripted policy replies.
                    "opencode_config": {
                        "agent": {"title": {"disable": True}, "summary": {"disable": True}},
                    },
                },
            },
        },
        "pi": {
            "harness_overrides": {"timeout": timeout},
            "legacy": {"harness_overrides": {"workspace_root": workspace}},
        },
        "copilot": {
            "harness_overrides": {"timeout": timeout},
            "legacy": {"harness_overrides": {"workspace_root": workspace}},
        },
        "crewai": {"harness_overrides": {"timeout": timeout}},
        "openclaw": {
            "probe": ".probes:OpenClawProbe",
            "harness_overrides": {"timeout": math.ceil(timeout)},
            "legacy": {"harness_overrides": {"workspace_root": workspace}},
        },
        "codex": {
            "legacy": {
                "requires_workspace": True,
                "harness_overrides": {
                    "cwd": workspace,
                    "timeout": timeout,
                    "sandbox_mode": "danger-full-access",
                    "extra_config": {"features": {"shell_snapshot": False}, "web_search": "disabled"},
                },
            },
        },
        "nooa": {
            "probe": ".probes:NOOAProbe",
            "required_imports": ["scripts.harness_conformance.nooa"],
            "inherit_adapter_fields": ["nooa"],
            "harness_overrides": {
                "sandbox_runner_timeout_seconds": timeout,
                "nooa": {
                    "execution_mode": "sandboxed",
                    "invocation_adapter": "${harness_conformance.nooa_observer:}:invoke_with_witness",
                },
            },
            "pop_fields": ["model", "concurrency"],
            "session": {"harness_overrides": None, "sandbox_backend": "conformance_local"},
        },
        "claude_code": {
            "harness_overrides": {
                "claude_code_version": "2.1.108",
                "timeout": int(timeout),
                "max_turns": 6,
                "allowed_tools": "Bash",
                "tools": "Bash",
                "session_name": "Conformance check",
                "thinking": "disabled",
            },
        },
        "miniswe_sandboxed": {
            "requires_sandbox": True,
            "harness_overrides": {
                "harness": {"step_limit": 6, "step_timeout_sec": 10},
                "agent_timeout_sec": timeout,
                "setup_timeout_sec": timeout,
                "artifacts_dir": str(directory / "native"),
            },
        },
        "terminus_2_sandboxed": {
            "requires_sandbox": True,
            "requires_tmux": True,
            "probe_overrides": {"final_responses": 2},
            "inherit_adapter_defaults": True,
            "harness_overrides": {
                "max_turns": 6,
                "enable_summarize": False,
                "llm_request_timeout": int(timeout),
                "sandbox_timeout": timeout,
            },
        },
        "hermes": {
            "harness_overrides": {
                "max_turns": 6,
                "enabled_toolsets": ["terminal"],
                "terminal_backend": "local",
                "terminal_timeout": 10,
                "compression_enabled": False,
                "checkpoints_enabled": False,
                "api_key": "conformance",
            },
        },
    }.get(harness, {})
    return OmegaConf.merge(
        {
            "probe": ".probes:Probe",
            # Whether the episode runner must prepare a sandbox for the agent to borrow.
            "requires_sandbox": False,
            "requires_tmux": False,
            "session": {
                # Session sandboxes replace provider settings inherited from adapter YAML.
                "pop_fields": ["resources_server", "sandbox_config"],
                "harness_overrides": {"sandbox_provider": "probe_sandbox", "sandbox_config": {"workdir": workspace}},
                "tool_transports": [],
                "sandbox_backend": "local",
            },
            "legacy": {
                "harness_overrides": {"resources_server": {"type": "resources_servers", "name": "probe_resources"}},
            },
        },
        settings,
    )


def _compaction_settings(harness: str, scenario: Scenario) -> DictConfig:
    """Harness overrides and scripted replies for the compaction scenario."""
    steps = scenario.tool_steps + 4
    context = {"context_window": 32768, "max_output_tokens": 8192}
    settings = {
        "simple": {"unsupported_reason": "Simple Agent does not implement compaction."},
        "opencode": {
            "harness_overrides": context,
            "probe_overrides": {
                "responses": [
                    {"kind": "tool", "input_tokens": 500, "output_tokens": 8},
                    {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    *[
                        {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                        for _ in range(scenario.tool_steps - 2)
                    ],
                    {"kind": "final", "input_tokens": 500, "output_tokens": 8},
                ],
            },
        },
        "pi": {
            "probe": ".probes:PiProbe",
            "harness_overrides": context,
            "probe_overrides": {
                "tool_output_chars": 20000,
                "responses": [
                    *[{"kind": "tool", "input_tokens": 500, "output_tokens": 8} for _ in range(scenario.tool_steps)],
                    {"kind": "context_overflow"},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                ],
            },
        },
        "kilocode": {
            "harness_overrides": {
                **context,
                "kilo_config": {"compaction": {"auto": True}},
            },
            "probe_overrides": {
                "responses": [
                    *[
                        {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                        for _ in range(scenario.tool_steps - 1)
                    ],
                    {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                ],
            },
        },
        "openclaw": {
            "harness_overrides": {"context_window": 131072, "max_output_tokens": 8192},
            "probe_overrides": {
                "context_window": 131072,
                "command_padding_chars": 12000,
                "responses": [
                    *[{"kind": "tool", "input_tokens": 500, "output_tokens": 8} for _ in range(scenario.tool_steps)],
                    {"kind": "context_overflow"},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                ],
            },
        },
        "codex": {
            "harness_overrides": {
                "model_context_window": 32768,
                "model_auto_compact_token_limit": 24000,
                "request_max_retries": 0,
                "stream_max_retries": 0,
            },
            "probe_overrides": {
                "responses": [
                    *[
                        {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                        for _ in range(scenario.tool_steps - 1)
                    ],
                    {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                ],
            },
        },
        "stirrup": {
            "harness_overrides": {
                "context_window_tokens": 32768,
                "agent_max_turns": steps,
            },
            "probe_overrides": {
                "responses": [
                    *[
                        {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                        for _ in range(scenario.tool_steps - 1)
                    ],
                    {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                ],
            },
        },
        "hermes": {
            "harness_overrides": {
                "compression_enabled": True,
                "context_length": 32768,
                "compression_threshold": 0.5,
                "max_turns": steps,
            },
            "probe_overrides": {
                "responses": [
                    *[
                        {"kind": "tool", "input_tokens": 1000, "output_tokens": 8}
                        for _ in range(scenario.tool_steps - 1)
                    ],
                    {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                ],
            },
        },
        "claude_code": {
            "harness_overrides": {
                "max_turns": steps,
                "settings": {
                    "autoCompactWindow": 100000,
                    "autoMemoryEnabled": False,
                    "promptSuggestionEnabled": False,
                    "env": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"},
                },
            },
            "probe_overrides": {
                "responses": [
                    *[
                        {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                        for _ in range(scenario.tool_steps - 1)
                    ],
                    {"kind": "tool", "input_tokens": 90000, "output_tokens": 8},
                    {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                    {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                ],
            },
        },
        "miniswe_sandboxed": {"unsupported_reason": "Mini-SWE does not implement compaction."},
        "terminus_2_sandboxed": {
            "harness_overrides": {
                "max_turns": steps,
                "enable_summarize": True,
                "model_context_limit": 32768,
                "proactive_summarization_threshold": 8000,
            },
            "probe_overrides": {
                "responses": [
                    *[
                        {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                        for _ in range(scenario.tool_steps - 1)
                    ],
                    {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
                    # Harbor summarizes, asks handoff questions, then answers them.
                    *[{"kind": "summary", "input_tokens": 500, "output_tokens": 11} for _ in range(3)],
                    *[{"kind": "final", "input_tokens": 100, "output_tokens": 8} for _ in range(2)],
                ],
            },
        },
    }.get(harness, {})
    return OmegaConf.create(settings)


def _settings(harness: str, directory: Path, timeout: float, scenario: Scenario | None = None) -> DictConfig:
    settings = _base_settings(harness, directory, timeout)
    scenario_settings = {"context_compaction": _compaction_settings}.get(getattr(scenario, "name", None))
    if scenario_settings is not None:
        settings = OmegaConf.merge(settings, scenario_settings(harness, scenario))
    return settings


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
    agent_cls = getattr(module, HARNESSES[harness])
    config_cls = agent_cls.model_fields["config"].annotation
    use_agent_sessions = agent_cls.supports_agent_sessions()
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
    settings = _settings(harness, directory, timeout, scenario)
    probe_module, probe_name = settings.probe.split(":")
    probe_cls = getattr(importlib.import_module(probe_module, package=__package__), probe_name)
    probe = probe_cls(scenario, directory, harness=harness, **settings.get("probe_overrides", {}))
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
            task = scenario.task()
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
