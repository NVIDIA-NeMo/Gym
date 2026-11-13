# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All per-harness conformance setup, importable before Gym dependencies are installed."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from omegaconf import DictConfig

    from .scenarios import Scenario


@dataclass(frozen=True)
class HarnessDefinition:
    """Adapter, runtime requirements, and base/scenario fixture settings for one harness."""

    agent: str
    base_settings: Callable[[Path, float], dict] = lambda directory, timeout: {}
    scenario_settings: dict[str, Callable[[Scenario], dict]] = field(default_factory=dict)
    install_module: str | None = None
    prepare_runtime: Callable[[], None] | None = None
    npm_version_key: str | None = None
    npm_version_default: str | None = None
    builtin: bool = False
    runtime_modules: tuple[str, ...] = ()
    runtime_files: dict[str, str] = field(default_factory=dict)
    runtime_metadata: dict[str, str] = field(default_factory=dict)
    requires_root: bool = False
    required_commands: tuple[str, ...] = ()
    session_requires_linux: bool = True
    ci_container: dict[str, str] = field(default_factory=dict)
    ci_packages: tuple[str, ...] = ()

    def settings(self, directory: Path, timeout: float, scenario: Scenario | None = None) -> DictConfig:
        """Merge shared defaults, this harness's base, then the selected scenario's overrides."""
        from omegaconf import OmegaConf

        settings = OmegaConf.merge(
            {
                "probe": ".probes:Probe",
                "requires_sandbox": False,
                "requires_tmux": False,
                "session": {
                    # Replace provider settings inherited from adapter YAML.
                    "pop_fields": ["resources_server", "sandbox_config"],
                    "harness_overrides": {
                        "sandbox_provider": "probe_sandbox",
                        "sandbox_config": {"workdir": str(directory / "workspace")},
                    },
                    "tool_transports": [],
                    "sandbox_backend": "local",
                },
                "legacy": {
                    "harness_overrides": {
                        "resources_server": {"type": "resources_servers", "name": "probe_resources"}
                    },
                },
            },
            self.base_settings(directory, timeout),
        )
        overrides = self.scenario_settings.get(getattr(scenario, "name", None))
        return OmegaConf.merge(settings, overrides(scenario)) if overrides else settings


def _prepare_crewai() -> None:
    from .ci import _build_crewai_sandbox_runtime

    _build_crewai_sandbox_runtime()


HARNESSES = {
    "simple": HarnessDefinition(
        agent="SimpleAgent",
        builtin=True,
        session_requires_linux=False,
        base_settings=lambda directory, timeout: {
            "harness_overrides": {"max_steps": 6},
            "session": {"harness_overrides": None, "tool_transports": ["direct_http"], "sandbox_backend": None},
            "task_overrides": {
                "responses_create_params": {
                    "model": "conformance-model",
                    "tools": [
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
                    ],
                }
            },
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
                "unsupported_reason": "Simple Agent does not implement compaction."
            }
        },
    ),
    "opencode": HarnessDefinition(
        agent="OpenCodeAgent",
        npm_version_key="opencode_version",
        npm_version_default="responses_api_agents.opencode_agent.runtime:OPENCODE_VERSION",
        base_settings=lambda directory, timeout: {
            "harness_overrides": {"timeout": timeout, "opencode_config": {"permission": {"bash": "allow"}}},
            "legacy": {
                "harness_overrides": {
                    "workspace_root": str(directory / "workspace"),
                    # Auxiliary requests would consume the scripted policy replies.
                    "opencode_config": {"agent": {"title": {"disable": True}, "summary": {"disable": True}}},
                }
            },
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
                "harness_overrides": {"context_window": 32768, "max_output_tokens": 8192},
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
                    ]
                },
            }
        },
    ),
    "pi": HarnessDefinition(
        agent="PiAgent",
        npm_version_key="pi_version",
        base_settings=lambda directory, timeout: {
            "harness_overrides": {"timeout": timeout},
            "legacy": {"harness_overrides": {"workspace_root": str(directory / "workspace")}},
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
                "probe": ".probes:PiProbe",
                "harness_overrides": {"context_window": 32768, "max_output_tokens": 8192},
                "probe_overrides": {
                    "tool_output_chars": 20000,
                    "responses": [
                        *[
                            {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                            for _ in range(scenario.tool_steps)
                        ],
                        {"kind": "context_overflow"},
                        {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                        {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                    ],
                },
            }
        },
    ),
    "codex": HarnessDefinition(
        agent="CodexAgent",
        npm_version_key="codex_version",
        base_settings=lambda directory, timeout: {
            "legacy": {
                "requires_workspace": True,
                "harness_overrides": {
                    "cwd": str(directory / "workspace"),
                    "timeout": timeout,
                    "sandbox_mode": "danger-full-access",
                    "extra_config": {"features": {"shell_snapshot": False}, "web_search": "disabled"},
                },
            }
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
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
                    ]
                },
            }
        },
    ),
    "hermes": HarnessDefinition(
        agent="HermesAgent",
        runtime_modules=("run_agent", "model_tools"),
        base_settings=lambda directory, timeout: {
            "harness_overrides": {
                "max_turns": 6,
                "enabled_toolsets": ["terminal"],
                "terminal_backend": "local",
                "terminal_timeout": 10,
                "compression_enabled": False,
                "checkpoints_enabled": False,
                "api_key": "conformance",
            }
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
                "harness_overrides": {
                    "compression_enabled": True,
                    "context_length": 32768,
                    "compression_threshold": 0.5,
                    "max_turns": scenario.tool_steps + 4,
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
                    ]
                },
            }
        },
    ),
    "kilocode": HarnessDefinition(
        agent="KiloCodeAgent",
        npm_version_key="kilo_version",
        base_settings=lambda directory, timeout: {},
        scenario_settings={
            "context_compaction": lambda scenario: {
                "harness_overrides": {
                    **{"context_window": 32768, "max_output_tokens": 8192},
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
                    ]
                },
            }
        },
    ),
    "stirrup": HarnessDefinition(
        agent="StirrupAgentWrapper",
        base_settings=lambda directory, timeout: {},
        scenario_settings={
            "context_compaction": lambda scenario: {
                "harness_overrides": {"context_window_tokens": 32768, "agent_max_turns": scenario.tool_steps + 4},
                "probe_overrides": {
                    "responses": [
                        *[
                            {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                            for _ in range(scenario.tool_steps - 1)
                        ],
                        {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
                        {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                        {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                    ]
                },
            }
        },
    ),
    "copilot": HarnessDefinition(
        agent="CopilotAgent",
        base_settings=lambda directory, timeout: {
            "harness_overrides": {"timeout": timeout},
            "legacy": {"harness_overrides": {"workspace_root": str(directory / "workspace")}},
        },
    ),
    "crewai": HarnessDefinition(
        agent="CrewAIAgent",
        prepare_runtime=_prepare_crewai,
        base_settings=lambda directory, timeout: {"harness_overrides": {"timeout": timeout}},
    ),
    "openclaw": HarnessDefinition(
        agent="OpenClawAgent",
        base_settings=lambda directory, timeout: {
            "probe": ".probes:OpenClawProbe",
            "harness_overrides": {"timeout": math.ceil(timeout)},
            "legacy": {"harness_overrides": {"workspace_root": str(directory / "workspace")}},
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
                "harness_overrides": {"context_window": 131072, "max_output_tokens": 8192},
                "probe_overrides": {
                    "context_window": 131072,
                    "command_padding_chars": 12000,
                    "responses": [
                        *[
                            {"kind": "tool", "input_tokens": 500, "output_tokens": 8}
                            for _ in range(scenario.tool_steps)
                        ],
                        {"kind": "context_overflow"},
                        {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
                        {"kind": "final", "input_tokens": 100, "output_tokens": 8},
                    ],
                },
            }
        },
    ),
    "nooa": HarnessDefinition(
        agent="NOOAAgent",
        install_module="scripts.harness_conformance.nooa",
        ci_container={"image": "python:3.13.14-slim", "options": "--init"},
        ci_packages=("git", "curl"),
        runtime_files={
            "pyproject_sha256": "responses_api_agents/nooa_agent/pyproject.toml",
            "lock_sha256": "responses_api_agents/nooa_agent/uv.lock",
            "bootstrap_sha256": "responses_api_agents/nooa_agent/sandbox_runtime.py",
            "observer_sha256": "scripts/harness_conformance/nooa_observer.py",
        },
        runtime_metadata={"sandbox_ownership": "resources", "execution_mode": "sandboxed"},
        base_settings=lambda directory, timeout: {
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
    ),
    "miniswe_sandboxed": HarnessDefinition(
        agent="MiniSWESandboxedAgent",
        base_settings=lambda directory, timeout: {
            "requires_sandbox": True,
            "harness_overrides": {
                "harness": {"step_limit": 6, "step_timeout_sec": 10},
                "agent_timeout_sec": timeout,
                "setup_timeout_sec": timeout,
                "artifacts_dir": str(directory / "native"),
            },
            "probe_overrides": {"finish_protocol": "bash_marker"},
        },
        scenario_settings={
            "context_compaction": lambda scenario: {"unsupported_reason": "Mini-SWE does not implement compaction."}
        },
    ),
    "claude_code": HarnessDefinition(
        agent="ClaudeCodeAgent",
        base_settings=lambda directory, timeout: {
            "harness_overrides": {
                "claude_code_version": "2.1.108",
                "timeout": int(timeout),
                "max_turns": 6,
                "allowed_tools": "Bash",
                "tools": "Bash",
                "session_name": "Conformance check",
                "thinking": "disabled",
            }
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
                "harness_overrides": {
                    "max_turns": scenario.tool_steps + 4,
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
                    ]
                },
            }
        },
    ),
    "terminus_2_sandboxed": HarnessDefinition(
        agent="Terminus2Agent",
        requires_root=True,
        required_commands=("tmux",),
        ci_packages=("tmux",),
        base_settings=lambda directory, timeout: {
            "requires_sandbox": True,
            "requires_tmux": True,
            "probe_overrides": {"final_responses": 2, "tool_protocol": "terminal_json"},
            "inherit_adapter_defaults": True,
            "harness_overrides": {
                "max_turns": 6,
                "enable_summarize": False,
                "llm_request_timeout": int(timeout),
                "sandbox_timeout": timeout,
            },
        },
        scenario_settings={
            "context_compaction": lambda scenario: {
                "harness_overrides": {
                    "max_turns": scenario.tool_steps + 4,
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
                    ]
                },
            }
        },
    ),
}
