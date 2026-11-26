# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configure existing Harbor tasks for group-sticky harnesses and explicit attribution."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from nemo_gym.core import HARNESS_NAMES, annotate_prompt_groups


def configure_harnesses(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    sources: Mapping[str, Mapping[str, str]],
    routing_seed: int,
    context_tokens: int,
    output_tokens: int,
    max_steps: int,
    codex_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse tasksets, providers and verifiers; only select the existing agent bridges."""
    from responses_api_agents.codex_agent.app import CodexAgentConfig
    from responses_api_agents.hermes_agent.app import HermesAgentConfig
    from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig

    prepared = annotate_prompt_groups(rows, sources=sources, routing_seed=routing_seed)
    output = copy.deepcopy(dict(config))
    configured: set[str] = set()
    for row in prepared:
        routing = row["harness_routing"]
        if routing["task_type"] != "agentic":
            continue
        environment = routing["env"]
        taskset = row["task_id"]["taskset"]
        if environment not in configured:
            implementations = output[environment]["environment_servers"]
            if len(implementations) != 1 or not ({"harbor", "unified_video_harbor"} & implementations.keys()):
                raise ValueError("Agentic harness routing requires an existing Harbor environment")
            kind, original = next(iter(implementations.items()))
            values = copy.deepcopy(original)
            agent_name = values["agent_server"]["name"]
            original_agents = output[agent_name]["responses_api_agents"]
            if set(original_agents) != {"opencode_agent"}:
                raise ValueError("The original Harbor agent must be the configured OpenCode bridge")
            # Composed YAML need not spell out defaults supplied by the agent's config model.
            # Runtime identity is only needed to validate the settings, and never enters the output.
            identity = {"host": "127.0.0.1", "port": 0, "name": agent_name}
            opencode = OpenCodeAgentConfig.model_validate(original_agents["opencode_agent"] | identity)
            if opencode.model_server is None:
                raise ValueError("Harness alternatives require a shared model server")
            if opencode.context_window != context_tokens or opencode.max_output_tokens != output_tokens:
                raise ValueError("Resolve the original model budgets before configuring harnesses")
            common = opencode.model_dump(
                mode="json",
                include={"entrypoint", "token_id_capture", "num_workers", "model_server", "model", "concurrency"},
            )
            common.update(resources_server=None, datasets=[])
            alternatives = {
                "codex": common
                | {
                    "timeout": opencode.timeout,
                    "model_context_window": context_tokens,
                    "codex_version": codex_version,
                },
                "hermes": common
                | {
                    "sandbox_runner_timeout_seconds": opencode.timeout,
                    "context_length": context_tokens,
                    "max_tokens": output_tokens,
                    "max_turns": max_steps,
                    "compression_enabled": False,
                    "chat_template_kwargs_enabled": True,
                },
            }
            config_models = {"codex": CodexAgentConfig, "hermes": HermesAgentConfig}
            required_fields = {
                "codex": {
                    "model_server",
                    "model",
                    "model_context_window",
                    "codex_version",
                    "token_id_capture",
                },
                "hermes": {
                    "model_server",
                    "model",
                    "context_length",
                    "max_tokens",
                    "max_turns",
                    "token_id_capture",
                    "chat_template_kwargs_enabled",
                },
            }
            refs = {"opencode": values["agent_server"]}
            for harness, agent_config in alternatives.items():
                name = agent_name + "_" + harness
                agent_identity = identity | {"name": name}
                expected = config_models[harness].model_validate(agent_config | agent_identity)
                component = {"responses_api_agents": {harness + "_agent": agent_config}}
                if name in output:
                    existing = output[name]["responses_api_agents"]
                    if set(existing) != {harness + "_agent"}:
                        raise ValueError("Harness agent name belongs to another implementation")
                    actual = existing[harness + "_agent"]
                    if actual.get("datasets"):
                        raise ValueError("Harness alternatives must declare no duplicate dataset")
                    validated = config_models[harness].model_validate(actual | agent_identity)
                    mismatches = sorted(
                        key for key in required_fields[harness] if getattr(validated, key) != getattr(expected, key)
                    )
                    if mismatches:
                        raise ValueError(
                            f"Existing {harness} configuration differs from the requested settings: "
                            + ", ".join(mismatches)
                        )
                else:
                    output[name] = component
                refs[harness] = {"type": "responses_api_agents", "name": name}
            if set(refs) != set(HARNESS_NAMES):
                raise ValueError("All configured harness bridges are required")
            values.update(harness_agents=refs, require_harness_routing=True, request_sampling_to_model_calls=True)
            implementations.pop(kind)
            implementations["unified_video_harbor"] = values
            configured.add(environment)
            # Codex and Hermes image tools must retain the same N4 image parts.
            model_name = opencode.model_server.name
            models = output[model_name]["responses_api_models"]
            if "vllm_model" in models:
                models["vllm_model"]["inline_tool_output_images"] = True
        values = output[environment]["environment_servers"]["unified_video_harbor"]
        if taskset not in values["tasksets"]:
            raise ValueError("Prepared taskset is absent from the original Harbor environment")
        routes = output.setdefault("environment_server_routes", {})
        if taskset in routes and routes[taskset] != environment:
            raise ValueError("Taskset already routes to another environment")
        routes[taskset] = environment
    output["environment_routing_mode"] = "agent"
    return output, prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("training-data", "gym-config", "sources", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("routing-seed", "context-tokens", "output-tokens", "max-steps"):
        parser.add_argument("--" + flag, type=int, required=True)
    parser.add_argument("--codex-version", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.training_data.read_text().splitlines() if line.strip()]
    config, prepared = configure_harnesses(
        yaml.safe_load(args.gym_config.read_text()),
        rows,
        sources=json.loads(args.sources.read_text()),
        routing_seed=args.routing_seed,
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        max_steps=args.max_steps,
        codex_version=args.codex_version,
    )
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    (args.output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prepared))
    (args.output / "gym_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
