# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate, validate, and materialize synthetic tool-use seed artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from resources_servers.synthetic_tool_use.assets import generation_asset_hashes, load_generation_assets
from resources_servers.synthetic_tool_use.common.artifacts import (
    RunArtifactStore,
    atomic_write_json,
)
from resources_servers.synthetic_tool_use.common.clients import OpenAIChatGenerator
from resources_servers.synthetic_tool_use.common.models import SeedGenerationConfig, StageState
from resources_servers.synthetic_tool_use_domain_generation.stage import DomainGenerationStage
from resources_servers.synthetic_tool_use_policy_tool_generation.stage import PolicyToolsGenerationStage
from resources_servers.synthetic_tool_use_scenario_generation.stage import ScenarioGenerationStage
from resources_servers.synthetic_tool_use_simulation.scripts.build_synthetic_tool_use_dataset import (
    build_sample_dataset,
    validate_domain_static_artifacts,
)


def load_config(path: Path) -> SeedGenerationConfig:
    expanded = os.path.expandvars(path.read_text(encoding="utf-8"))
    raw = yaml.safe_load(expanded)
    return SeedGenerationConfig.model_validate(raw)


def selected_source_indexes(store: RunArtifactStore, start: int | None, end: int | None) -> set[int] | None:
    if start is None and end is None:
        return None
    lower = 0 if start is None else start
    upper = float("inf") if end is None else end
    return {entry.source_index for entry in store.load_manifest().domains if lower <= entry.source_index < upper}


def validate_run(store: RunArtifactStore) -> dict[str, Any]:
    report: dict[str, Any] = {
        "domains_seen": 0,
        "domains_eligible": 0,
        "domains_incomplete": 0,
        "domains_valid": 0,
        "failures": [],
    }
    for entry in store.load_manifest().domains:
        report["domains_seen"] += 1
        if entry.stages["scenarios"].state != StageState.COMPLETE:
            report["domains_incomplete"] += 1
            continue
        report["domains_eligible"] += 1
        domain_dir = store.domains_dir / entry.artifact_dir
        try:
            validate_domain_static_artifacts(domain_dir)
            report["domains_valid"] += 1
        except Exception as exc:
            report["failures"].append(
                {
                    "domain_id": entry.domain_id,
                    "source_index": entry.source_index,
                    "reason": getattr(exc, "reason", "validation_error"),
                    "detail": str(exc),
                }
            )
    atomic_write_json(store.run_dir / "validation_report.json", report)
    return report


def materialize(
    config: SeedGenerationConfig,
    store: RunArtifactStore,
    output_path: Path | None,
    dataset_name: str | None,
    parallel_tool_calls: bool,
) -> dict[str, Any]:
    dataset_name = dataset_name or config.source_name
    output_path = output_path or store.run_dir / f"{dataset_name}.jsonl"
    return build_sample_dataset(
        source_dirs=[store.domains_dir],
        output_path=output_path,
        report_path=output_path.with_suffix(".report.json"),
        max_rows=None,
        dataset_name=dataset_name,
        source_names=[config.source_name],
        max_rows_per_domain=None,
        scan_domains_per_source=None,
        domain_generator_model=config.domain_model.model,
        policy_tools_model=config.policy_tools_model.model,
        scenario_generator_model=config.scenario_model.model,
        parallel_tool_calls=parallel_tool_calls,
    )


async def run_model_stages(args: argparse.Namespace, config: SeedGenerationConfig, store: RunArtifactStore) -> None:
    assets = load_generation_assets(config.generation_profile)
    store.record_asset_hashes(generation_asset_hashes(config.generation_profile))
    indexes = selected_source_indexes(store, args.domain_start, args.domain_end)
    clients: list[OpenAIChatGenerator] = []
    try:
        if args.command in {"domains", "all"}:
            domain_client = OpenAIChatGenerator(config.domain_model)
            clients.append(domain_client)
            await DomainGenerationStage(config, assets.domain_prompt, store, domain_client).run(resume=args.resume)
            indexes = selected_source_indexes(store, args.domain_start, args.domain_end)
        if args.command in {"policies", "all"}:
            policy_client = OpenAIChatGenerator(config.policy_tools_model)
            clients.append(policy_client)
            judge_client = None
            if config.policy_tools.judge_enabled:
                if config.judge_model is None:
                    raise ValueError("judge_model is required when policy_tools.judge_enabled is true")
                judge_client = OpenAIChatGenerator(config.judge_model)
                clients.append(judge_client)
            await PolicyToolsGenerationStage(config, assets.policy_tools, store, policy_client, judge_client).run(
                resume=args.resume,
                source_indexes=indexes,
            )
        if args.command in {"scenarios", "all"}:
            scenario_client = OpenAIChatGenerator(config.scenario_model)
            clients.append(scenario_client)
            await ScenarioGenerationStage(config, assets.scenarios, store, scenario_client).run(
                resume=args.resume,
                source_indexes=indexes,
            )
    finally:
        await asyncio.gather(*(client.close() for client in clients))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--domain-start", type=int, default=None, help="Inclusive source-domain index.")
    parser.add_argument("--domain-end", type=int, default=None, help="Exclusive source-domain index.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("domains", "policies", "scenarios", "validate"):
        subparsers.add_parser(command)
    for command in ("materialize", "all"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--dataset-name", default=None)
        command_parser.add_argument("--output-path", type=Path, default=None)
        command_parser.add_argument("--parallel-tool-calls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    store = RunArtifactStore.create(config, generation_asset_hashes(config.generation_profile))
    if args.command in {"domains", "policies", "scenarios", "all"}:
        asyncio.run(run_model_stages(args, config, store))
    result: dict[str, Any]
    if args.command == "validate":
        result = validate_run(store)
    elif args.command in {"materialize", "all"}:
        validation = validate_run(store)
        if validation["failures"]:
            raise ValueError(f"cannot materialize: {len(validation['failures'])} domains failed validation")
        if validation["domains_valid"] == 0:
            raise ValueError("cannot materialize: no completed domains passed validation")
        result = materialize(config, store, args.output_path, args.dataset_name, args.parallel_tool_calls)
    else:
        result = store.write_generation_report()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
