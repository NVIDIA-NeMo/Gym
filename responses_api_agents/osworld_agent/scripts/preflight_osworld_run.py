#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail fast on OSWorld inputs and runtime references before a full rollout."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_PORT_LOCK = Path("/tmp/docker_port_allocation.lck")


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _load_config_paths(config_paths: str) -> list[dict[str, Any]]:
    configs = []
    for raw_path in config_paths.split(","):
        path = _resolve(raw_path.strip())
        if not path.is_file():
            raise ValueError(f"missing config file: {path}")
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            raise ValueError(f"config must contain a mapping: {path}")
        configs.append(value)
    return configs


def _validate_agent_classes(configs: list[dict[str, Any]]) -> list[str]:
    from responses_api_agents.osworld_agent.runner_registry import resolve_runner_spec

    class_paths = {
        value
        for config in configs
        for key, value in _walk(config)
        if key == "agent_class_path" and isinstance(value, str) and value.strip()
    }
    runner_names = {
        value
        for config in configs
        for key, value in _walk(config)
        if key == "runner_name" and isinstance(value, str) and value.strip()
    }
    for runner_name in runner_names:
        runner_spec = resolve_runner_spec(runner_name)
        if runner_spec.agent_class_path:
            class_paths.add(runner_spec.agent_class_path)
    for class_path in sorted(class_paths):
        module_name, separator, attribute = class_path.rpartition(".")
        if not separator:
            raise ValueError(f"agent_class_path must be module-qualified: {class_path}")
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute):
            raise ValueError(f"agent class does not exist: {class_path}")
    return sorted(class_paths)


def _validate_docker_lock(configs: list[dict[str, Any]]) -> None:
    uses_docker = any(
        key == "provider_name" and value == "docker"
        for config in configs
        for key, value in _walk(config)
    )
    if not uses_docker:
        return
    if DOCKER_PORT_LOCK.exists() and not os.access(DOCKER_PORT_LOCK, os.R_OK | os.W_OK):
        stat = DOCKER_PORT_LOCK.stat()
        raise PermissionError(
            f"Docker port lock is not readable/writable: {DOCKER_PORT_LOCK} "
            f"(uid={stat.st_uid}, gid={stat.st_gid}, mode={oct(stat.st_mode & 0o777)}). "
            "Fix this shared lock before launching rollouts."
        )


def _validate_input(path: Path, expected_rows: int | None) -> tuple[int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    task_ids = []
    for index, row in enumerate(rows, 1):
        metadata = row.get("verifier_metadata") or {}
        task = metadata.get("osworld_task") or {}
        task_id = metadata.get("task_id") or task.get("id") or task.get("task_id")
        if not task_id:
            raise ValueError(f"input row {index} has no OSWorld task ID")
        task_ids.append(str(task_id))
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("input contains duplicate OSWorld task IDs")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} input rows, found {len(rows)}")
    return len(rows), len(set(task_ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-paths", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    configs = _load_config_paths(args.config_paths)
    class_paths = _validate_agent_classes(configs)
    _validate_docker_lock(configs)
    rows, unique_task_ids = _validate_input(args.input_jsonl, args.expected_rows)
    print(
        json.dumps(
            {
                "preflight": "ok",
                "rows": rows,
                "unique_task_ids": unique_task_ids,
                "agent_class_paths": class_paths,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
