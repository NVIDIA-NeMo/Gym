# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process bridge to the local Claw-Eval fork; deliberately has no Gym dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


def source_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    for relative in (
        "src/claw_eval/config.py",
        "evaluation/run_multimodal.py",
        "evaluation/task_catalog.py",
    ):
        if not (root / relative).is_file():
            raise FileNotFoundError(f"CLAW_EVAL_ROOT is missing {relative}: {root}")
    return root


def task_path(root: Path, task_id: str, digest: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task_id):
        raise ValueError("task_id must be a task directory name, without path separators")
    path = (root / "tasks" / task_id / "task.yaml").resolve()
    if not path.is_relative_to(root / "tasks"):
        raise ValueError("task definition is outside the task root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Task digest mismatch for {task_id}; prepare the dataset again")
    return path


def runtime_provenance(root: Path, config_path: Path) -> dict[str, str]:
    # Include working-tree edits: HEAD alone does not identify this local fork.
    paths = sorted((root / "src/claw_eval").rglob("*.py"))
    paths += sorted((root / "evaluation").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else "unknown",
        "runtime_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value


def prepare(payload: dict[str, Any], root: Path) -> None:
    from claw_eval.models.task import TaskDefinition
    from evaluation.task_catalog import discover_tasks

    paths = discover_tasks(root / "tasks", payload["split"])
    selected = set(payload.get("task_ids", []))
    if selected:
        missing = selected - {path.parent.name for path in paths}
        if missing:
            raise ValueError(f"Unknown task IDs for {payload['split']}: {sorted(missing)}")
        paths = [path for path in paths if path.parent.name in selected]
    output = Path(payload["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for path in paths:
                # Adapt the native public task contract to Gym's request schema.
                # Loading attachments, tools, and graders remains the runner's job.
                task = TaskDefinition.from_yaml(path)
                row = {
                    "agent_ref": {"type": "responses_api_agents", "name": payload["agent_name"]},
                    "responses_create_params": {"input": [{"role": "user", "content": task.prompt.text}]},
                    "verifier_metadata": {
                        "task_id": task.task_id,
                        "task_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "split": payload["split"],
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {len(paths)} {payload['split']} tasks in {output}")


def run_trial(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    from claw_eval.config import Config, load_config
    from claw_eval.models.task import TaskDefinition
    from evaluation.run_multimodal import (
        build_judge,
        build_provider,
        run_one,
        task_contract_fingerprint,
        validate_assets,
    )

    metadata = payload["verifier_metadata"]
    path = task_path(root, metadata["task_id"], metadata["task_sha256"])
    task = TaskDefinition.from_yaml(path)
    if task.task_id != metadata["task_id"] or task.prompt.text != payload["prompt"]:
        raise ValueError("Gym row does not match the native task ID and prompt; prepare the dataset again")
    config_path = Path(payload["claweval_config"])
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Claw-Eval config does not exist: {config_path}")
    cfg = load_config(config_path)
    raw = cfg.model_dump()
    for section in ("model", "judge", "user_agent_model"):
        merge_config(raw[section], payload.get(f"{section}_overrides", {}))
    cfg = Config.model_validate(raw)
    # One Gym rollout is one native trial. Match the local three-trial seed protocol.
    cfg.model.extra_body = dict(cfg.model.extra_body or {})
    cfg.model.extra_body["seed"] = payload["seed"]
    if task.user_agent.enabled and not (cfg.user_agent_model.api_key or cfg.judge.api_key):
        raise ValueError("This task requires a configured user-agent model API key")
    fixture_root = Path(payload.get("fixture_root") or root / "tasks").expanduser().resolve()
    missing = validate_assets([path], fixture_root)
    if missing:
        raise FileNotFoundError("Missing Claw-Eval assets: " + "; ".join(missing))
    judge = build_judge(cfg, no_judge=payload["no_judge"])
    provider = build_provider(cfg)
    try:
        result = run_one(
            task_yaml=path,
            fixture_root=fixture_root,
            output_dir=Path(payload["output_dir"]),
            cfg=cfg,
            provider=provider,
            judge=judge,
            sandbox_image=payload["sandbox_image"],
            sandbox_dependencies=Path(payload["sandbox_dependencies"]),
            sandbox_server=root / "src/claw_eval/sandbox/server.py",
            sandbox_port=payload["sandbox_port"],
            sandbox_ready_timeout=payload["sandbox_ready_timeout"],
            max_turns=-1,  # preserve each task's native turn limit
        )
    finally:
        provider.client.close()
    result["provenance"] = runtime_provenance(root, config_path)
    result["task_contract"] = task_contract_fingerprint(path, fixture_root)
    result["seed"] = payload["seed"]
    result["judge_enabled"] = judge is not None
    result["model"] = cfg.model.model_id
    return result


def main() -> None:
    # Allow the native context managers to stop services and the Pyxis step on timeout.
    def terminate(signum, frame):
        raise KeyboardInterrupt("Claw-Eval worker terminated")

    signal.signal(signal.SIGTERM, terminate)
    payload = json.load(sys.stdin)
    root = source_root(payload["claweval_root"])
    sys.path[:0] = [str(root / "src"), str(root)]
    os.chdir(root)
    if payload["operation"] == "prepare":
        prepare(payload, root)
    elif payload["operation"] == "run":
        result = run_trial(payload, root)
        Path(payload["output_dir"], "result.json").write_text(json.dumps(result), encoding="utf-8")
    else:
        raise ValueError(f"Unknown operation: {payload['operation']}")


if __name__ == "__main__":
    main()
