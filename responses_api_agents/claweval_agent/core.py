# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ClawEval task inventory, Harbor staging, and native-result validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


TEXT_EXTENSIONS = {
    ".bash",
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_package_digest(task_dir: Path, definition: dict[str, Any]) -> str:
    """Hash private scoring inputs without hashing multi-gigabyte media."""
    paths = [task_dir / "task.yaml", task_dir / "grader.py"]
    paths.extend(task_dir / relative for relative in definition.get("sandbox_grader_files", []))
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.relative_to(task_dir.resolve())).encode())
        digest.update(b"\0")
        digest.update(_digest_file(path).encode())
        digest.update(b"\0")
    # Reference images are numerous. Their immutable path/size/mtime identity is
    # sufficient for start-up validation; the staged verifier receives the files
    # themselves and records the task package digest used by the dataset.
    for relative in sorted(definition.get("local_grader_files", [])):
        path = (task_dir / relative).resolve(strict=True)
        stat = path.stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return digest.hexdigest()


def inventory(task_roots: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Load immutable public prompts and private task-package provenance."""
    tasks: dict[str, dict[str, Any]] = {}
    for variant, root_value in task_roots.items():
        if not variant or "::" in variant:
            raise ValueError(f"Invalid ClawEval task variant: {variant!r}")
        root = Path(root_value).resolve(strict=True)
        for task_yaml in sorted(root.glob("*/task.yaml")):
            definition = yaml.safe_load(task_yaml.read_text())
            task_id = definition.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"Missing task_id in {task_yaml}")
            key = f"{variant}::{task_id}"
            if key in tasks:
                raise ValueError(f"Duplicate ClawEval task variant {key}: {task_yaml}")
            prompt = definition.get("prompt", {}).get("text")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Missing prompt text in {task_yaml}")
            fixture_records = []
            for relative in definition.get("sandbox_files", []):
                source = task_yaml.parent / relative
                if not source.exists():
                    raise FileNotFoundError(source)
                resolved = source.resolve(strict=True)
                fixture_records.append(
                    {
                        "path": relative,
                        "source": str(resolved),
                        "size": resolved.stat().st_size,
                    }
                )
            tasks[key] = {
                "task_id": task_id,
                "task_variant": variant,
                "family": str(definition.get("category") or "unknown"),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "task_yaml": str(task_yaml.resolve()),
                "task_package_sha256": _task_package_digest(task_yaml.parent, definition),
                "fixtures": fixture_records,
            }
    if not tasks:
        raise ValueError("No ClawEval task.yaml files found")
    return tasks


def _message_value(message: dict[str, Any]) -> str | None:
    role = message.get("from") or message.get("role")
    if role not in {"human", "user"}:
        return None
    value = message.get("value", message.get("content"))
    return value if isinstance(value, str) and value else None


def source_id_task_id(source_id: str) -> str:
    """Extract the task ID without folding the trial suffix into it."""
    marker = source_id.rfind("_trial")
    if marker < 0 or not source_id[marker + len("_trial") :].isdigit():
        raise ValueError(f"ClawEval SFT id has no numeric trial suffix: {source_id}")
    task_id = source_id[source_id.find("__") + 2 : marker] if "__" in source_id else source_id[:marker]
    if not task_id.startswith(("G_video_", "B_video_")):
        raise ValueError(f"Unsupported ClawEval task id in {source_id}")
    return task_id


def dataset_rows(sft_path: str | Path, tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert successful SFT trajectories into prompt-only on-policy RL rows."""
    rows = []
    seen_source_ids: set[str] = set()
    with Path(sft_path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            source_id = record.get("id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"Missing id at SFT line {line_number}")
            if source_id in seen_source_ids:
                raise ValueError(f"Duplicate SFT id: {source_id}")
            seen_source_ids.add(source_id)
            task_id = source_id_task_id(source_id)
            task_variant = source_id.split("__", 1)[0]
            task = tasks.get(f"{task_variant}::{task_id}")
            if task is None:
                raise ValueError(f"No task package for {source_id} ({task_variant}::{task_id})")
            user_messages = [
                value for message in record.get("conversations", []) if (value := _message_value(message)) is not None
            ]
            if not user_messages:
                raise ValueError(f"No initial user prompt in {source_id}")
            prompt = user_messages[0]
            if prompt != task["prompt"]:
                raise ValueError(f"SFT prompt differs from the immutable ClawEval task package: {source_id}")
            prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            rows.append(
                {
                    "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                    "verifier_metadata": {
                        "source_id": source_id,
                        "task_id": task_id,
                        "task_variant": task_variant,
                        "family": task["family"],
                        "prompt_sha256": prompt_sha256,
                        "task_package_sha256": task["task_package_sha256"],
                    },
                }
            )
    return rows


def dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = [row["verifier_metadata"]["task_id"] for row in rows]
    families = Counter(row["verifier_metadata"]["family"] for row in rows)
    variants = Counter(row["verifier_metadata"]["task_variant"] for row in rows)
    return {
        "rows": len(rows),
        "unique_tasks": len(set(task_ids)),
        "families": dict(sorted(families.items())),
        "variants": dict(sorted(variants.items())),
    }


DOCKERFILE = """\
FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

RUN apt-get -o APT::Sandbox::User=root update \\
    && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends \\
        ffmpeg git ripgrep unzip \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
        \"httpx>=0.27,<1\" \"matplotlib>=3.9,<4\" \"numpy<3\" \\
        \"openai>=1.60,<3\" \"opencv-python-headless==4.13.*\" \\
        \"pillow>=10,<13\" \"pydantic>=2.10,<3\" \"pyyaml>=6,<7\"

WORKDIR /workspace
RUN mkdir -p /workspace/fixtures /workspace/output /workspace/work
"""


TEST_SCRIPT = """\
#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier /logs/artifacts
python3 /tests/judge_claweval.py
find /workspace -maxdepth 2 -type f -not -path '/workspace/fixtures/*' -exec cp --parents '{}' /logs/artifacts/ \\; 2>/dev/null || true
"""


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve(strict=True)
    # Docker Compose preserves a hard link's numeric ownership while uploading
    # a Harbor workdir.  Shared media caches can be owned by an unmapped service
    # uid (commonly 65534), which makes rootless Docker reject the upload at
    # lchown time.  Copy foreign-owned fixtures so the staged inode is owned by
    # the launching user; retain hard links for user-owned media to avoid
    # needlessly duplicating large videos.
    if source.stat().st_uid != os.geteuid():
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_harbor_task(
    task: dict[str, Any],
    prompt: str,
    destination: str | Path,
    *,
    claw_eval_source: str | Path,
) -> Path:
    """Create a Harbor task while keeping all grader material verifier-only."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source_dir = Path(task["task_yaml"]).parent.resolve(strict=True)
    claw_eval_source = Path(claw_eval_source).resolve(strict=True)
    environment_dir = destination / "environment"
    step_dir = destination / "steps" / "solve"
    workdir = step_dir / "workdir"
    tests = step_dir / "tests"
    solution = step_dir / "solution"
    for directory in (environment_dir, workdir, tests, solution):
        directory.mkdir(parents=True, exist_ok=False)

    definition = yaml.safe_load(Path(task["task_yaml"]).read_text())
    definition["prompt"]["text"] = prompt
    timeout = int(definition.get("environment", {}).get("timeout_seconds", 1200))
    verifier_timeout = max(1800, int(definition.get("environment", {}).get("env_snapshot_timeout", 600)) + 600)
    (destination / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[task]",
                f'name = "claweval/{task["task_id"]}"',
                "",
                "[metadata]",
                f"category = {json.dumps(task['family'])}",
                'source = "ClawEval task package; prompt selected from successful OpenCode trace"',
                "",
                "[environment]",
                "build_timeout_sec = 1200.0",
                "cpus = 4",
                "memory_mb = 16384",
                "storage_mb = 32768",
                "allow_internet = true",
                "",
                "[[steps]]",
                'name = "solve"',
                "",
                "[steps.agent]",
                f"timeout_sec = {float(timeout):.1f}",
                "",
                "[steps.verifier]",
                f"timeout_sec = {float(verifier_timeout):.1f}",
                "",
                "[steps.verifier.env]",
                'CLAW_EVAL_JUDGE_API_KEY = "${CLAW_EVAL_JUDGE_API_KEY}"',
                'CLAW_EVAL_JUDGE_BASE_URL = "${CLAW_EVAL_JUDGE_BASE_URL}"',
                'CLAW_EVAL_JUDGE_MODEL = "${CLAW_EVAL_JUDGE_MODEL}"',
                "",
            ]
        )
    )
    (environment_dir / "Dockerfile").write_text(DOCKERFILE)
    (step_dir / "instruction.md").write_text(prompt)
    (workdir / "AGENTS.md").write_text(
        "# ClawEval media tools\n\n"
        "The task fixture is under `/workspace/fixtures`. Use `claw_ReadMedia` "
        "to inspect images or bounded video clips; it returns chronological image "
        "frames to the model. At most 64 media frames are retained in one episode. "
        "Use `claw_BrowserScreenshot` to inspect generated webpages or animations. "
        "You may also use ffmpeg and the ordinary OpenCode file tools.\n"
    )
    tools_dir = workdir / ".claw_tools"
    tools_dir.mkdir()
    shutil.copy2(Path(__file__).with_name("mcp_server.py"), tools_dir / "mcp_server.py")
    (solution / "solve.sh").write_text("#!/bin/bash\n# No golden solution is exposed to training.\n")
    (tests / "test.sh").write_text(TEST_SCRIPT)
    shutil.copy2(Path(__file__).with_name("judge_claweval.py"), tests / "judge_claweval.py")
    shutil.copy2(Path(__file__).with_name("judge_transport.py"), tests / "judge_transport.py")
    (tests / "task.yaml").write_text(yaml.safe_dump(definition, sort_keys=False, allow_unicode=True))
    shutil.copy2(source_dir / "grader.py", tests / "grader.py")

    for relative in definition.get("sandbox_files", []):
        source = source_dir / relative
        if not source.exists():
            raise FileNotFoundError(source)
        _copy_or_link(source, workdir / relative)
    for key in ("sandbox_grader_files", "local_grader_files"):
        for relative in definition.get(key, []):
            source = source_dir / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            target = tests / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    package_source = claw_eval_source / "src" / "claw_eval"
    if not package_source.is_dir():
        raise FileNotFoundError(package_source)
    shutil.copytree(package_source, tests / "claw_eval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "staging_manifest.json").write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "task_variant": task["task_variant"],
                "family": task["family"],
                "source_task_yaml": task["task_yaml"],
                "task_package_sha256": task["task_package_sha256"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "claw_eval_source": str(claw_eval_source),
                "fixtures": task["fixtures"],
                "private_grader_material": "steps/solve/tests (uploaded only for verification)",
                "agent_media_tools": ["claw_ReadMedia", "claw_BrowserScreenshot"],
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def read_result(output: Path, task: dict[str, Any]) -> dict[str, Any]:
    task_results = sorted(output.glob("jobs/trial/*/result.json"))
    if len(task_results) != 1:
        raise RuntimeError(f"Expected one Harbor trial result, found {len(task_results)}")
    trial = json.loads(task_results[0].read_text())
    task_name = trial.get("task_name", "").removeprefix("claweval/")
    if task_name != task["task_id"]:
        raise ValueError(f"Harbor returned mismatched ClawEval task {task_name}")
    trial_dir = task_results[0].parent
    reward_files = list(trial_dir.glob("steps/solve/verifier/reward.json"))
    if len(reward_files) != 1:
        raise RuntimeError(f"Missing native ClawEval reward: {trial_dir}")
    reward_record = json.loads(reward_files[0].read_text())
    reward = reward_record.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise ValueError(f"Invalid ClawEval reward: {reward}")
    trajectories = list(trial_dir.glob("steps/solve/agent/trajectory.json"))
    timed_out = any(
        isinstance(info, dict) and info.get("exception_type") in {"AgentTimeoutError", "TimeoutError"}
        for info in [trial.get("exception_info")]
        + [(step or {}).get("exception_info") for step in trial.get("step_results") or []]
    )
    if trajectories:
        if len(trajectories) != 1:
            raise RuntimeError(f"Expected one model trajectory: {trial_dir}")
        trajectory = json.loads(trajectories[0].read_text())
    elif timed_out:
        trajectory = {"steps": []}
    else:
        raise RuntimeError(f"Missing model trajectory: {trial_dir}")
    return {
        "reward": float(reward),
        "status": "AGENT_TIMEOUT" if timed_out else "OK",
        "trajectory": trajectory,
        "artifacts": str(output),
        "completion": reward_record.get("completion"),
        "robustness": reward_record.get("robustness"),
        "safety": reward_record.get("safety"),
        "passed": reward_record.get("passed"),
    }


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    rewards = [float(row["reward"]) for row in rows]
    result = {"mean/reward": sum(rewards) / len(rewards), "task_count": float(len(rows))}
    by_family: dict[str, list[float]] = {}
    for row in rows:
        by_family.setdefault(str(row.get("family", "unknown")), []).append(float(row["reward"]))
    result.update({f"mean/reward_{family}": sum(values) / len(values) for family, values in by_family.items()})
    return result
