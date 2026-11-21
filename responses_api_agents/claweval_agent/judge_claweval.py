# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-only entrypoint for a staged ClawEval task."""

from __future__ import annotations

import base64
import glob
import inspect
import json
import mimetypes
import os
import subprocess
import sys
from pathlib import Path


TESTS = Path("/tests")
WORKSPACE = Path("/workspace")
sys.path.insert(0, str(TESTS))

from claw_eval.graders.registry import get_grader  # noqa: E402
from claw_eval.models.scoring import compute_task_score, is_pass  # noqa: E402
from claw_eval.models.task import TaskDefinition  # noqa: E402
from judge_transport import create_judge  # noqa: E402


def _read_file(path: Path) -> dict:
    if not path.is_file():
        return {"error": f"not found: {path}"}
    mime = mimetypes.guess_type(path.name)[0]
    if mime and (mime.startswith("text/") or mime in {"application/json", "application/xml"}):
        return {"encoding": "utf-8", "content": path.read_text(errors="replace")}
    if path.suffix.lower() in {".csv", ".html", ".json", ".md", ".txt", ".xml"}:
        return {"encoding": "utf-8", "content": path.read_text(errors="replace")}
    return {
        "encoding": "base64",
        "content": base64.b64encode(path.read_bytes()).decode(),
        "mime_type": mime or "application/octet-stream",
    }


def collect_snapshot(task: TaskDefinition) -> dict:
    snapshot: dict = {}
    timeout = task.environment.env_snapshot_timeout
    for command in task.env_snapshot_commands:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        snapshot[f"cmd:{command}"] = {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    for pattern in task.env_snapshot_files:
        matches = sorted(glob.glob(pattern)) if any(marker in pattern for marker in "*?") else [pattern]
        if not matches:
            snapshot[f"file:{pattern}"] = {"error": f"no files matched: {pattern}"}
        for value in matches[:50]:
            snapshot[f"file:{value}"] = _read_file(Path(value))
    for relative in task.local_grader_files:
        snapshot[f"local_file:{relative}"] = _read_file(TESTS / relative)
    return snapshot


def main() -> None:
    task = TaskDefinition.from_yaml(TESTS / "task.yaml")
    for relative in task.sandbox_grader_files:
        source = TESTS / relative
        destination = WORKSPACE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    snapshot = collect_snapshot(task)
    key = os.environ.get("CLAW_EVAL_JUDGE_API_KEY")
    if not key:
        raise RuntimeError("CLAW_EVAL_JUDGE_API_KEY is required by the native ClawEval judge")
    base_url = os.environ.get("CLAW_EVAL_JUDGE_BASE_URL")
    model = os.environ.get("CLAW_EVAL_JUDGE_MODEL")
    if not base_url or not model:
        raise RuntimeError("ClawEval judge base URL and model must be configured")
    judge = create_judge(
        model_id=model,
        api_key=key,
        base_url=base_url,
    )
    grader = get_grader(task.task_id, tasks_dir=TESTS.parent, task_dir=TESTS)
    kwargs = {
        "audit_data": None,
        "judge": judge,
        "media_events": [],
        "env_snapshot": snapshot,
    }
    accepted = inspect.signature(grader.grade).parameters
    scores = grader.grade([], [], task, **{key: value for key, value in kwargs.items() if key in accepted})
    reward = compute_task_score(scores)
    record = {
        "reward": reward,
        "completion": scores.completion,
        "robustness": scores.robustness,
        "safety": scores.safety,
        "passed": is_pass(reward),
        "task_id": task.task_id,
    }
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(json.dumps(record, indent=2) + "\n")
    (logs / "reward.txt").write_text(f"{reward}\n")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
