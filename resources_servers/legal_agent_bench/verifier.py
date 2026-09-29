# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Harbor entrypoint and result writer for Legal Agent Bench verification.

Generated Harbor tasks place the adjacent ``lab_harbor`` verifier package in
``/tests``. Repository imports use the vendored package at the pinned source
revision.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__:
    from .vendor.harvey_labs.lab_harbor.judge import (
        OpenAICompatibleJudge,
        write_transcript_event,
    )
    from .vendor.harvey_labs.lab_harbor.scoring import score_rubric
else:
    from lab_harbor.judge import OpenAICompatibleJudge, write_transcript_event
    from lab_harbor.scoring import score_rubric


DEFAULT_REWARD_MODE = "full_task"
REWARD_MODE_ENV_KEY = "LEGAL_AGENT_BENCH_REWARD_MODE"
REWARD_MODES = {"full_task", "criteria_pass_rate"}


def _validate_reward_mode(value: str) -> str:
    if value not in REWARD_MODES:
        raise ValueError(f"Invalid {REWARD_MODE_ENV_KEY}: {value!r}; expected one of {sorted(REWARD_MODES)}")
    return value


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_report(scores: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for item in scores.get("criteria_results", []):
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(item.get('id', '')))}</td>"
            f"<td>{html_lib.escape(str(item.get('verdict', '')).upper())}</td>"
            f"<td>{html_lib.escape(str(item.get('title', '')))}</td>"
            f"<td>{html_lib.escape(str(item.get('reasoning', '')))}</td>"
            "</tr>"
        )
    report_html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>LAB Score - {html_lib.escape(str(scores['run_id']))}</title>"
        "<style>body{font-family:sans-serif;max-width:1000px;margin:32px auto;}"
        "td,th{border:1px solid #ddd;padding:8px;vertical-align:top;}"
        "table{border-collapse:collapse;width:100%;}</style></head><body>"
        f"<h1>{html_lib.escape(str(scores['summary']))}</h1>"
        f"<p>Score: {scores['score']:.2f} | Judge: {html_lib.escape(str(scores['judge_model']))}</p>"
        "<table><thead><tr><th>ID</th><th>Verdict</th><th>Title</th><th>Reasoning</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")


def run_verifier(args: argparse.Namespace) -> dict[str, Any]:
    task_config = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    verifier_dir = Path(args.verifier_dir)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = verifier_dir / "transcript.jsonl"
    config = _read_json_if_exists(run_dir / "config.json")
    metrics = _read_json_if_exists(run_dir / "metrics.json")

    judge_model = os.environ.get("LAB_JUDGE_MODEL", args.judge_model)
    judge_kwargs = {
        "model": judge_model,
        "base_url": os.environ.get("LAB_JUDGE_BASE_URL") or args.judge_base_url,
        "api_key": os.environ.get("LAB_JUDGE_API_KEY") or args.judge_api_key,
        "temperature": _optional_float(os.environ.get("LAB_JUDGE_TEMPERATURE")),
        "timeout_seconds": float(
            os.environ.get("LAB_JUDGE_REQUEST_TIMEOUT_SECONDS") or os.environ.get("LAB_JUDGE_TIMEOUT_SECONDS", "90")
        ),
        "max_retries": int(os.environ.get("LAB_JUDGE_MAX_RETRIES", "1")),
        "structured_output": _optional_bool(
            os.environ.get("LAB_JUDGE_STRUCTURED_OUTPUT"),
            default=True,
        ),
        "parse_repair_attempts": int(os.environ.get("LAB_JUDGE_PARSE_REPAIR_ATTEMPTS", "1")),
        "repair_max_tokens": int(os.environ.get("LAB_JUDGE_REPAIR_MAX_TOKENS", "4096")),
        "transcript_path": transcript_path,
    }

    def judge_factory() -> OpenAICompatibleJudge:
        return OpenAICompatibleJudge(**judge_kwargs)

    judge = judge_factory()
    judge_parallelism = max(1, int(os.environ.get("LAB_JUDGE_PARALLELISM", "6")))

    result = score_rubric(
        criteria=task_config["criteria"],
        run_dir=run_dir,
        judge=judge,
        task_desc=f"{task_config['title']}\n\n{task_config['instructions']}",
        judge_max_tokens=int(os.environ.get("LAB_JUDGE_MAX_TOKENS", "4096")),
        transcript_path=transcript_path,
        judge_factory=judge_factory if judge_parallelism > 1 else None,
        parallelism=judge_parallelism,
    )

    scores = {
        **result,
        "run_id": config.get("run_id") or metrics.get("run_id") or "harbor-run",
        "task": config.get("task") or metrics.get("task") or task_config.get("title", ""),
        "judge_model": judge_model,
        "judge_model_base_url": judge.base_url,
        "judge_parallelism": judge_parallelism,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    if metrics:
        scores["cost"] = {
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "wall_clock_seconds": metrics.get("wall_clock_seconds", 0),
        }
        scores["doc_coverage"] = {
            "documents_read": metrics.get("documents_read", 0),
            "total_vdr_files": metrics.get("total_vdr_files", 0),
            "documents_skipped": metrics.get("documents_skipped", 0),
            "documents_read_list": metrics.get("documents_read_list", []),
            "documents_skipped_list": metrics.get("documents_skipped_list", []),
        }
    if config:
        scores["agent"] = {
            key: value
            for key, value in {
                "id": config.get("agent_id"),
                "config_id": config.get("agent_config_id"),
                "model": config.get("model") or metrics.get("model"),
                "model_base_url": config.get("agent_model_base_url"),
                "reasoning_effort": config.get("reasoning_effort"),
                "temperature": config.get("temperature"),
                "top_p": config.get("agent_model_top_p"),
                "max_tokens": config.get("agent_model_max_tokens"),
                "timeout_seconds": config.get("agent_model_timeout_seconds"),
                "tool_runtime": config.get("tool_runtime"),
            }.items()
            if value is not None
        }

    reward_mode = _validate_reward_mode(os.environ.get(REWARD_MODE_ENV_KEY, DEFAULT_REWARD_MODE))
    reward = _build_reward(scores, reward_mode)
    scores["reported_reward"] = reward["reward"]
    scores["reported_reward_mode"] = reward_mode
    (verifier_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    _write_report(scores, verifier_dir)
    reward_path = Path(args.reward_json)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps(reward, indent=2), encoding="utf-8")
    return scores


def _build_reward(scores: dict[str, Any], reward_mode: str) -> dict[str, float | int]:
    reward_mode = _validate_reward_mode(reward_mode)
    criteria_pass_rate = float(scores["n_passed"]) / float(scores["n_criteria"]) if scores["n_criteria"] else 0.0
    reported_reward = float(scores["score"]) if reward_mode == "full_task" else criteria_pass_rate
    return {
        "reward": reported_reward,
        "score": float(scores["score"]),
        "criteria_pass_rate": criteria_pass_rate,
        "judge_error_count": int(scores.get("judge_error_count") or 0),
        "judge_error_rate": (
            float(scores.get("judge_error_count") or 0) / float(scores["n_criteria"]) if scores["n_criteria"] else 0.0
        ),
        "n_passed": int(scores["n_passed"]),
        "n_criteria": int(scores["n_criteria"]),
        "all_pass": 1 if scores["all_pass"] else 0,
    }


def write_failure_outputs(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    task_config = _read_json_if_exists(Path(args.task_json))
    run_dir = Path(args.run_dir)
    config = _read_json_if_exists(run_dir / "config.json")
    metrics = _read_json_if_exists(run_dir / "metrics.json")
    criteria = task_config.get("criteria", [])
    error_type = type(exc).__name__
    message = str(exc)

    scores = {
        "score": 0.0,
        "all_pass": False,
        "n_passed": 0,
        "n_criteria": len(criteria),
        "criteria_results": [],
        "summary": f"Verifier error: {error_type}",
        "error": message,
        "error_type": error_type,
        "run_id": config.get("run_id") or metrics.get("run_id") or "harbor-run",
        "task": config.get("task") or metrics.get("task") or task_config.get("title", ""),
        "judge_model": os.environ.get("LAB_JUDGE_MODEL", args.judge_model),
        "judge_model_base_url": (os.environ.get("LAB_JUDGE_BASE_URL") or args.judge_base_url),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }

    verifier_dir = Path(args.verifier_dir)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    error = {"error": message, "error_type": error_type}
    write_transcript_event(
        verifier_dir / "transcript.jsonl",
        {"type": "verifier_error", **error},
    )
    (verifier_dir / "error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
    (verifier_dir / "scores.json").write_text(
        json.dumps(scores, indent=2),
        encoding="utf-8",
    )
    _write_report(scores, verifier_dir)
    reward = {
        "reward": 0.0,
        "score": 0.0,
        "criteria_pass_rate": 0.0,
        "judge_error_count": len(criteria),
        "judge_error_rate": 1.0 if criteria else 0.0,
        "n_passed": 0,
        "n_criteria": len(criteria),
        "all_pass": 0,
        "verifier_error": 1,
    }
    reward_path = Path(args.reward_json)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps(reward, indent=2), encoding="utf-8")
    return scores


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bool(value: str | None, *, default: bool) -> bool:
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a Legal Agent Bench Harbor trial.")
    parser.add_argument("--task-json", default="/tests/task.json")
    parser.add_argument("--run-dir", default="/logs/agent/artifacts/lab-run")
    parser.add_argument("--verifier-dir", default="/logs/verifier")
    parser.add_argument("--reward-json", default="/logs/verifier/reward.json")
    parser.add_argument("--judge-model", default="openai-compatible/model")
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scores = run_verifier(args)
    except Exception as exc:
        # Harbor expects reward.json even when judging fails. Preserve a
        # structured trial result and expose verifier_error/judge_error metrics
        # instead of turning the infrastructure failure into a missing result.
        scores = write_failure_outputs(args, exc)
        print(scores["summary"], file=sys.stderr)
        return 0
    print(scores["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
