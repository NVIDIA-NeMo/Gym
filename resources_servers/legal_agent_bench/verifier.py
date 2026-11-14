# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Original Harvey LAB portions remain under MIT; NVIDIA modifications follow.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Harvey LAB source (MIT): https://github.com/harveyai/harvey-labs/blob/
# f46ef86e4788545622db25dcffa3aebb7a139929/evaluation/scoring.py
"""Harbor entrypoint and result writer for Legal Agent Bench verification.

Generated Harbor tasks place the adjacent ``lab_harbor`` verifier package in
``/tests``. Repository imports use the adjacent package with adaptations from the pinned
source revision.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


if __package__:
    from .lab_harbor.document_ids import CORRECTION_VERSION, document_id_aliases
    from .lab_harbor.judge import (
        OpenAICompatibleJudge,
        write_transcript_event,
    )
    from .lab_harbor.judge_config import validate_judges
    from .lab_harbor.scoring import score_rubric
else:
    from lab_harbor.document_ids import CORRECTION_VERSION, document_id_aliases
    from lab_harbor.judge import OpenAICompatibleJudge, write_transcript_event
    from lab_harbor.judge_config import validate_judges
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
        f"<p>Score: {scores['score']:.2f} | Judge: {html_lib.escape(str(scores.get('judge_model', 'unavailable')))}</p>"
        "<table><thead><tr><th>ID</th><th>Verdict</th><th>Title</th><th>Reasoning</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")


def _document_tool_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("python-docx", "openpyxl", "pandas", "python-pptx", "pdfplumber", "markitdown"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _run_single_verifier(args: argparse.Namespace, *, judge_override: dict[str, Any]) -> dict[str, Any]:
    task_config = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
    metadata = task_config.get("metadata") or {}
    aliases = document_id_aliases(
        task_id=metadata.get("lab_task_id"), source_revision=metadata.get("lab_source_revision")
    )
    run_dir = Path(args.run_dir)
    verifier_dir = Path(args.verifier_dir)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = verifier_dir / "transcript.jsonl"
    config = _read_json_if_exists(run_dir / "config.json")
    metrics = _read_json_if_exists(run_dir / "metrics.json")

    override = judge_override
    judge_model = override["model_name"]
    judge_kwargs = {
        "model": judge_model,
        "base_url": override.get("base_url") or os.environ.get("LAB_JUDGE_BASE_URL") or args.judge_base_url,
        "api_key": (
            override["api_key"]
            if override.get("api_key") is not None
            else os.environ.get("LAB_JUDGE_API_KEY") or args.judge_api_key
        ),
        "temperature": _optional_float(os.environ.get("LAB_JUDGE_TEMPERATURE")),
        "reasoning_effort": override.get("reasoning_effort") or os.environ.get("LAB_JUDGE_REASONING_EFFORT") or None,
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
    judge_parallelism = max(1, int(getattr(args, "judge_parallelism", os.environ.get("LAB_JUDGE_PARALLELISM", "6"))))

    result = score_rubric(
        criteria=task_config["criteria"],
        run_dir=run_dir,
        judge=judge,
        task_desc=f"{task_config['title']}\n\n{task_config['instructions']}",
        judge_max_tokens=int(os.environ.get("LAB_JUDGE_MAX_TOKENS", "4096")),
        transcript_path=transcript_path,
        judge_factory=judge_factory if judge_parallelism > 1 else None,
        parallelism=judge_parallelism,
        document_id_aliases=aliases,
    )

    scores = {
        **result,
        "run_id": config.get("run_id") or metrics.get("run_id") or "harbor-run",
        "task": config.get("task") or metrics.get("task") or task_config.get("title", ""),
        "judge_model": judge_model,
        "judge_model_base_url": judge.base_url,
        "judge_parallelism": judge_parallelism,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": (task_config.get("metadata") or {}).get("lab_source_revision"),
        "harness_pin": config.get("harness_pin") or config.get("harness_config"),
        "gym_commit": config.get("gym_commit"),
        "completion": {"status": metrics.get("status"), "finish_reason": metrics.get("finish_reason")},
        "usage": metrics.get("usage", {}),
        "judge_profile": "single",
        "document_tool_versions": _document_tool_versions(),
        "grading_corrections": [CORRECTION_VERSION] if aliases else [],
        "document_id_aliases": aliases,
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
    if scores.get("judge_error_count"):
        raise RuntimeError(f"LAB criteria could not be graded: {scores['judge_error_count']}")
    reward_path = Path(args.reward_json)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps(reward, indent=2), encoding="utf-8")
    return scores


def run_verifier(args: argparse.Namespace) -> dict[str, Any]:
    """Score with one explicit judge or average the official pair's task scores."""
    verifier_dir = Path(args.verifier_dir)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = verifier_dir / "scores_dual.json"
    # Replays may reuse an output directory. Never mix an earlier successful
    # judge report/reward with this attempt's failure diagnostics.
    for directory in (verifier_dir, verifier_dir / "judge-1", verifier_dir / "judge-2"):
        for name in (
            "scores.json",
            "scores_dual.json",
            "reward.json",
            "error.json",
            "report.html",
            "transcript.jsonl",
        ):
            (directory / name).unlink(missing_ok=True)
    Path(args.reward_json).unlink(missing_ok=True)
    try:
        raw = json.loads(os.environ.get("LAB_JUDGES", "null"))
    except json.JSONDecodeError:
        raise ValueError("LAB_JUDGES must be a JSON list of one or two judge configurations") from None
    judges = validate_judges(raw)
    if len(judges) == 1:
        return _run_single_verifier(args, judge_override=judges[0])
    canonical = [judge["model_name"].split("/")[-1] for judge in judges]
    per_judge = {}
    errors = []
    total_parallelism = max(1, int(os.environ.get("LAB_JUDGE_PARALLELISM", "6")))
    workers = min(len(judges), total_parallelism)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        for index, spec in enumerate(judges):
            child_args = copy(args)
            child_dir = verifier_dir / f"judge-{index + 1}"
            child_args.verifier_dir = str(child_dir)
            child_args.reward_json = str(child_dir / "reward.json")
            child_args.judge_parallelism = max(1, total_parallelism // workers)
            pending.append(pool.submit(_run_single_verifier, child_args, judge_override=spec))
        for model, future in zip(canonical, pending, strict=True):
            try:
                per_judge[model] = future.result()
            except Exception as exc:
                # Each judge retains its own diagnostics; wait for both before rejecting the aggregate.
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("Dual grading incomplete: " + "; ".join(errors))
    values = list(per_judge.values())
    task_score = sum(float(value["score"]) for value in values) / 2
    criterion_rate = (
        sum(value["n_passed"] / value["n_criteria"] if value["n_criteria"] else 0.0 for value in values) / 2
    )
    scores = {
        "judge_profile": "lab-standard-dual-v1",
        "judges": canonical,
        "per_judge": per_judge,
        "score": task_score,
        "dual_all_pass_rate": task_score,
        "dual_criterion_pass": criterion_rate,
        "all_pass": task_score == 1.0,
        "n_passed": sum(value["n_passed"] for value in values),
        "n_criteria": sum(value["n_criteria"] for value in values),
        "judge_error_count": 0,
        "run_id": values[0]["run_id"],
        "task": values[0]["task"],
        "source_revision": values[0].get("source_revision"),
        "harness_pin": values[0].get("harness_pin"),
        "gym_commit": values[0].get("gym_commit"),
        "pandoc_version": values[0].get("pandoc_version"),
        "document_tool_versions": values[0]["document_tool_versions"],
        "grading_corrections": values[0].get("grading_corrections", []),
        "document_id_aliases": values[0].get("document_id_aliases", {}),
        "summary": f"Dual task all-pass: {task_score:.1f}; criterion pass: {criterion_rate:.4f}",
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    mode = _validate_reward_mode(os.environ.get(REWARD_MODE_ENV_KEY, DEFAULT_REWARD_MODE))
    reward = _build_reward(scores, mode)
    scores.update(reported_reward=reward["reward"], reported_reward_mode=mode)
    aggregate_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    output = Path(args.reward_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reward, indent=2), encoding="utf-8")
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
    verifier_dir = Path(args.verifier_dir)

    scores = {
        "score": 0.0,
        "all_pass": False,
        "n_passed": 0,
        "n_criteria": len(criteria),
        "criteria_results": [],
        "run_id": config.get("run_id") or metrics.get("run_id") or "harbor-run",
        "task": config.get("task") or metrics.get("task") or task_config.get("title", ""),
        "judge_model_base_url": (os.environ.get("LAB_JUDGE_BASE_URL") or args.judge_base_url),
        # A single judge can complete some criteria before failing. Keep those
        # results and their provenance alongside the operational failure.
        **_read_json_if_exists(verifier_dir / "scores.json"),
        "summary": f"Verifier error: {error_type}",
        "error": message,
        "error_type": error_type,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }

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
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scores = run_verifier(args)
    except Exception as exc:
        # Preserve error diagnostics while allowing the caller to route an
        # unavailable verifier as a failure, never as a completed scored zero.
        scores = write_failure_outputs(args, exc)
        (Path(args.verifier_dir) / "scores_dual.json").unlink(missing_ok=True)
        print(scores["summary"], file=sys.stderr)
        return 1
    print(scores["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
