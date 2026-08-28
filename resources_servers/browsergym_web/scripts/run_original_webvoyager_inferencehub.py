# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an unmodified WebVoyager task with credentials loaded outside argv/logs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI as OpenAIClient


TASK = {
    "web_name": "ArXiv",
    "id": "ArXiv--13",
    "ques": "How many articles on ArXiv with 'SimCSE' in the title?",
    "web": "https://arxiv.org/",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[str, str, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("InferenceHub config must be a YAML mapping")
    base_url = str(config.get("policy_base_url") or "").rstrip("/")
    api_key = str(config.get("policy_api_key") or "").strip()
    model = str(config.get("policy_model_name") or "").strip()
    if not base_url or not api_key or not model:
        raise ValueError("InferenceHub config requires policy_base_url, policy_api_key, and policy_model_name")
    openai_base_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
    return openai_base_url, api_key, model


def _import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_task_dir(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("*/taskArXiv--13"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError("upstream WebVoyager did not create taskArXiv--13")
    return candidates[-1]


def _extract_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        match = re.search(r"ANSWER[; ]+\[?([^\]]*)\]?", content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _write_reconstructed_events(out_dir: Path, messages: list[dict[str, Any]], task_dir: Path) -> None:
    events = []
    sequence = 0
    step = 0
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            continue
        sequence += 1
        content = message.get("content")
        event = "model_response" if role == "assistant" else "observation"
        data: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp_source": "reconstructed_after_upstream_run",
        }
        if role == "assistant":
            match = re.search(r"Action:\s*(.+)", str(content), flags=re.IGNORECASE | re.DOTALL)
            data["raw_model_output"] = content
            data["benchmark_action"] = match.group(1).strip() if match else None
        else:
            screenshot = task_dir / f"screenshot{step + 1}.png"
            if screenshot.exists():
                data["screenshot"] = {
                    "path": str(screenshot.relative_to(out_dir)),
                    "size_bytes": screenshot.stat().st_size,
                    "sha256": _sha256(screenshot),
                }
            step += 1
        events.append(
            {
                "sequence": sequence,
                "timestamp": _utc_now(),
                "runtime": "selenium-original",
                "event": event,
                "task_id": TASK["id"],
                "step": max(0, step - (role == "assistant")),
                "data": data,
            }
        )
    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webvoyager-root", type=Path, required=True)
    parser.add_argument("--inferencehub-config", type=Path, required=True)
    parser.add_argument("--chromium-binary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--judge-images", type=int, default=3)
    args = parser.parse_args()

    started_at = _utc_now()
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_root = args.out_dir / "upstream-results"
    downloads = args.out_dir / "downloads"
    results_root.mkdir(exist_ok=True)
    downloads.mkdir(exist_ok=True)
    task_file = args.out_dir / "task.jsonl"
    task_file.write_text(json.dumps(TASK, ensure_ascii=False) + "\n", encoding="utf-8")

    base_url, api_key, model = _load_config(args.inferencehub_config)
    manifest = {
        "schema_version": 1,
        "created_at": started_at,
        "runtime": "selenium-original",
        "task": TASK,
        "policy": {
            "provider": "InferenceHub",
            "base_url": base_url.removesuffix("/v1"),
            "model": model,
            "max_tokens": 1000,
            "seed": None,
            "temperature": "upstream SDK default (upstream --temperature is not forwarded)",
        },
        "judge": {
            "provider": "InferenceHub",
            "model": model,
            "same_as_policy": True,
            "temperature": 0,
            "seed": 42,
            "screenshots": args.judge_images,
        },
        "headless": True,
        "viewport": {"width": 1024, "height": 768},
        "max_iter": args.max_iter,
        "webvoyager_root": str(args.webvoyager_root.resolve()),
        "chromium_binary": str(args.chromium_binary.resolve()),
        "credential_source": str(args.inferencehub_config.resolve()),
        "credential_recorded": False,
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    (args.out_dir / "commands.log").write_text(
        "run_original_webvoyager_inferencehub.py --webvoyager-root <path> "
        "--inferencehub-config <redacted-path> --chromium-binary <path> --out-dir <path>\n",
        encoding="utf-8",
    )

    webvoyager_root = args.webvoyager_root.resolve()
    sys.path.insert(0, str(webvoyager_root))
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(webvoyager_root)
        upstream = _import_from_path("webvoyager_upstream_run", webvoyager_root / "run.py")

        def client_factory(*_client_args, **_client_kwargs):
            return OpenAIClient(api_key=api_key, base_url=base_url)

        original_driver_config = upstream.driver_config

        def driver_config(run_args):
            options = original_driver_config(run_args)
            options.binary_location = str(args.chromium_binary.resolve())
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            return options

        upstream.OpenAI = client_factory
        upstream.driver_config = driver_config
        sys.argv = [
            str(webvoyager_root / "run.py"),
            "--test_file",
            str(task_file),
            "--max_iter",
            str(args.max_iter),
            "--api_model",
            model,
            "--output_dir",
            str(results_root),
            "--download_dir",
            str(downloads),
            "--max_attached_imgs",
            "1",
            "--headless",
            "--force_device_scale",
            "--fix_box_color",
            "--window_width",
            "1024",
            "--window_height",
            "768",
        ]
        upstream.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    task_dir = _find_task_dir(results_root)
    messages_path = task_dir / "interact_messages.json"
    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    answer = _extract_answer(messages)
    _write_reconstructed_events(args.out_dir, messages, task_dir)

    evaluator = _import_from_path("webvoyager_upstream_eval", webvoyager_root / "evaluation" / "auto_eval.py")
    judge_stdout = io.StringIO()
    judge_started = time.monotonic()
    with contextlib.redirect_stdout(judge_stdout):
        judge_score = evaluator.auto_eval_by_gpt4v(
            str(task_dir),
            OpenAIClient(api_key=api_key, base_url=base_url),
            model,
            args.judge_images,
        )
    judge_ms = round((time.monotonic() - judge_started) * 1000)
    (args.out_dir / "judge.log").write_text(judge_stdout.getvalue(), encoding="utf-8")

    screenshots = []
    for screenshot in sorted(task_dir.glob("screenshot*.png")):
        screenshots.append(
            {
                "path": str(screenshot.relative_to(args.out_dir)),
                "size_bytes": screenshot.stat().st_size,
                "sha256": _sha256(screenshot),
            }
        )
    summary = {
        "status": "completed",
        "runtime": "selenium-original",
        "task_id": TASK["id"],
        "answer": answer,
        "model_turns": sum(message.get("role") == "assistant" for message in messages),
        "screenshots": screenshots,
        "judge_score": judge_score,
        "judge_duration_ms": judge_ms,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "task_dir": str(task_dir),
    }
    _write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
