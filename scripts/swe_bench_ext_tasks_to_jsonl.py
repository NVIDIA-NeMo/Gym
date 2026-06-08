#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert SWE-Bench-Ext task folders into NeMo Gym rollout JSONL.

The input paths can be task roots containing many task subdirectories, or
individual task directories. No delivery root is assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


INTERFACE_JOINER = "\n\nFollowing are the new interfaces that need to be implemented:\n"

METADATA_FILES = {
    "prompt_statement": "prompt_statement.md",
    "test_patch": "test.patch",
    "golden_patch": "golden.patch",
    "test_metadata": "test_metadata.json",
    "rubric": "rubric/rubric.json",
    "knowledge_base": "knowledge_base.md",
    "requirements": "requirements.json",
    "interface": "interface.md",
    "tool_schema": "tool_schema.json",
}

JSON_FIELDS = {"test_metadata", "rubric", "requirements", "tool_schema"}
REQUIRED_FIELDS = ("prompt_statement", "test_patch", "golden_patch", "test_metadata", "rubric")


def read_file(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    return text if text.strip() else None


def repo_from_url(repo_url: str) -> str:
    repo_url = repo_url.strip()
    if not repo_url:
        return ""
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    if "github.com/" in repo_url:
        return repo_url.split("github.com/", 1)[1].strip("/")
    if repo_url.startswith("git@") and ":" in repo_url:
        return repo_url.split(":", 1)[1].strip("/")
    return ""


def looks_like_task_dir(path: Path) -> bool:
    return (path / METADATA_FILES["prompt_statement"]).is_file() and (
        path / METADATA_FILES["test_metadata"]
    ).is_file()


def iter_task_dirs(paths: list[Path]) -> list[Path]:
    task_dirs: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {path}")
        if looks_like_task_dir(path):
            task_dirs.append(path)
            continue
        task_dirs.extend(sorted(child for child in path.iterdir() if child.is_dir()))
    return task_dirs


def load_raw_task_files(task_dir: Path) -> Optional[dict[str, str]]:
    raw: dict[str, str] = {}
    for key, filename in METADATA_FILES.items():
        content = read_file(task_dir / filename)
        if content is None:
            if key in REQUIRED_FIELDS:
                return None
            raw[key] = ""
            continue

        if key in JSON_FIELDS:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                if key in REQUIRED_FIELDS:
                    return None
                raw[key] = content
            else:
                raw[key] = json.dumps(parsed, indent=2)
        else:
            raw[key] = content
    return raw


def build_record(
    task_dir: Path,
    *,
    dataset_name: str,
    split: str,
    agent_ref_type: str,
    agent_ref_name: str,
    model: str,
    temperature: float,
    top_p: float,
) -> Optional[dict]:
    raw = load_raw_task_files(task_dir)
    if raw is None:
        return None

    instance_id = task_dir.name
    prompt = raw["prompt_statement"]
    interface = raw["interface"]
    problem_statement = prompt + (INTERFACE_JOINER + interface if interface else "")

    try:
        test_metadata = json.loads(raw["test_metadata"]) if raw["test_metadata"] else {}
    except json.JSONDecodeError:
        test_metadata = {}

    language = test_metadata.get("language", "") or ""
    test_framework = test_metadata.get("test_framework", "") or ""
    test_command = test_metadata.get("test_command", "") or ""
    test_files = test_metadata.get("test_files", []) or []
    fail_to_pass = test_metadata.get("FAIL_TO_PASS", []) or []
    pass_to_pass = test_metadata.get("PASS_TO_PASS", []) or []
    repo = (
        test_metadata.get("repo")
        or test_metadata.get("repo_name")
        or repo_from_url(test_metadata.get("repo_url", "") or "")
    )
    base_commit = test_metadata.get("base_commit", "") or ""

    instance_dict = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "patch": raw["golden_patch"],
        "golden_patch": raw["golden_patch"],
        "test_patch": raw["test_patch"],
        "problem_statement": problem_statement,
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
        "language": language,
        "repo_language": language,
        "interface": interface,
        "test_framework": test_framework,
        "test_command": test_command,
        "test_files": test_files,
        "requirements": raw["requirements"],
        "rubric": raw["rubric"],
        "knowledge_base": raw["knowledge_base"],
        "prompt_statement": prompt,
        "tool_schema": raw["tool_schema"],
    }
    for key in ("command_framework", "parser_framework"):
        if key in test_metadata:
            instance_dict[key] = test_metadata[key]

    metadata = {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "prompt_statement": prompt,
        "test_patch": raw["test_patch"],
        "golden_patch": raw["golden_patch"],
        "test_metadata": raw["test_metadata"],
        "rubric": raw["rubric"],
        "knowledge_base": raw["knowledge_base"],
        "requirements": raw["requirements"],
        "interface": interface,
        "tool_schema": raw["tool_schema"],
        "dataset_name": dataset_name,
        "split": split,
        "repo": repo,
        "base_commit": base_commit,
        "instance_dict": json.dumps(instance_dict),
    }

    return {
        "instance_id": instance_id,
        "responses_create_params": {
            "input": [],
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "metadata": metadata,
        },
        "agent_ref": {"type": agent_ref_type, "name": agent_ref_name},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_paths", nargs="+", help="Task roots or individual task directories")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL path")
    parser.add_argument("--dataset-name", default="swe-bench-ext")
    parser.add_argument("--split", default="train")
    parser.add_argument("--agent-ref-type", default="responses_api_agents")
    parser.add_argument("--agent-ref-name", default="swe_agents")
    parser.add_argument("--model", default="default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_paths = [Path(path) for path in args.task_paths]

    try:
        task_dirs = iter_task_dirs(task_paths)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    records: dict[str, dict] = {}
    skipped = 0
    dupes = 0

    for task_dir in task_dirs:
        record = build_record(
            task_dir,
            dataset_name=args.dataset_name,
            split=args.split,
            agent_ref_type=args.agent_ref_type,
            agent_ref_name=args.agent_ref_name,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if record is None:
            skipped += 1
            continue
        if record["instance_id"] in records:
            dupes += 1
        records[record["instance_id"]] = record

    if not records:
        print("Error: no valid task records were produced", file=sys.stderr)
        print(f"Scanned {len(task_dirs)} candidate task directories; skipped {skipped}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records.values():
            f.write(json.dumps(record) + "\n")

    print(
        f"Wrote {len(records)} task(s) to {output_path} "
        f"(scanned {len(task_dirs)}, skipped {skipped}, deduped {dupes})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
