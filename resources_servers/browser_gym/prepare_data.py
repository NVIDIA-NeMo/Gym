#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch tasks from a Browser Gym and write Gym-compatible JSONL.

Queries a gym's /api/v1/get_expected_state endpoint, converts each task
into the standard NeMo Gym JSONL format (responses_create_params +
verifier_metadata), and writes the result to a local file.

Usage:
    # Fetch all tasks:
    python prepare_data.py \
        --gym-url https://your-gym-url.com \
        --output data/tasks.jsonl

    # Fetch specific tasks by ID:
    python prepare_data.py \
        --gym-url https://your-gym-url.com \
        --task-id TASK-001 TASK-002 \
        --output data/tasks.jsonl

Then run rollout collection as usual:
    ng_collect_rollouts \
        +agent_name=browser_openai_agent \
        +input_jsonl_fpath=resources_servers/browser_gym/data/tasks.jsonl \
        +output_jsonl_fpath=results/cua_rollouts.jsonl \
        +num_repeats=5
"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


def fetch_gym_tasks(gym_url: str, task_ids: Optional[List[str]] = None) -> List[Dict]:
    """Fetch tasks from a gym's /api/v1/get_expected_state endpoint.

    Returns a list of dicts in NeMo Gym JSONL format, each containing
    responses_create_params and verifier_metadata.
    """
    endpoint = f"{gym_url.rstrip('/')}/api/v1/get_expected_state"
    print(f"Fetching tasks from {endpoint} ...")

    req = urllib.request.Request(endpoint, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ValueError(f"Failed to fetch tasks from {endpoint}: HTTP {e.code} — {e.reason}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Could not connect to gym at {endpoint}: {e.reason}") from e

    verifiers = payload.get("verifiers", {})
    if not verifiers:
        raise ValueError(f"No verifiers found in response from {endpoint}")

    if task_ids:
        missing_ids = [tid for tid in task_ids if tid not in verifiers]
        if missing_ids:
            available = ", ".join(sorted(verifiers.keys())[:20])
            raise ValueError(
                f"Task ID(s) not found in gym at {gym_url}: {', '.join(missing_ids)}. "
                f"Available task IDs ({len(verifiers)} total): {available}"
            )
        verifiers = {tid: verifiers[tid] for tid in task_ids}

    rows: List[Dict] = []
    for tid, details in verifiers.items():
        prompt = ""
        if isinstance(details, dict):
            prompt = details.get("task_statement") or details.get("prompt", "")

        start_url = details.get("start_url", gym_url) if isinstance(details, dict) else gym_url

        viewport = details.get("viewport_size") if isinstance(details, dict) else None
        if isinstance(viewport, list) and len(viewport) == 2:
            viewport = {"width": viewport[0], "height": viewport[1]}
        elif not isinstance(viewport, dict):
            viewport = {"width": 1280, "height": 720}

        rows.append(
            {
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "verifier_metadata": {
                    "task_id": tid,
                    "gym_url": gym_url,
                    "start_url": start_url,
                    "viewport": viewport,
                },
            }
        )

    print(f"Fetched {len(rows)} task(s) from gym")
    return rows


def write_jsonl(rows: List[Dict], output_path: str) -> int:
    """Write rows as JSONL. Returns the number of rows written."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch Browser Gym tasks and write Gym-compatible JSONL")
    parser.add_argument("--gym-url", required=True, help="Base URL of the gym (e.g. https://your-gym-url.com)")
    parser.add_argument(
        "--task-id",
        nargs="+",
        default=None,
        help="Optional task ID(s) to fetch. If omitted, fetches all tasks.",
    )
    parser.add_argument(
        "--output",
        default="data/tasks.jsonl",
        help="Path to output JSONL file (default: data/tasks.jsonl)",
    )
    args = parser.parse_args()

    rows = fetch_gym_tasks(args.gym_url, args.task_id)
    count = write_jsonl(rows, args.output)
    print(f"Wrote {count} task(s) to {args.output}")


if __name__ == "__main__":
    main()
