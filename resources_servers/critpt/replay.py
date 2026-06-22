# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""CritPt scoring-only replay tool.

Reads submissions and prior AA responses persisted by `CritPtResourcesServer`
(see `app.py`'s `cache_dir` config) and re-ships any unscored submissions to
the Artificial Analysis CritPt scoring endpoint.

Use this after a live run dies from AA rate-limit exhaustion: once the daily
quota resets, run this once to recover the missing batch scores without
rerunning model inference.

Batches that the live run already scored successfully are skipped, so partial
quota is never wasted re-scoring submissions that already have a verdict.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from resources_servers.critpt.app import (
    CritPtRateLimitExceeded,
    _call_api,
)


def _load_jsonl(path: Path) -> List[Dict]:
    """Load all non-empty lines as JSON objects from a JSONL file."""
    out: List[Dict] = []
    if not path.exists():
        return out
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _pack_into_batches(submissions: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Greedy bin-packing matching the in-server batching policy.

    Each batch may contain a given problem_id at most once. A new submission
    is placed into the first existing batch lacking its problem_id; a new
    batch is opened only when no existing batch can accept it.
    """
    batches: List[List[Dict]] = []
    for sub in submissions:
        pid = sub["submission"]["problem_id"]
        placed = False
        for b in batches:
            if pid not in {s["submission"]["problem_id"] for s in b}:
                b.append(sub)
                placed = True
                break
        if not placed:
            batches.append([sub])
    return batches


async def main_async(args: argparse.Namespace) -> int:
    cache_dir: Path = args.cache_dir
    submissions_path = cache_dir / "submissions.jsonl"
    aa_responses_path = cache_dir / "aa_responses.jsonl"

    if not submissions_path.exists():
        print(
            f"No submissions cache at {submissions_path}; nothing to replay.",
            file=sys.stderr,
        )
        return 2

    submissions = _load_jsonl(submissions_path)
    prior_responses = _load_jsonl(aa_responses_path)

    scored_ids = set()
    for rec in prior_responses:
        scored_ids.update(rec["submission_ids"])

    pending = [s for s in submissions if s["submission_id"] not in scored_ids]
    print(f"submissions on disk: {len(submissions)}")
    print(f"already scored:      {len(scored_ids)}")
    print(f"pending replay:      {len(pending)}")

    if not pending:
        print("Nothing to replay.")
        return 0

    batches = _pack_into_batches(pending, args.batch_size)
    full_batches = [b for b in batches if len(b) == args.batch_size]
    short_batches = [b for b in batches if len(b) != args.batch_size]
    print(
        f"packed pending into {len(batches)} batches: "
        f"{len(full_batches)} full, {len(short_batches)} short "
        f"(AA only accepts full batches; short ones will be skipped)."
    )

    rejudged = 0
    for batch in full_batches:
        sub_ids = [b["submission_id"] for b in batch]
        sub_payload = [b["submission"] for b in batch]
        print(f"shipping batch of {len(sub_payload)} submissions ...")
        try:
            response = await _call_api(
                api_url=args.api_url,
                api_key=args.api_key,
                submissions=sub_payload,
                max_retries=args.max_retries,
                backoff_seconds=args.backoff_seconds,
            )
        except CritPtRateLimitExceeded as e:
            print(
                f"AA quota exhausted: retry_after={e.retry_after_seconds}s, "
                f"reset_unix={e.reset_unix}. Rerun this tool after that time.",
                file=sys.stderr,
            )
            return 3

        with aa_responses_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "batch_id": f"replay-{rejudged}",
                        "submission_ids": sub_ids,
                        "response": response,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
        rejudged += 1
        print(
            f"  scored: accuracy={response.get('accuracy')}, "
            f"judge_errors={response.get('judge_error_count')}, "
            f"timeout_rate={response.get('timeout_rate')}"
        )

    print(f"Replay complete. Rejudged {rejudged} batches.")
    if short_batches:
        print(
            f"Note: {len(short_batches)} short batch(es) were not shipped because "
            "AA requires exactly batch_size submissions per call. The corresponding "
            "submissions remain in submissions.jsonl unscored."
        )
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-ship unscored CritPt submissions to AA without rerunning inference.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Path to the directory holding submissions.jsonl + aa_responses.jsonl, "
        "matching the `cache_dir` set on the resource server config.",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY"),
        help="AA API key (defaults to $ARTIFICIAL_ANALYSIS_API_KEY).",
    )
    p.add_argument(
        "--api-url",
        type=str,
        default="https://artificialanalysis.ai/api/v2/critpt/evaluate",
    )
    p.add_argument("--batch-size", type=int, default=70)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--backoff-seconds", type=float, default=5.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.api_key:
        print(
            "AA API key required: pass --api-key or set $ARTIFICIAL_ANALYSIS_API_KEY.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
