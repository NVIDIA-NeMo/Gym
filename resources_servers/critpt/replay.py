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

By default the tool authenticates with a single key from
$ARTIFICIAL_ANALYSIS_API_KEY. Pass --api-key-file (one key per line) to load
multiple keys, then combine with --rotate-on-rate-limit to fail over on
HTTP 429. When every configured key is rate-limited the tool exits cleanly
with exit code 3; already-scored batches stay in aa_responses.jsonl, so
simply rerun the tool after the AA daily quota resets.

Examples:
    ARTIFICIAL_ANALYSIS_API_KEY=... python -m resources_servers.critpt.replay \\
        --cache-dir /path/to/critpt_cache

    python -m resources_servers.critpt.replay \\
        --cache-dir /path/to/critpt_cache \\
        --api-key-file ~/.aa_keys \\
        --rotate-on-rate-limit
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from resources_servers.critpt.app import (
    CritPtRateLimitExceeded,
    _call_api,
)


AA_API_KEY_ENV_VAR = "ARTIFICIAL_ANALYSIS_API_KEY"


def _load_api_keys(args: argparse.Namespace) -> List[str]:
    """Resolve AA API keys.

    If --api-key-file is given, every non-empty, non-`#`-comment line is taken
    as one key (preserving order, deduped). Otherwise fall back to a single
    key read from $ARTIFICIAL_ANALYSIS_API_KEY. Returns [] if neither yields a
    key; the caller is responsible for surfacing that.
    """
    if args.api_key_file:
        keys: List[str] = []
        with args.api_key_file.open("r") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and stripped not in keys:
                    keys.append(stripped)
        return keys
    value = os.environ.get(AA_API_KEY_ENV_VAR)
    return [value] if value else []


async def _call_api_with_rotation(
    api_keys: List[str],
    api_url: str,
    submissions: List[Dict],
    max_retries: int,
    backoff_seconds: float,
    rotate_on_rate_limit: bool,
    key_index_in: int,
) -> Tuple[Dict, int]:
    """Call AA with key-rotation on HTTP 429.

    On a 429: if rotation is enabled and >1 keys are configured, advance to
    the next key and retry immediately. Once all keys have cycled back to the
    starting index (i.e. every key was rate-limited this round) re-raise so
    the caller fails fast — AA's daily quota means waiting in-process is not
    productive, just rerun after the quota resets.

    Returns (response, new_key_index). The caller threads the returned index
    back in so successive successful batches round-robin across keys.
    """
    key_index = key_index_in
    while True:
        api_key = api_keys[key_index % len(api_keys)]
        try:
            response = await _call_api(
                api_url=api_url,
                api_key=api_key,
                submissions=submissions,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            return response, key_index + 1
        except CritPtRateLimitExceeded:
            if not (rotate_on_rate_limit and len(api_keys) > 1):
                raise
            key_index += 1
            if key_index % len(api_keys) == 0:
                raise
            print(
                f"  rate-limited on key idx={(key_index - 1) % len(api_keys)}; "
                f"rotating to key idx={key_index % len(api_keys)}"
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


async def main_async(args: argparse.Namespace, api_keys: List[str]) -> int:
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
    print(f"AA keys configured:  {len(api_keys)}")

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
    key_index = 0
    for batch in full_batches:
        sub_ids = [b["submission_id"] for b in batch]
        sub_payload = [b["submission"] for b in batch]
        print(f"shipping batch of {len(sub_payload)} submissions ...")
        try:
            response, key_index = await _call_api_with_rotation(
                api_keys=api_keys,
                api_url=args.api_url,
                submissions=sub_payload,
                max_retries=args.max_retries,
                backoff_seconds=args.backoff_seconds,
                rotate_on_rate_limit=args.rotate_on_rate_limit,
                key_index_in=key_index,
            )
        except CritPtRateLimitExceeded as e:
            print(
                f"AA quota exhausted on all {len(api_keys)} key(s): "
                f"retry_after={e.retry_after_seconds}s, reset_unix={e.reset_unix}. "
                f"Already-scored batches remain in aa_responses.jsonl; rerun this "
                f"tool after the quota resets.",
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
        "--api-key-file",
        type=Path,
        default=None,
        help=f"Optional file with one AA API key per line (blank lines and '#' "
        f"comments ignored). When omitted, falls back to a single key from "
        f"${AA_API_KEY_ENV_VAR}.",
    )
    p.add_argument(
        "--rotate-on-rate-limit",
        action="store_true",
        help="On HTTP 429, advance to the next configured AA key and retry immediately. "
        "No-op when only one key is configured.",
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
    api_keys = _load_api_keys(args)
    if not api_keys:
        print(
            f"No AA API key resolved: set ${AA_API_KEY_ENV_VAR} or pass --api-key-file.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(main_async(args, api_keys))


if __name__ == "__main__":
    sys.exit(main())
