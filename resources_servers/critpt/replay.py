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

API keys are read from $ARTIFICIAL_ANALYSIS_API_KEY.  # pragma: allowlist secret

    # Single key (free tier, 10 CritPt scorings / 24h):
    ARTIFICIAL_ANALYSIS_API_KEY="aa-xxxxx"  # pragma: allowlist secret

    # Multiple keys for rotation on HTTP 429 (bracketed list literal,
    # comma-separated; whitespace ok):
    ARTIFICIAL_ANALYSIS_API_KEY="[aa-key-A,aa-key-B,aa-key-C]"  # pragma: allowlist secret

Rotation is always on when more than one key is configured: every 429
advances to the next key. When every configured key 429s on the same batch
(one full cycle), the tool exits with code 3; already-scored batches stay
in aa_responses.jsonl, so simply rerun after the AA daily quota resets.

Example:
    ARTIFICIAL_ANALYSIS_API_KEY="[k1,k2,k3]" python -m resources_servers.critpt.replay \\  # pragma: allowlist secret
        --cache-dir /path/to/critpt_cache

AA only accepts full batches of `--batch-size` (70) submissions, so a partial
run (e.g. a 5-problem smoke test) leaves a short batch that is skipped by
default. Pass `--fire-after N` to pad short batches up to batch_size with empty
padding submissions and ship them anyway, mirroring the live server's smoke-test fire_after:

    ARTIFICIAL_ANALYSIS_API_KEY="aa-xxx" python -m resources_servers.critpt.replay \\  # pragma: allowlist secret
        --cache-dir /path/to/critpt_cache --fire-after 5
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
    _ALL_PROBLEM_IDS,
    CritPtRateLimitExceeded,
    _call_api,
    refresh_partial_metrics,
)


AA_API_KEY_ENV_VAR = "ARTIFICIAL_ANALYSIS_API_KEY"  # pragma: allowlist secret


def _parse_api_keys_env(raw: str) -> List[str]:
    """Parse $ARTIFICIAL_ANALYSIS_API_KEY into a non-empty ordered list.  # pragma: allowlist secret

    Accepts two shapes:

    * Single key:        ``aa-xxx``      → ``["aa-xxx"]``
    * Bracketed list:    ``[k1,k2,k3]``  → ``["k1", "k2", "k3"]``

    For the list form we tolerate whitespace and surrounding single/double
    quotes around each item (a common shape after a shell `export` round-trip
    of a `.env` line with comma-separated quoted strings). Order is preserved
    and duplicates are dropped while keeping the first occurrence.

    Raises ValueError on an empty value or an empty/all-dup list — callers
    fail fast rather than try to ship to AA with no key.
    """
    s = raw.strip()
    if not s:
        raise ValueError(f"${AA_API_KEY_ENV_VAR} is set but empty; cannot resolve any AA key.")
    if not (s.startswith("[") and s.endswith("]")):
        return [s]

    inner = s[1:-1].strip()
    if not inner:
        raise ValueError(f"${AA_API_KEY_ENV_VAR}={raw!r} parsed as an empty list.")
    keys: List[str] = []
    for piece in inner.split(","):
        cleaned = piece.strip().strip('"').strip("'")
        if cleaned and cleaned not in keys:
            keys.append(cleaned)
    if not keys:
        raise ValueError(f"${AA_API_KEY_ENV_VAR}={raw!r} parsed to zero non-empty keys.")
    return keys


def _load_api_keys() -> List[str]:
    """Resolve AA API keys from $ARTIFICIAL_ANALYSIS_API_KEY.  # pragma: allowlist secret

    Single-key and bracketed-list shapes are both accepted by
    `_parse_api_keys_env`. Returns [] when the env var is unset so the
    caller can surface a uniform "no key resolved" error and exit 2.
    """
    value = os.environ.get(AA_API_KEY_ENV_VAR)
    if value is None:
        return []
    return _parse_api_keys_env(value)


async def _call_api_with_rotation(
    api_keys: List[str],
    api_url: str,
    submissions: List[Dict],
    max_retries: int,
    backoff_seconds: float,
    key_index_in: int,
) -> Tuple[Dict, int]:
    """Call AA with key-rotation on HTTP 429.

    Sticky behaviour matching the live server: start on `key_index_in`,
    advance only on a 429, and re-raise the last `CritPtRateLimitExceeded`
    once every key has 429'd in one cycle. On success the cursor stays on
    the key that worked so successive batches keep hitting it until AA
    rate-limits it.

    Returns (response, key_index_used). Callers thread the returned index
    back in so the next batch resumes on the same (working) key.
    """
    n = len(api_keys)
    last_exc = None
    for attempt in range(n):
        current = (key_index_in + attempt) % n
        try:
            response = await _call_api(
                api_url=api_url,
                api_key=api_keys[current],
                submissions=submissions,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            return response, current
        except CritPtRateLimitExceeded as exc:
            last_exc = exc
            remaining = n - (attempt + 1)
            if remaining > 0:
                next_idx = (current + 1) % n
                print(f"  rate-limited on key {current + 1}/{n}; rotating to key {next_idx + 1}/{n}")
    if last_exc is None:
        raise RuntimeError("_call_api_with_rotation invoked with no api_keys")
    raise last_exc


def _load_jsonl(path: Path) -> List[Dict]:
    """Load all non-empty lines as JSON objects from a JSONL file."""
    out: List[Dict] = []
    if not path.exists():
        return out
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                out.append(json.loads(line))
    return out


def _pad_to_batch_size(sub_payload: List[Dict], batch_size: int) -> List[Dict]:
    """Top a short batch up to batch_size with empty padding submissions.

    Mirrors the live server's smoke-test padding (see `CritPtResourcesServer`):
    AA PUBLIC mode only accepts exactly batch_size submissions, so a partial
    batch (e.g. a 5-problem smoke run) is filled with empty-code padding for the
    canonical problem_ids not already present. Padding entries are synthetic — they
    carry no submission_id and are never recorded as scored — so a later replay
    still treats only the real submissions as done.
    """
    existing = {s["problem_id"] for s in sub_payload}
    padded = list(sub_payload)
    for pid in _ALL_PROBLEM_IDS:
        if len(padded) >= batch_size:
            break
        if pid not in existing:
            padded.append(
                {
                    "problem_id": pid,
                    "generated_code": "```python\n```",
                    "model": "unknown",
                    "generation_config": {},
                }
            )
    return padded


def _pack_into_batches(submissions: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Greedy bin-packing matching the in-server batching policy.

    Each batch may contain a given problem_id at most once. A new submission
    is placed into the first existing batch lacking its problem_id; a new
    batch is opened only when no existing batch can accept it.
    """
    batches: List[List[Dict]] = []
    for sub in submissions:
        pid = sub["submission"]["problem_id"]
        target = next(
            (b for b in batches if pid not in {s["submission"]["problem_id"] for s in b}),
            None,
        )
        if target is None:
            batches.append([sub])
        else:
            target.append(sub)
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
    # AA only accepts exactly batch_size submissions. Normally we ship only full
    # batches; with --fire-after (smoke-test parity), batches holding at least
    # fire_after real submissions are shipped and padded up to batch_size.
    padding_enabled = bool(args.fire_after) and args.fire_after < args.batch_size
    min_real = args.fire_after if padding_enabled else args.batch_size
    eligible = [b for b in batches if len(b) >= min_real]
    skipped = [b for b in batches if len(b) < min_real]
    if padding_enabled:
        print(
            f"packed pending into {len(batches)} batches: "
            f"{len(eligible)} shippable (>= fire_after={args.fire_after} real submissions, "
            f"padded up to {args.batch_size}), {len(skipped)} skipped (too few submissions)."
        )
    else:
        print(
            f"packed pending into {len(batches)} batches: "
            f"{len(eligible)} full, {len(skipped)} short "
            f"(AA only accepts full batches; short ones will be skipped)."
        )

    rejudged = 0
    key_index = 0
    for batch in eligible:
        sub_ids = [b["submission_id"] for b in batch]
        sub_payload = [b["submission"] for b in batch]
        if len(sub_payload) < args.batch_size:
            sub_payload = _pad_to_batch_size(sub_payload, args.batch_size)
            print(
                f"shipping padded batch: {len(sub_ids)} real + "
                f"{len(sub_payload) - len(sub_ids)} padded = {len(sub_payload)} submissions ..."
            )
        else:
            print(f"shipping batch of {len(sub_payload)} submissions ...")
        try:
            response, key_index = await _call_api_with_rotation(
                api_keys=api_keys,
                api_url=args.api_url,
                submissions=sub_payload,
                max_retries=args.max_retries,
                backoff_seconds=args.backoff_seconds,
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
            if rejudged > 0:
                refresh_partial_metrics(args.cache_dir)
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
    if skipped:
        print(
            f"Note: {len(skipped)} batch(es) were not shipped because they held fewer "
            f"than {min_real} submissions and AA requires exactly batch_size per call. "
            "The corresponding submissions remain in submissions.jsonl unscored"
            + (
                "."
                if padding_enabled
                else " (pass --fire-after N to pad and ship short batches, matching a smoke run)."
            )
        )
    refresh_partial_metrics(args.cache_dir)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-ship unscored CritPt submissions to AA without rerunning inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Path to the directory holding submissions.jsonl + aa_responses.jsonl, "
        "matching the `cache_dir` set on the resource server config.",
    )
    p.add_argument(
        "--api-url",
        type=str,
        default="https://artificialanalysis.ai/api/v2/critpt/evaluate",
    )
    p.add_argument("--batch-size", type=int, default=70)
    p.add_argument(
        "--fire-after",
        type=int,
        default=None,
        help="Smoke-test only: ship batches with at least this many real submissions, "
        "padding up to --batch-size with empty padding submissions (mirrors the server's fire_after). "
        "Leave unset to ship only full batches.",
    )
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--backoff-seconds", type=float, default=5.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # _call_api() reuses NeMo Gym's shared request() helper, which lazily
    # initializes the global config via Hydra — and Hydra re-parses sys.argv.
    # Our replay-specific flags (--cache-dir, --fire-after, ...) are unknown to
    # Hydra and would abort with an "unrecognized arguments" usage error, so we
    # clear argv now that argparse has consumed it.
    sys.argv = sys.argv[:1]
    api_keys = _load_api_keys()
    if not api_keys:
        print(
            f"No AA API key resolved: set ${AA_API_KEY_ENV_VAR} (single key or a `[k1,k2,k3]` list for rotation).",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(main_async(args, api_keys))


if __name__ == "__main__":
    sys.exit(main())
