# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`gym eval intake` — submit a finished rundir to a benchmark-run data layer.

```bash
export NEMO_GYM_INTAKE_BASE_URL=https://<your data layer>

# Name a rundir the service can read itself. Nothing is copied; the reply is a job id.
gym eval intake --lustre <cluster>:/<shared storage>/.../<rundir>

# What became of an earlier submission.
gym eval intake --status <job id>
```
"""

import argparse
import asyncio
import json
import sys

from nemo_gym import intake as intake_api
from nemo_gym.intake import IntakeError, IntakeOutcome, Submission


# Distinct codes so a driving script can tell the three non-successes apart.
EXIT_REFUSED = 1
EXIT_STILL_PARSING = 3
EXIT_FAILED = 4


def _status_line(message: str) -> None:
    sys.stderr.write("\r\033[K" + message)
    sys.stderr.flush()


def _emit(payload: dict, *, as_json: bool, human: str) -> None:
    """Write the result to stdout, flushed, so it cannot interleave with the stderr progress."""
    sys.stderr.write("\n")
    sys.stderr.flush()
    sys.stdout.write(json.dumps(payload, indent=2) if as_json else human)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _outcome_payload(submission: Submission | None, outcome: IntakeOutcome | None) -> dict:
    payload: dict = {}
    if submission is not None:
        payload.update({"job_id": submission.id, "status_path": submission.status_path})
    if outcome is not None:
        payload.update({"state": outcome.state, "run_ids": outcome.run_ids, "detail": outcome.detail})
    return payload


def intake(args: argparse.Namespace) -> None:
    as_json = bool(getattr(args, "json", False))
    try:
        base_url = intake_api.resolve_base_url(args.base_url)
        if args.status:
            raise SystemExit(_report_status(base_url, args.status, as_json=as_json))
        raise SystemExit(_submit(base_url, args, as_json=as_json))
    except IntakeError as error:
        sys.stderr.write(f"\n{error}\n")
        raise SystemExit(EXIT_REFUSED) from None


def _report_status(base_url: str, job_id: str, *, as_json: bool) -> int:
    async def run() -> IntakeOutcome:
        async with intake_api.new_session() as session:
            return await intake_api.fetch_status(session, base_url, job_id)

    outcome = asyncio.run(run())
    _emit(
        _outcome_payload(None, outcome),
        as_json=as_json,
        human=f"{job_id}: {outcome.state}" + (f" -> {', '.join(outcome.run_ids)}" if outcome.run_ids else ""),
    )
    return 0 if outcome.succeeded else EXIT_FAILED


def _submit(base_url: str, args: argparse.Namespace, *, as_json: bool) -> int:
    timeout_s = args.timeout if args.timeout is not None else intake_api.DEFAULT_POLL_TIMEOUT_S

    def on_wait(waited: float, outcome: IntakeOutcome) -> None:
        _status_line(f"{outcome.state} · waited {int(waited)}s (safe to Ctrl-C; the request is already durable)")

    submission, outcome = asyncio.run(
        intake_api.run_intake(
            base_url=base_url,
            locator=args.lustre,
            label=args.label,
            wait=not args.no_wait,
            timeout_s=timeout_s,
            on_wait=on_wait,
        )
    )
    payload = _outcome_payload(submission, outcome)
    poll_at = f"poll: {base_url}{submission.status_path}"
    if outcome is None:
        headline = (
            f"submitted · job {submission.id}"
            if args.no_wait
            else f"still parsing after {timeout_s:.0f}s · job {submission.id}"
        )
        _emit(payload, as_json=as_json, human=f"{headline}\n{poll_at}")
        return 0 if args.no_wait else EXIT_STILL_PARSING
    runs = ", ".join(outcome.run_ids) or "none yet"
    _emit(payload, as_json=as_json, human=f"{outcome.state} · job {submission.id} · run_ids: {runs}")
    return 0 if outcome.succeeded else EXIT_FAILED
