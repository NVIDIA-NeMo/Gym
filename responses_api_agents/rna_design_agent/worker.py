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
"""Run one RNA design rollout in its own process: read a job as JSON on stdin, print one JSON result.

``run_one`` sets process-wide environment variables and a module-level target table, so each rollout
needs its own process."""

import json
import sys
from pathlib import Path
from typing import Any


# ``rds`` sits next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rds.benchmark import run_one  # noqa: E402
from rds.evaluate import ground_truth  # noqa: E402
from rds.llm import GatewayProvider  # noqa: E402
from rds.state import CampaignState  # noqa: E402
from rds.suite import Suite  # noqa: E402
from rds.targets import from_dict, register  # noqa: E402


def find_campaign(suite_path: str, target_name: str):
    suite = Suite.load(suite_path)
    for campaign in suite.campaigns():
        if campaign.inline_target["name"] == target_name:
            return suite, campaign
    raise ValueError(f"target {target_name!r} is not in suite {suite.name!r}")


def score_rollout(job: dict[str, Any]) -> dict[str, Any]:
    """Run the campaign and return the reward inputs.

    ``best`` is the pool's ground-truth score from ``rds.evaluate.ground_truth``, None when no design passes
    the gates. ``solved`` applies the suite's ``pass_threshold``.
    """
    suite, campaign = find_campaign(job["suite_path"], job["target_name"])
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    provider = None
    if job["arm"] == "agent":
        # A Gym model server needs no key, and the rollout-prefixed URL attributes each call to this rollout.
        provider = GatewayProvider(model=job["model"], base_url=job["base_url"], api_key="unused", timeout_s=600)

    result = run_one(
        campaign,
        run_dir,
        arm=job["arm"],
        generator=job.get("generator") or suite.generator,
        provider=provider,
        max_tokens=job.get("max_tokens", 8192),
        seed=job.get("seed", 101),
    )

    # Re-score the pool by the ground-truth protocol, whatever the visibility setting.
    register(from_dict(campaign.inline_target))
    state = CampaignState(run_dir)
    try:
        pool = [row["sequence"] for row in state.candidates()]
    finally:
        state.close()
    # An empty pool's result has no `top`.
    truth = ground_truth(pool, campaign)

    run = result["run"]
    tokens = result.get("tokens") or {}
    transcript, tool_calls, tool_errors, environment_faults = read_trace(run_dir)
    return {
        "transcript": transcript,
        "tool_calls": tool_calls,
        "best": truth["best"],
        "solved": bool(truth["solved"]),
        "pool_size": truth["n"],
        "n_passing_gates": truth["n_passing_gates"],
        "median": truth["median"],
        "top": truth.get("top", []),
        "metric": campaign.scoring.primary,
        "protocol": truth["protocol"],
        "pass_threshold": campaign.scoring.pass_threshold,
        "tokens_in": tokens.get("in", 0),
        "tokens_out": tokens.get("out", 0),
        "rounds": run["rounds"],
        "stop_reason": run["reason"],
        "tool_failures": tool_errors,
        "environment_faults": environment_faults,
        "code": run["code"],
    }


def read_trace(run_dir: Path) -> tuple[list[str], int, int, int]:
    """One line per round (tool, arguments, any error), the tool call count, how many raised, and how many of those
    the machine caused."""
    lines, calls, errors, environment = [], 0, 0, 0
    for line in (run_dir / "trace.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "action" and event["tool"] != "stop":
            calls += 1
            lines.append(f"round {event['round']}: {event['tool']} {json.dumps(event.get('arguments'))}")
        elif event["kind"] == "tool_error":
            errors += 1
            environment += bool(event.get("environment"))
            lines.append(f"  failed: {event['error']}: {event['message']}")
    return lines, calls, errors, environment


def main() -> int:
    job = json.loads(sys.stdin.read())
    try:
        out = {"ok": True, "result": score_rollout(job)}
    except Exception as exc:  # reported as data so the agent can mask the sample instead of crashing
        out = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:2000]}
    sys.stdout.write("\n" + json.dumps(out, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
