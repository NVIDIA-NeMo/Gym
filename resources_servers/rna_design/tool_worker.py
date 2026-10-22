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
"""One rollout's design campaign, driven one tool call at a time. Reads JSON lines on stdin, writes JSON lines.

The rds tools keep process-wide state, so each rollout gets its own process. The first line opens the campaign.
"""

import json
import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[2] / "responses_api_agents" / "rna_design_agent"
sys.path.insert(0, str(AGENT_DIR))

import worker  # noqa: E402
from rds.budget import Budget, Limits  # noqa: E402
from rds.campaign import applicable_tools, objective_view  # noqa: E402
from rds.evaluate import ground_truth  # noqa: E402
from rds.runner import SELECTORS, apply_tool_call  # noqa: E402
from rds.state import CampaignState  # noqa: E402
from rds.targets import from_dict, register  # noqa: E402


class Campaign:
    def __init__(self, suite_path: str, target_name: str, run_dir: str) -> None:
        _, self.campaign = worker.find_campaign(suite_path, target_name)
        register(from_dict(self.campaign.inline_target))
        self.state = CampaignState(run_dir, self.campaign.scoring.direction)
        self.budget = Budget(Limits(**self.campaign.budget))
        self.rank_by = self.campaign.scoring.primary
        self.objective = objective_view(self.campaign)
        self.allowed = {spec.name for spec in applicable_tools(self.campaign)}
        self.failures: list[str] = []
        self.state.record("campaign_start", controller="harness", rank_by=self.rank_by, budget=self.budget.snapshot())

    def call(self, tool: str, arguments: dict) -> dict:
        if tool not in self.allowed:
            raise ValueError(f"{tool} is not offered for this task; available: {', '.join(sorted(self.allowed))}")
        spent = self.budget.exhausted()
        if spent:
            raise RuntimeError(f"the budget is spent ({spent}); no further tool calls run")
        self.state.round += 1
        # The same event the campaign loop writes, so the transcript and call count agree.
        self.state.record("action", tool=tool, arguments=arguments, rationale="")
        apply_tool_call(tool, arguments, self.state, self.budget, self.rank_by, self.objective, self.failures)
        return self.view()

    def view(self) -> dict:
        """The campaign state, the same view the agent's model gets at the start of every round. Costs nothing."""
        view = self.state.view(
            self.budget.snapshot(),
            rank_by=self.rank_by,
            failures=self.failures[-5:],
            objective=self.objective,
            selectors=SELECTORS,
        )
        return json.loads(view.to_prompt())

    def score(self) -> dict:
        pool = [row["sequence"] for row in self.state.candidates()]
        truth = ground_truth(pool, self.campaign)
        transcript, tool_calls, tool_errors = worker.read_trace(Path(self.state.dir))
        self.state.record("campaign_end", controller="harness", reason="harness_done", rounds=self.state.round)
        return {
            "best": truth["best"],
            "solved": bool(truth["solved"]),
            "pool_size": truth["n"],
            "n_passing_gates": truth["n_passing_gates"],
            "median": truth["median"],
            "top": truth.get("top", []),
            "metric": self.rank_by,
            "protocol": truth["protocol"],
            "pass_threshold": self.campaign.scoring.pass_threshold,
            "rounds": self.state.round,
            "tool_calls": tool_calls,
            "tool_failures": tool_errors,
            "transcript": transcript,
        }


def main() -> None:
    campaign = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            if request["op"] == "open":
                campaign = Campaign(request["suite_path"], request["target_name"], request["run_dir"])
                reply = {"ok": True, "result": {"objective": campaign.objective, "tools": sorted(campaign.allowed)}}
            elif request["op"] == "view":
                reply = {"ok": True, "result": campaign.view()}
            elif request["op"] == "call":
                reply = {"ok": True, "result": campaign.call(request["tool"], request.get("arguments") or {})}
            else:
                reply = {"ok": True, "result": campaign.score()}
        except Exception as exc:  # a failed tool is an answer for the model, not a crash of the rollout
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps({**reply, "id": request.get("id")}, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
