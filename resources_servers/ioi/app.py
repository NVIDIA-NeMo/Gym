# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IOI (International Olympiad in Informatics) resource server.

Thin subclass of the mainline ``competitive_coding_challenges`` (CCC) server
that adds IOI-specific aggregate metrics: ``ioi_total_score`` and
``per_problem_subtask_scores`` (max per-subtask score pooled across all
rollouts of a problem, summed).

All verify-path logic (sandbox compile/run, test scoring, min(scores) *
subtask_cap aggregation) is inherited unchanged from CCC.
"""

from typing import Any, Dict, List

from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from resources_servers.competitive_coding_challenges.app import (
    CompetitiveCodingChallengesResourcesServer,
    CompetitiveCodingChallengesResourcesServerConfig,
)


class IOIResourcesServer(CompetitiveCodingChallengesResourcesServer):
    config: CompetitiveCodingChallengesResourcesServerConfig

    @staticmethod
    def _score_fn(result: dict) -> Dict[str, float]:
        """Per-rollout accuracy: 1.0 iff every subtask the rollout covered scored > 0.

        CCC stores per-subtask scores inside ``details.test_case_results``.
        """
        details = result.get("details") or {}
        tcr = details.get("test_case_results") or {}
        if not tcr:
            return {"accuracy": 0.0}
        return {"accuracy": 1.0 if all((r.get("score", 0) or 0) > 0 for r in tcr.values()) else 0.0}

    def _lookup_subtask_cap(self, competition_id: str | None, problem_id: str, subtask: str) -> float:
        """Look up the max score a subtask can award, via the CCC evaluator's loaded metadata.

        NB: named ``_lookup_subtask_cap`` rather than ``_subtask_max_score`` to
        avoid shadowing CCC's parent-class method, which has a different
        signature (``(self, body)``) and is called from the inherited
        ``_compute_reward`` path.
        """
        if not self._evaluator:
            return 0.0
        try:
            meta = self._evaluator.get_problem_metadata(problem_id, competition_id)
        except Exception:
            return 0.0
        st_meta = (meta.get("subtasks") or {}).get(subtask, {})
        return float(st_meta.get("score", 0) or 0)

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Pass@k accuracy + IOI-style total_score aggregated across problems.

        For each problem (grouped by ``problem_id``), take the max per-subtask
        score across ALL rollouts of ALL subtasks in that problem, sum across
        subtasks for a per-problem total, then sum across problems.
        """
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_code",
        )

        problem_subtask_max: Dict[str, Dict[str, float]] = {}
        problem_subtask_cap: Dict[str, Dict[str, float]] = {}
        for rollouts in tasks:
            for r in rollouts:
                problem_id = r.get("problem_id") or r.get("ioi_id")
                if not problem_id:
                    continue
                competition_id = r.get("competition_id")
                details = r.get("details") or {}
                tcr = details.get("test_case_results") or {}
                for st, sub in tcr.items():
                    achieved = float((sub or {}).get("score", 0) or 0)
                    problem_subtask_max.setdefault(problem_id, {})
                    if achieved > problem_subtask_max[problem_id].get(st, 0):
                        problem_subtask_max[problem_id][st] = achieved
                    if st not in problem_subtask_cap.setdefault(problem_id, {}):
                        problem_subtask_cap[problem_id][st] = self._lookup_subtask_cap(competition_id, problem_id, st)

        total_score = 0.0
        per_problem: Dict[str, Dict[str, Any]] = {}
        for problem_id, subs in problem_subtask_max.items():
            problem_total = sum(subs.values())
            max_total = sum(problem_subtask_cap.get(problem_id, {}).values())
            per_problem[problem_id] = {
                "total": {"score": problem_total, "max_score": max_total},
                "subtasks": {
                    st: {
                        "score": subs[st],
                        "max_score": problem_subtask_cap.get(problem_id, {}).get(st, 0),
                    }
                    for st in subs
                },
            }
            total_score += problem_total

        metrics["ioi_total_score"] = int(total_score)
        metrics["per_problem_subtask_scores"] = per_problem
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        if "ioi_total_score" in agent_metrics:
            key["ioi_total_score"] = agent_metrics["ioi_total_score"]
        return key


if __name__ == "__main__":
    IOIResourcesServer.run_webserver()
