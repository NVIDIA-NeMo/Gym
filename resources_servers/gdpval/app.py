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
"""GDPVal resources server.

Scores Stirrup agent deliverables for the GDPVal benchmark. Two modes,
selected via ``reward_mode`` config:

- ``rubric``: score deliverables against a per-task rubric using an LLM
  judge. Reward in [0.0, 1.0].
- ``comparison``: pairwise-judge the eval deliverable against a reference
  rollout's deliverable for the same ``task_id``. Reward in {0.0, 0.5, 1.0}.
  ``aggregate_metrics`` then reduces win/loss/tie counts into an ELO rating.

Scoring internals live in ``scoring.py`` (rubric) and ``comparison.py``
(pairwise judge + ELO math).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.server_utils import get_server_url


_DEFAULT_JUDGE_PROMPT_FPATH = str(Path(__file__).parent / "prompts" / "judge_prompt.j2")
_DEFAULT_REFERENCE_ELO = 1000.0


def _iter_ref_repeat_dirs(task_dir: Path) -> List[Path]:
    """All reference deliverable dirs for a task, supporting both layouts.

    New: ``task_<id>/repeat_<n>/`` — return every repeat dir, sorted. Old:
    flat ``task_<id>/`` — return ``[task_dir]``. Missing → ``[]``.

    Returning every repeat lets the comparison verifier judge each eval
    rollout against *all* reference rollouts so the win rate (and ELO)
    averages over reference variance instead of being anchored to a single
    sample.
    """
    if not task_dir.is_dir():
        return []
    repeats = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    return repeats or [task_dir]


def _safe_output_text(response: Any) -> str:
    """Extract concatenated assistant text from a response without relying on
    ``response.output_text`` — that property raises ``AttributeError`` when
    ``output[*].content`` contains raw strings (e.g. input messages carried
    through by the Stirrup agent)."""
    parts: List[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if d.get("type") != "message":
            continue
        if d.get("role") and d.get("role") != "assistant":
            continue
        content = d.get("content") or []
        if isinstance(content, str):
            parts.append(content)
            continue
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(c.get("text") or "")
    return "\n".join(p for p in parts if p)


class GDPValResourcesServerConfig(BaseResourcesServerConfig):
    reward_mode: Literal["rubric", "comparison"] = "rubric"

    # Comparison-mode: directory tree containing the reference model's
    # deliverables, laid out as ``<reference_deliverables_dir>/task_<task_id>/``
    # with the same files the agent would persist (deliverable artifacts +
    # finish_params.json + reference_files/). Required when
    # ``reward_mode=comparison``.
    reference_deliverables_dir: Optional[str] = None

    # Pairwise judge trials per task. 4 is the historical default; alternates
    # swap/no-swap to debias position effects.
    num_comparison_trials: int = 4

    # ELO assigned to the reference model in pairwise mode. ``aggregate_metrics``
    # reports the eval model's ELO relative to this anchor.
    reference_elo: float = _DEFAULT_REFERENCE_ELO

    # Office→PDF preconversion for deliverables before pairwise judging.
    # Most office docs render poorly as raw text; PDFs let multimodal judges
    # read tables/charts. Costs ~5-30s per Office file.
    preconvert_office_to_pdf: bool = True
    preconvert_max_concurrent: int = 1

    judge_model_server: ModelServerRef
    judge_responses_create_params_overrides: Dict[str, Any] = {}
    judge_prompt_template_fpath: Optional[str] = None

    # Rubric-mode scoring backend:
    # - ``"binary"`` (default, legacy): judge emits a JSON ``{criteria_scores:
    #   [{score: 0|1, ...}], overall_score: float}``; reward is the overall
    #   score (0-1). Treats every criterion as equal weight.
    # - ``"structured"``: judge emits ``CRITERION_NUMBER[N]: GRADE[X] out of
    #   MAX_POSSIBLE_POINTS[Y]`` tagged output and ``FINAL_SCORE[…] / MAX_POSSIBLE_SCORE[…]``.
    #   Honors per-criterion point weights when the rubric carries them in
    #   ``rubric_json[i].score`` or ``rubric_json[i].weight``. For datasets
    #   without weights, every criterion contributes max-points 1, giving a
    #   signal equivalent to binary mode. Multi-trial averaged for stability.
    #   The tagged output is also more compact than the JSON-with-rationale
    #   format used by binary mode, so it rarely runs into the judge's
    #   ``finish_reason: length`` truncation on rubrics with many criteria.
    rubric_scoring_mode: Literal["binary", "structured"] = "binary"
    rubric_structured_num_trials: int = 2
    rubric_structured_formatting_retries: int = 3

    # When True, every judge call's raw response text is preserved on
    # ``verify_response.judge_response`` (per-trial in comparison mode under
    # ``per_ref_repeat[i].raw_responses``; under top-level ``raw_responses``
    # in rubric modes). Off by default — raw responses are 10-50 KB each and
    # multiply by num_trials × num_ref_repeats × num_tasks. Turn on for debug
    # runs to post-mortem judge verdicts.
    persist_raw_judge_responses: bool = False


class GDPValVerifyRequest(BaseVerifyRequest):
    task_id: str
    sector: Optional[str] = None
    occupation: Optional[str] = None
    prompt: Optional[str] = None
    rubric_json: Optional[Any] = None
    rubric_pretty: Optional[str] = None
    reference_file_urls: Optional[List[str]] = None
    deliverables_dir: Optional[str] = None


class GDPValVerifyResponse(GDPValVerifyRequest, BaseVerifyResponse):
    verify_mode: Literal["rubric", "comparison"] = "rubric"
    judge_response: Optional[Dict[str, Any]] = None
    invalid_judge_response: Optional[bool] = None
    # Majority-decision flags across all (ref_repeat × trial) judge votes —
    # kept for back-compat with older verify responses (still bool-valued).
    win: Optional[bool] = None
    loss: Optional[bool] = None
    tie: Optional[bool] = None
    # Raw judge vote counts aggregated over every reference repeat × trial.
    # ``aggregate_metrics`` prefers these so the win rate reflects all
    # comparisons rather than treating each verify call as a single vote.
    total_wins: Optional[int] = None
    total_losses: Optional[int] = None
    total_ties: Optional[int] = None


class GDPValResourcesServer(SimpleResourcesServer):
    config: GDPValResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._judge_prompt_fpath: str = self.config.judge_prompt_template_fpath or _DEFAULT_JUDGE_PROMPT_FPATH
        if self.config.reward_mode == "comparison" and not self.config.reference_deliverables_dir:
            raise ValueError("reward_mode=comparison requires reference_deliverables_dir to be set")
        super().model_post_init(context)

    async def verify(self, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        if self.config.reward_mode == "comparison":
            return await self._verify_comparison(body)

        return await self._verify_rubric(body)

    async def _verify_rubric(self, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        if not (body.rubric_json or body.rubric_pretty):
            return GDPValVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verify_mode="rubric",
                invalid_judge_response=True,
            )

        overrides = dict(self.config.judge_responses_create_params_overrides or {})
        judge_base_url = get_server_url(self.config.judge_model_server.name) + "/v1"
        judge_model_name = overrides.pop("model", "judge")
        judge_api_key = overrides.pop("api_key", "dummy")
        # Anything left in `overrides` (max_tokens, temperature, top_p, …) is
        # merged into the judge's chat.completions.create kwargs.
        judge_create_overrides = overrides or None

        deliverable_text = _safe_output_text(body.response)
        deliverable_content_blocks: Optional[List[Dict[str, Any]]] = None

        if body.deliverables_dir and Path(body.deliverables_dir).is_dir():
            from responses_api_agents.stirrup_agent.file_reader import (
                convert_deliverables_to_content_blocks,
                read_deliverable_files,
            )

            read = read_deliverable_files(body.deliverables_dir)
            if read:
                deliverable_text = read
            blocks = convert_deliverables_to_content_blocks(body.deliverables_dir)
            if blocks:
                deliverable_content_blocks = blocks

        task_prompt = body.prompt or ""
        rubric_pretty = body.rubric_pretty or ""

        # Visual scoring when deliverable renders (PDFs/images) are available —
        # the judge model is expected to be multimodal (configured via
        # ``judge_model_server`` in the benchmark YAML). Falls back to text
        # scoring only when no content blocks could be built.
        if self.config.rubric_scoring_mode == "structured":
            from resources_servers.gdpval.scoring import score_with_rubric_structured

            reward, judge_result = await score_with_rubric_structured(
                deliverable_text=deliverable_text,
                rubric_json=body.rubric_json,
                rubric_pretty=rubric_pretty,
                task_prompt=task_prompt,
                model_base_url=judge_base_url,
                model_name=judge_model_name,
                api_key=judge_api_key,
                num_trials=self.config.rubric_structured_num_trials,
                formatting_retries=self.config.rubric_structured_formatting_retries,
                deliverable_content_blocks=deliverable_content_blocks,
                include_raw_responses=self.config.persist_raw_judge_responses,
            )
        elif deliverable_content_blocks:
            from resources_servers.gdpval.scoring import score_with_rubric_visual

            reward, judge_result = await score_with_rubric_visual(
                deliverable_content_blocks=deliverable_content_blocks,
                rubric_json=body.rubric_json,
                rubric_pretty=rubric_pretty,
                task_prompt=task_prompt,
                judge_prompt_template=self._judge_prompt_fpath,
                model_base_url=judge_base_url,
                model_name=judge_model_name,
                api_key=judge_api_key,
                create_overrides=judge_create_overrides,
                include_raw_responses=self.config.persist_raw_judge_responses,
            )
        else:
            from resources_servers.gdpval.scoring import score_with_rubric

            reward, judge_result = await score_with_rubric(
                deliverable_text=deliverable_text,
                rubric_json=body.rubric_json,
                rubric_pretty=rubric_pretty,
                task_prompt=task_prompt,
                judge_prompt_template=self._judge_prompt_fpath,
                model_base_url=judge_base_url,
                model_name=judge_model_name,
                api_key=judge_api_key,
                create_overrides=judge_create_overrides,
                include_raw_responses=self.config.persist_raw_judge_responses,
            )

        return GDPValVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            verify_mode="rubric",
            judge_response=judge_result,
            invalid_judge_response=(judge_result is None),
        )

    async def _verify_comparison(self, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        from openai import OpenAI

        from resources_servers.gdpval.comparison import (
            build_file_section,
            run_trials,
            task_attempted,
        )
        from resources_servers.gdpval.preconvert import preconvert_dir_async

        ref_root = Path(self.config.reference_deliverables_dir)
        ref_task_root = ref_root / f"task_{body.task_id}"
        ref_task_dirs = [d for d in _iter_ref_repeat_dirs(ref_task_root) if task_attempted(str(d))]
        eval_task_dir = Path(body.deliverables_dir) if body.deliverables_dir else None

        if not ref_task_dirs:
            print(f"[gdpval] no reference deliverable for task {body.task_id}", flush=True)
            return GDPValVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verify_mode="comparison",
                judge_response={"error": "reference_missing"},
            )

        if eval_task_dir is None or not task_attempted(str(eval_task_dir)):
            print(f"[gdpval] eval deliverable missing for task {body.task_id}", flush=True)
            return GDPValVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verify_mode="comparison",
                judge_response={"error": "eval_missing"},
                loss=True,
            )

        if self.config.preconvert_office_to_pdf:
            await preconvert_dir_async(eval_task_dir, max_concurrent=self.config.preconvert_max_concurrent)
            for ref_dir in ref_task_dirs:
                await preconvert_dir_async(ref_dir, max_concurrent=self.config.preconvert_max_concurrent)

        eval_submission = build_file_section(str(eval_task_dir))

        overrides = dict(self.config.judge_responses_create_params_overrides or {})
        judge_base_url = get_server_url(self.config.judge_model_server.name) + "/v1"
        judge_model_name = overrides.get("model", "judge")
        judge_api_key = overrides.get("api_key", "dummy")
        client = OpenAI(base_url=judge_base_url, api_key=judge_api_key)

        # Judge eval submission against every available reference repeat. Raw
        # vote counts (not just per-matchup majority) are summed so the win
        # rate averages over reference variance — see ``_iter_ref_repeat_dirs``.
        total_wins = 0
        total_losses = 0
        total_ties = 0
        per_ref_results: List[Dict[str, Any]] = []
        for ref_dir in ref_task_dirs:
            refs_subdir = ref_dir / "reference_files"
            refs = build_file_section(str(refs_subdir) if refs_subdir.is_dir() else None)
            ref_submission = build_file_section(str(ref_dir))
            result = await asyncio.to_thread(
                run_trials,
                client=client,
                model=judge_model_name,
                task_prompt=body.prompt or "",
                refs=refs,
                submission_a=ref_submission,
                submission_b=eval_submission,
                num_trials=self.config.num_comparison_trials,
                return_raw_responses=self.config.persist_raw_judge_responses,
            )
            # ``run_trials`` casts submission_a=ref, submission_b=eval, so
            # ``win_count_b`` is eval wins.
            total_wins += result["win_count_b"]
            total_losses += result["win_count_a"]
            total_ties += result["tie_count"]
            per_ref_results.append({"ref_repeat": ref_dir.name, **result})

        total_judged = total_wins + total_losses + total_ties
        if total_wins > total_losses:
            reward = 1.0
        elif total_losses > total_wins:
            reward = 0.0
        else:
            reward = 0.5

        return GDPValVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verify_mode="comparison",
            judge_response={
                "per_ref_repeat": per_ref_results,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "total_ties": total_ties,
                "total_judged": total_judged,
                "ref_repeat_count": len(ref_task_dirs),
            },
            win=reward == 1.0,
            loss=reward == 0.0,
            tie=reward == 0.5,
            total_wins=total_wins,
            total_losses=total_losses,
            total_ties=total_ties,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        if self.config.reward_mode != "comparison":
            return await super().aggregate_metrics(body)

        from resources_servers.gdpval.comparison import calculate_elo

        # Prefer the raw judge vote counts (``total_wins``/``total_losses``/
        # ``total_ties``) when present so the win rate reflects every
        # eval×ref_repeat×trial comparison. Fall back to the bool flags for
        # verify responses produced before this field existed — those count as
        # one vote each.
        def _votes(vr: Dict[str, Any]) -> tuple[int, int, int]:
            tw, tl, tt = vr.get("total_wins"), vr.get("total_losses"), vr.get("total_ties")
            if tw is not None or tl is not None or tt is not None:
                return int(tw or 0), int(tl or 0), int(tt or 0)
            return int(bool(vr.get("win"))), int(bool(vr.get("loss"))), int(bool(vr.get("tie")))

        wins = losses = ties = 0
        for vr in body.verify_responses:
            w, ls, t = _votes(vr)
            wins += w
            losses += ls
            ties += t
        judged = wins + losses + ties

        if judged == 0:
            return await super().aggregate_metrics(body)

        win_rate = (wins + 0.5 * ties) / judged
        eval_elo, normalized_elo = calculate_elo(win_rate, self.config.reference_elo)

        base = await super().aggregate_metrics(body)
        extra = {
            "comparison/wins": wins,
            "comparison/losses": losses,
            "comparison/ties": ties,
            "comparison/judged": judged,
            "comparison/win_rate": win_rate,
            "comparison/eval_elo": eval_elo,
            "comparison/normalized_elo": normalized_elo,
            "comparison/reference_elo": self.config.reference_elo,
        }
        merged_agent = {**base.agent_metrics, **extra}
        merged_key = {**base.key_metrics, **extra}
        return AggregateMetrics(
            group_level_metrics=base.group_level_metrics,
            agent_metrics=merged_agent,
            key_metrics=merged_key,
        )


if __name__ == "__main__":
    GDPValResourcesServer.run_webserver()
