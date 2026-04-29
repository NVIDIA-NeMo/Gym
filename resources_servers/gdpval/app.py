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
    win: Optional[bool] = None
    loss: Optional[bool] = None
    tie: Optional[bool] = None


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
        if deliverable_content_blocks:
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
            compute_comparison_reward,
            run_trials,
            task_attempted,
        )
        from resources_servers.gdpval.preconvert import preconvert_dir_async

        ref_root = Path(self.config.reference_deliverables_dir)
        ref_task_dir = ref_root / f"task_{body.task_id}"
        eval_task_dir = Path(body.deliverables_dir) if body.deliverables_dir else None

        if not task_attempted(str(ref_task_dir)):
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
            await preconvert_dir_async(ref_task_dir, max_concurrent=self.config.preconvert_max_concurrent)

        refs_dir = ref_task_dir / "reference_files"
        refs = build_file_section(str(refs_dir) if refs_dir.is_dir() else None)
        ref_submission = build_file_section(str(ref_task_dir))
        eval_submission = build_file_section(str(eval_task_dir))

        overrides = dict(self.config.judge_responses_create_params_overrides or {})
        judge_base_url = get_server_url(self.config.judge_model_server.name) + "/v1"
        judge_model_name = overrides.get("model", "judge")
        judge_api_key = overrides.get("api_key", "dummy")

        client = OpenAI(base_url=judge_base_url, api_key=judge_api_key)
        result = await asyncio.to_thread(
            run_trials,
            client=client,
            model=judge_model_name,
            task_prompt=body.prompt or "",
            refs=refs,
            submission_a=ref_submission,
            submission_b=eval_submission,
            num_trials=self.config.num_comparison_trials,
        )

        reward = compute_comparison_reward(result["winner"])
        return GDPValVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verify_mode="comparison",
            judge_response=result,
            win=reward == 1.0,
            loss=reward == 0.0,
            tie=reward == 0.5,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        if self.config.reward_mode != "comparison":
            return await super().aggregate_metrics(body)

        from resources_servers.gdpval.comparison import calculate_elo

        wins = sum(1 for vr in body.verify_responses if vr.get("win"))
        losses = sum(1 for vr in body.verify_responses if vr.get("loss"))
        ties = sum(1 for vr in body.verify_responses if vr.get("tie"))
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
