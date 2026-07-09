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

Both modes grade with a multi-judge *panel* (``judge_panel``): a set of frontier
judges (e.g. GPT-5.5, Gemini 3.1 Pro Preview, Claude Opus 4.8, each at their own
reasoning settings) that are sampled per comparison/scoring call. Panel sampling
+ seeding lives in ``judge_panel.py``. A single judge is just the degenerate
one-member panel synthesized from ``judge_model_server`` when ``judge_panel`` is
unset — there is one panel-based code path either way.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.server_utils import get_server_url
from resources_servers.gdpval.judge_panel import (
    ResolvedJudge,
    dir_contains_audio_video,
    make_rng,
    panel_summary,
    select_av_judges,
)


LOGGER = logging.getLogger(__name__)

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


class ReferenceModelConfig(BaseModel):
    """A single reference model for comparison-mode pairwise ELO.

    ``deliverables_dir`` is a directory tree of the reference model's
    deliverables, laid out as ``<deliverables_dir>/task_<task_id>/`` (optionally
    with ``repeat_<n>/`` subdirs) containing the same files the agent would
    persist (deliverable artifacts + finish_params.json + reference_files/).
    ``elo`` is the reference's known rating (e.g. a published Arena/AA number),
    held fixed when the eval model's MLE rating is fit.
    """

    deliverables_dir: str
    elo: float = _DEFAULT_REFERENCE_ELO


class JudgePanelMember(BaseModel):
    """One judge in a multi-judge panel.

    Each member is a distinct frontier model + reasoning configuration. For every
    comparison/scoring call one member is sampled (see
    ``judge_panel.sample_judge``). Members may share the single
    ``judge_model_server`` proxy and differ only by ``model`` + reasoning knobs,
    or point at distinct servers via ``model_server``.
    """

    # Human-readable label used in logs and the per-judge metrics breakdown,
    # e.g. "gpt-5.5", "gemini-3.1-pro", "claude-opus-4.8". Defaults to ``model``.
    name: Optional[str] = None
    # Upstream model id as the judge endpoint expects (e.g. "openai/gpt-5.5").
    # Falls back to ``create_params_overrides.model`` then the legacy default.
    model: Optional[str] = None
    # Defaults to the server-level ``judge_model_server`` when omitted.
    model_server: Optional[ModelServerRef] = None
    # Provider-specific generation/reasoning knobs merged into
    # ``chat.completions.create`` (e.g. ``{reasoning_effort: high}`` or
    # ``{extra_body: {...}}``). A ``None`` value drops the matching default.
    create_params_overrides: Dict[str, Any] = {}
    # Relative sampling weight; defaults to equal weighting across the panel.
    weight: float = 1.0
    # Set True for a member that natively reads audio/video (e.g. Gemini 3.1 Pro
    # Preview). Tasks whose deliverables/references contain audio or video files
    # are routed to the AV-capable member(s) instead of the sampled panel.
    handles_audio_video: bool = False


class GDPValResourcesServerConfig(BaseResourcesServerConfig):
    reward_mode: Literal["rubric", "comparison"] = "rubric"

    # Comparison-mode: one or more reference models the eval deliverable is
    # pairwise-judged against. The eval model's ELO is then estimated globally
    # via an anchored Bradley-Terry MLE over all references (see
    # ``comparison.calculate_mle_elo``). Keyed by an arbitrary reference id used
    # in the per-reference aggregate metrics.
    #
    #   reference_models:
    #     kimi_k2.5_thinking: {deliverables_dir: /gdpval/refs/kimi, elo: 1290}
    #     glm5.1:            {deliverables_dir: /gdpval/refs/glm5.1, elo: 1535}
    #
    # For back-compat the legacy single-reference fields
    # ``reference_deliverables_dir`` + ``reference_elo`` are still honored when
    # ``reference_models`` is empty (treated as a single reference id
    # ``"reference"``).
    reference_models: Dict[str, ReferenceModelConfig] = {}

    # Legacy single-reference fields. Prefer ``reference_models``.
    reference_deliverables_dir: Optional[str] = None

    # Pairwise judge trials per task. 4 is the historical default; alternates
    # swap/no-swap to debias position effects.
    num_comparison_trials: int = 4

    # ELO assigned to the (legacy single) reference model in pairwise mode.
    # Ignored when ``reference_models`` is set (each carries its own ``elo``).
    reference_elo: float = _DEFAULT_REFERENCE_ELO

    # Office→PDF preconversion for deliverables before pairwise judging.
    # Most office docs render poorly as raw text; PDFs let multimodal judges
    # read tables/charts. Costs ~5-30s per Office file.
    preconvert_office_to_pdf: bool = True
    preconvert_max_concurrent: int = 4

    judge_model_server: ModelServerRef
    judge_responses_create_params_overrides: Dict[str, Any] = {}
    judge_prompt_template_fpath: Optional[str] = None

    # Multi-judge panel. Every comparison/scoring call samples one member. A
    # single judge is just the degenerate case: when this is None a one-member
    # panel is synthesized from ``judge_model_server`` +
    # ``judge_responses_create_params_overrides`` (see ``_effective_panel``), so
    # there is a single panel-based code path with no separate single-judge
    # branch. Applies to every judge mode (rubric text/visual/structured and
    # pairwise comparison, including multi-stage ELO).
    judge_panel: Optional[List[JudgePanelMember]] = None
    # Seed for reproducible per-call judge sampling. The RNG is seeded per
    # (task_id, mode/ref_repeat) so reruns of the same task draw the same
    # judges. Leave None to seed only on those identity parts (still stable per
    # task); set an int to additionally shift the whole stream.
    judge_sampling_seed: Optional[int] = None

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
    # Optional per-request filter (comparison mode): judge the eval deliverable
    # only against this subset of the configured ``reference_models``. Unknown
    # ids are ignored; ``None`` (default) judges against every configured
    # reference. Used by the multi-stage ELO driver to select a different set of
    # reference models per judgementstage without reconfiguring the server.
    reference_ids: Optional[List[str]] = None


class GDPValVerifyResponse(GDPValVerifyRequest, BaseVerifyResponse):
    verify_mode: Literal["rubric", "comparison"] = "rubric"
    judge_response: Optional[Dict[str, Any]] = None
    invalid_judge_response: Optional[bool] = None
    # Majority-decision flags across all (ref_repeat × trial) judge votes —
    # kept for back-compat with older verify responses (still bool-valued).
    win: Optional[bool] = None
    loss: Optional[bool] = None
    tie: Optional[bool] = None
    # Raw judge vote counts aggregated over every reference (model × repeat ×
    # trial). ``aggregate_metrics`` prefers these so the win rate reflects all
    # comparisons rather than treating each verify call as a single vote.
    total_wins: Optional[int] = None
    total_losses: Optional[int] = None
    total_ties: Optional[int] = None
    # Per-reference-model vote breakdown for multi-reference comparison mode.
    # Maps reference id -> {wins, losses, ties, reference_elo, ref_repeat_count}.
    # ``aggregate_metrics`` uses these to build the per-reference battle table
    # that the anchored Bradley-Terry MLE is fit over.
    per_reference: Optional[Dict[str, Dict[str, Any]]] = None


class GDPValResourcesServer(SimpleResourcesServer):
    config: GDPValResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._judge_prompt_fpath: str = self.config.judge_prompt_template_fpath or _DEFAULT_JUDGE_PROMPT_FPATH
        # Normalize the reference-model set: prefer the multi-reference
        # ``reference_models`` mapping; fall back to the legacy single-reference
        # fields (treated as a single reference id ``"reference"``).
        self._references: Dict[str, ReferenceModelConfig] = {}
        if self.config.reward_mode == "comparison":
            if self.config.reference_models:
                self._references = dict(self.config.reference_models)
            elif self.config.reference_deliverables_dir:
                self._references = {
                    "reference": ReferenceModelConfig(
                        deliverables_dir=self.config.reference_deliverables_dir,
                        elo=self.config.reference_elo,
                    )
                }
            else:
                raise ValueError(
                    "reward_mode=comparison requires reference_deliverables_dir or reference_models to be set"
                )
        if self.config.preconvert_office_to_pdf:
            from resources_servers.gdpval.setup_libreoffice import ensure_libreoffice

            if not ensure_libreoffice() and self.config.reward_mode == "comparison":
                raise RuntimeError(
                    "preconvert_office_to_pdf=True and reward_mode='comparison' but libreoffice "
                    "could not be ensured on the host. Office deliverables would reach the multimodal "
                    "judge as filename-only stubs, biasing the win rate. Install libreoffice in the "
                    "deployment container, or set preconvert_office_to_pdf=false to opt out."
                )
        super().model_post_init(context)

    def _effective_panel(self) -> List[JudgePanelMember]:
        """The panel to grade with — always a non-empty list of members.

        A single judge is just the degenerate case: when ``judge_panel`` is unset
        we synthesize a one-member panel from the legacy single-judge fields
        (``judge_model_server`` + ``judge_responses_create_params_overrides``), so
        every code path downstream is panel-based.
        """
        if self.config.judge_panel:
            return self.config.judge_panel
        # Degenerate 1-member panel: inherit the legacy create-params overrides
        # (model/api_key are split out during resolution, the rest become the
        # member's reasoning/generation knobs).
        return [
            JudgePanelMember(create_params_overrides=dict(self.config.judge_responses_create_params_overrides or {}))
        ]

    def _resolve_judges(self) -> List[ResolvedJudge]:
        """Resolve the (always non-empty) panel to concrete upstream coordinates.

        Every judge — including the single-judge special case (see
        :meth:`_effective_panel`) — is resolved through this one loop. Members may
        share ``judge_model_server`` (differing only by model + reasoning) or
        point at their own ``model_server``. Per-member ``model`` / ``api_key``
        fall back to the legacy ``judge_responses_create_params_overrides`` then to
        sane defaults.
        """
        legacy_overrides = dict(self.config.judge_responses_create_params_overrides or {})

        def _url(server: ModelServerRef) -> str:
            return get_server_url(server.name) + "/v1"

        judges: List[ResolvedJudge] = []
        for i, member in enumerate(self._effective_panel()):
            server = member.model_server or self.config.judge_model_server
            overrides = dict(member.create_params_overrides or {})
            model = member.model or overrides.pop("model", None) or legacy_overrides.get("model", "judge")
            api_key = overrides.pop("api_key", None) or legacy_overrides.get("api_key", "dummy")
            judges.append(
                ResolvedJudge(
                    name=member.name or model or f"judge_{i}",
                    base_url=_url(server),
                    model=model,
                    api_key=api_key,
                    create_overrides=overrides or None,
                    weight=member.weight,
                    handles_audio_video=member.handles_audio_video,
                )
            )
        return judges

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

        judges = self._resolve_judges()
        # Route tasks with audio/video deliverables to the AV-capable judge(s) —
        # most judges can't read those modalities natively.
        if dir_contains_audio_video(body.deliverables_dir):
            av_judges = select_av_judges(judges)
            if [j.name for j in av_judges] != [j.name for j in judges]:
                print(
                    f"[gdpval] task {body.task_id} has audio/video deliverables; routing to "
                    f"{[j.name for j in av_judges]}",
                    flush=True,
                )
            judges = av_judges
        # Seed per task so a rerun samples the same judge(s); the ``rubric`` tag
        # keeps the stream distinct from the comparison path.
        rng = make_rng(self.config.judge_sampling_seed, body.task_id, "rubric")

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
                judges=judges,
                rng=rng,
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
                judges=judges,
                rng=rng,
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
                judges=judges,
                rng=rng,
                include_raw_responses=self.config.persist_raw_judge_responses,
            )

        return GDPValVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            verify_mode="rubric",
            judge_response=judge_result,
            invalid_judge_response=(judge_result is None),
        )

    async def _preconvert_and_log(self, target_dir: Path, *, label: str) -> None:
        from resources_servers.gdpval.preconvert import preconvert_dir_async

        n_ok, n_fail, errors = await preconvert_dir_async(
            target_dir, max_concurrent=self.config.preconvert_max_concurrent
        )
        if n_ok or n_fail:
            LOGGER.info("preconvert %s: ok=%d fail=%d", label, n_ok, n_fail)
        if n_fail:
            for msg in errors[:5]:
                LOGGER.warning("preconvert %s: %s", label, msg)

    async def _verify_comparison(self, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        from openai import OpenAI

        from resources_servers.gdpval.comparison import (
            JUDGE_REQUEST_TIMEOUT_SECONDS,
            Judge,
            build_file_section,
            clean_up_paths,
            run_trials,
            task_attempted,
        )

        eval_task_dir = Path(body.deliverables_dir) if body.deliverables_dir else None

        # Optional per-request reference subset (multi-stage ELO). When set, only
        # the named references are judged this call; unknown ids are ignored.
        active_references = self._references
        if body.reference_ids is not None:
            requested = set(body.reference_ids)
            active_references = {rid: cfg for rid, cfg in self._references.items() if rid in requested}

        # Resolve, per reference model, the available (attempted) repeat dirs
        # for this task. A reference that has no deliverable for this task is
        # simply skipped — the eval model just isn't judged against it here.
        ref_dirs_by_id: Dict[str, List[Path]] = {}
        for ref_id, ref_cfg in active_references.items():
            ref_task_root = Path(ref_cfg.deliverables_dir) / f"task_{body.task_id}"
            dirs = [d for d in _iter_ref_repeat_dirs(ref_task_root) if task_attempted(str(d))]
            if dirs:
                ref_dirs_by_id[ref_id] = dirs

        if not ref_dirs_by_id:
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
            await self._preconvert_and_log(eval_task_dir, label=f"eval/{body.task_id}")
            for ref_id, dirs in ref_dirs_by_id.items():
                for ref_dir in dirs:
                    await self._preconvert_and_log(ref_dir, label=f"ref/{ref_id}/{body.task_id}/{ref_dir.name}")

        clean_up_list: List[Path] = []
        # Build the judge panel. Members may share a single proxy server (so we
        # reuse one OpenAI client per distinct upstream) and differ only by model
        # + reasoning settings. run_trials samples one member per trial.
        resolved_judges = self._resolve_judges()
        client_cache: Dict[tuple, Any] = {}

        def _client_for(judge: ResolvedJudge) -> Any:
            key = (judge.base_url, judge.api_key)
            if key not in client_cache:
                client_cache[key] = OpenAI(
                    base_url=judge.base_url,
                    api_key=judge.api_key,
                    timeout=JUDGE_REQUEST_TIMEOUT_SECONDS,
                )
            return client_cache[key]

        judges = [
            Judge(
                name=rj.name,
                client=_client_for(rj),
                model=rj.model,
                create_overrides=rj.create_overrides or None,
                weight=rj.weight,
                handles_audio_video=rj.handles_audio_video,
            )
            for rj in resolved_judges
        ]

        # Route tasks with audio/video files (in the eval submission or any
        # reference) to the AV-capable judge(s) — most judges can't read those
        # modalities natively. Detection peeks into zip archives too.
        av_routed = dir_contains_audio_video(eval_task_dir) or any(
            dir_contains_audio_video(d) for dirs in ref_dirs_by_id.values() for d in dirs
        )
        if av_routed:
            av_judges = select_av_judges(judges)
            if [j.name for j in av_judges] != [j.name for j in judges]:
                print(
                    f"[gdpval] task {body.task_id} has audio/video files; routing comparison to "
                    f"{[j.name for j in av_judges]}",
                    flush=True,
                )
            judges = av_judges

        total_wins = 0
        total_losses = 0
        total_ties = 0
        # Per-reference-model vote tallies + a flat list of every (ref × repeat)
        # matchup for back-compat with the single-reference judge_response shape.
        per_reference: Dict[str, Dict[str, Any]] = {}
        per_ref_results: List[Dict[str, Any]] = []
        # eval-perspective per-judge tally pooled across every reference × repeat.
        per_judge_totals: Dict[str, Dict[str, int]] = {}
        # Per-reference judge failures (timeouts, upstream 5xx, oversize/context
        # payloads). With multiple references a single failed matchup must NOT
        # discard the whole rollout — we skip just that (ref × repeat) and keep
        # every reference that judged successfully.
        ref_errors: Dict[str, List[str]] = {}
        attempted_matchups = 0
        last_error: Optional[Exception] = None
        try:
            eval_submission = build_file_section(str(eval_task_dir), clean_up_list)

            # Judge the eval submission against every reference model, and within
            # each model against every available reference repeat. Raw vote
            # counts (not just per-matchup majority) are summed so the win rate
            # averages over reference variance — see ``_iter_ref_repeat_dirs``.
            for ref_id, dirs in ref_dirs_by_id.items():
                ref_wins = ref_losses = ref_ties = 0
                ref_judged_repeats = 0
                for ref_dir in dirs:
                    refs_subdir = ref_dir / "reference_files"
                    refs = build_file_section(
                        str(refs_subdir) if refs_subdir.is_dir() else None,
                        clean_up_list,
                    )
                    ref_submission = build_file_section(str(ref_dir), clean_up_list)
                    attempted_matchups += 1
                    # Seed per (task, ref_id, ref_repeat) so judge sampling is
                    # reproducible and each reference subset draws independently —
                    # this makes multi-stage ELO reruns replayable per stage.
                    rng = make_rng(self.config.judge_sampling_seed, body.task_id, ref_id, ref_dir.name)
                    try:
                        result = await asyncio.to_thread(
                            run_trials,
                            judges=judges,
                            task_prompt=body.prompt or "",
                            refs=refs,
                            submission_a=ref_submission,
                            submission_b=eval_submission,
                            num_trials=self.config.num_comparison_trials,
                            return_raw_responses=self.config.persist_raw_judge_responses,
                            rng=rng,
                        )
                    except Exception as e:  # noqa: BLE001 — isolate per-matchup judge failures
                        last_error = e
                        ref_errors.setdefault(ref_id, []).append(f"{ref_dir.name}: {e!r}")
                        print(
                            f"[gdpval] judge failed for task {body.task_id} ref {ref_id}/{ref_dir.name}: {e!r}",
                            flush=True,
                        )
                        continue
                    # ``run_trials`` casts submission_a=ref, submission_b=eval, so
                    # ``win_count_b`` is eval wins.
                    ref_wins += result["win_count_b"]
                    ref_losses += result["win_count_a"]
                    ref_ties += result["tie_count"]
                    ref_judged_repeats += 1
                    # Fold per-judge counts into eval-perspective panel totals
                    # (B=eval, A=ref) so the per-member balance is auditable.
                    for jname, jc in (result.get("per_judge") or {}).items():
                        agg = per_judge_totals.setdefault(jname, {"wins": 0, "losses": 0, "ties": 0, "trials": 0})
                        agg["wins"] += jc.get("win_count_b", 0)
                        agg["losses"] += jc.get("win_count_a", 0)
                        agg["ties"] += jc.get("tie_count", 0)
                        agg["trials"] += jc.get("trials", 0)
                    per_ref_results.append({"ref_id": ref_id, "ref_repeat": ref_dir.name, **result})

                # Only record references that produced at least one valid matchup;
                # a reference whose every repeat failed contributes no votes (and
                # must not appear as a 0/0/0 battle in aggregate_metrics).
                if ref_judged_repeats > 0:
                    per_reference[ref_id] = {
                        "wins": ref_wins,
                        "losses": ref_losses,
                        "ties": ref_ties,
                        "reference_elo": self._references[ref_id].elo,
                        "ref_repeat_count": ref_judged_repeats,
                    }
                    total_wins += ref_wins
                    total_losses += ref_losses
                    total_ties += ref_ties
        finally:
            clean_up_paths(clean_up_list)

        # Every matchup failed → this rollout is genuinely unjudgeable. Surface
        # it as a failure (matches pre-resilience behavior) rather than emitting
        # a fake neutral reward that would pollute the metrics.
        if attempted_matchups > 0 and not per_reference:
            raise RuntimeError(
                f"all {attempted_matchups} judge matchup(s) failed for task {body.task_id}; last error: {last_error!r}"
            )

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
                "per_reference": per_reference,
                "per_ref_repeat": per_ref_results,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "total_ties": total_ties,
                "total_judged": total_judged,
                "reference_count": len(per_reference),
                # Back-compat: total matchups across all references × repeats.
                "ref_repeat_count": len(per_ref_results),
                # References (and their repeats) whose judge calls failed and
                # were skipped. Empty when every matchup succeeded.
                "ref_errors": ref_errors,
                # Multi-judge panel that graded this rollout + the pooled
                # per-member vote tally (eval-perspective).
                "judge_panel": panel_summary(judges),
                "per_judge": per_judge_totals,
                # True when this task's audio/video content forced routing to the
                # AV-capable judge subset (``judge_panel`` above reflects it).
                "av_routed": av_routed,
            },
            win=reward == 1.0,
            loss=reward == 0.0,
            tie=reward == 0.5,
            total_wins=total_wins,
            total_losses=total_losses,
            total_ties=total_ties,
            per_reference=per_reference,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        if self.config.reward_mode != "comparison":
            return await super().aggregate_metrics(body)

        from resources_servers.gdpval.comparison import (
            calculate_elo,
            calculate_mle_elo,
            predict_win_rate,
        )

        # Prefer the raw judge vote counts (``total_wins``/``total_losses``/
        # ``total_ties``) when present so the win rate reflects every
        # eval×ref×repeat×trial comparison. Fall back to the bool flags for
        # verify responses produced before this field existed — those count as
        # one vote each.
        def _votes(vr: Dict[str, Any]) -> tuple[int, int, int]:
            tw, tl, tt = vr.get("total_wins"), vr.get("total_losses"), vr.get("total_ties")
            if tw is not None or tl is not None or tt is not None:
                return int(tw or 0), int(tl or 0), int(tt or 0)
            return int(bool(vr.get("win"))), int(bool(vr.get("loss"))), int(bool(vr.get("tie")))

        # Pool a set of verify responses into total win stats + per-reference
        # battle totals (ref_id -> [wins, losses, ties, ref_elo]). Factored out
        # so it can be applied to all rollouts (descriptive metrics) and, for
        # multi-stage runs, to each stage's rollouts independently.
        def _accumulate(verify_responses: List[Dict[str, Any]]) -> tuple[int, int, int, Dict[str, List[Any]]]:
            w_total = l_total = t_total = 0
            ref_totals: Dict[str, List[Any]] = {}
            for vr in verify_responses:
                w, ls, t = _votes(vr)
                w_total += w
                l_total += ls
                t_total += t
                for ref_id, counts in (vr.get("per_reference") or {}).items():
                    entry = ref_totals.setdefault(ref_id, [0, 0, 0, None])
                    entry[0] += int(counts.get("wins", 0) or 0)
                    entry[1] += int(counts.get("losses", 0) or 0)
                    entry[2] += int(counts.get("ties", 0) or 0)
                    if entry[3] is None:
                        # Prefer the ELO from config; fall back to whatever the
                        # verify response recorded at judging time.
                        cfg_ref = self._references.get(ref_id)
                        entry[3] = cfg_ref.elo if cfg_ref is not None else counts.get("reference_elo")
            return w_total, l_total, t_total, ref_totals

        # Fit the anchored Bradley-Terry MLE over a per-reference battle table.
        # Returns ``(eval_elo, normalized_elo, num_references)``; the elos are
        # ``None`` when no reference had both a known anchor ELO and a judged
        # game, or when the MLE could not produce a rating.
        def _fit_mle(ref_totals: Dict[str, List[Any]]) -> tuple[Optional[float], Optional[float], int]:
            stage_battles = [
                (float(ref_elo), rw, rl, rt)
                for (rw, rl, rt, ref_elo) in ref_totals.values()
                if ref_elo is not None and (rw + rl + rt) > 0
            ]
            if not stage_battles:
                return None, None, 0
            fit = calculate_mle_elo(stage_battles)
            if fit is None:
                return None, None, len(stage_battles)
            return fit[0], fit[1], len(stage_battles)

        # Multi-stage runs tag each rollout with the stage that produced it
        # (``stage_index``, stamped by the multi-stage orchestrator). Detect them
        # up front: a task may recur across stages (judged against a different
        # reference subset each time), so the same ``(task_index, rollout_index)``
        # appears once per stage — distinguished only by ``stage_index``.
        staged: Dict[int, List[Dict[str, Any]]] = {}
        for vr in body.verify_responses:
            stage_index = vr.get("stage_index")
            if stage_index is not None:
                staged.setdefault(int(stage_index), []).append(vr)

        # RewardProfiler (the base aggregation) keys rollouts by
        # ``(task_index, rollout_index)`` and rejects duplicates. Multi-stage
        # rollouts collide on that key by design, so feed the base profiler the
        # LAST stage alone — the headline stage, whose keys are unique — instead
        # of the pooled set. Single-stage / untagged runs use the full body.
        base_body = body
        if staged:
            base_body = AggregateMetricsRequest(verify_responses=staged[max(staged)])

        # Pooled (across every stage / reference) win stats — always emitted as
        # descriptive metrics regardless of staging.
        wins, losses, ties, per_ref_totals = _accumulate(list(body.verify_responses))

        judged = wins + losses + ties
        if judged == 0:
            return await super().aggregate_metrics(base_body)

        win_rate = (wins + 0.5 * ties) / judged

        base = await super().aggregate_metrics(base_body)
        # Total win stats (always emitted).
        extra: Dict[str, Any] = {
            "comparison/wins": wins,
            "comparison/losses": losses,
            "comparison/ties": ties,
            "comparison/judged": judged,
            "comparison/win_rate": win_rate,
        }

        # Per-reference win stats (always emitted when present).
        for ref_id, (rw, rl, rt, ref_elo) in per_ref_totals.items():
            r_judged = rw + rl + rt
            # Keep every emitted metric numeric (downstream coerces each metric
            # into a float ``Score``): use 0.0 rather than NaN when unjudged.
            r_win_rate = (rw + 0.5 * rt) / r_judged if r_judged else 0.0
            extra[f"comparison/ref/{ref_id}/wins"] = rw
            extra[f"comparison/ref/{ref_id}/losses"] = rl
            extra[f"comparison/ref/{ref_id}/ties"] = rt
            extra[f"comparison/ref/{ref_id}/judged"] = r_judged
            extra[f"comparison/ref/{ref_id}/win_rate"] = r_win_rate
            if ref_elo is not None:
                extra[f"comparison/ref/{ref_id}/reference_elo"] = ref_elo

        # When stages are present, fit each stage's ELO independently and report
        # the LAST stage's fit as the headline ``comparison/eval_elo`` (the
        # multi-stage design refines on a larger task set vs nearby references in
        # later stages), while exposing every stage's estimate as a
        # ``comparison/stage_<k>/*`` extra for visibility. Untagged runs keep the
        # original single-pass behavior below.
        if staged:
            extra["comparison/num_stages"] = len(staged)
            headline: Optional[tuple[Optional[float], Optional[float], int]] = None
            last_index = max(staged)
            for stage_index in sorted(staged):
                stage_responses = staged[stage_index]
                _, _, _, stage_ref_totals = _accumulate(stage_responses)
                stage_elo, stage_norm, stage_nref = _fit_mle(stage_ref_totals)
                prefix = f"comparison/stage_{stage_index}"
                if stage_elo is not None:
                    extra[f"{prefix}/eval_elo"] = stage_elo
                    extra[f"{prefix}/normalized_elo"] = stage_norm
                extra[f"{prefix}/num_references"] = stage_nref
                extra[f"{prefix}/num_tasks"] = len({vr.get("task_id") for vr in stage_responses})
                if stage_index == last_index:
                    headline = (stage_elo, stage_norm, stage_nref)

            # Headline = last stage's fit. Fall back to the pooled MLE only if
            # the last stage failed to produce a rating, so the run still
            # surfaces a number.
            if headline is not None and headline[0] is not None:
                eval_elo, normalized_elo, num_references = headline
            else:
                eval_elo, normalized_elo, num_references = _fit_mle(per_ref_totals)
            if eval_elo is not None:
                extra["comparison/eval_elo"] = eval_elo
                extra["comparison/normalized_elo"] = normalized_elo
                extra["comparison/num_references"] = num_references
                for ref_id, (rw, rl, rt, ref_elo) in per_ref_totals.items():
                    if ref_elo is not None:
                        extra[f"comparison/ref/{ref_id}/predicted_win_rate"] = predict_win_rate(
                            eval_elo, float(ref_elo)
                        )
        else:
            # ELO estimate. With per-reference battles we fit an anchored
            # Bradley-Terry MLE across all references; otherwise fall back to the
            # legacy single-anchor closed form.
            eval_elo, normalized_elo, num_references = _fit_mle(per_ref_totals)
            if eval_elo is not None:
                extra["comparison/eval_elo"] = eval_elo
                extra["comparison/normalized_elo"] = normalized_elo
                # Number of references the MLE was fit over (>1 ⇒ multi-reference
                # Bradley-Terry). All metric values must stay numeric — downstream
                # coerces each into a float ``Score`` — so we encode the method as a
                # count rather than a descriptive string.
                extra["comparison/num_references"] = num_references
                # Predicted (model-implied) win rate vs each reference, useful to
                # eyeball MLE fit against the observed per-reference win rate.
                for ref_id, (rw, rl, rt, ref_elo) in per_ref_totals.items():
                    if ref_elo is not None:
                        extra[f"comparison/ref/{ref_id}/predicted_win_rate"] = predict_win_rate(
                            eval_elo, float(ref_elo)
                        )
            else:
                eval_elo, normalized_elo = calculate_elo(win_rate, self.config.reference_elo)
                extra["comparison/eval_elo"] = eval_elo
                extra["comparison/normalized_elo"] = normalized_elo
                extra["comparison/reference_elo"] = self.config.reference_elo
                extra["comparison/num_references"] = 1

        merged_agent = {**base.agent_metrics, **extra}
        merged_key = {**base.key_metrics, **extra}
        return AggregateMetrics(
            group_level_metrics=base.group_level_metrics,
            agent_metrics=merged_agent,
            key_metrics=merged_key,
        )


if __name__ == "__main__":
    GDPValResourcesServer.run_webserver()
