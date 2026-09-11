# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AA-Briefcase-Lite rubric and local pairwise evaluation server.

The binary path follows the public AA-Briefcase-Lite contract: one independent
strict pass/fail call per A/C check, using the released judge prompts. The
pairwise path evaluates each AQ/P criterion against configured public example
submissions. Artificial Analysis did not release the production pairwise
prompt or private comparison graph, so those scores are local diagnostics.

Artifact parsing, Office rendering, media routing, judge-panel sampling, and
position-debiased pairwise trials are reused from the GDPval implementation.
The judge never receives the AA source pool.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse, SimpleResourcesServer
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig
from resources_servers.gdpval.comparison import (
    JUDGE_REQUEST_TIMEOUT_SECONDS,
    Judge,
    build_file_section,
    run_trials,
)
from resources_servers.gdpval.judge_panel import (
    ResolvedJudge,
    dir_media_modalities,
    make_rng,
    merge_create_kwargs,
    sample_judge,
)
from resources_servers.gdpval.preconvert import preconvert_dir_async


_BINARY_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class AABriefcaseLiteResourcesServerConfig(GDPValResourcesServerConfig):
    """Public Lite grading configuration layered on GDPval judge support."""

    name: str = "aa_briefcase_lite"
    reward_mode: Literal["binary", "pairwise", "all"] = "binary"
    dataset_dir: str
    pairwise_reference_ids: List[str] = ["gpt-5-5"]
    pairwise_num_trials: int = Field(default=2, ge=1)
    binary_formatting_retries: int = Field(default=1, ge=0, le=3)


class AABriefcaseLiteVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    deliverables_dir: Optional[str] = None
    ng_task_index: Optional[int] = Field(default=None, alias="_ng_task_index")
    ng_rollout_index: Optional[int] = Field(default=None, alias="_ng_rollout_index")
    ng_attempt_index: Optional[int] = Field(default=None, alias="_ng_attempt_index")


class AABriefcaseLiteVerifyResponse(AABriefcaseLiteVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verify_mode: Literal["binary", "pairwise", "all"]
    judge_response: Optional[Dict[str, Any]] = None
    invalid_judge_response: bool = False
    invalid_judge_retryable: Optional[bool] = None
    rubric_passed: int = 0
    rubric_total: int = 0
    pairwise_wins: int = 0
    pairwise_losses: int = 0
    pairwise_ties: int = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_binary_judgement(text: str) -> Optional[dict[str, Any]]:
    """Parse the released binary judge shape without coercing malformed output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    match = _BINARY_JSON_RE.search(stripped)
    if match and match.group(0) != stripped:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get("passed"), bool):
            continue
        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            continue
        return {"passed": parsed["passed"], "reasoning": reasoning}
    return None


def _requested_filenames(checks: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for check in checks:
        for value in str(check.get("taskdoer_output_file", "")).split(","):
            name = value.strip()
            if not name:
                continue
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"invalid taskdoer_output_file entry: {name!r}")
            names.add(name)
    return sorted(names)


def _stage_submission(source_dir: Optional[str], filenames: list[str], stack: ExitStack) -> tuple[Path, list[str]]:
    """Create a disposable artifact view without mutating submitted files."""

    stage = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="aa_bclite_judge_")))
    source = Path(source_dir) if source_dir else None
    missing: list[str] = []
    for name in filenames:
        candidate = source / name if source is not None else None
        if candidate is None or not candidate.is_file():
            missing.append(name)
            continue
        (stage / name).symlink_to(candidate.resolve())
    return stage, missing


def _pairwise_task_prompt(task_markdown: str, check: dict[str, Any]) -> str:
    """Criterion-specific task supplied to GDPval's A/B trial runner."""

    return (
        "Judge only the following AA-Briefcase-Lite pairwise criterion. Do not use or infer "
        "facts from external source files; you receive only the task and submitted artifacts.\n\n"
        f"<ORIGINAL_TASK>\n{task_markdown}\n</ORIGINAL_TASK>\n\n"
        f"<CHECK_DESCRIPTION>\n{check['check_description']}\n</CHECK_DESCRIPTION>\n\n"
        f"<SUBMISSION_A_WINS>\n{check['score_1_criteria']}\n</SUBMISSION_A_WINS>\n\n"
        f"<SUBMISSION_B_WINS>\n{check['score_0_criteria']}\n</SUBMISSION_B_WINS>"
    )


class AABriefcaseLiteResourcesServer(GDPValResourcesServer):
    """Grade public Lite artifacts with official binary and local pairwise checks."""

    config: AABriefcaseLiteResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        root = Path(self.config.dataset_dir).resolve()
        if not (root / "checks.jsonl").is_file():
            raise FileNotFoundError(f"AA-Briefcase-Lite checks not found under {root}")
        self._aa_dataset_root = root
        self._aa_checks = _read_jsonl(root / "checks.jsonl")
        if len(self._aa_checks) != 63:
            raise ValueError(f"expected 63 AA-Briefcase-Lite checks, found {len(self._aa_checks)}")
        self._aa_binary_system = (root / "prompts" / "judge_system.txt").read_text(encoding="utf-8")
        self._aa_binary_user = (root / "prompts" / "judge_user.txt").read_text(encoding="utf-8")
        super().model_post_init(context)

    def _checks_for_task(self, task_id: str, scoring_type: str) -> list[dict[str, Any]]:
        checks = [
            check
            for check in self._aa_checks
            if check.get("task_id") == task_id and check.get("scoring_type") == scoring_type
        ]
        if not checks:
            raise ValueError(f"no {scoring_type} checks found for AA-Briefcase-Lite task {task_id!r}")
        return checks

    def _task_markdown(self, task_id: str) -> str:
        if not re.fullmatch(r"w\d+_t\d+", task_id):
            raise ValueError(f"invalid AA-Briefcase-Lite task id: {task_id!r}")
        path = self._aa_dataset_root / "tasks" / f"{task_id}.md"
        if not path.is_file():
            raise FileNotFoundError(f"task markdown not found: {path}")
        return path.read_text(encoding="utf-8")

    async def _preconvert(self, stage: Path) -> None:
        if not self.config.preconvert_office_to_pdf:
            return
        _ok, failed, errors = await preconvert_dir_async(stage, self.config.preconvert_max_concurrent)
        if failed:
            raise RuntimeError(f"AA artifact Office-to-PDF conversion failed: {errors}")

    async def _section(self, stage: Path, judge: ResolvedJudge, missing: list[str]) -> list[dict[str, Any]]:
        blocks = await asyncio.to_thread(
            build_file_section,
            str(stage),
            [],
            media_mode=judge.media_mode,
            render_dpi=self.config.judge_pdf_render_dpi,
            max_pages=self.config.judge_pdf_max_pages,
            include_text=self.config.judge_pdf_include_text,
            audio_capable=judge.handles_audio,
            video_capable=judge.handles_video,
        )
        blocks.extend({"type": "text", "text": f"[required submitted file missing: {name}]"} for name in missing)
        return blocks

    async def _binary_call(
        self,
        judge: ResolvedJudge,
        task_markdown: str,
        check: dict[str, Any],
        artifact_blocks: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], str]:
        user_text = self._aa_binary_user.format(
            task_markdown=task_markdown,
            check_description=check["check_description"],
            score_1_criteria=check["score_1_criteria"],
            score_0_criteria=check["score_0_criteria"],
        )
        messages = [
            {"role": "system", "content": self._aa_binary_system},
            {"role": "user", "content": [{"type": "text", "text": user_text}, *artifact_blocks]},
        ]
        client = AsyncOpenAI(
            base_url=judge.base_url,
            api_key=judge.api_key,
            timeout=JUDGE_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
        raw = ""
        for _attempt in range(self.config.binary_formatting_retries + 1):
            kwargs = merge_create_kwargs(
                {
                    "model": judge.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                judge.create_overrides,
            )
            response = await client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
            parsed = _parse_binary_judgement(raw)
            if parsed is not None:
                return parsed, raw
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            'Return only one JSON object with boolean key "passed" and string key "reasoning".'
                        ),
                    },
                ]
            )
        return None, raw

    async def _verify_binary(
        self,
        body: AABriefcaseLiteVerifyRequest,
        task_markdown: str,
        judges: list[ResolvedJudge],
        stage: Path,
        missing: list[str],
    ) -> tuple[float, list[dict[str, Any]], int]:
        checks = self._checks_for_task(body.task_id, "binary")
        section_cache: dict[str, list[dict[str, Any]]] = {}
        results: list[dict[str, Any]] = []
        invalid = 0
        passed = 0
        for check in checks:
            rng = make_rng(self.config.judge_sampling_seed, body.task_id, check["check_id"], "binary")
            selected = sample_judge(judges, rng)
            if selected.name not in section_cache:
                section_cache[selected.name] = await self._section(stage, selected, missing)
            parsed, raw = await self._binary_call(selected, task_markdown, check, section_cache[selected.name])
            result: dict[str, Any] = {
                "check_id": check["check_id"],
                "check_type": check["check_type"],
                "judge_name": selected.name,
                "passed": parsed["passed"] if parsed else None,
                "reasoning": parsed["reasoning"] if parsed else "invalid judge response",
            }
            if self.config.persist_raw_judge_responses:
                result["raw_response"] = raw
            if parsed is None:
                invalid += 1
            elif parsed["passed"]:
                passed += 1
            results.append(result)
        return passed / len(checks), results, invalid

    @staticmethod
    def _pairwise_judges(resolved: list[ResolvedJudge]) -> list[Judge]:
        clients: dict[tuple[str, str], OpenAI] = {}
        output: list[Judge] = []
        for item in resolved:
            key = (item.base_url, item.api_key)
            clients.setdefault(
                key,
                OpenAI(
                    base_url=item.base_url,
                    api_key=item.api_key,
                    timeout=JUDGE_REQUEST_TIMEOUT_SECONDS,
                    max_retries=0,
                ),
            )
            output.append(
                Judge(
                    name=item.name,
                    client=clients[key],
                    model=item.model,
                    create_overrides=item.create_overrides,
                    weight=item.weight,
                    handles_audio=item.handles_audio,
                    handles_video=item.handles_video,
                    media_mode=item.media_mode,
                )
            )
        return output

    async def _verify_pairwise(
        self,
        body: AABriefcaseLiteVerifyRequest,
        task_markdown: str,
        resolved_judges: list[ResolvedJudge],
        eval_stage: Path,
        eval_missing: list[str],
        stack: ExitStack,
    ) -> tuple[float, list[dict[str, Any]], int, int, int, int]:
        checks = self._checks_for_task(body.task_id, "pairwise")
        filenames = _requested_filenames(checks)
        results: list[dict[str, Any]] = []
        wins = losses = ties = invalid = 0
        for reference_id in self.config.pairwise_reference_ids:
            ref_source = self._aa_dataset_root / "submissions" / reference_id / body.task_id / "submission"
            if not ref_source.is_dir():
                raise FileNotFoundError(
                    f"public AA pairwise reference {reference_id!r} has no submission for {body.task_id}: {ref_source}"
                )
            ref_stage, ref_missing = _stage_submission(str(ref_source), filenames, stack)
            await self._preconvert(ref_stage)
            matchup_judges = list(resolved_judges)
            modalities = dir_media_modalities(eval_stage) | dir_media_modalities(ref_stage)
            if modalities:
                matchup_judges, _audio, _video = self._route_media_judges(
                    matchup_judges,
                    task_id=body.task_id,
                    modalities=modalities,
                    label=f"eval/reference {reference_id} artifacts",
                )
            eval_sections: dict[str, list[dict[str, Any]]] = {}
            ref_sections: dict[str, list[dict[str, Any]]] = {}
            for judge in matchup_judges:
                eval_sections[judge.name] = await self._section(eval_stage, judge, eval_missing)
                ref_sections[judge.name] = await self._section(ref_stage, judge, ref_missing)
            judges = self._pairwise_judges(matchup_judges)
            sections_by_judge = {
                judge.name: {
                    "refs": [],
                    "submission_a": eval_sections[judge.name],
                    "submission_b": ref_sections[judge.name],
                }
                for judge in judges
            }
            for check in checks:
                rng = make_rng(
                    self.config.judge_sampling_seed,
                    body.task_id,
                    check["check_id"],
                    reference_id,
                    "pairwise",
                )
                result = await asyncio.to_thread(
                    run_trials,
                    judges=judges,
                    task_prompt=_pairwise_task_prompt(task_markdown, check),
                    refs=[],
                    submission_a=eval_sections[judges[0].name],
                    submission_b=ref_sections[judges[0].name],
                    sections_by_judge=sections_by_judge,
                    num_trials=self.config.pairwise_num_trials,
                    return_raw_responses=self.config.persist_raw_judge_responses,
                    rng=rng,
                )
                check_wins = int(result["win_count_a"])
                check_losses = int(result["win_count_b"])
                check_ties = int(result["tie_count"])
                check_invalid = int(result["invalid_count"])
                wins += check_wins
                losses += check_losses
                ties += check_ties
                invalid += check_invalid
                results.append(
                    {
                        "check_id": check["check_id"],
                        "check_type": check["check_type"],
                        "reference_id": reference_id,
                        **result,
                    }
                )
        judged = wins + losses + ties
        reward = (wins + 0.5 * ties) / judged if judged else 0.0
        return reward, results, wins, losses, ties, invalid

    async def verify(self, body: AABriefcaseLiteVerifyRequest) -> AABriefcaseLiteVerifyResponse:
        task_markdown = self._task_markdown(body.task_id)
        binary_checks = self._checks_for_task(body.task_id, "binary")
        pairwise_checks = self._checks_for_task(body.task_id, "pairwise")
        filenames = _requested_filenames(binary_checks + pairwise_checks)

        with ExitStack() as stack:
            eval_stage, missing = _stage_submission(body.deliverables_dir, filenames, stack)
            await self._preconvert(eval_stage)
            resolved = self._resolve_judges()
            modalities = dir_media_modalities(eval_stage)
            if modalities:
                resolved, _audio, _video = self._route_media_judges(
                    resolved,
                    task_id=body.task_id,
                    modalities=modalities,
                    label="submitted artifact",
                )

            binary_reward = 0.0
            binary_results: list[dict[str, Any]] = []
            binary_invalid = 0
            if self.config.reward_mode in {"binary", "all"}:
                binary_reward, binary_results, binary_invalid = await self._verify_binary(
                    body, task_markdown, resolved, eval_stage, missing
                )

            pairwise_reward = 0.0
            pairwise_results: list[dict[str, Any]] = []
            wins = losses = ties = pairwise_invalid = 0
            if self.config.reward_mode in {"pairwise", "all"}:
                if not self.config.pairwise_reference_ids:
                    raise ValueError("pairwise mode requires at least one pairwise_reference_id")
                pairwise_reward, pairwise_results, wins, losses, ties, pairwise_invalid = await self._verify_pairwise(
                    body,
                    task_markdown,
                    resolved,
                    eval_stage,
                    missing,
                    stack,
                )

        invalid = binary_invalid + pairwise_invalid
        if self.config.reward_mode == "binary":
            reward = binary_reward
        elif self.config.reward_mode == "pairwise":
            reward = pairwise_reward
        else:
            # Convenience scalar only. Official AA uses a private MLE-Elo graph.
            reward = (binary_reward + pairwise_reward) / 2.0
        judge_response = {
            "protocol": ("official_public_binary" if self.config.reward_mode == "binary" else "local_unofficial"),
            "binary_score": binary_reward if binary_results else None,
            "binary_results": binary_results,
            "pairwise_score": pairwise_reward if pairwise_results else None,
            "pairwise_results": pairwise_results,
            "pairwise_warning": (
                "Local diagnostic only: AA has not released its production pairwise prompt "
                "or private comparison graph."
                if pairwise_results
                else None
            ),
        }
        return AABriefcaseLiteVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verify_mode=self.config.reward_mode,
            judge_response=judge_response,
            invalid_judge_response=bool(invalid),
            invalid_judge_retryable=True if invalid else None,
            rubric_passed=sum(1 for item in binary_results if item.get("passed") is True),
            rubric_total=len(binary_results),
            pairwise_wins=wins,
            pairwise_losses=losses,
            pairwise_ties=ties,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        valid = [row for row in body.verify_responses if not row.get("invalid_judge_response")]
        base = (
            await SimpleResourcesServer.aggregate_metrics(self, AggregateMetricsRequest(verify_responses=valid))
            if valid
            else AggregateMetrics()
        )
        passed = sum(int(row.get("rubric_passed", 0)) for row in valid)
        rubric_total = sum(int(row.get("rubric_total", 0)) for row in valid)
        wins = sum(int(row.get("pairwise_wins", 0)) for row in valid)
        losses = sum(int(row.get("pairwise_losses", 0)) for row in valid)
        ties = sum(int(row.get("pairwise_ties", 0)) for row in valid)
        judged = wins + losses + ties
        extra: dict[str, Any] = {
            "aa_lite/rows_total": len(body.verify_responses),
            "aa_lite/rows_valid": len(valid),
            "aa_lite/binary_passed": passed,
            "aa_lite/binary_total": rubric_total,
            "aa_lite/binary_pass_rate": passed / rubric_total if rubric_total else 0.0,
            "aa_lite/pairwise_wins": wins,
            "aa_lite/pairwise_losses": losses,
            "aa_lite/pairwise_ties": ties,
            "aa_lite/pairwise_judged": judged,
            "aa_lite/pairwise_win_rate": (wins + 0.5 * ties) / judged if judged else 0.0,
            # Numeric for downstream metric serializers. Always false for Lite.
            "aa_lite/official_leaderboard_comparable": 0.0,
        }
        return AggregateMetrics(
            group_level_metrics=base.group_level_metrics,
            agent_metrics={**base.agent_metrics, **extra},
            key_metrics={**base.key_metrics, **extra},
        )


if __name__ == "__main__":
    AABriefcaseLiteResourcesServer.run_webserver()
