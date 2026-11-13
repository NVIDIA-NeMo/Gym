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
import logging
import re
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import httpx
from fastapi import HTTPException, Request
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import (
    VERIFIER_ERROR,
    FailureBoundary,
    NgFailureEvent,
    current_verifier_invocation_id,
    observe_failure,
)
from nemo_gym.server_runtime import BaseVerifyResponse, SimpleResourcesServer
from nemo_gym.verification import JUDGE_FAILURES, JudgeError, is_judge_failure, reraise_judge_errors
from resources_servers.gdpval import persisted_layout
from resources_servers.gdpval.app import (
    GDPValResourcesServer,
    GDPValResourcesServerConfig,
    GDPValVerifyRequest,
    _TaskSession,
    resolve_deliverables,
)
from resources_servers.gdpval.comparison import (
    JUDGE_REQUEST_TIMEOUT_SECONDS,
    MAX_SECTION_TEXT_CHARS_FOR_JUDGE,
    MAX_TEXT_FILE_CHARS_FOR_JUDGE,
    Judge,
    _bounded_text,
    _load_raw_text,
    build_file_section,
    run_trials,
)
from resources_servers.gdpval.judge_panel import (
    ResolvedJudge,
    dir_media_modalities,
    is_audio_file,
    is_video_file,
    make_rng,
    merge_create_kwargs,
    sample_judge,
)
from resources_servers.gdpval.preconvert import OFFICE_EXTENSIONS, preconvert_dir_async, sidecar_pdf
from resources_servers.gdpval.task_sandbox import collect_deliverables


SANDBOX_WORKDIR = "/home/user"
# The read-only input directories of AA's protocol, under the working directory.
SOURCE_DIRECTORIES = ("shared", "week")

logger = logging.getLogger(__name__)


class AAEvaluationFault(RuntimeError):
    """The evaluation could not stage, convert or read what it grades; the model is not at fault."""


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
_BINARY_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_BINARY_JSON_INSTRUCTION = 'Return only one JSON object with boolean key "passed" and string key "reasoning".'


_BINARY_TRANSPORT_LOGGER = logging.getLogger(__name__ + ".binary_transport")
_BINARY_TRANSPORT_LOGGER.setLevel(logging.INFO)


_JUDGE_USAGE_LOGGER = logging.getLogger(__name__ + ".judge_usage")
_JUDGE_USAGE_LOGGER.setLevel(logging.INFO)
_PAIRWISE_TRANSPORT_LOGGER = logging.getLogger(__name__ + ".pairwise_transport")
_PAIRWISE_TRANSPORT_LOGGER.setLevel(logging.INFO)


def _log_judge_usage(response: httpx.Response, *, mode: str, model: str) -> None:
    """Retain usage for each returned attempt, without logging submitted content."""
    if not response.is_success:
        return
    try:
        body = response.json()
    except ValueError:
        return
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    choices = body.get("choices") or []
    _JUDGE_USAGE_LOGGER.info(
        "Judge usage: mode=%s model=%s prompt_tokens=%s completion_tokens=%s "
        "total_tokens=%s reasoning_tokens=%s cached_tokens=%s finish_reason=%s",
        mode,
        model,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
        details.get("reasoning_tokens"),
        prompt_details.get("cached_tokens"),
        choices[0].get("finish_reason") if choices else None,
    )


class _PairwiseJudgeHttpClient(DefaultHttpxClient):
    """Capture usage before the shared pairwise helper reduces responses to text."""

    def __init__(self, *, model: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        started = time.monotonic()
        status = None
        error_type = None
        try:
            response = super().send(request, **kwargs)
            status = response.status_code
            response.read()
            _log_judge_usage(response, mode="pairwise", model=self.model)
            return response
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            _PAIRWISE_TRANSPORT_LOGGER.info(
                "Pairwise judge transport: model=%s retry_count=%s status=%s error=%s duration_seconds=%.3f",
                self.model,
                request.headers.get("x-stainless-retry-count"),
                status,
                error_type,
                time.monotonic() - started,
            )


class _BinaryJudgeHttpClient(DefaultAsyncHttpxClient):
    """Observe each SDK transport attempt without changing its retry policy."""

    def __init__(self, *, check_id: str, model: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.check_id = check_id
        self.model = model
        self.format_attempt = 0
        self.transport_attempt = 0

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        self.transport_attempt += 1
        started = time.monotonic()
        status = None
        error_type = None
        try:
            response = await super().send(request, **kwargs)
            status = response.status_code
            await response.aread()
            _log_judge_usage(response, mode="binary", model=self.model)
            return response
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            _BINARY_TRANSPORT_LOGGER.info(
                "Binary judge transport: check_id=%s model=%s format_attempt=%d "
                "transport_attempt=%d status=%s error=%s duration_seconds=%.3f",
                self.check_id,
                self.model,
                self.format_attempt,
                self.transport_attempt,
                status,
                error_type,
                time.monotonic() - started,
            )


class AABriefcaseReference(BaseModel):
    """One pairwise opponent.

    Leave ``deliverables_dir`` unset for a reference shipped with the dataset. Set it to a local
    run's ``deliverables_cache`` to use that run's ``task_<id>/repeat_<repeat>/`` output instead.
    ``exclude_judges`` lists panel judges, by name, that must not score this reference, such as
    a judge from the same model family.
    """

    deliverables_dir: Optional[str] = None
    repeat: int = Field(default=0, ge=0)
    exclude_judges: List[str] = Field(default_factory=list)


class AABriefcaseLiteResourcesServerConfig(GDPValResourcesServerConfig):
    """Public Lite grading configuration layered on GDPval judge support."""

    name: str = "aa_briefcase_lite"
    reward_mode: Literal["binary", "pairwise", "all"] = "binary"
    dataset_dir: str
    # One entry per opponent; when empty, pairwise_reference_ids is used.
    pairwise_references: Dict[str, AABriefcaseReference] = Field(default_factory=dict)
    pairwise_reference_ids: List[str] = ["gpt-5-5"]
    pairwise_num_trials: int = Field(default=2, ge=1)
    binary_formatting_retries: int = Field(default=2, ge=0, le=3)
    binary_max_tokens_by_judge: Dict[str, int] = Field(default_factory=dict)
    # GDPVal's web tools need a key at startup. The prepared rows offer no web tool, and the routes refuse requests.
    tavily_api_key: Union[str, List[str]] = "unused"


class AABriefcaseLiteVerifyRequest(GDPValVerifyRequest):
    """GDPVal's request: the task id, the deliverable directory, and the rollout and attempt indices."""


class AABriefcaseLiteVerifyResponse(AABriefcaseLiteVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verify_mode: Literal["binary", "pairwise", "all"]
    judge_response: Optional[Dict[str, Any]] = None
    rubric_passed: int = 0
    rubric_total: int = 0
    pairwise_wins: int = 0
    pairwise_losses: int = 0
    pairwise_ties: int = 0
    # References that were not judged for this task; their votes are absent from the counts above.
    pairwise_references_skipped: int = 0


def _is_lfs_pointer(path: Path) -> bool:
    """Whether *path* is the stub Git LFS leaves for a file it has not downloaded."""
    if path.stat().st_size > 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX


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
                raise AAEvaluationFault(f"invalid taskdoer_output_file entry: {name!r}")
            names.add(name)
    return sorted(names)


def _existing_render(office_file: Path) -> Optional[Path]:
    """An existing PDF render of *office_file*, or None when it has none.

    Preconversion writes ``<stem>.pdf`` and switches to the injective
    ``<name>.pdf`` sidecar only when two Office files share a stem, so both
    names count. A plain ``<stem>.pdf`` is attributable to one source only when
    a single Office file carries that stem; otherwise it is left out rather
    than risk grading an unrelated document as the deliverable.
    """

    sidecar = sidecar_pdf(office_file)
    if sidecar.is_file():
        return sidecar
    plain = office_file.with_suffix(".pdf")
    if not plain.is_file():
        return None
    same_stem = [
        sibling
        for sibling in office_file.parent.iterdir()
        if sibling.suffix.lower() in OFFICE_EXTENSIONS and sibling.stem == office_file.stem
    ]
    return plain if len(same_stem) == 1 else None


def _stage_submission(
    source_dir: Optional[str], filenames: list[str], stack: ExitStack, *, carry_renders: bool
) -> tuple[Path, list[str]]:
    """Create a disposable artifact view without mutating submitted files.

    With ``carry_renders`` the caller does not convert Office deliverables, so an
    existing render is carried in and one is required: without it the judge would
    receive a filename-only stub and score a deliverable it never saw. When the
    caller does convert, renders are left out, so the judge always sees our
    conversion of the Office file rather than a PDF the model wrote itself.
    """

    stage = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="aa_bclite_judge_")))
    source = Path(source_dir) if source_dir else None
    missing: list[str] = []
    staged: list[Path] = []
    for name in filenames:
        candidate = source / name if source is not None else None
        if candidate is None or not candidate.is_file():
            missing.append(name)
            continue
        if _is_lfs_pointer(candidate):
            raise AAEvaluationFault(f"{candidate} is an unhydrated Git LFS pointer, not the file it stands for")
        (stage / name).symlink_to(candidate.resolve())
        staged.append(candidate)
    if not carry_renders:
        return stage, missing
    # The Office original stays the deliverable; the render is how the judge reads it.
    for candidate in staged:
        if candidate.suffix.lower() not in OFFICE_EXTENSIONS:
            continue
        render = _existing_render(candidate)
        if render is None:
            raise AAEvaluationFault(
                f"{candidate.name} has no PDF render beside it and Office-to-PDF conversion is "
                "off, so the judge would receive a filename-only stub. Place a render next to "
                "the deliverable, or set preconvert_office_to_pdf=true on a host with LibreOffice."
            )
        if (stage / render.name).exists():
            continue
        (stage / render.name).symlink_to(render.resolve())
    return stage, missing


def _judges_for_reference(reference: AABriefcaseReference, judges: list[ResolvedJudge]) -> list[ResolvedJudge]:
    """The panel for one matchup, without the reference's excluded judges; may be empty."""
    excluded = set(reference.exclude_judges)
    return [judge for judge in judges if judge.name not in excluded]


def _skipped_reference(reference_id: str, reference: AABriefcaseReference, reason: str) -> dict[str, Any]:
    """A result for a reference that was not judged; it has no vote counts, so it is never a tie."""
    return {
        "reference_id": reference_id,
        "skipped_reason": reason,
        "excluded_judges": sorted(reference.exclude_judges),
    }


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

    # AA's prompts name /home/user, and the task inputs sit in its shared/ and week/ directories.
    _sandbox_workdir: ClassVar[Optional[str]] = SANDBOX_WORKDIR

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
        self._require_office_rendering()
        self._require_pairwise_references()
        super().model_post_init(context)

    @property
    def _carry_renders(self) -> bool:
        """Renders beside a deliverable are used only when we do not convert ourselves."""

        return not self.config.preconvert_office_to_pdf

    def _require_office_rendering(self) -> None:
        """Refuse to start when Office deliverables could not be rendered for the judge.

        GDPval only guards comparison mode, so binary and pairwise grading would
        otherwise reach the judge with filename-only stubs after the rollouts are
        already spent.
        """

        if not self.config.preconvert_office_to_pdf or self.config.libreoffice_command is not None:
            # GDPval's own startup check probes the configured command.
            return
        from resources_servers.gdpval.setup_libreoffice import ensure_libreoffice

        if not ensure_libreoffice():
            raise RuntimeError(
                "preconvert_office_to_pdf=True but libreoffice could not be ensured on the host. "
                "Office deliverables would reach the judge as filename-only stubs. Install "
                "libreoffice, or place a PDF render beside every Office deliverable and set "
                "preconvert_office_to_pdf=false."
            )

    def _require_pairwise_references(self) -> None:
        if self.config.reward_mode in {"pairwise", "all"} and not self._resolved_references():
            raise ValueError("pairwise mode requires at least one pairwise reference")

    def _checks_for_task(self, task_id: str, scoring_type: str) -> list[dict[str, Any]]:
        checks = [
            check
            for check in self._aa_checks
            if check.get("task_id") == task_id and check.get("scoring_type") == scoring_type
        ]
        if not checks:
            raise AAEvaluationFault(f"no {scoring_type} checks found for AA-Briefcase-Lite task {task_id!r}")
        return checks

    def _task_markdown(self, task_id: str) -> str:
        if not re.fullmatch(r"w\d+_t\d+", task_id):
            raise AAEvaluationFault(f"invalid AA-Briefcase-Lite task id: {task_id!r}")
        path = self._aa_dataset_root / "tasks" / f"{task_id}.md"
        if not path.is_file():
            raise AAEvaluationFault(f"task markdown not found: {path}")
        return path.read_text(encoding="utf-8")

    async def _preconvert(self, stage: Path) -> None:
        if not self.config.preconvert_office_to_pdf:
            return
        _ok, failed, errors = await preconvert_dir_async(
            stage, self.config.preconvert_max_concurrent, libreoffice_command=self.config.libreoffice_command
        )
        if failed:
            raise AAEvaluationFault(f"AA artifact Office-to-PDF conversion failed: {errors}")

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
        # GDPval renders unknown extensions through a sibling PDF. AA also grades
        # the declared LaTeX source, so retain that source alongside the rendering.
        for source in sorted(stage.glob("*.tex")):
            # The shared converter labels a sibling PDF as its source file.
            # Identify the PDF actually attached, including the preferred sidecar.
            pdf = sidecar_pdf(source)
            if not pdf.is_file():
                pdf = source.with_suffix(".pdf")
            if pdf.is_file():
                for index, block in enumerate(blocks[:-1]):
                    content = blocks[index + 1]
                    has_rendering = content["type"] != "text" or content.get("text", "").startswith(
                        "[extracted text]\n"
                    )
                    if block.get("text") == f"\n{source.name}:\n" and has_rendering:
                        block["text"] = f"\n{pdf.name} (submitted alongside {source.name}):\n"
            text_used = sum(len(block.get("text", "")) for block in blocks)
            remaining = max(0, MAX_SECTION_TEXT_CHARS_FOR_JUDGE - text_used)
            label = _bounded_text(f"\n{source.name} (LaTeX source):\n", remaining)
            text = await asyncio.to_thread(
                _load_raw_text, source, min(MAX_TEXT_FILE_CHARS_FOR_JUDGE, remaining - len(label))
            )
            blocks.append({"type": "text", "text": label + text})
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
        before_artifact, after_artifact = user_text.split("<<<SUBMISSION CONTENT MESSAGES>>>")
        messages = [
            {"role": "system", "content": self._aa_binary_system + "\n\n" + _BINARY_JSON_INSTRUCTION},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": before_artifact},
                    *artifact_blocks,
                    {"type": "text", "text": after_artifact},
                ],
            },
        ]
        async with _BinaryJudgeHttpClient(check_id=check["check_id"], model=judge.model) as http_client:
            client = AsyncOpenAI(
                base_url=judge.base_url,
                api_key=judge.api_key,
                timeout=JUDGE_REQUEST_TIMEOUT_SECONDS,
                max_retries=2,
                http_client=http_client,
            )
            raw = ""
            for _attempt in range(self.config.binary_formatting_retries + 1):
                http_client.format_attempt = _attempt + 1
                http_client.transport_attempt = 0
                kwargs = merge_create_kwargs(
                    {
                        "model": judge.model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 4096,
                    },
                    judge.create_overrides,
                )
                # Binary budgets are independent of the larger pairwise comparison budget.
                if judge.name in self.config.binary_max_tokens_by_judge:
                    kwargs["max_tokens"] = self.config.binary_max_tokens_by_judge[judge.name]
                response = await client.chat.completions.create(**kwargs)
                raw = (response.choices[0].message.content or "").strip()
                parsed = _parse_binary_judgement(raw)
                if parsed is not None:
                    return parsed, raw
                if getattr(response.choices[0], "finish_reason", None) == "length":
                    logging.getLogger(__name__).warning(
                        "Invalid binary judge answer reached its token limit: check_id=%s model=%s "
                        "format_attempt=%d completion_tokens=%s",
                        check["check_id"],
                        judge.model,
                        _attempt + 1,
                        getattr(getattr(response, "usage", None), "completion_tokens", None),
                    )
                # Empty generations have nothing to repair: retry the same check without
                # adding empty assistant turns or rerunning already completed checks.
                if not raw:
                    continue
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": _BINARY_JSON_INSTRUCTION},
                    ]
                )
            return None, raw

    async def _verify_binary(
        self,
        body: AABriefcaseLiteVerifyRequest,
        task_markdown: str,
        judges: list[ResolvedJudge],
    ) -> tuple[float, list[dict[str, Any]]]:
        checks = self._checks_for_task(body.task_id, "binary")
        section_cache: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = {}
        stages: dict[tuple[str, ...], tuple[Path, list[str]]] = {}
        results: list[dict[str, Any]] = []
        passed = 0
        with ExitStack() as stack:
            for check in checks:
                filenames = tuple(_requested_filenames([check]))
                if filenames not in stages:
                    stages[filenames] = _stage_submission(
                        body.deliverables_dir, list(filenames), stack, carry_renders=self._carry_renders
                    )
                    await self._preconvert(stages[filenames][0])
                stage, missing = stages[filenames]
                if missing:
                    # A deliverable the model did not submit is the model's zero; the judge is not asked.
                    results.append(
                        {
                            "check_id": check["check_id"],
                            "check_type": check["check_type"],
                            "judge_name": None,
                            "passed": False,
                            "reasoning": f"required deliverable missing: {', '.join(missing)}",
                            "missing_deliverables": missing,
                        }
                    )
                    continue
                eligible = judges
                modalities = dir_media_modalities(stage)
                # Missing required media must not change this check's judge panel
                # across submissions; retain capability checks for attached media too.
                if any(is_audio_file(name) for name in filenames):
                    modalities.add("audio")
                if any(is_video_file(name) for name in filenames):
                    modalities.add("video")
                if modalities:
                    eligible, _audio, _video = self._route_media_judges(
                        judges, task_id=body.task_id, modalities=modalities, label="binary check artifact"
                    )
                rng = make_rng(self.config.judge_sampling_seed, body.task_id, check["check_id"], "binary")
                selected = sample_judge(eligible, rng)
                cache_key = (filenames, selected.name)
                if cache_key not in section_cache:
                    section_cache[cache_key] = await self._section(stage, selected, [])
                # A failed judge call masks the row as judge_failed and keeps the saved deliverable.
                parsed, raw = await reraise_judge_errors(
                    self._binary_call(selected, task_markdown, check, section_cache[cache_key])
                )
                if parsed is None:
                    # A check the judge never answered is no verdict on the model: scoring it as a fail would lower
                    # the row, and leaving it out would score a partial rubric.
                    raise JudgeError(
                        f"judge {selected.name} gave no usable verdict for check {check['check_id']} "
                        f"after {self.config.binary_formatting_retries + 1} attempts"
                    )
                result: dict[str, Any] = {
                    "check_id": check["check_id"],
                    "check_type": check["check_type"],
                    "judge_name": selected.name,
                    "passed": parsed["passed"],
                    "reasoning": parsed["reasoning"],
                }
                if self.config.persist_raw_judge_responses:
                    result["raw_response"] = raw
                passed += parsed["passed"]
                results.append(result)
        return passed / len(checks), results

    @staticmethod
    def _pairwise_judges(resolved: list[ResolvedJudge]) -> list[Judge]:
        clients: dict[tuple[str, str, str], OpenAI] = {}
        output: list[Judge] = []
        for item in resolved:
            key = (item.base_url, item.api_key, item.model)
            clients.setdefault(
                key,
                OpenAI(
                    base_url=item.base_url,
                    api_key=item.api_key,
                    timeout=JUDGE_REQUEST_TIMEOUT_SECONDS,
                    max_retries=2,
                    http_client=_PairwiseJudgeHttpClient(model=item.model),
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

    async def _judge_matchup(
        self,
        body: AABriefcaseLiteVerifyRequest,
        task_markdown: str,
        checks: list[dict[str, Any]],
        reference_id: str,
        reference: AABriefcaseReference,
        matchup_judges: list[ResolvedJudge],
        eval_stage: Path,
        eval_missing: list[str],
        ref_stage: Path,
        ref_missing: list[str],
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        """Judge every pairwise criterion of one reference; a judge failure raises and drops the reference."""
        eval_sections: dict[str, list[dict[str, Any]]] = {}
        ref_sections: dict[str, list[dict[str, Any]]] = {}
        for judge in matchup_judges:
            eval_sections[judge.name] = await self._section(eval_stage, judge, [])
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
        results: list[dict[str, Any]] = []
        wins = losses = ties = 0
        for check in checks:
            absent = [name for name in _requested_filenames([check]) if name in eval_missing]
            if absent:
                # A deliverable the model did not submit loses every trial; the judge is not asked.
                trials = self.config.pairwise_num_trials
                losses += trials
                results.append(
                    {
                        "check_id": check["check_id"],
                        "check_type": check["check_type"],
                        "reference_id": reference_id,
                        "excluded_judges": sorted(reference.exclude_judges),
                        "win_count_a": 0,
                        "win_count_b": trials,
                        "tie_count": 0,
                        "invalid_count": 0,
                        "missing_deliverables": absent,
                    }
                )
                continue
            rng = make_rng(
                self.config.judge_sampling_seed,
                body.task_id,
                check["check_id"],
                reference_id,
                "pairwise",
            )
            result = await reraise_judge_errors(
                asyncio.to_thread(
                    run_trials,
                    judges=judges,
                    task_prompt=_pairwise_task_prompt(task_markdown, check),
                    refs=[],
                    submission_a=eval_sections[judges[0].name],
                    submission_b=ref_sections[judges[0].name],
                    sections_by_judge=sections_by_judge,
                    num_trials=self.config.pairwise_num_trials,
                    # The SDK owns transport retries, matching the binary caller.
                    request_attempts=1,
                    invalid_response_retries=2,
                    return_raw_responses=self.config.persist_raw_judge_responses,
                    rng=rng,
                )
            )
            if result["invalid_count"]:
                # A vote the judge never cast would leave the criterion decided by the others.
                raise JudgeError(
                    f"{result['invalid_count']} of {self.config.pairwise_num_trials} judge answers for check "
                    f"{check['check_id']} against reference {reference_id} were unusable"
                )
            wins += int(result["win_count_a"])
            losses += int(result["win_count_b"])
            ties += int(result["tie_count"])
            results.append(
                {
                    "check_id": check["check_id"],
                    "check_type": check["check_type"],
                    "reference_id": reference_id,
                    "excluded_judges": sorted(reference.exclude_judges),
                    **result,
                }
            )
        return results, wins, losses, ties

    async def _verify_pairwise(
        self,
        body: AABriefcaseLiteVerifyRequest,
        task_markdown: str,
        resolved_judges: list[ResolvedJudge],
        eval_stage: Path,
        eval_missing: list[str],
        stack: ExitStack,
    ) -> tuple[float, list[dict[str, Any]], int, int, int]:
        checks = self._checks_for_task(body.task_id, "pairwise")
        filenames = _requested_filenames(checks)
        results: list[dict[str, Any]] = []
        wins = losses = ties = 0
        attempted = 0
        failed: dict[str, BaseException] = {}
        for reference_id, reference in self._resolved_references().items():
            ref_source = self._reference_source(reference_id, reference, body.task_id)
            if not ref_source.is_dir():
                # Not every shipped reference covers every task (o3 has no w1_t1).
                results.append(_skipped_reference(reference_id, reference, "no_submission_for_task"))
                logger.warning(
                    "AA-Briefcase-Lite task %s: skipped pairwise reference %s (no submission at %s)",
                    body.task_id,
                    reference_id,
                    ref_source,
                )
                continue
            ref_stage, ref_missing = _stage_submission(
                str(ref_source), filenames, stack, carry_renders=self._carry_renders
            )
            await self._preconvert(ref_stage)
            matchup_judges = _judges_for_reference(reference, resolved_judges)
            if not matchup_judges:
                results.append(_skipped_reference(reference_id, reference, "exclusion_empties_panel"))
                logger.warning(
                    "AA-Briefcase-Lite task %s: skipped pairwise reference %s (its excluded judges leave no judge)",
                    body.task_id,
                    reference_id,
                )
                continue
            modalities = dir_media_modalities(eval_stage) | dir_media_modalities(ref_stage)
            if modalities:
                matchup_judges, _audio, _video = self._route_media_judges(
                    matchup_judges,
                    task_id=body.task_id,
                    modalities=modalities,
                    label=f"eval/reference {reference_id} artifacts",
                )
            attempted += 1
            try:
                matchup = await self._judge_matchup(
                    body,
                    task_markdown,
                    checks,
                    reference_id,
                    reference,
                    matchup_judges,
                    eval_stage,
                    eval_missing,
                    ref_stage,
                    ref_missing,
                )
            except JUDGE_FAILURES as error:
                if not is_judge_failure(error):
                    raise
                # Like GDPVal's matchups, one reference whose judging failed does not fail the others.
                failed[reference_id] = error
                logger.warning(
                    "AA-Briefcase-Lite task %s: judging against reference %s failed: %s",
                    body.task_id,
                    reference_id,
                    error,
                )
                results.append(
                    {**_skipped_reference(reference_id, reference, "judge_failed"), "error": str(error)[:500]}
                )
                continue
            matchup_results, matchup_wins, matchup_losses, matchup_ties = matchup
            results.extend(matchup_results)
            wins += matchup_wins
            losses += matchup_losses
            ties += matchup_ties
        if not attempted:
            # Every reference was left out, so no vote was cast: the row has no pairwise score, and 0.0 would read as
            # the model losing every matchup.
            skipped = {item["reference_id"]: item["skipped_reason"] for item in results}
            raise AAEvaluationFault(f"no pairwise reference could be judged for task {body.task_id}: {skipped}")
        if len(failed) == attempted:
            # Nothing was judged, so the task has no pairwise score; the last failure masks the row.
            raise list(failed.values())[-1]
        judged = wins + losses + ties
        return (wins + 0.5 * ties) / judged, results, wins, losses, ties

    async def web_search(self, body: Any) -> str:
        raise HTTPException(status_code=404, detail="AA-Briefcase-Lite offers no web tools")

    async def fetch_web_page(self, body: Any) -> str:
        raise HTTPException(status_code=404, detail="AA-Briefcase-Lite offers no web tools")

    async def _download_reference_files(self, task: Any, reference_dir: Path) -> List[str]:
        """Stage the dataset's shared and week inputs for upload under the working directory."""
        staged: List[str] = []
        for name in SOURCE_DIRECTORIES:
            source = self._aa_dataset_root / "source_files" / name
            files = [path for path in sorted(source.rglob("*")) if path.is_file()]
            # An empty or unhydrated input directory would start every episode without the task's files, and the
            # model's work on nothing would be scored as its own.
            if not files:
                raise RuntimeError(f"AA-Briefcase-Lite input directory {source} has no files")
            pointers = [path.name for path in files if _is_lfs_pointer(path)]
            if pointers:
                raise RuntimeError(
                    f"AA-Briefcase-Lite input files under {source} are unhydrated Git LFS pointers: {pointers[:5]}"
                )
            for path in files:
                relative = f"{name}/{path.relative_to(source).as_posix()}"
                target = reference_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(path.resolve())
                staged.append(relative)
        return staged

    async def _collect_deliverables(self, session: _TaskSession) -> str:
        # The inputs are the dataset's, so unlike GDPVal they are not copied beside the submitted files.
        if session.deliverables_dir is None:
            task_dir = persisted_layout.repeat_dir(
                self.config.persist_deliverables_dir, session.task_id.task_id, session.repeat
            )
            with tempfile.TemporaryDirectory(prefix="aa_bclite_no_inputs_") as empty:
                await collect_deliverables(
                    session.sandbox, session.finish_call, Path(empty), task_dir, self._sandbox_workdir
                )
            await self._release(session)
            session.deliverables_dir = str(task_dir)
        return session.deliverables_dir

    async def verify(self, request: Request, body: AABriefcaseLiteVerifyRequest) -> AABriefcaseLiteVerifyResponse:
        requested = body
        body, masked = await resolve_deliverables(self, request, body)
        # A judge failure leaves verify through ``judge_failsafe``, which builds its masked row from the request's own
        # body. That row must name the persisted deliverable of a live session, or ``gym eval reverify`` has nothing
        # to judge again.
        requested.deliverables_dir = body.deliverables_dir
        if masked is not None:
            return AABriefcaseLiteVerifyResponse(**body.model_dump(), verify_mode=self.config.reward_mode, **masked)
        try:
            return await self._grade(body)
        except AAEvaluationFault as fault:
            # The verifier failed, not the model: the row keeps its deliverable and is masked, and a
            # reverify can grade it once the cause is fixed.
            logger.error("AA-Briefcase-Lite task %s could not be graded: %s", body.task_id, fault)
            event = NgFailureEvent(
                boundary=FailureBoundary.RESOURCES_SERVER,
                failure_kind=VERIFIER_ERROR,
                failure_reason=str(fault)[:2000],
                operation="verify",
                operation_id=current_verifier_invocation_id(),
                stage="verification",
                exception_type=type(fault).__name__,
            )
            event.__cause__ = fault
            observe_failure(event)
            return AABriefcaseLiteVerifyResponse(
                **body.model_dump(),
                verify_mode=self.config.reward_mode,
                reward=0.0,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason=str(fault)[:2000],
            )

    async def _grade(self, body: AABriefcaseLiteVerifyRequest) -> AABriefcaseLiteVerifyResponse:
        if not (body.deliverables_dir and Path(body.deliverables_dir).is_dir()):
            # Every file would read as one the model did not submit, and each check would score as its zero.
            raise AAEvaluationFault(f"deliverables_dir {body.deliverables_dir!r} is not a directory")
        task_markdown = self._task_markdown(body.task_id)
        binary_checks = self._checks_for_task(body.task_id, "binary")
        pairwise_checks = self._checks_for_task(body.task_id, "pairwise")
        filenames = _requested_filenames(binary_checks + pairwise_checks)

        with ExitStack() as stack:
            judges = self._resolve_judges()
            resolved = judges

            binary_reward = 0.0
            binary_results: list[dict[str, Any]] = []
            if self.config.reward_mode in {"binary", "all"}:
                binary_reward, binary_results = await self._verify_binary(body, task_markdown, judges)

            pairwise_reward = 0.0
            pairwise_results: list[dict[str, Any]] = []
            wins = losses = ties = 0
            if self.config.reward_mode in {"pairwise", "all"}:
                eval_stage, missing = _stage_submission(
                    body.deliverables_dir, filenames, stack, carry_renders=self._carry_renders
                )
                await self._preconvert(eval_stage)
                modalities = dir_media_modalities(eval_stage)
                if modalities:
                    resolved, _audio, _video = self._route_media_judges(
                        resolved,
                        task_id=body.task_id,
                        modalities=modalities,
                        label="submitted artifact",
                    )
                pairwise_reward, pairwise_results, wins, losses, ties = await self._verify_pairwise(
                    body,
                    task_markdown,
                    resolved,
                    eval_stage,
                    missing,
                    stack,
                )

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
            "pairwise_skipped_references": [
                {"reference_id": item["reference_id"], "reason": item["skipped_reason"]}
                for item in pairwise_results
                if "skipped_reason" in item
            ],
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
            rubric_passed=sum(1 for item in binary_results if item.get("passed") is True),
            rubric_total=len(binary_results),
            pairwise_wins=wins,
            pairwise_losses=losses,
            pairwise_ties=ties,
            pairwise_references_skipped=sum(1 for item in pairwise_results if "skipped_reason" in item),
        )

    def _resolved_references(self) -> Dict[str, AABriefcaseReference]:
        """``pairwise_references`` when set, else one default entry per ``pairwise_reference_ids``."""
        if self.config.pairwise_references:
            return dict(self.config.pairwise_references)
        return {reference_id: AABriefcaseReference() for reference_id in self.config.pairwise_reference_ids}

    def _reference_source(self, reference_id: str, reference: AABriefcaseReference, task_id: str) -> Path:
        """This reference's submission directory for one task."""
        if reference.deliverables_dir:
            return Path(reference.deliverables_dir) / f"task_{task_id}" / f"repeat_{reference.repeat}"
        return self._aa_dataset_root / "submissions" / reference_id / task_id / "submission"

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        rows = body.verify_responses
        base = await SimpleResourcesServer.aggregate_metrics(self, body) if rows else AggregateMetrics()
        passed = sum(int(row.get("rubric_passed", 0)) for row in rows)
        rubric_total = sum(int(row.get("rubric_total", 0)) for row in rows)
        wins = sum(int(row.get("pairwise_wins", 0)) for row in rows)
        losses = sum(int(row.get("pairwise_losses", 0)) for row in rows)
        ties = sum(int(row.get("pairwise_ties", 0)) for row in rows)
        skipped_references = sum(int(row.get("pairwise_references_skipped", 0)) for row in rows)
        judged = wins + losses + ties
        extra: dict[str, Any] = {
            "aa_lite/rows_total": len(rows),
            "aa_lite/binary_passed": passed,
            "aa_lite/binary_total": rubric_total,
            "aa_lite/pairwise_wins": wins,
            "aa_lite/pairwise_losses": losses,
            "aa_lite/pairwise_ties": ties,
            "aa_lite/pairwise_judged": judged,
            "aa_lite/pairwise_references_skipped": skipped_references,
            # Numeric for downstream metric serializers. Always false for Lite.
            "aa_lite/official_leaderboard_comparable": 0.0,
        }
        # A path the run did not take has no rate: 0.0 would read as the model failing every check.
        if rubric_total:
            extra["aa_lite/binary_pass_rate"] = passed / rubric_total
        if judged:
            extra["aa_lite/pairwise_win_rate"] = (wins + 0.5 * ties) / judged
        return AggregateMetrics(
            group_level_metrics=base.group_level_metrics,
            agent_metrics={**base.agent_metrics, **extra},
            key_metrics={**base.key_metrics, **extra},
        )


if __name__ == "__main__":
    AABriefcaseLiteResourcesServer.run_webserver()
