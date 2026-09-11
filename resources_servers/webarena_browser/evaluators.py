# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-specific evaluators colocated with the WebArena browser."""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Mapping
from typing import Protocol

from nemo_gym.web.evaluation_collision import build_collision_plan, has_collision_mitigation
from nemo_gym.web.models import WebBenchmark, WebObservation, WebTask, WebVerifierResult
from nemo_gym.web.session import EvaluatorConfigurationError
from nemo_gym.web.visual_browser import VisualBrowserEvaluationContext
from resources_servers.webarena_browser.config import WebArenaBrowserResourcesServerConfig
from resources_servers.webarena_browser.site_auth import configured_site_urls, resolve_site_templates


LOG = logging.getLogger("nemo_gym.resources_servers.webarena_browser")


class WebArenaFamilyEvaluator(Protocol):
    """One benchmark's scoring contract inside the WebArena browser process."""

    def prepare(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        browser_context: VisualBrowserEvaluationContext,
    ) -> None: ...

    def evaluate(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        final_answer: str | None,
        browser_context: VisualBrowserEvaluationContext,
    ) -> WebVerifierResult: ...

    def close(self) -> None: ...


def _task_needs_judge(task_config: dict) -> bool:
    eval_config = task_config.get("eval") or {}
    reference_answers = eval_config.get("reference_answers") or {}
    return bool(set(reference_answers).intersection({"exact_match", "must_include", "fuzzy_match"}))


def _reference_task_config(
    task: WebTask,
    config: WebArenaBrowserResourcesServerConfig,
) -> tuple[dict, dict]:
    """Build the reference evaluator's normalized, deployment-resolved input."""

    site_urls = configured_site_urls(task)
    task_config = copy.deepcopy(task.original_metadata)
    task_config.setdefault("id", task.task_id)
    task_config["task_id"] = int(task.task_id) if task.task_id.isdigit() else task.task_id
    task_config["intent"] = task.intent
    task_config["sites"] = list(task.sites)
    task_config["start_url"] = list(task.start_urls)
    task_config = resolve_site_templates(task_config, site_urls)
    if not isinstance(task_config.get("eval"), dict):
        raise EvaluatorConfigurationError("WebArena task requires an eval object")
    if not isinstance(task_config["eval"].get("eval_types"), list):
        raise EvaluatorConfigurationError("WebArena task requires eval.eval_types")

    source_plan = task.task_kwargs.get("collision_plan")
    collision_plan = copy.deepcopy(source_plan) if isinstance(source_plan, dict) else build_collision_plan(task_config)
    collision_plan = resolve_site_templates(collision_plan, site_urls)
    return task_config, collision_plan


class WebArenaClassicEvaluator:
    """Run the pinned local evaluator while its Playwright page is still live."""

    def __init__(
        self,
        config: WebArenaBrowserResourcesServerConfig,
    ) -> None:
        self.config = config
        self._task_key: tuple[WebBenchmark, str] | None = None
        self._task_config: dict | None = None
        self._collision_plan: dict | None = None
        self._before: dict | None = None

    def prepare(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        browser_context: VisualBrowserEvaluationContext,
    ) -> None:
        del observation
        if task.benchmark != WebBenchmark.WEBARENA:
            raise EvaluatorConfigurationError(f"WebArena evaluator received benchmark {task.benchmark.value!r}")
        task_config, collision_plan = _reference_task_config(task, self.config)
        if _task_needs_judge(task_config) and not os.environ.get("WEBARENA_JUDGE_API_KEY", "").strip():
            raise EvaluatorConfigurationError("WEBARENA_JUDGE_API_KEY is required by this WebArena task")

        from resources_servers.webarena_browser.reference_evaluation import (
            collect_browser_snapshots_sync,
            collect_snapshots,
            merge_snapshots,
        )

        before = merge_snapshots(
            collect_snapshots(collision_plan),
            collect_browser_snapshots_sync(browser_context.page, collision_plan),
        )
        self._task_key = (task.benchmark, task.task_id)
        self._task_config = task_config
        self._collision_plan = collision_plan
        self._before = before
        LOG.info(
            "event=webarena_evaluator_prepared benchmark=%s task=%s collision_mitigation=%s snapshot_groups=%d",
            task.benchmark.value,
            task.task_id,
            has_collision_mitigation(collision_plan),
            len(before),
        )

    def evaluate(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        final_answer: str | None,
        browser_context: VisualBrowserEvaluationContext,
    ) -> WebVerifierResult:
        del observation
        if self._task_key != (task.benchmark, task.task_id):
            raise RuntimeError("WebArena evaluator is not prepared for this task")
        if self._task_config is None or self._collision_plan is None or self._before is None:
            raise RuntimeError("WebArena evaluator state is incomplete")

        from resources_servers.webarena_browser.reference_evaluation import (
            build_snapshot_context,
            collect_browser_snapshots_sync,
            collect_snapshots,
            evaluate_classic_task_sync,
            merge_snapshots,
        )

        after = merge_snapshots(
            collect_snapshots(self._collision_plan),
            collect_browser_snapshots_sync(browser_context.page, self._collision_plan),
        )
        eval_context = build_snapshot_context(self._collision_plan, self._before, after)
        judge_log_path = (
            browser_context.artifact_dir / "webarena-evaluator-judge.jsonl"
            if browser_context.artifact_dir is not None
            else None
        )
        agent_result = {"answer": final_answer or ""}
        score, message = evaluate_classic_task_sync(
            self._task_config,
            agent_result,
            browser_context.page,
            judge_log_path=judge_log_path,
            eval_context=eval_context,
        )
        verifier_version = "webarena-reference-3b775dc"
        score = max(0.0, min(1.0, float(score)))
        LOG.info(
            "event=webarena_evaluator_complete benchmark=%s task=%s score=%.3f message=%r",
            task.benchmark.value,
            task.task_id,
            score,
            message[:500],
        )
        return WebVerifierResult(
            reward=score,
            raw_score=score,
            task_success=score >= 1.0,
            valid_sample=True,
            evidence=list(browser_context.evidence),
            verifier_version=verifier_version,
            metadata={
                "evaluation_message": message,
                "collision_mitigation": has_collision_mitigation(self._collision_plan),
                "snapshot_groups": sorted(set(self._before) | set(after)),
            },
        )

    def close(self) -> None:
        self._task_key = None
        self._task_config = None
        self._collision_plan = None
        self._before = None


class WebArenaTaskEvaluator:
    """Select the local WebArena-family evaluator without browser coupling.

    WebArena-family evaluators plug into the two-phase lifecycle: ``prepare``
    captures before-state and ``evaluate`` scores against the still-live page.
    WebVoyager is owned by the dedicated ``visual_browser`` server.

    The router deliberately fails closed when a benchmark evaluator has not
    been installed. Merely allowing a benchmark in server configuration must
    never turn an unevaluated rollout into a zero or a false success.
    """

    def __init__(
        self,
        evaluators: Mapping[WebBenchmark, WebArenaFamilyEvaluator] | None = None,
        *,
        config: WebArenaBrowserResourcesServerConfig | None = None,
    ) -> None:
        if evaluators is None:
            resolved_evaluators: dict[WebBenchmark, WebArenaFamilyEvaluator] = {}
            if config is not None:
                if WebBenchmark.WEBARENA in config.allowed_benchmarks:
                    resolved_evaluators[WebBenchmark.WEBARENA] = WebArenaClassicEvaluator(config)
            self._evaluators = resolved_evaluators
        else:
            self._evaluators = dict(evaluators)
        self._active: WebArenaFamilyEvaluator | None = None
        self._active_task: tuple[WebBenchmark, str] | None = None

    def _evaluator_for(self, task: WebTask) -> WebArenaFamilyEvaluator:
        evaluator = self._evaluators.get(task.benchmark)
        if evaluator is None:
            raise EvaluatorConfigurationError(
                f"WebArena-family evaluator for benchmark {task.benchmark.value!r} is not installed"
            )
        return evaluator

    def prepare(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        browser_context: VisualBrowserEvaluationContext,
    ) -> None:
        if self._active is not None:
            raise RuntimeError("WebArena-family evaluator is already bound to a task")
        evaluator = self._evaluator_for(task)
        evaluator.prepare(
            task=task,
            observation=observation,
            browser_context=browser_context,
        )
        self._active = evaluator
        self._active_task = (task.benchmark, task.task_id)

    def evaluate(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        final_answer: str | None,
        browser_context: VisualBrowserEvaluationContext,
    ) -> WebVerifierResult:
        task_key = (task.benchmark, task.task_id)
        if self._active is None or self._active_task != task_key:
            raise RuntimeError("WebArena-family evaluator must be prepared for the active task before evaluation")
        return self._active.evaluate(
            task=task,
            observation=observation,
            final_answer=final_answer,
            browser_context=browser_context,
        )

    def close(self) -> None:
        evaluator = self._active
        self._active = None
        self._active_task = None
        if evaluator is not None:
            evaluator.close()
