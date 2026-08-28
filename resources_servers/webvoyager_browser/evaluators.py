# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence-only evaluator for post-close WebVoyager judging."""

from __future__ import annotations

import logging

from nemo_gym.web.models import WebBenchmark, WebObservation, WebTask, WebVerifierResult
from nemo_gym.web.native_browser import NativeBrowserEvaluationContext
from nemo_gym.web.session import EvaluatorConfigurationError


LOG = logging.getLogger("nemo_gym.resources_servers.webvoyager_browser")


class WebVoyagerEvidenceEvaluator:
    """Return immutable browser evidence for the standard judge resource server."""

    def prepare(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        browser_context: NativeBrowserEvaluationContext,
    ) -> None:
        del observation, browser_context
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise EvaluatorConfigurationError(f"WebVoyager evaluator received benchmark {task.benchmark.value!r}")

    def evaluate(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        final_answer: str | None,
        browser_context: NativeBrowserEvaluationContext,
    ) -> WebVerifierResult:
        del observation
        evidence = list(browser_context.evidence)
        LOG.info(
            "event=webvoyager_evidence_complete benchmark=%s task=%s screenshots=%d final_answer_present=%s",
            task.benchmark.value,
            task.task_id,
            len(evidence),
            bool(final_answer),
        )
        return WebVerifierResult(
            valid_sample=False,
            failure_kind="external_judge_required",
            evidence=evidence,
            verifier_version="native-webvoyager-gemini-v1",
            metadata={"final_answer": final_answer or "", "screenshots": len(evidence)},
        )

    def close(self) -> None:
        return None
