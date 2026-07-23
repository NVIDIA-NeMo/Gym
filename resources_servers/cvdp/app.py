# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CVDP resources server.

This module owns the *policy* side of CVDP verification: the HTTP ``verify``
contract, the request/response schemas, and how a model/agent answer is turned
into a reward. ``verify`` routes by category — code-comprehension tasks are
scored with BLEU/ROUGE against a reference answer (``_verify_subjective``),
while code-generation tasks are graded by actually running the task's test
harness (``_verify_objective``). The objective path either grades the files an
agent already wrote (``rtl_files``) or parses RTL out of the model's text, then
delegates execution to :class:`resources_servers.cvdp.testbench_runner.TestbenchRunner`,
which owns the *mechanism* (docker-compose → Apptainer translation, the SIF
cache, and the sandbox provider). Keeping execution in ``testbench_runner.py``
lets this file stay focused on the contract and scoring.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from cvdp_lib.cvdp_constants import (
    BLEU_THRESHOLD,
    CODE_COMPREHENSION_CATEGORIES,
    N_GRAM_DEFAULT,
    ROUGE_THRESHOLD,
    VERIF_EDA_CATEGORIES,
    is_score_based_category,
)
from cvdp_lib.model_helpers import ModelHelpers
from cvdp_lib.subjective import calculate_BLEU, calculate_ROUGE
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.cvdp.testbench_runner import TestbenchRunner


_helpers = ModelHelpers()


# ----------------------------
# Config
# ----------------------------


class CVDPResourcesServerConfig(BaseResourcesServerConfig):
    oss_sim_image: str = "ghcr.io/hdl/sim/osvb"
    oss_pnr_image: str = "ghcr.io/hdl/impl/pnr"
    eda_sim_image: str = ""  # Set to a commercial EDA image (e.g. Cadence Xcelium)
    container_timeout: int = 600
    num_processes: int = 4  # Max concurrent Apptainer jobs
    sif_cache_dir: str = ""  # Defaults to ~/.cache/nemo-gym/sif
    harness_workspace_dir: str = ""  # Optional host directory for per-rollout temp workspaces
    container_tmp_bind_path: str = ""  # If set, redirect in-container temp (e.g. /tmp) to per-rollout host storage
    sandbox_provider: Dict[str, Any] = {"apptainer": {}}


# ----------------------------
# Schemas
# ----------------------------


class CVDPVerifierMetadata(BaseModel):
    task_id: str
    categories: List[str] = []
    difficulty: str = ""
    target_files: List[str] = []  # Empty for code-comprehension categories
    harness_files: Dict[str, Optional[str]] = {}  # Empty for code-comprehension categories
    context_files: Dict[str, str] = {}  # Companion RTL from input.context (non-target files needed for compilation)
    subjective_reference: Optional[str] = None  # Reference answer for code-comprehension categories (6,8,9,10)


class CVDPRunRequest(BaseRunRequest):
    pass


class CVDPVerifyRequest(CVDPRunRequest, BaseVerifyRequest):
    verifier_metadata: Dict[str, Any]
    rtl_files: Optional[Dict[str, str]] = (
        None  # files the agent already wrote to disk in the sandbox (agentic flow). When present, these are graded directly instead of re-parsing RTL out of the model's chat text.
    )


class CVDPVerifyResponse(BaseVerifyResponse):
    task_id: Optional[str] = None
    category: Optional[str] = None  # e.g. "cid003" — for CVDP report
    difficulty: Optional[str] = None  # e.g. "easy" — for CVDP report
    extracted_rtl: Optional[Dict[str, str]] = None
    container_exit_code: Optional[int] = None
    container_stderr: Optional[str] = None
    container_services: Optional[List[Dict]] = None  # per-service results: [{"service", "exit_code", "stderr"}]
    execution_time: Optional[float] = None  # total harness wall time in seconds
    parse_failed: bool = False  # True when model produced output but RTL extraction failed
    bleu_score: Optional[float] = None  # BLEU score for code-comprehension categories
    rouge_score: Optional[float] = None  # ROUGE score for code-comprehension categories


# ----------------------------
# Code extraction helpers
# ----------------------------


def _parse_model_response(res: str, target_files: List[str]) -> Optional[Dict[str, str]]:
    """
    Parse model output using ModelHelpers.parse_model_response().
    Returns {filename: code} or None on failure.
    """
    if not target_files:
        return None

    no_schema = len(target_files) == 1

    # Match CVDP's openai_llm.py: strip response before parsing
    res = res.strip()

    # Match CVDP's openai_llm.py: fix JSON formatting for multi-file responses
    if not no_schema and res.startswith("{") and res.endswith("}"):
        res = _helpers.fix_json_formatting(res)

    output, success = _helpers.parse_model_response(res, files=target_files, no_schema=no_schema)

    if not success:
        return None

    if no_schema:  # schema is one first or multiple
        code = output.get("direct_text") or output.get("response")
        return {target_files[0]: code} if code else None

    result: Dict[str, str] = {}
    if "code" in output and isinstance(output["code"], list):
        for item in output["code"]:
            if isinstance(item, dict):
                result.update(item)

    return result if result else None


# ----------------------------
# Server
# ----------------------------


class CVDPResourcesServer(SimpleResourcesServer):
    config: CVDPResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)
        # Sandbox execution (SIF cache, provider, compose translation) lives in
        # the harness runner; this server only owns the HTTP contract + scoring.
        self._harness = TestbenchRunner(self.config)

        # Warn if commercial EDA image is not configured.
        # Categories 12, 13, 14 require a commercial EDA image (e.g. Cadence Xcelium).
        # Apptainer uses host networking so no license network setup is needed
        # (unlike Docker which requires a dedicated license network).
        # This mirrors CVDP's validate_commercial_eda_setup() — warn but don't block.
        if not self.config.eda_sim_image:
            logging.warning(
                "eda_sim_image is not configured. "
                "Categories %s (commercial EDA) will fail if __VERIF_EDA_IMAGE__ "
                "is referenced in harness files.",
                VERIF_EDA_CATEGORIES,
            )

    async def verify(self, body: CVDPVerifyRequest) -> CVDPVerifyResponse:
        meta = CVDPVerifierMetadata.model_validate(body.verifier_metadata)

        category, difficulty = (
            meta.categories[0],
            meta.categories[1],
        )  # categories is [category_id, difficulty], e.g. ["cid003", "medium"]
        category_num = int(category[3:])  # "cid003" -> 3

        model_out = body.response.output_text

        # Fallback: if output_text is empty, try extracting RTL from reasoning content.
        # Mirrors cvdp's logic: if message.content is None, fall back to message.reasoning_content.
        if not model_out or not model_out.strip():
            for item in body.response.output:
                if getattr(item, "type", None) == "reasoning":
                    reasoning_texts = [s.text for s in (item.summary or []) if s.text]
                    if reasoning_texts:
                        model_out = "\n".join(reasoning_texts)
                        break

        has_model_output = bool(model_out and model_out.strip())
        if not has_model_output:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                extracted_rtl=None,
                container_exit_code=None,
                container_stderr=None,
                container_services=None,
                execution_time=0.0,
                parse_failed=False,
            )

        # Route: code-comprehension categories use subjective scoring,
        # code-generation categories use the docker-compose harness.
        if category_num in CODE_COMPREHENSION_CATEGORIES:
            return self._verify_subjective(body, meta, category, difficulty, category_num, model_out)

        return await self._verify_objective(body, meta, category, difficulty, model_out)

    def _verify_subjective(
        self,
        body: CVDPVerifyRequest,
        meta: CVDPVerifierMetadata,
        category: str,
        difficulty: str,
        category_num: int,
        model_out: str,
    ) -> CVDPVerifyResponse:
        """
        Subjective scoring for code-comprehension categories (6, 8, 9, 10).

        Mirrors repository.sbj() + dataset_processor.run_subjective_scoring():
        - Categories 6, 8: BLEU/ROUGE n-gram scoring
        - Categories 9, 10: Also BLEU/ROUGE (LLM subjective scoring requires a
          separate judge model endpoint which is not wired up here; BLEU/ROUGE
          serves as the default fallback, matching CVDP's behavior when no
          sbj_llm_model is configured)
        """
        reference = meta.subjective_reference or ""
        if not reference:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                container_stderr="No subjective_reference provided for code-comprehension category",
            )

        n_gram = N_GRAM_DEFAULT
        bleu_val = calculate_BLEU(model_out.strip(), reference, n_gram)
        rouge_val = calculate_ROUGE(model_out.strip(), reference, n_gram)

        # Score-based categories (6, 8, 9, 10) return the BLEU score directly
        # as a fractional reward, matching CVDP's SCORING_MODE_SCORE behavior.
        if is_score_based_category(category_num):
            reward = bleu_val
        else:
            # Threshold-based: both ROUGE and BLEU must pass
            rouge_pass = rouge_val > ROUGE_THRESHOLD
            bleu_pass = bleu_val > BLEU_THRESHOLD
            reward = 1.0 if (rouge_pass and bleu_pass) else 0.0

        return CVDPVerifyResponse(
            **body.model_dump(),
            reward=reward,
            task_id=meta.task_id,
            category=category,
            difficulty=difficulty,
            bleu_score=bleu_val,
            rouge_score=rouge_val,
        )

    async def _verify_objective(
        self,
        body: CVDPVerifyRequest,
        meta: CVDPVerifierMetadata,
        category: str,
        difficulty: str,
        model_out: str,
    ) -> CVDPVerifyResponse:
        """
        Objective scoring for code-generation categories via docker-compose harness.
        """
        # Agentic flow: the agent ran in its own sandbox and reports the files it
        # wrote on disk. Grade those directly. Model-only flow: fall back to
        # parsing RTL out of the model's text response.
        if body.rtl_files:
            rtl_files = dict(body.rtl_files)
        else:
            rtl_files = _parse_model_response(model_out, meta.target_files)

        # If model produced output but parsing failed, signal parse_failed so the
        # agent can retry with a fresh model completion — mirrors CVDP's
        # LLM_RETRY_COUNT loop in dataset_processor.py.
        if rtl_files is None:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                extracted_rtl=None,
                container_exit_code=None,
                container_stderr="parse_failed: could not extract RTL from model output",
                container_services=[],
                execution_time=0.0,
                parse_failed=True,
            )

        async with self._semaphore:
            t0 = time.time()
            exit_code, stderr, service_results = await self._harness.run(
                rtl_files=rtl_files or {},
                harness_files=meta.harness_files,
                task_id=meta.task_id,
                context_files=meta.context_files,
            )
            execution_time = time.time() - t0

        return CVDPVerifyResponse(
            **body.model_dump(),
            reward=1.0 if exit_code == 0 else 0.0,
            task_id=meta.task_id,
            category=category,
            difficulty=difficulty,
            extracted_rtl=rtl_files,
            container_exit_code=exit_code,
            container_stderr=stderr,
            container_services=service_results,
            execution_time=execution_time,
        )


if __name__ == "__main__":
    CVDPResourcesServer.run_webserver()
