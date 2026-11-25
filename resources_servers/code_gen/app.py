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

import base64
import functools
import json
import logging
import zlib
from asyncio import Lock, Semaphore
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union
from weakref import WeakValueDictionary

from lcb_integration.extraction_utils import LMStyle, extract_code
from pydantic import BaseModel, Field, model_validator
from sandbox_execution import check_assertions_in_sandbox, check_correctness_in_sandbox, task_deadline_s
from task_data import AssertionTests, ResourceLimits

from nemo_gym.config import get_global_config_dict
from nemo_gym.core import PROVIDER_UNAVAILABLE
from nemo_gym.metrics import (
    add_avg_sample_std_dev,
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.sandbox import SandboxAccess, resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    VerifierMode,
    borrow,
    observe_programmatic_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import SandboxVerificationConfig, SandboxVerificationError


LOG = logging.getLogger(__name__)

# Assertion tests without a task memory limit get 4 GB by default.
_ASSERTION_MEMORY_LIMIT_MB = 4096


# ----------------------------
# Config
# ----------------------------
class CompCodingResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    num_processes: int
    unit_test_timeout_secs: int
    debug: bool
    reasoning_format_penalty: float = 0.0
    # The unit tests run in the sandbox the Environment Server lends /verify. For a caller that lends none, such as a
    # re-verification, they run in a fresh sandbox per verification from this provider: a top-level sandbox block,
    # such as `sandbox` from nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml, or an inline provider
    # mapping.
    sandbox_provider: Union[str, Dict[str, Any]]
    sandbox_config: SandboxVerificationConfig = Field(default_factory=SandboxVerificationConfig)

    @model_validator(mode="after")
    def _require_sandbox_image(self) -> "CompCodingResourcesServerConfig":
        if not self.sandbox_config.image:
            raise ValueError("sandbox_config.image is required for code verification")
        return self


# ----------------------------
# Schemas
# ----------------------------


# This is LiveCodeBench format
class UnitTests(BaseModel):
    type: Literal["unit-tests"] = "unit-tests"
    inputs: List[str]
    outputs: List[str]
    fn_name: Optional[str] = None


class CompCodingRunRequest(BaseRunRequest):
    pass


class CompCodingVerifyRequest(CompCodingRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None
    # The verifier's sandbox (body key NG_SANDBOX_ACCESS_KEY), lent by the Environment Server that owns it and starts
    # it right before /verify. The tests run in it instead of in a sandbox of this server's own. Excluded from
    # dumps, so no response or result carries it.
    ng_sandbox_access: SandboxAccess | None = Field(default=None, exclude=True)


class CompCodingVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    result: Optional[List[Union[int, bool]]] = None
    metadata: Optional[Dict[str, Any]] = None
    unit_tests_time_taken: Optional[float] = None
    reasoning_format_violation_rate: float = 0.0
    difficulty: Optional[str] = None
    # The limits applied; a None memory limit is the fixed 4 GB guard.
    limits: ResourceLimits | None = None


# ----------------------------
# Server
# ----------------------------
class CompCodingResourcesServer(SimpleResourcesServer):
    # Each verification runs its candidate code on its own: there is no session state to export,
    # and a checkpoint can re-run a verification after a crash instead of waiting for it.
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"
    ray_enabled = False
    # The verifier runs the candidate's code, so /verify runs only in a separate group: a fresh one with no network
    # that the Environment Server starts right before /verify and lends it. "separate" is also the mode when a pairing
    # leaves verifier.mode unset. "shared" would grade in the agent's group, with the agent's image, network and
    # leftovers, and "none" would leave the sandbox to this server. A caller that lends no sandbox, such as a
    # re-verification, gets a fresh sandbox of this server's own.
    verifier_mode: ClassVar[VerifierMode] = "separate"
    supported_verifier_modes: ClassVar[frozenset[VerifierMode]] = frozenset({"separate"})
    config: CompCodingResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)
        self._sandbox_target: Optional[Tuple[Dict[str, Any], Dict[str, str]]] = None
        # One grade at a time per lent sandbox: a /verify the caller resent after a dropped connection waits for the
        # first instead of sharing the sandbox's CPU with it, which would turn per-test time limits into failures.
        self._lent_sandbox_locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()

    def _resolve_sandbox_target(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Provider config and sandbox metadata, resolved once from the global config."""
        if self._sandbox_target is None:
            global_config_dict = get_global_config_dict()
            provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
            metadata = {
                **resolve_provider_metadata(self.config.sandbox_provider, global_config_dict),
                "nemo_gym_server": self.config.name,
            }
            self._sandbox_target = (provider, metadata)
        return self._sandbox_target

    @staticmethod
    def _has_reasoning_format_violation(response) -> bool:
        open_tag = "<think>"
        close_tag = "</think>"

        # Final answer (output_text) should not contain any think tags.
        final_answer = response.output_text or ""
        if open_tag in final_answer or close_tag in final_answer:
            return True

        # Reasoning content should not have more than 1 think tag.
        reasoning = ""
        for item in response.output or []:
            if getattr(item, "type", None) == "reasoning":
                for summary in getattr(item, "summary", []) or []:
                    text = getattr(summary, "text", None)
                    if isinstance(text, str):
                        reasoning += text
        if reasoning.count(open_tag) > 1 or reasoning.count(close_tag) > 1:
            return True

        return False

    @staticmethod
    def _code_score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r["reward"] > 0)}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute code generation metrics: pass@k, majority@k, per-sample statistics.

        Produces overall metrics (with avg_sample_std_dev) and per-difficulty-subset metrics.
        """
        metrics, all_score_dicts, score_names, max_k = compute_pass_majority_metrics(
            tasks,
            score_fn=self._code_score_fn,
            answer_key="extracted_model_code",
        )
        add_avg_sample_std_dev(metrics, all_score_dicts, score_names, max_k)
        metrics.update(compute_subset_metrics(tasks, "difficulty", self._code_score_fn, "extracted_model_code"))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for code generation benchmarks."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=["accuracy"]))

        # Per-difficulty-subset headlines
        for prefix in {k.split("/pass@")[0] for k in agent_metrics if "/pass@" in k and k[0].islower()}:
            key.update(highest_k_metrics(agent_metrics, f"{prefix}/pass@1[avg-of-{{k}}]", score_names=["accuracy"]))

        return key

    async def verify(self, body: CompCodingVerifyRequest) -> CompCodingVerifyResponse:
        model_out = body.response.output_text
        difficulty = (body.verifier_metadata or {}).get("difficulty")

        if not model_out or not model_out.strip():
            # A response existed but had no usable text -> model failure
            return observe_programmatic_grading(
                lambda: CompCodingVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    difficulty=difficulty,
                )
            )

        # Tests are stored compressed by benchmarks/livecodebench/prepare_utils.py
        # (_pack_unit_tests) to keep the materialized-inputs file from reaching
        # 12 GB. Decode here, immediately before use, so the plaintext exists only
        # for this verification instead of for the whole run. Falls back to the
        # old plaintext shape so pre-patch artifacts still verify.
        _raw_tests = body.verifier_metadata["unit_tests"]
        if (
            isinstance(_raw_tests, dict)
            and "packed" in _raw_tests
            and _raw_tests.get("type", "unit-tests") == "unit-tests"
        ):
            _decoded = json.loads(zlib.decompress(base64.b64decode(_raw_tests["packed"])))
            _raw_tests = {
                "inputs": _decoded["inputs"],
                "outputs": _decoded["outputs"],
                "fn_name": _raw_tests.get("fn_name"),
            }
        if isinstance(_raw_tests, dict) and _raw_tests.get("type") == "assertions":
            tests = AssertionTests.model_validate(_raw_tests)
        else:
            tests = UnitTests.model_validate(_raw_tests)
        raw_limits = body.verifier_metadata.get("resource_limits")
        task_limits = ResourceLimits.model_validate({} if raw_limits is None else raw_limits)
        timeout = (
            task_limits.time_limit_s if task_limits.time_limit_s is not None else self.config.unit_test_timeout_secs
        )
        memory_limit_mb = task_limits.memory_limit_mb
        if isinstance(tests, AssertionTests) and memory_limit_mb is None:
            memory_limit_mb = _ASSERTION_MEMORY_LIMIT_MB
        limits = ResourceLimits(time_limit_s=timeout or None, memory_limit_mb=memory_limit_mb)

        # 3) extract code (code fence or raw)
        code = extract_code(model_out, LMStyle.OpenAIChat)
        if not code:
            return observe_programmatic_grading(
                lambda: CompCodingVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    extracted_model_output=model_out,
                    difficulty=difficulty,
                )
            )

        # 4) run in the sandbox lent to /verify, or else in a fresh provider sandbox. num_processes caps only the
        # sandboxes this server starts: a lent sandbox is its owner's, so its grade does not queue behind them.
        async with nullcontext() if body.ng_sandbox_access is not None else self._semaphore:
            """
            Sample looks like this:
            ```json
            {
                "input_output": "{\"inputs\": [...], ...}",
            }
            ```
            `input_output` looks like this:
            ```json
            {
                "inputs": [
                    "6\n4 13 2 3 2 6",
                    ...
                ],
                "outputs": [
                    "4 30 2 13 2 13",
                    ...
                ],
                "fn_name": null
            }
            ```
            """

            # Measured from here: a fresh sandbox's slot is held, and a grade in a lent sandbox includes borrowing the
            # sandbox and waiting for an earlier grade in it.
            start_time = time()

            grade = self._grader(
                code, tests, timeout=timeout, memory_limit_mb=memory_limit_mb, task_limits=task_limits
            )
            try:
                result, metadata = await self._run_tests(body.ng_sandbox_access, grade)
            except SandboxVerificationError as e:
                # The environment failed, not the policy: exclude the sample rather than score it.
                LOG.warning("code_gen sandbox verification failed (%s): %s", e.failure_kind, e)
                return CompCodingVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    mask_sample=True,
                    failure_kind=e.failure_kind,
                    failure_reason=str(e),
                    extracted_model_output=model_out,
                    extracted_model_code=code,
                    difficulty=difficulty,
                    limits=limits,
                )

            unit_tests_time_taken = time() - start_time

        def _grade() -> CompCodingVerifyResponse:
            has_violation = self._has_reasoning_format_violation(body.response)
            return CompCodingVerifyResponse(
                **body.model_dump(),
                reward=(
                    self.config.reasoning_format_penalty
                    if has_violation
                    else (1.0 if all(r == True for r in result) else 0.0)
                ),
                extracted_model_output=model_out,
                extracted_model_code=code,
                result=result,
                metadata=metadata,
                unit_tests_time_taken=unit_tests_time_taken,
                reasoning_format_violation_rate=1.0 if has_violation else 0.0,
                difficulty=difficulty,
                limits=limits,
            )

        return observe_programmatic_grading(_grade)

    def _grader(
        self,
        code: str,
        tests: UnitTests | AssertionTests,
        *,
        timeout: float,
        memory_limit_mb: int | None,
        task_limits: ResourceLimits,
    ) -> Callable[..., Awaitable[tuple[list[bool | int], dict[str, Any] | None]]]:
        """How to grade ``code`` against ``tests`` under the task's limits; call it with the sandbox to grade in
        (``sandbox=``) or the provider of a fresh one (``provider=``, ``metadata=``)."""
        if isinstance(tests, AssertionTests):
            return functools.partial(
                check_assertions_in_sandbox,
                code,
                tests.tests,
                time_limit_s=timeout,
                memory_limit_mb=memory_limit_mb,
                config=self.config.sandbox_config,
            )
        # A task's own time limit also bounds compilation; otherwise check_correctness keeps its deadline.
        deadline_s = task_deadline_s(timeout, len(tests.inputs)) if task_limits.time_limit_s is not None else None
        return functools.partial(
            check_correctness_in_sandbox,
            {"input_output": tests.model_dump_json()},  # sample
            code,  # generation
            timeout,
            self.config.debug,
            num_tests=len(tests.inputs),
            config=self.config.sandbox_config,
            memory_limit_mb=memory_limit_mb,
            deadline_s=deadline_s,
        )

    async def _run_tests(
        self,
        access: SandboxAccess | None,
        grade: Callable[..., Awaitable[tuple[list[bool | int], dict[str, Any] | None]]],
    ) -> tuple[list[bool | int], dict[str, Any] | None]:
        """Run the tests in the sandbox lent to /verify, or else in a fresh sandbox of this server's own.

        A lent sandbox is borrowed: this server releases its client after the tests, and the Environment Server that
        owns the sandbox stops it.

        Raises:
            SandboxVerificationError: When no sandbox can run them; the caller masks the sample.
        """
        if access is not None:
            key = json.dumps(access.connection.model_dump(mode="json"), sort_keys=True)
            lock = self._lent_sandbox_locks.setdefault(key, Lock())
            async with lock:
                return await self._run_tests_in_lent_sandbox(access, grade)
        provider, sandbox_metadata = self._resolve_sandbox_target()
        return await grade(provider=provider, metadata=sandbox_metadata)

    async def _run_tests_in_lent_sandbox(
        self,
        access: SandboxAccess,
        grade: Callable[..., Awaitable[tuple[list[bool | int], dict[str, Any] | None]]],
    ) -> tuple[list[bool | int], dict[str, Any] | None]:
        """Borrow the lent sandbox, run the tests in it, and release the client; its owner stops it."""
        try:
            sandbox = await borrow(access)
        except (ImportError, ValueError):
            # A missing provider SDK or a provider block this server cannot resolve would mask every sample, and some
            # providers (Enroot, Apptainer, bwrap) raise ValueError for a sandbox that is gone. Neither is the
            # policy's failure: fail the request, which the Environment Server retries as an infrastructure failure.
            raise
        except Exception as e:
            raise SandboxVerificationError(PROVIDER_UNAVAILABLE, f"borrowing the lent sandbox failed: {e!r}") from e
        try:
            return await grade(sandbox=sandbox)
        finally:
            # A failed release must not replace the grade or the error above it.
            try:
                await sandbox.disconnect()
            except Exception:
                LOG.warning("code_gen could not release its client of the lent sandbox", exc_info=True)


if __name__ == "__main__":
    CompCodingResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = CompCodingResourcesServer.run_webserver()  # noqa: F401
