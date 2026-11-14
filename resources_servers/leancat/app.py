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

"""LeanCat resources server: formal category theory in Lean 4.

LeanCat (arXiv:2512.24796) is 100 statement-level 1-category-theory problems in Lean 4 /
Mathlib v4.19.0.

The task is *whole-file*: the model returns the entire Lean file (imports, preamble, any
auxiliary definitions, the target theorem with its proof), rather than a proof body the server
reassembles a file around. Because the model owns the whole file it could also weaken the
theorem, so ``proof_utils.check_statement_preserved`` compares it against the reference.

Verification runs through ``lean_proof.lean_sandbox``: one sandbox per server process,
created from an image carrying Lean and Mathlib at the version the rows are written against
(``lean_proof/lean_image`` builds one per version), reused across rollouts. Each attempt
is compiled with ``lake env lean``, which is what upstream's ``verify_lean`` does.

The text checks, status vocabulary, toolchain probe and sandbox runner come from
``resources_servers/lean_proof``, the library the Lean benchmarks share. Only what is
specific to LeanCat lives here: its row schema, its Easy/Medium/High metrics, and the
decision to require statement preservation by default.
"""

import logging
import uuid
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult
from resources_servers.lean_proof.lean_sandbox import (
    DEFAULT_LEAN_PROJECT_DIR,
    CompilerOutput,
    LeanSandbox,
)
from resources_servers.lean_proof.proof_utils import (
    axiom_check_passed,
    axiom_check_suffix,
    check_statement_preserved,
    extract_lean_code,
    find_banned_declarations,
)
from resources_servers.lean_proof.status import (
    STATUS_BANNED_TOKENS,
    STATUS_COMPLETED,
    STATUS_EMPTY_GENERATION,
    STATUS_SANDBOX_ERROR,
    STATUS_STATEMENT_MODIFIED,
    determine_proof_status,
)


logger = logging.getLogger(__name__)


def score_leancat_rollout(rollout: Dict[str, Any]) -> Dict[str, float]:
    """Named scores for aggregate metrics (see ``LeanCatResourcesServer.compute_metrics``)."""
    return {
        "accuracy": rollout["reward"],
        "statement_preserved": float(bool(rollout.get("statement_preserved", False))),
    }


class LeanCatResourcesServerConfig(BaseResourcesServerConfig):
    # verify() is a pure function of the request body and this config, so `gym eval reverify`
    # can rescore stored rollouts.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Name of a top-level sandbox config block, or an inline {provider: {...}} mapping.
    sandbox_provider: str = "sandbox"
    # image / snapshot_id / resources / ttl, as in the shipped provider configs. A snapshot
    # is the practical route here: there is no published image at Mathlib v4.19.0, and one
    # built inside a sandbox can be snapshotted once and reused (provider_options.snapshot_id).
    sandbox_config: Dict[str, Any] = {}
    # Lake project the compile runs in; lean_image/ builds Mathlib here.
    lean_project_dir: str = DEFAULT_LEAN_PROJECT_DIR

    # Upstream's per-attempt verification budget (EVALUATION.md); LeanCat proofs import all
    # of Mathlib and rely on heavy typeclass search.
    compilation_timeout: float = 300.0
    max_output_characters: int = 4000
    require_statement_preserved: bool = True
    ban_proof_shortcuts: bool = True

    # Probe the sandbox's Lean/Mathlib version once and log an error on a mismatch. A row's
    # own `lean_toolchain` overrides the default.
    check_lean_version: bool = True
    expected_lean_version: str = "4.19.0"


class LeanCatRunRequest(BaseRunRequest):
    # Fields arrive as flat row columns (see prepare.py); the validator below also accepts
    # them nested under `verifier_metadata`.
    verifier_metadata: Optional[Dict[str, Any]] = None

    formal_statement: str
    problem_id: Optional[str] = None
    level: Optional[str] = None
    tag: Optional[List[str]] = None
    domain: Optional[List[str]] = None
    natural_language_statement: Optional[str] = None
    lean_toolchain: Optional[str] = None
    mathlib_version: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _lift_verifier_metadata(cls, data: Any) -> Any:
        """Accept the row's fields nested under `verifier_metadata` or at the top level.

        Gym posts `verifier_metadata` to /verify still nested; lifting it here keeps the
        typed fields and lets `level` reach the response for compute_subset_metrics.
        Top-level keys win.
        """
        if isinstance(data, dict) and isinstance(data.get("verifier_metadata"), dict):
            return {**data["verifier_metadata"], **data}
        return data


class LeanCatVerifyRequest(LeanCatRunRequest, BaseVerifyRequest):
    pass


class LeanCatVerifyResponse(LeanCatVerifyRequest, BaseVerifyResponse):
    # Inherits the request fields so `level` reaches the rollout dict that
    # `compute_subset_metrics` groups by.
    proof_status: str
    predicted_proof: str
    statement_preserved: bool
    compiler_output: Optional[CompilerOutput] = None


class LeanCatResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: LeanCatResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._lean = LeanSandbox(
            sandbox_provider=self.config.sandbox_provider,
            sandbox_config=self.config.sandbox_config,
            project_dir=self.config.lean_project_dir,
            server_name=self.config.name,
        )
        self._toolchain_checked = False

    async def _check_toolchain_once(self, expected: Optional[str]) -> None:
        """Probe the sandbox's Lean version once per process; see LeanSandbox.check_toolchain."""
        if not self.config.check_lean_version or self._toolchain_checked:
            return
        # Set only after it succeeds: a probe that failed because the sandbox was not up yet
        # would otherwise never run again, and a wrong Mathlib would go unlogged all run.
        await self._lean.check_toolchain(
            expected or self.config.expected_lean_version,
            compile_fn=lambda code, timeout_s: self._run_lean(code, timeout_s=timeout_s),
        )
        self._toolchain_checked = True

    async def _run_lean(self, code: str, timeout_s: Optional[float] = None) -> SandboxExecResult:
        """Compile one submission."""
        return await self._lean.compile(
            code, timeout_s=self.config.compilation_timeout if timeout_s is None else timeout_s
        )

    async def verify(self, body: LeanCatVerifyRequest) -> LeanCatVerifyResponse:
        """Score one attempt: 1.0 only if it is a valid LeanCat proof, else 0.0.

        The text checks run first because they are free and a Mathlib compile is not.
        """
        body_dict = body.model_dump()
        code = extract_lean_code(body.response.output_text)

        if not code:
            return LeanCatVerifyResponse(
                **body_dict,
                reward=0.0,
                proof_status=STATUS_EMPTY_GENERATION,
                predicted_proof="",
                statement_preserved=False,
                failure_reason="No Lean code found in the response.",
            )

        if self.config.ban_proof_shortcuts:
            banned = find_banned_declarations(code)
            if banned:
                return LeanCatVerifyResponse(
                    **body_dict,
                    reward=0.0,
                    proof_status=STATUS_BANNED_TOKENS,
                    predicted_proof=code,
                    statement_preserved=False,
                    failure_reason=f"Submission uses banned declarations: {', '.join(banned)}.",
                )

        preserved, reason = check_statement_preserved(body.formal_statement, code)
        if self.config.require_statement_preserved and not preserved:
            return LeanCatVerifyResponse(
                **body_dict,
                reward=0.0,
                proof_status=STATUS_STATEMENT_MODIFIED,
                predicted_proof=code,
                statement_preserved=False,
                failure_reason=reason,
            )

        try:
            # Inside the handler: this is the first thing to touch the sandbox, so a sandbox
            # that will not start surfaced here as a 500 while the identical failure during
            # the compile came back as a status.
            await self._check_toolchain_once(body.lean_toolchain)
            token = f"AXIOMS_{uuid.uuid4().hex}"
            suffix = axiom_check_suffix(token) if self.config.ban_proof_shortcuts else ""
            result = await self._run_lean(code + suffix)
        except Exception as exc:  # noqa: BLE001 - any start/exec failure is a sandbox outcome
            # Without this a sandbox that will not start raises out of verify() as a 500,
            # while the same failure during exec comes back as `sandbox_error`. One rollout
            # should not take the run down, and the two paths should look the same.
            logger.error("sandbox failed for problem %s: %s: %s", body.problem_id, type(exc).__name__, exc)
            return LeanCatVerifyResponse(
                **body_dict,
                reward=0.0,
                proof_status=STATUS_SANDBOX_ERROR,
                predicted_proof=code,
                statement_preserved=preserved,
                failure_reason=f"Sandbox unavailable: {type(exc).__name__}: {exc}",
                # The model did not fail here, the infrastructure did; masking keeps an
                # outage from lowering pass@k. A `timeout` stays unmasked: upstream counts
                # it as a failed attempt.
                mask_sample=True,
            )
        proof_status, failure_reason = determine_proof_status(
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "error_type": result.error_type,
            }
        )
        if suffix and proof_status == STATUS_COMPLETED and not axiom_check_passed(result.stdout or "", token):
            proof_status = STATUS_BANNED_TOKENS
            failure_reason = "Axiom check failed: an axiom outside the standard set, or the file stopped early."
        limit = self.config.max_output_characters
        compiler_output = CompilerOutput(
            process_status=proof_status,
            stdout=(result.stdout or "")[:limit],
            stderr=(result.stderr or "")[:limit],
        )

        return LeanCatVerifyResponse(
            **body_dict,
            reward=1.0 if proof_status == STATUS_COMPLETED else 0.0,
            proof_status=proof_status,
            predicted_proof=code,
            statement_preserved=preserved,
            compiler_output=compiler_output,
            failure_reason=failure_reason,
            # Same reasoning as the start-failure path above: an infrastructure error is not
            # the model failing. `timeout` is left unmasked -- upstream counts it as a fail.
            mask_sample=proof_status == STATUS_SANDBOX_ERROR,
        )

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics
    # ──────────────────────────────────────────────────────────

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Pooled pass@k plus the paper's Easy/Medium/High breakdown.

        ``statement_preserved`` is reported as a second score so the guard's rejection rate
        is visible in the aggregate metrics.
        """
        if not tasks:
            return {}

        metrics = compute_pass_majority_metrics(tasks, score_fn=score_leancat_rollout)[0]
        metrics.update(compute_subset_metrics(tasks, subset_key="level", score_fn=score_leancat_rollout))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline: token counts, plus highest-k pass@k and pass@1[avg-of-k] accuracy."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))

        # The tier split is the paper's central claim -- High is a flat zero -- and the pooled
        # number hides it, so the headline carries it too.
        for tier in ("Easy", "Medium", "High"):
            key.update(highest_k_metrics(agent_metrics, tier + "/pass@{k}", score_names=["accuracy"]))

        return key


if __name__ == "__main__":
    LeanCatResourcesServer.run_webserver()
