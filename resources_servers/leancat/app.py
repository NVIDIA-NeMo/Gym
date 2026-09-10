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
Mathlib v4.19.0, built to stress library-grounded abstraction rather than search depth.
The published headline is how little of it current models solve: 12.0% pass@4 for the
best model, and 0.0% on the High tier.

The task here is *whole-file*, which is what separates this server from
``math_formal_lean``. There, the model writes a proof body and the harness reassembles
the file around it. Here the model returns the entire file -- imports, ``open`` and
``variable`` preamble, any auxiliary definitions it wants, then the target theorem --
because many LeanCat problems set up their own structures and instances before the
statement, and a reassembly step would have to guess where the model's additions belong.
Handing the model the whole file removes the guess.

That freedom is why ``proof_utils.check_statement_preserved`` exists: a model that owns
the whole file can also weaken the theorem it was asked to prove, and the weakened
version compiles. See ``proof_utils`` for what is enforced.
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from resources_servers.leancat.proof_utils import (
    check_statement_preserved,
    extract_lean_code,
    find_banned_tokens,
)
from resources_servers.leancat.sandbox_client import Lean4SandboxClient


# Terminal values of `proof_status`. Everything except COMPLETED scores 0.0; they are kept
# distinct because "the model cheated" and "the sandbox was down" need very different
# responses from whoever reads the run.
#
# "completed", "empty_generation" and "timeout" are spelled as math_formal_lean spells them,
# so a reader moving between the two servers reads the same word for the same outcome. The
# rest have no counterpart there: that server reassembles the file itself, so it has nothing
# to catch a tampered statement and folds every non-timeout compiler failure into the raw
# process status.
STATUS_COMPLETED = "completed"
STATUS_EMPTY_GENERATION = "empty_generation"
STATUS_BANNED_TOKENS = "banned_tokens"
STATUS_STATEMENT_MODIFIED = "statement_modified"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_TIMEOUT = "timeout"
STATUS_SANDBOX_ERROR = "sandbox_error"


def determine_proof_status(compiler_output: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Map a sandbox result onto a proof status and, when it failed, a one-line reason.

    Takes the raw sandbox dict rather than the parsed model, as math_formal_lean's function
    of the same name does, so the two read the same and neither depends on the other's types.

    ``process_status == "completed"`` means the sandbox got a zero exit code, but that is
    checked *and* the output is scanned for ``error:``/``sorry``: a warning-only build that
    declared a sorry still exits zero, and that must not score as a proof.
    """
    process_status = compiler_output.get("process_status", "unknown")

    if process_status == "timeout":
        return STATUS_TIMEOUT, "Lean compilation timed out."
    if process_status == "failed":
        # NeMo-Skills' sandbox reports "failed" for any non-zero `lake env lean` exit
        # (local_sandbox_server.py: completed iff returncode == 0). That is an ordinary
        # compile error, not an infrastructure problem -- calling it sandbox_error made 59%
        # of a real run look like the sandbox was broken.
        return STATUS_COMPILE_ERROR, "Lean rejected the proof."
    if process_status != "completed":
        return STATUS_SANDBOX_ERROR, f"Sandbox reported status {process_status!r}."

    # The Gym sandbox backend reports `lake env lean`'s exit status, which is 0 only when
    # Lean accepted the file. Trust it over string-matching the output: a non-zero exit
    # with no "error:" in the captured text would otherwise score as a proof. The HTTP
    # backend omits the key, so this is a no-op there.
    return_code = compiler_output.get("return_code")
    if return_code is not None and return_code != 0:
        return STATUS_COMPILE_ERROR, f"Lean exited with status {return_code}."

    stdout = compiler_output.get("stdout", "")
    stderr = compiler_output.get("stderr", "")
    combined = f"{stdout}\n{stderr}".lower()
    if "error:" in combined:
        return STATUS_COMPILE_ERROR, "Lean reported compilation errors."
    if re.search(r"\bsorry\b", combined) is not None:
        return STATUS_COMPILE_ERROR, "Lean reported a declaration that uses 'sorry'."

    return STATUS_COMPLETED, None


def score_leancat_rollout(rollout: Dict[str, Any]) -> Dict[str, float]:
    """Named scores for aggregate metrics (see ``LeanCatResourcesServer.compute_metrics``)."""
    return {
        "accuracy": rollout["reward"],
        "statement_preserved": float(bool(rollout.get("statement_preserved", False))),
    }


class LeanCatResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_host: str = "127.0.0.1"
    sandbox_port: int = 6000
    # Upstream's per-attempt verification budget (EVALUATION.md). LeanCat proofs import
    # all of Mathlib and lean on heavy typeclass search, so the 30s that miniF2F gets is
    # not enough here.
    compilation_timeout: float = 300.0
    max_output_characters: int = 4000
    # Both guards default on: they are what make a reward of 1.0 mean what the paper
    # means by "solved". Turn them off only to measure how often they fire.
    require_statement_preserved: bool = True
    ban_proof_shortcuts: bool = True


class LeanCatRunRequest(BaseRunRequest):
    # Fields arrive as flat row columns (see prepare.py); the validator below also
    # accepts them nested under `verifier_metadata` for hand-written rows.
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

        Gym posts each row to /verify with `verifier_metadata` still nested -- it is not
        spliced onto the body, which is why math_formal_lean can declare its fields flat
        (its rows carry no wrapper at all) while ours cannot. Lifting here keeps the typed
        fields and lets `level` reach the response, which compute_subset_metrics needs.

        Top-level keys win, so an explicit override is never clobbered by the metadata.
        """
        if isinstance(data, dict) and isinstance(data.get("verifier_metadata"), dict):
            return {**data["verifier_metadata"], **data}
        return data


class LeanCatVerifyRequest(LeanCatRunRequest, BaseVerifyRequest):
    pass


class CompilerOutput(BaseModel):
    process_status: str
    stdout: str
    stderr: str


class LeanCatVerifyResponse(LeanCatVerifyRequest, BaseVerifyResponse):
    # Inherits the request fields so `level` survives onto the rollout dict that
    # `compute_subset_metrics` groups by; a response that only carried the reward would
    # make the per-difficulty breakdown impossible to compute.
    proof_status: str
    predicted_proof: str
    statement_preserved: bool
    compiler_output: Optional[CompilerOutput] = None


class LeanCatResourcesServer(SimpleResourcesServer):
    config: LeanCatResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sandbox_client = Lean4SandboxClient(
            host=self.config.sandbox_host,
            port=self.config.sandbox_port,
            max_output_characters=self.config.max_output_characters,
        )

    async def verify(self, body: LeanCatVerifyRequest) -> LeanCatVerifyResponse:
        """Score one attempt: 1.0 only if it is a valid LeanCat proof, else 0.0.

        The static checks run before the sandbox call, not after, because they are free
        and a five-minute Mathlib compile is not -- there is no point compiling a file
        that already lost on ``sorry``.
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
            banned = find_banned_tokens(code)
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

        raw_output = await self._sandbox_client.execute_lean4(
            code=code,
            timeout=self.config.compilation_timeout,
        )
        proof_status, failure_reason = determine_proof_status(raw_output)
        compiler_output = CompilerOutput(
            process_status=raw_output.get("process_status", "unknown"),
            stdout=raw_output.get("stdout", ""),
            stderr=raw_output.get("stderr", ""),
        )

        return LeanCatVerifyResponse(
            **body_dict,
            reward=1.0 if proof_status == STATUS_COMPLETED else 0.0,
            proof_status=proof_status,
            predicted_proof=code,
            statement_preserved=preserved,
            compiler_output=compiler_output,
            failure_reason=failure_reason,
        )

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics
    # ──────────────────────────────────────────────────────────

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Pooled pass@k plus a per-difficulty breakdown.

        The paper reports Easy/Medium/High separately and pooled, and the split is the
        whole point of its argument -- the pooled number hides that High is a flat zero.
        ``statement_preserved`` rides along as a second score so a run that collapses
        because the guard is rejecting everything is visible without opening rollouts.
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

        return key


if __name__ == "__main__":
    LeanCatResourcesServer.run_webserver()
