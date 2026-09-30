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

"""Formal Conjectures resources server: fill one hole in a Lean 4 file, verified by Lean.

Upstream (https://github.com/google-deepmind/formal-conjectures) is a library of formalized
mathematics, most of it **open** -- the headline conjectures carry a `sorry` that nobody can
fill. Those are not a benchmark. What is scored here is the other half of the repo: theorems
that ship with real Lean proofs (`test` sanity checks, `API` supporting lemmas, textbook
exercises, and formalized `research solved` results). `extract.py` strips such a proof off and
`prepare.py` keeps only the tasks whose reference version was **observed to compile**.

The scoring question is *"is this particular declaration proved?"*, and that is what makes this
server distinctive:

* **The file may legitimately contain other `sorry`s.** An FC file typically pairs a proved
  lemma with the open conjecture it sanity-checks, and the task keeps every declaration before
  the target because proofs depend on them. So "no `sorry` in the file" is the wrong check --
  it rejects correct answers. Hence `determine_proof_status(..., sorry_is_error=False)` and
  `find_banned_declarations(..., declarations_only=True)`.
* **`#print axioms <target>` is the right check.** An incomplete Lean proof depends on
  `sorryAx`, and `#print axioms` reports that per declaration. It also catches indirection a
  textual scan cannot: a proof that leans on a sorry'd lemma earlier in the file. During
  dataset validation this rejected upstream FC "proofs" that were not actually proofs.
* **The statement still has to survive.** The whole-file format lets a model weaken the theorem
  and hand back something that compiles, so `check_target_statement_preserved` runs first. It
  compares the target's signature rather than splitting the file on `sorry`, for the same
  reason as the first bullet: the hole is not textually unique here.

Everything that is not FC-specific -- the sandbox, the toolchain probe, the text checks and the
status vocabulary -- comes from `resources_servers/lean_proof`, the library the Lean benchmarks
share. `STATUS_UNPROVED` is the one status FC adds, and it sits next to the shared ones rather
than redefining them.
"""

import logging
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.lean_proof.lean_sandbox import (
    DEFAULT_LEAN_PROJECT_DIR,
    CompilerOutput,
    LeanSandbox,
)
from resources_servers.lean_proof.proof_utils import (
    DECLARED_SHORTCUT_TOKENS,
    check_target_statement_preserved,
    extract_lean_code,
    find_banned_declarations,
    has_unterminated_block_comment,
)
from resources_servers.lean_proof.status import (
    STATUS_BANNED_TOKENS,
    STATUS_COMPILE_ERROR,
    STATUS_COMPLETED,
    STATUS_EMPTY_GENERATION,
    STATUS_SANDBOX_ERROR,
    STATUS_STATEMENT_MODIFIED,
    determine_proof_status,
)


logger = logging.getLogger(__name__)

# The one status the shared vocabulary does not cover: the file compiled, but the declaration
# being scored still rests on `sorryAx`. Distinct from `compile_error` on purpose -- the model
# produced valid Lean, it just did not produce a proof.
STATUS_UNPROVED = "unproved"

# The two forms `#print axioms` emits. The second is easy to miss and costly to miss: a fully
# constructive proof reports "does not depend on any axioms", so matching only the first reads
# the *strongest* possible result as "no axiom line found" and scores it 0.
_AXIOM_LINE_RE = re.compile(
    r"'(?P<name>[^']*)'\s+(?:depends on axioms:\s*\[(?P<axioms>[^\]]*)\]|does not depend on any axioms)"
)


def score_rollout(rollout: Dict[str, Any]) -> Dict[str, float]:
    """Named scores for aggregate metrics.

    `statement_preserved` rides along as a second score so a run collapsing because the guard
    rejects everything is visible without opening rollouts.
    """
    return {
        "accuracy": rollout["reward"],
        "statement_preserved": float(bool(rollout.get("statement_preserved", False))),
    }


def target_is_proved(compiler_output: Dict[str, Any], full_name: Optional[str] = None) -> Optional[bool]:
    """Whether ``#print axioms <target>`` reported a proof free of ``sorryAx``.

    Returns None when the target's axiom line was not found at all -- the declaration does not
    exist under the expected name, which is a failure, not a pass.

    ``full_name`` anchors the answer to the declaration being scored. The submission is a whole
    file and may contain ``#print axioms`` calls of its own; without the anchor a reply that
    prints the axioms of some proved Mathlib lemma next to its own sorry'd theorem would be
    read as a proof. The server's probe is appended last, so the last matching line wins.
    """
    combined = f"{compiler_output.get('stdout', '')}\n{compiler_output.get('stderr', '')}"
    matches = list(_AXIOM_LINE_RE.finditer(combined))
    if not matches:
        return None

    if full_name is not None:
        matches = [m for m in matches if m.group("name") == full_name]
        if not matches:
            return None

    axioms = matches[-1].group("axioms")
    # `axioms is None` is the "does not depend on any axioms" branch: no axioms at all, which
    # trivially includes no `sorryAx`.
    return axioms is None or "sorryAx" not in axioms


class FormalConjecturesResourcesServerConfig(BaseResourcesServerConfig):
    # verify() is a pure function of the request body and this config, so `gym eval reverify`
    # can rescore stored rollouts.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Name of a top-level sandbox config block, or an inline {provider: {...}} mapping.
    sandbox_provider: str = "sandbox"
    # image / resources / ttl, as in the shipped provider configs. Build the image with
    # `resources_servers/lean_proof/lean_image/build.sh v4.33.1`.
    sandbox_config: Dict[str, Any] = {}
    # Lake project the compile runs in; lean_image/ builds Mathlib here.
    lean_project_dir: str = DEFAULT_LEAN_PROJECT_DIR

    # Every task imports all of Mathlib, so a compile is never instant.
    compilation_timeout: float = 300.0
    max_output_characters: int = 4000

    # The whole-file format lets a model weaken the theorem; on by default because a reward of
    # 1.0 should mean the stated theorem was proved. Turn it off only to measure how often it
    # fires -- `statement_preserved` is reported either way.
    require_statement_preserved: bool = True
    ban_proof_shortcuts: bool = True

    # FC pins Lean/Mathlib v4.33.1. An older Mathlib does not fail loudly; it fails individual
    # tasks with ordinary-looking errors, which reads as a model result and is not one.
    check_lean_version: bool = True
    expected_lean_version: str = "4.33.1"


class FormalConjecturesRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_file: str
    full_name: str
    target_statement: str
    declaration: Optional[str] = None
    task_id: Optional[str] = None
    source_path: Optional[str] = None
    category: Optional[str] = None
    ams: Optional[str] = None
    lean_toolchain: Optional[str] = None
    mathlib_version: Optional[str] = None


class FormalConjecturesVerifyRequest(FormalConjecturesRunRequest, BaseVerifyRequest):
    pass


class FormalConjecturesVerifyResponse(FormalConjecturesVerifyRequest, BaseVerifyResponse):
    # Request fields ride along so `compute_subset_metrics` can group by `category` on the
    # rollout dict; a response carrying only the reward makes that breakdown impossible.
    proof_status: str
    predicted_proof: str
    statement_preserved: bool
    compiler_output: Optional[CompilerOutput] = None


class FormalConjecturesVerifier:
    """Scoring logic, separated from the server so a test can drive it without a sandbox.

    Subclasses supply ``config`` and ``_run_lean``; the server below wires those to a real
    ``LeanSandbox``.
    """

    config: FormalConjecturesResourcesServerConfig

    async def _run_lean(self, code: str, timeout_s: Optional[float] = None) -> SandboxExecResult:
        raise NotImplementedError

    async def _check_toolchain_once(self, expected: Optional[str]) -> None:
        raise NotImplementedError

    async def verify(self, body: FormalConjecturesVerifyRequest) -> FormalConjecturesVerifyResponse:
        """Score one attempt: 1.0 only if Lean says the target declaration is proved, else 0.0.

        The text checks run first because they are free and a Mathlib compile is not.
        """
        body_dict = body.model_dump()
        code = extract_lean_code(body.response.output_text)

        def fail(
            status: str,
            reason: Optional[str],
            preserved: bool = False,
            output: Optional[CompilerOutput] = None,
            mask_sample: bool = False,
        ) -> FormalConjecturesVerifyResponse:
            return FormalConjecturesVerifyResponse(
                **body_dict,
                reward=0.0,
                proof_status=status,
                predicted_proof=code,
                statement_preserved=preserved,
                failure_reason=reason,
                compiler_output=output,
                mask_sample=mask_sample,
            )

        if not code:
            return fail(STATUS_EMPTY_GENERATION, "No Lean code found in the response.")

        if self.config.ban_proof_shortcuts:
            # Declarations only, and `sorry`/`admit` are not on the list: an FC file is
            # *supposed* to keep the open conjecture's hole. `#print axioms` below is what
            # decides whether the target itself is honest.
            banned = find_banned_declarations(code, DECLARED_SHORTCUT_TOKENS, declarations_only=True)
            if banned:
                return fail(STATUS_BANNED_TOKENS, f"Submission declares: {', '.join(banned)}.")

        if has_unterminated_block_comment(code):
            # A mangled `-/` swallows the rest of the file, so every later check sees an empty
            # file. Say "malformed", not "statement modified" -- the model produced bad output,
            # it did not try to weaken the theorem.
            return fail(STATUS_COMPILE_ERROR, "Submission has an unterminated block comment.")

        preserved, reason = check_target_statement_preserved(body.target_statement, code)
        if self.config.require_statement_preserved and not preserved:
            return fail(STATUS_STATEMENT_MODIFIED, reason)

        # Ask Lean itself whether the target is proved. Appended by the server, never trusted
        # to the model: the point is to check the model's work, not to take its word.
        probe = f"{code}\n\n#print axioms {body.full_name}\n"
        try:
            # Inside the handler: this is the first thing to touch the sandbox, so a sandbox
            # that will not start surfaces here as a status rather than as a 500, matching the
            # identical failure during the compile.
            await self._check_toolchain_once(body.lean_toolchain)
            result = await self._run_lean(probe)
        except Exception as exc:  # noqa: BLE001 - any start/exec failure is a sandbox outcome
            logger.error("sandbox failed for task %s: %s: %s", body.task_id, type(exc).__name__, exc)
            return fail(
                STATUS_SANDBOX_ERROR,
                f"Sandbox unavailable: {type(exc).__name__}: {exc}",
                preserved,
                # The model did not fail here, the infrastructure did; masking keeps an outage
                # from lowering pass@k. A `timeout` stays unmasked -- it is a failed attempt.
                mask_sample=True,
            )

        raw = {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "return_code": result.return_code,
            "error_type": result.error_type,
        }
        # `sorry_is_error=False`: the file is allowed to keep the open conjecture's hole, so
        # Lean's "declaration uses 'sorry'" warning says nothing about the target. The
        # `#print axioms` check below is what stands in for it.
        proof_status, failure_reason = determine_proof_status(raw, sorry_is_error=False)

        limit = self.config.max_output_characters
        compiler_output = CompilerOutput(
            process_status=proof_status,
            stdout=raw["stdout"][:limit],
            stderr=raw["stderr"][:limit],
        )

        if proof_status != STATUS_COMPLETED:
            return fail(
                proof_status,
                failure_reason,
                preserved,
                compiler_output,
                mask_sample=proof_status == STATUS_SANDBOX_ERROR,
            )

        proved = target_is_proved(raw, body.full_name)
        if proved is None:
            return fail(
                STATUS_COMPILE_ERROR,
                f"`#print axioms {body.full_name}` produced no output; the declaration is missing or renamed.",
                preserved,
                compiler_output,
            )
        if not proved:
            return fail(STATUS_UNPROVED, "The target theorem still depends on `sorryAx`.", preserved, compiler_output)

        return FormalConjecturesVerifyResponse(
            **body_dict,
            reward=1.0,
            proof_status=STATUS_COMPLETED,
            predicted_proof=code,
            statement_preserved=preserved,
            failure_reason=None,
            compiler_output=compiler_output,
        )


class FormalConjecturesResourcesServer(FormalConjecturesVerifier, SimpleResourcesServer):
    config: FormalConjecturesResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._lean = LeanSandbox(
            sandbox_provider=self.config.sandbox_provider,
            sandbox_config=self.config.sandbox_config,
            project_dir=self.config.lean_project_dir,
            server_name=self.config.name,
        )
        self._toolchain_checked = False

    async def _run_lean(self, code: str, timeout_s: Optional[float] = None) -> SandboxExecResult:
        """Compile one submission."""
        return await self._lean.compile(
            code, timeout_s=self.config.compilation_timeout if timeout_s is None else timeout_s
        )

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

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Pooled pass@k plus a breakdown by FC's own difficulty proxy, `category`.

        The pooled number hides the gradient that matters here: `test` sanity checks are
        near-solved while `research solved` is not, and a run that only moves the easy tier
        should not read as progress.
        """
        if not tasks:
            return {}
        metrics = compute_pass_majority_metrics(tasks, score_fn=score_rollout)[0]
        metrics.update(compute_subset_metrics(tasks, subset_key="category", score_fn=score_rollout))
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


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=FormalConjecturesVerifier,
    request_model=FormalConjecturesVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    FormalConjecturesResourcesServer.run_webserver()
