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
server differ from `math_formal_lean`:

* **The file may legitimately contain other `sorry`s.** An FC file typically pairs a proved
  lemma with the open conjecture it sanity-checks, and the task keeps every declaration before
  the target because proofs depend on them. So "no `sorry` in the file" is the wrong check --
  it rejects correct answers.
* **`#print axioms <target>` is the right check.** An incomplete Lean proof depends on
  `sorryAx`, and `#print axioms` reports that per declaration. It also catches indirection a
  textual scan cannot: a proof that leans on a sorry'd lemma earlier in the file. During
  dataset validation this rejected upstream FC "proofs" that were not actually proofs.
* **The statement still has to survive.** The whole-file format lets a model weaken the theorem
  and hand back something that compiles, so `check_statement_preserved` runs first.
"""

import logging
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.lean_proof.proof_utils import (
    check_statement_preserved,
    extract_lean_code,
    find_banned_declarations,
    has_unterminated_block_comment,
)
from resources_servers.lean_proof.sandbox_client import Lean4SandboxClient
from resources_servers.lean_proof.toolchain import ToolchainCheck


LOG = logging.getLogger(__name__)

STATUS_COMPLETED = "completed"
STATUS_EMPTY_GENERATION = "empty_generation"
STATUS_BANNED_TOKENS = "banned_tokens"
STATUS_STATEMENT_MODIFIED = "statement_modified"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_UNPROVED = "unproved"  # compiled, but the target still rests on `sorryAx`
STATUS_TIMEOUT = "timeout"
STATUS_SANDBOX_ERROR = "sandbox_error"

# `import Mathlib` is part of the probe on purpose: a bare version query would happily report a
# version from a sandbox with no Mathlib at all.
_AXIOMS_RE = re.compile(r"depends on axioms:\s*\[([^\]]*)\]")


def score_rollout(rollout: Dict[str, Any]) -> Dict[str, float]:
    """Named scores for aggregate metrics.

    `statement_preserved` rides along as a second score so a run collapsing because the guard
    rejects everything is visible without opening rollouts.
    """
    return {
        "accuracy": rollout["reward"],
        "statement_preserved": float(bool(rollout.get("statement_preserved", False))),
    }


def target_is_proved(compiler_output: Dict[str, Any]) -> Optional[bool]:
    """Whether ``#print axioms <target>`` reported a proof free of ``sorryAx``.

    Returns None when no axiom line was found at all -- that means the declaration does not
    exist under the expected name, which is a failure, not a pass.
    """
    combined = f"{compiler_output.get('stdout', '')}\n{compiler_output.get('stderr', '')}"
    match = _AXIOMS_RE.search(combined)
    if not match:
        return None
    return "sorryAx" not in match.group(1)


class FormalConjecturesResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    sandbox_host: str = "127.0.0.1"
    sandbox_port: int = 6000
    # Every task imports all of Mathlib, so a compile is never instant.
    compilation_timeout: float = 300.0
    max_output_characters: int = 4000

    # The whole-file format lets a model weaken the theorem; on by default because a reward of
    # 1.0 should mean the stated theorem was proved. Turn it off only to measure how often it
    # fires -- `statement_preserved` is reported either way.
    require_statement_preserved: bool = True

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


class CompilerOutput(BaseModel):
    process_status: str
    stdout: str
    stderr: str


class FormalConjecturesVerifyResponse(FormalConjecturesVerifyRequest, BaseVerifyResponse):
    # Request fields ride along so `compute_subset_metrics` can group by `category` on the
    # rollout dict; a response carrying only the reward makes that breakdown impossible.
    proof_status: str
    predicted_proof: str
    statement_preserved: bool
    compiler_output: Optional[CompilerOutput] = None


class FormalConjecturesVerifier:
    """Scoring logic, separated from the server so the fixture can exercise it directly."""

    config: FormalConjecturesResourcesServerConfig

    async def verify(self, body: FormalConjecturesVerifyRequest) -> FormalConjecturesVerifyResponse:
        body_dict = body.model_dump()
        code = extract_lean_code(body.response.output_text)

        def fail(status: str, reason: str, preserved: bool = False, out: Optional[Dict[str, Any]] = None):
            return FormalConjecturesVerifyResponse(
                **body_dict,
                reward=0.0,
                proof_status=status,
                predicted_proof=code,
                statement_preserved=preserved,
                failure_reason=reason,
                compiler_output=CompilerOutput(
                    process_status=out.get("process_status", "unknown"),
                    stdout=out.get("stdout", ""),
                    stderr=out.get("stderr", ""),
                )
                if out
                else None,
            )

        if not code:
            return fail(STATUS_EMPTY_GENERATION, "No Lean code found in the response.")

        banned = find_banned_declarations(code)
        if banned:
            return fail(STATUS_BANNED_TOKENS, f"Submission declares: {', '.join(banned)}.")

        if has_unterminated_block_comment(code):
            # A mangled `-/` swallows the rest of the file, so every later check sees an empty
            # file. Say "malformed", not "statement modified" -- the model produced bad output,
            # it did not try to weaken the theorem.
            return fail(STATUS_COMPILE_ERROR, "Submission has an unterminated block comment.")

        preserved, reason = check_statement_preserved(body.target_statement, code)
        if self.config.require_statement_preserved and not preserved:
            return fail(STATUS_STATEMENT_MODIFIED, reason)

        await self._check_toolchain_once(body.lean_toolchain)

        # Ask Lean itself whether the target is proved. Appended by the server, never trusted
        # to the model: the point is to check the model's work, not to take its word.
        probe = f"{code}\n\n#print axioms {body.full_name}\n"
        raw = await self._sandbox_client.execute_lean4(code=probe, timeout=self.config.compilation_timeout)

        process_status = raw.get("process_status", "unknown")
        if process_status == "timeout":
            return fail(STATUS_TIMEOUT, "Lean compilation timed out.", preserved, raw)
        if process_status not in ("completed", "failed"):
            return fail(STATUS_SANDBOX_ERROR, f"Sandbox reported status {process_status!r}.", preserved, raw)

        combined = f"{raw.get('stdout', '')}\n{raw.get('stderr', '')}"
        if "error:" in combined.lower():
            return fail(STATUS_COMPILE_ERROR, "Lean reported compilation errors.", preserved, raw)

        proved = target_is_proved(raw)
        if proved is None:
            return fail(
                STATUS_COMPILE_ERROR,
                f"`#print axioms {body.full_name}` produced no output; the declaration is missing or renamed.",
                preserved,
                raw,
            )
        if not proved:
            return fail(STATUS_UNPROVED, "The target theorem still depends on `sorryAx`.", preserved, raw)

        return FormalConjecturesVerifyResponse(
            **body_dict,
            reward=1.0,
            proof_status=STATUS_COMPLETED,
            predicted_proof=code,
            statement_preserved=preserved,
            failure_reason=None,
            compiler_output=CompilerOutput(
                process_status=process_status,
                stdout=raw.get("stdout", ""),
                stderr=raw.get("stderr", ""),
            ),
        )


class FormalConjecturesResourcesServer(FormalConjecturesVerifier, SimpleResourcesServer):
    config: FormalConjecturesResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sandbox_client = Lean4SandboxClient(
            host=self.config.sandbox_host,
            port=self.config.sandbox_port,
            max_output_characters=self.config.max_output_characters,
        )
        self._toolchain = ToolchainCheck(self.config.expected_lean_version)

    async def _check_toolchain_once(self, expected: Optional[str]) -> None:
        if self.config.check_lean_version:
            await self._toolchain.run(self._sandbox_client, expected)

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
