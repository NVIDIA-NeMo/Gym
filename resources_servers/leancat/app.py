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
auxiliary definitions, the target theorem with its proof), unlike ``math_formal_lean``
where the model writes only a proof body and the server reassembles the file. Because the
model owns the whole file it could also weaken the theorem, so
``proof_utils.check_statement_preserved`` compares the submission against the reference.

Verification runs through ``nemo_gym.sandbox``: one sandbox per server process, created
from a snapshot (or image) carrying Lean 4.19.0 and Mathlib v4.19.0, reused across
rollouts. Each attempt is written into it and compiled with ``lake env lean``, which is
what upstream's ``verify_lean`` does. A sandbox per rollout is not viable here -- pod
allocation costs minutes and a run is thousands of rollouts.

``CompilerOutput`` and the Lean comment stripper are imported from ``math_formal_lean``.
Only the whole-file logic lives here.
"""

import asyncio
import logging
import re
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
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.utils import cpu_cap_env
from resources_servers.leancat.proof_utils import (
    check_statement_preserved,
    extract_lean_code,
    find_banned_tokens,
)
from resources_servers.math_formal_lean.app import CompilerOutput
from resources_servers.math_formal_lean.toolchain import normalize_version, parse_lean_version


# Terminal values of `proof_status`. Only COMPLETED scores 1.0. "completed",
# "empty_generation" and "timeout" match math_formal_lean's vocabulary; the rest are
# specific to the whole-file task.
logger = logging.getLogger(__name__)

STATUS_COMPLETED = "completed"
STATUS_EMPTY_GENERATION = "empty_generation"
STATUS_BANNED_TOKENS = "banned_tokens"
STATUS_STATEMENT_MODIFIED = "statement_modified"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_TIMEOUT = "timeout"
STATUS_SANDBOX_ERROR = "sandbox_error"


def determine_proof_status(compiler_output: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Map a sandbox result onto a proof status and, when it failed, a one-line reason.

    A zero exit is not sufficient: a build that declares a ``sorry`` exits zero with only a
    warning, so the output is scanned for ``error:`` and ``sorry`` as well.

    ``error_type`` distinguishes the sandbox failing to run the command from Lean rejecting
    the proof. A non-zero exit with no ``error_type`` is an ordinary compile error, which is
    what most wrong proofs look like and must not be reported as infrastructure trouble.
    """
    error_type = compiler_output.get("error_type")
    if error_type:
        if "timeout" in str(error_type).lower():
            return STATUS_TIMEOUT, "Lean compilation timed out."
        return STATUS_SANDBOX_ERROR, f"Sandbox reported {error_type!r}."

    stdout = compiler_output.get("stdout") or ""
    stderr = compiler_output.get("stderr") or ""
    combined = f"{stdout}\n{stderr}".lower()

    if compiler_output.get("return_code", 0) != 0:
        return STATUS_COMPILE_ERROR, "Lean rejected the proof."
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
    # verify() is a pure function of the request body and this config, so `gym eval reverify`
    # can rescore stored rollouts.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Name of a top-level sandbox config block, or an inline {provider: {...}} mapping.
    sandbox_provider: str = "sandbox"
    # image / snapshot_id / resources / ttl, as in the shipped provider configs. A snapshot
    # is the practical route here: there is no published image at Mathlib v4.19.0, and one
    # built inside a sandbox can be snapshotted once and reused (provider_options.snapshot_id).
    sandbox_config: Dict[str, Any] = {}
    # Directory of the lake project the compile runs in, so `lake env lean` resolves Mathlib.
    lean_project_dir: str = "/lean4/my_project"

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
    config: LeanCatResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # One sandbox per server process, created on the first verify. Creating one per
        # rollout is not viable: pod allocation costs minutes and a run is thousands of
        # rollouts. `_sandbox_lock` keeps concurrent verifies from creating several.
        # There is no server shutdown hook, so cleanup is `sandbox_config.ttl_s`: the
        # sandbox expires on its own if the process dies without releasing it.
        self._sandbox: Optional[AsyncSandbox] = None
        self._sandbox_lock = asyncio.Lock()
        self._toolchain_checked = False

    async def _ensure_sandbox(self) -> AsyncSandbox:
        """Create the shared sandbox on first use, or return the running one."""
        if self._sandbox is not None:
            return self._sandbox

        async with self._sandbox_lock:
            if self._sandbox is not None:
                return self._sandbox

            provider = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            default_metadata = resolve_provider_metadata(self.config.sandbox_provider, get_global_config_dict())
            sandbox_config = dict(self.config.sandbox_config)

            resources = SandboxResources.from_mapping(sandbox_config.get("resources", {}))
            env = dict(sandbox_config.get("env", {}))
            if sandbox_config.get("derive_cpu_env", True):
                # `lake` sizes its worker pool from the host core count, which leaks through on
                # clusters without LXCFS. Explicit env keys win over the derived caps.
                env = cpu_cap_env(resources.cpu) | env

            spec = SandboxSpec(
                image=sandbox_config.get("image"),
                ttl_s=sandbox_config.get("ttl_s"),
                ready_timeout_s=sandbox_config.get("ready_timeout_s"),
                workdir=self.config.lean_project_dir,
                env=env,
                metadata=default_metadata | sandbox_config.get("metadata", {}) | {"nemo_gym_agent": self.config.name},
                resources=resources,
                entrypoint=sandbox_config.get("entrypoint"),
                provider_options=sandbox_config.get("provider_options", {}),
            )
            sandbox = AsyncSandbox(provider)
            await sandbox.start(spec)
            self._sandbox = sandbox
            return sandbox

    async def _check_toolchain_once(self, expected: Optional[str]) -> None:
        """Log an error if the sandbox's Mathlib is not the one the rows were written against.

        A wrong Mathlib fails statements with ordinary compile errors, so the score would be
        meaningless but look plausible: on Mathlib v4.12.0, 36 of the 100 reference statements
        fail to compile with their `sorry` still intact. Runs once per process.
        """
        if not self.config.check_lean_version or self._toolchain_checked:
            return
        self._toolchain_checked = True

        want = normalize_version(expected or self.config.expected_lean_version)
        result = await self._run_lean("import Mathlib\n#eval Lean.versionString\n", timeout_s=600)
        found = parse_lean_version({"stdout": result.stdout or "", "stderr": result.stderr or ""})

        if found is None:
            logger.error(
                "LEAN VERSION UNKNOWN: could not determine the sandbox's Lean version. "
                "If `import Mathlib` does not compile, every task will fail for reasons that "
                "have nothing to do with the model."
            )
        elif want and found != want:
            logger.error(
                "MATHLIB MISMATCH: sandbox is Lean %s but the rows are written against %s. "
                "Statements will fail with ordinary compile errors and the score will be "
                "meaningless but plausible.",
                found,
                want,
            )

    async def _run_lean(self, code: str, timeout_s: Optional[float] = None) -> SandboxExecResult:
        """Write ``code`` to a fresh file in the sandbox and compile it with `lake env lean`.

        This is upstream's `verify_lean` (``scripts/eval_common.py``): a temp file inside the
        lake project, compiled with the project's toolchain. The file is passed through a
        heredoc rather than interpolated into the command, so quotes, backslashes and unicode
        in a proof need no escaping. Each call uses a unique name because one sandbox serves
        many concurrent verifies.
        """
        sandbox = await self._ensure_sandbox()
        timeout = self.config.compilation_timeout if timeout_s is None else timeout_s
        path = f"attempt_{uuid.uuid4().hex}.lean"
        delimiter = f"LEANCAT_EOF_{uuid.uuid4().hex}"

        # `cat` writes the file, then lake compiles it; the file is removed either way so a
        # long-lived sandbox does not accumulate one file per rollout.
        command = (
            f"cat > {path} <<'{delimiter}'\n{code}\n{delimiter}\n"
            f"lake env lean {path}; status=$?; rm -f {path}; exit $status"
        )
        return await sandbox.exec(
            command,
            cwd=self.config.lean_project_dir,
            # The sandbox, not the client, should report the timeout.
            timeout_s=timeout + 30,
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

        await self._check_toolchain_once(body.lean_toolchain)

        result = await self._run_lean(code)
        proof_status, failure_reason = determine_proof_status(
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "error_type": result.error_type,
            }
        )
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

        return key


if __name__ == "__main__":
    LeanCatResourcesServer.run_webserver()
