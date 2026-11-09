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

"""SciCode resources server.

Runs the agent's accumulated per-sub-step Python solutions against each sub-step's test cases
(targets loaded from test_data.h5) and returns a binary reward: 1.0 iff every sub-step passes.
Per-sub-step counts are also returned so sub-step accuracy can be computed downstream.

Each sub-step runs in a subprocess using the configured grading interpreter. Optional
sandbox execution moves that subprocess and its scientific dependencies off the server host.

The agent's solutions, ``step_usage`` and ``token_usage_version`` arrive as top-level fields from the
agent's ``/run``, or under the response's ``scicode`` field from an agent session. This server also
owns the SciCode benchmark metrics: sub-step-weighted accuracy, its across-run variability, and the
whole-problem token accounting built from ``step_usage``.
"""

import asyncio
import hashlib
import os
import shutil
import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from scicode_integration.runner import (
    build_test_program,
    preflight_grading_interpreter,
    process_exit,
    run_substep,
    sanitize_test,
)
from scicode_integration.sandbox_runner import ScicodeSandboxConfig, run_sandbox_substep

from nemo_gym import PARENT_DIR
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.metrics import Stat
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


# Agent sentinel for a sub-step it could not generate (ran out of context); always fails.
_OUT_OF_CONTEXT = "_ran_out_of_context_"
TOKEN_USAGE_VERSION = 1
# The response field in which scicode_agent's session activation reports its solutions and step usage.
SCICODE_RESPONSE_KEY = "scicode"


class ScicodeGradingInterpreter(BaseModel):
    """One Python interpreter used to grade generated SciCode programs."""

    name: str
    python_executable: str


class ScicodeResourcesServerConfig(BaseResourcesServerConfig):
    # Grading is a pure function of the request, so the same saved answer can be graded again.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    num_processes: int = Field(default=20, ge=1)
    # Optional remote execution; interpreter and test-data paths then refer to the image.
    sandbox: Optional[ScicodeSandboxConfig] = None
    # Per-sub-step execution timeout
    timeout_secs: float = 30.0
    # Local path to SciCode's test_data.h5 (staged manually)
    test_data_fpath: Optional[str] = None
    # Optional checksum for binding problem/test releases. Existing SciCode configs leave this unset.
    test_data_md5: Optional[str] = None
    # Empty preserves the existing behavior of grading with this server's interpreter.
    grading_interpreters: List[ScicodeGradingInterpreter] = Field(default_factory=list)
    # SciCode-Verified sets this to two so a single-interpreter run cannot be mistaken for canonical.
    required_grading_interpreters: int = Field(default=1, ge=1)
    # The canonical OR rule short-circuits after the first pass. Enable this only for diagnostics.
    run_all_grading_interpreters: bool = False
    # Opt-in: keep each scored sub-step's execution record (exit code, timeout, duration, exception class
    # and bounded stdout/stderr tails) in the verify response as `step_outputs`, aligned with `step_results`.
    # Off by default: the record carries model-generated output text and grows every rollout row.
    record_step_outputs: bool = False
    # Last N characters of stdout and of stderr kept per sub-step when record_step_outputs is on (0 keeps none).
    output_tail_chars: int = Field(default=2000, ge=0)
    # Opt-in on top of record_step_outputs: each record also carries `code`, the accumulated solution the
    # sub-step's tests ran against (the agent's solutions[step_id]); None for a sentinel step that never ran.
    # Off by default: the code is model-generated and grows a row by the size of every scored program.
    record_step_code: bool = False


class ScicodeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    problem_id: str
    sub_steps: List[dict]
    # {"<problem_id>.<step>": accumulated_code} produced by the agent.
    solutions: Optional[Dict[str, str]] = None


class ScicodeVerifyRequest(ScicodeRunRequest, BaseVerifyRequest):
    pass


class ScicodeVerifyResponse(BaseVerifyResponse):
    # Retain the agent's compact accounting records through verification. None
    # distinguishes historical final-step-only rollouts from whole-problem usage.
    token_usage_version: Optional[int] = None
    step_usage: Optional[List[dict]] = None
    # Declared so it survives into the rollout output (identifies the problem); the request's
    # sub_steps/solutions are intentionally not carried through to keep rollout rows small.
    problem_id: str = ""
    step_results: List[bool] = []
    num_steps_passed: int = 0
    num_steps_total: int = 0
    problem_accuracy: bool = False
    # Per-rollout sub-step pass fraction. As a numeric verify-response field it
    # gets the full generic statistics (mean/std/min/max/median) from the
    # RewardProfiler, which the aggregate-only pooled scalar cannot provide.
    subtask_accuracy: float = 0.0
    # Audit fields are aligned with step_results and contain only scored (non-prefilled) steps.
    scored_step_ids: List[str] = Field(default_factory=list)
    step_environment_results: List[Dict[str, bool]] = Field(default_factory=list)
    step_environment_errors: List[Dict[str, str]] = Field(default_factory=list)
    # Per scored sub-step, aligned with step_results, only when record_step_outputs is on (None otherwise):
    # {step_id, step_number, environment, status (executed | out_of_context | empty), passed, returncode,
    #  timed_out, duration_s, exception_class, stdout_tail, stderr_tail} plus `code` when record_step_code
    # is on too. The record describes the grading run that decided the step: the environment that passed
    # under the OR rule, else the last one tried; `environment` is None for a sentinel step that never ran.
    step_outputs: Optional[List[dict]] = None
    # Per scored sub-step, aligned with step_results, always recorded: how the grading run that decided the step
    # ended, {returncode, timed_out, exit}, `exit` typed by process_exit (`killed:<SIGNAL>` for a negative return
    # code; `not_run` for an out-of-context or empty step). Unlike step_outputs it carries no model output.
    step_exits: List[dict] = Field(default_factory=list)


def _lift_session_results(body: ScicodeVerifyRequest) -> ScicodeVerifyRequest:
    """Read the solutions and step usage that an agent session's response carries, as /run sends them.

    The field leaves the response either way, so the rollout row keeps one copy of ``step_usage`` and
    ``token_usage_version``. Like a ``/run`` row, it keeps no solutions. A field without a solutions dict breaks
    the contract between the agent and this server, not the model's answer, so it fails verification instead of
    grading nothing.
    """
    data = body.model_dump()
    results = data["response"].pop(SCICODE_RESPONSE_KEY, None)
    if results is None:
        return body
    if not isinstance(results, dict) or not isinstance(results.get("solutions"), dict):
        raise ValueError(f"The response's {SCICODE_RESPONSE_KEY!r} field carries no solutions dict")
    if data.get("solutions") is None:
        data |= {key: results.get(key) for key in ("solutions", "step_usage", "token_usage_version")}
    return ScicodeVerifyRequest.model_validate(data)


def _across_run_stats(tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Across-run (repeat-to-repeat) variability of the two headline metrics.

    Run i is "take repeat i of every problem" (repeats aligned by their rollout
    index, run count = minimum repeat count so the matrix stays rectangular).
    Per run, problem accuracy is the mean of problem_accuracy over problems and
    subtask accuracy is the sub-step-weighted pool - the same definitions as
    the headline mean/problem_accuracy and subtask_accuracy, so the std-dev
    over the per-run values (sample std, ddof=1) is the run-to-run variability
    of exactly those metrics. Single-repeat collections emit nothing.
    """
    max_k = min((len(t) for t in tasks if t), default=0)
    if max_k < 2:
        return {}

    rows = [sorted(t, key=lambda r: r.get(ROLLOUT_INDEX_KEY_NAME, 0))[:max_k] for t in tasks if len(t) >= max_k]

    problem_runs: List[float] = []
    subtask_runs: List[float] = []
    for i in range(max_k):
        acc = [float(r[i]["problem_accuracy"]) for r in rows if r[i].get("problem_accuracy") is not None]
        if acc:
            problem_runs.append(sum(acc) / len(acc))
        passed = sum(r[i].get("num_steps_passed", 0) for r in rows)
        total = sum(r[i].get("num_steps_total", 0) for r in rows)
        if total:
            subtask_runs.append(passed / total)

    metrics: Dict[str, Any] = {}
    for key, runs in (("mean/problem_accuracy", problem_runs), ("subtask_accuracy", subtask_runs)):
        if len(runs) < 2:
            continue
        std_dev = 0.0 if all(v == runs[0] for v in runs) else statistics.stdev(runs)
        metrics[f"{key}/std_dev_across_runs"] = std_dev
    return metrics


def _token_metrics(tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Pool generated steps across attempts, rather than averaging per-problem ratios."""
    rows = [r for task in tasks for r in task]
    versions = {r.get("token_usage_version") for r in rows}
    if not rows or versions == {None}:
        # Historical rollouts contain only the last step's usage.
        return {}
    if versions != {TOKEN_USAGE_VERSION}:
        raise ValueError("Cannot aggregate SciCode rollouts with different token accounting versions")

    steps = [step for r in rows for step in r["step_usage"] if step["status"] != "prefilled"]
    generated = [step for step in steps if step["status"] == "generated"]
    metrics = {
        # Aggregate values are exported as numeric scores by NeMo Evaluator.
        "token_usage_version": TOKEN_USAGE_VERSION,
        "num_subproblems": len(steps),
        "num_generated_steps": len(generated),
        "num_steps_with_usage": sum(step["usage"] is not None for step in generated),
        "generation_coverage": len(generated) / len(steps) if steps else None,
    }
    complete = metrics["num_steps_with_usage"] == len(generated)
    metrics["token_usage_complete"] = complete
    # Override the generic means too: pandas would otherwise silently average only
    # the rollouts with known usage, presenting a partial measurement as a full one.
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        total = sum(step["usage"][name] for step in steps) if complete else None
        metrics[f"mean/{name}"] = total / len(rows) if total is not None else None
        metrics[f"mean/{name}_per_problem"] = metrics[f"mean/{name}"]
        metrics[f"mean/{name}_per_subproblem"] = total / len(steps) if complete and steps else None
    return metrics


class ScicodeResourcesServer(SimpleResourcesServer):
    config: ScicodeResourcesServerConfig
    _resolved_test_data_path: Optional[str] = PrivateAttr(default=None)
    _resolved_grading_interpreters: Optional[List[tuple[str, str]]] = PrivateAttr(default=None)

    def model_post_init(self, context):
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)
        if self.config.test_data_md5 and not self.config.sandbox:
            self._resolve_test_data()
        if self.config.sandbox or self.config.grading_interpreters or self.config.required_grading_interpreters > 1:
            self._resolve_grading_interpreters()

    def _resolve_test_data(self) -> str:
        if self.config.sandbox:
            if not self.config.test_data_fpath or not Path(self.config.test_data_fpath).is_absolute():
                raise ValueError("Sandbox test_data_fpath must be an absolute path inside the image")
            return self.config.test_data_fpath
        if self._resolved_test_data_path is not None:
            return self._resolved_test_data_path
        if not self.config.test_data_fpath:
            raise RuntimeError(
                "test_data_fpath is not configured. Stage SciCode's test_data.h5 and set "
                "test_data_fpath (see benchmarks/scicode/README.md)."
            )
        path = Path(self.config.test_data_fpath).expanduser()
        # Resolve relative paths against the Gym root, since the server's cwd is its own dir.
        if not path.is_absolute():
            path = PARENT_DIR / path
        if not path.is_file():
            raise RuntimeError(
                f"SciCode test_data.h5 not found at {path}. Download and stage it "
                "(see benchmarks/scicode/README.md) before running."
            )
        if self.config.test_data_md5:
            digest = hashlib.md5()  # noqa: S324 - release integrity, not cryptographic security
            with path.open("rb") as test_data:
                for chunk in iter(lambda: test_data.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_md5 = digest.hexdigest()
            if actual_md5 != self.config.test_data_md5.lower():
                raise RuntimeError(
                    f"SciCode test data checksum mismatch at {path}: expected "
                    f"{self.config.test_data_md5.lower()}, got {actual_md5}."
                )
        self._resolved_test_data_path = str(path)
        return self._resolved_test_data_path

    def _resolve_grading_interpreters(self) -> List[tuple[str, str]]:
        if self._resolved_grading_interpreters is not None:
            return self._resolved_grading_interpreters

        configured = self.config.grading_interpreters
        if self.config.sandbox:
            self._resolve_test_data()
            if len(configured) < self.config.required_grading_interpreters:
                raise ValueError("Sandbox grading requires explicit grading_interpreters inside the image")
            if len({item.name for item in configured}) != len(configured):
                raise ValueError("Duplicate SciCode grading interpreter name")
            if any(not Path(item.python_executable).is_absolute() for item in configured):
                raise ValueError("Sandbox grading interpreter paths must be absolute")
            return [(item.name, item.python_executable) for item in configured]
        if configured:
            resolved = []
            seen_names = set()
            for interpreter in configured:
                if interpreter.name in seen_names:
                    raise RuntimeError(f"Duplicate SciCode grading interpreter name: {interpreter.name!r}.")
                seen_names.add(interpreter.name)

                configured_path = Path(interpreter.python_executable).expanduser()
                if configured_path.is_absolute() or configured_path.parent != Path("."):
                    if not configured_path.is_absolute():
                        configured_path = PARENT_DIR / configured_path
                    # Keep the configured path rather than resolving symlinks. A virtual
                    # environment's ``bin/python`` is normally a symlink; dereferencing it
                    # changes argv[0] to the base interpreter and prevents Python from finding
                    # the venv's pyvenv.cfg and site-packages.
                    executable = os.path.abspath(configured_path)
                else:
                    executable = shutil.which(interpreter.python_executable) or ""
                if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
                    raise RuntimeError(
                        f"SciCode grading interpreter {interpreter.name!r} is not executable: "
                        f"{interpreter.python_executable!r}."
                    )
                resolved.append((interpreter.name, executable))
        else:
            resolved = [("current", sys.executable)]

        if len(resolved) < self.config.required_grading_interpreters:
            raise RuntimeError(
                "SciCode requires at least "
                f"{self.config.required_grading_interpreters} grading interpreters, but "
                f"{len(resolved)} resolved. Configure grading_interpreters explicitly."
            )
        for name, executable in resolved:
            # Cold imports on a network filesystem need the same budget as grading, not a tighter cutoff.
            if error := preflight_grading_interpreter(executable, timeout_secs=self.config.timeout_secs):
                raise RuntimeError(f"SciCode grading environment preflight failed for {name!r}:\n{error}")
        self._resolved_grading_interpreters = resolved
        return self._resolved_grading_interpreters

    async def verify(self, body: ScicodeVerifyRequest) -> ScicodeVerifyResponse:
        body = _lift_session_results(body)
        solutions = body.solutions or {}
        record = self.config.record_step_outputs
        record_code = record and self.config.record_step_code
        # Score only sub-steps the agent produced a solution for. Sub-steps absent from solutions
        # (prefilled steps) are excluded from the denominator entirely; out-of-context sentinels
        # are present and counted as failures.
        scored = [i for i in range(len(body.sub_steps)) if f"{body.problem_id}.{i + 1}" in solutions]
        if not scored:
            return ScicodeVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                step_results=[],
                step_outputs=[] if record else None,
                num_steps_passed=0,
                num_steps_total=0,
                problem_accuracy=False,
            )

        h5_path = self._resolve_test_data()
        grading_interpreters = self._resolve_grading_interpreters()
        loop = asyncio.get_running_loop()

        def _step_output(
            i: int,
            status: str,
            passed: bool,
            result: Optional[dict] = None,
            code: Optional[str] = None,
            environment: Optional[str] = None,
        ) -> dict:
            result = result or {}
            output = {
                "step_id": f"{body.problem_id}.{i + 1}",
                "step_number": body.sub_steps[i]["step_number"],
                "environment": environment,
                "status": status,
                "passed": passed,
                "returncode": result.get("returncode"),
                "timed_out": bool(result.get("timed_out", False)),
                "exit": process_exit(result),
                "duration_s": result.get("duration_s"),
                "exception_class": result.get("exception_class"),
                "stdout_tail": result.get("stdout_tail", ""),
                "stderr_tail": result.get("stderr_tail", ""),
            }
            if record_code:
                output["code"] = code
            return output

        def _step_exit(result: dict) -> dict:
            return {
                "returncode": result.get("returncode"),
                "timed_out": bool(result.get("timed_out", False)),
                "exit": process_exit(result),
            }

        async def _run_substep(i: int) -> tuple[bool, Dict[str, bool], Dict[str, str], Optional[dict], dict]:
            sub_step = body.sub_steps[i]
            code = solutions[f"{body.problem_id}.{i + 1}"]
            if not code or code == _OUT_OF_CONTEXT:
                status = "out_of_context" if code == _OUT_OF_CONTEXT else "empty"
                return False, {}, {}, _step_output(i, status, False) if record else None, _step_exit({})
            sanitized = [sanitize_test(tc) for tc in sub_step["test_cases"]]
            program = build_test_program(code, h5_path, sub_step["step_number"], sanitized)
            per_environment = {}
            per_environment_errors = {}
            attempts: List[tuple[str, dict]] = []
            for name, executable in grading_interpreters:
                if record:
                    call = partial(
                        run_substep,
                        program,
                        self.config.timeout_secs,
                        executable,
                        output_tail_chars=self.config.output_tail_chars,
                    )
                else:
                    call = partial(run_substep, program, self.config.timeout_secs, executable)
                async with self._semaphore:
                    if self.config.sandbox:
                        result = await run_sandbox_substep(
                            program,
                            config=self.config.sandbox,
                            python_executable=executable,
                            timeout_secs=self.config.timeout_secs,
                            test_data_path=h5_path,
                            test_data_md5=self.config.test_data_md5,
                            output_tail_chars=self.config.output_tail_chars if record else 2000,
                        )
                    else:
                        result = await loop.run_in_executor(None, call)
                per_environment[name] = bool(result["passed"])
                attempts.append((name, result))
                if result["error"]:
                    per_environment_errors[name] = result["error"]
                if result.get("infrastructure_error"):
                    raise RuntimeError(
                        "SciCode grading infrastructure failure for "
                        f"problem {body.problem_id}, step {sub_step['step_number']}, "
                        f"environment {name}:\n{result['error']}"
                    )
                if per_environment[name] and not self.config.run_all_grading_interpreters:
                    break
            passed = any(per_environment.values())
            # The run that decided the step: the environment that passed under the OR rule, else the last tried.
            name, result = next(((n, r) for n, r in attempts if r["passed"]), attempts[-1])
            output = _step_output(i, "executed", passed, result, code=code, environment=name) if record else None
            return passed, per_environment, per_environment_errors, output, _step_exit(result)

        tasks = [asyncio.create_task(_run_substep(i)) for i in scored]
        try:
            evaluated_steps = list(await asyncio.gather(*tasks))
        except BaseException:
            # A failed request must not leave sibling sandboxes running after its caller retries.
            # Canceling a local executor await cannot stop its subprocess or release its permit safely.
            if self.config.sandbox:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            raise
        step_results = [passed for passed, _, _, _, _ in evaluated_steps]
        step_environment_results = [per_environment for _, per_environment, _, _, _ in evaluated_steps]
        step_environment_errors = [per_environment_errors for _, _, per_environment_errors, _, _ in evaluated_steps]
        step_outputs = [output for _, _, _, output, _ in evaluated_steps] if record else None
        scored_step_ids = [body.sub_steps[i]["step_number"] for i in scored]
        num_passed = sum(step_results)
        all_passed = num_passed == len(scored)

        return ScicodeVerifyResponse(
            **body.model_dump(),
            reward=1.0 if all_passed else 0.0,
            step_results=step_results,
            num_steps_passed=num_passed,
            num_steps_total=len(scored),
            problem_accuracy=all_passed,
            subtask_accuracy=num_passed / len(scored),
            scored_step_ids=scored_step_ids,
            step_environment_results=step_environment_results,
            step_environment_errors=step_environment_errors,
            step_outputs=step_outputs,
            step_exits=[step_exit for _, _, _, _, step_exit in evaluated_steps],
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        if any(
            step["status"] == "generated" and step["usage"] is None
            for row in body.verify_responses
            for step in (row.get("step_usage") or [])
        ):
            # Suppress generic token statistics at every level (including per-task
            # and per-repeat). Keep the caller's raw rollout records unchanged.
            body = body.model_copy(
                update={
                    "verify_responses": [
                        {**row, "response": {**(row.get("response") or {}), "usage": None}}
                        for row in body.verify_responses
                    ]
                }
            )
        metrics = await super().aggregate_metrics(body)
        # Custom repeat estimates are added after the shared headline selection.
        metrics.key_metrics.update(self.get_key_metrics(metrics.agent_metrics))
        # NeMo Evaluator requires numeric scores. Omit unavailable values rather
        # than exporting nulls or inventing zero consumption.
        metrics.agent_metrics = {k: v for k, v in metrics.agent_metrics.items() if v is not None}
        metrics.key_metrics = {k: v for k, v in metrics.key_metrics.items() if v is not None}
        return metrics

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Headline SciCode metric: sub-step-weighted accuracy = total passed / total over all rollouts."""
        passed = sum(r.get("num_steps_passed", 0) for task in tasks for r in task)
        total = sum(r.get("num_steps_total", 0) for task in tasks for r in task)
        metrics = {"subtask_accuracy": passed / total if total else 0.0}
        metrics.update(_across_run_stats(tasks))
        metrics.update(_token_metrics(tasks))
        return metrics

    def compute_repeat_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in agent_metrics.items()
            if k.startswith("mean/")
            or k.startswith("subtask_accuracy")
            or k
            in {
                f"{stat.across_repeats_prefix}{metric}"
                for stat in (Stat.MEAN, Stat.CI_LOW_95, Stat.CI_HIGH_95)
                for metric in ("subtask_accuracy", "mean/problem_accuracy")
            }
            or k in {"generation_coverage", "token_usage_complete"}
        }


if __name__ == "__main__":
    ScicodeResourcesServer.run_webserver()
