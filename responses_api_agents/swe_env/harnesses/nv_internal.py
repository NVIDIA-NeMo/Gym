# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""nv-internal-1 harness: flat, host-graded NVIDIA-internal family.

This family does not run any in-container grading harness: it ships a per-instance
``run_script.sh`` + ``parsing_script.py`` that emit a structured ``output.json``
test report. The recipe is a 3-hop sequence:

    1. ``bash run_script.sh <test_files> > stdout.log 2> stderr.log``  (keep streams separate)
    2. ``python parsing_script.py stdout.log stderr.log output.json``  (parse to JSON report)
    3. read ``output.json`` back host-side

Grading is then a pure host-side parse of that report's ``{tests: [{name, status}]}``
shape. Because the family is flat and host-graded, it runs on any exec-capable
provider (e.g. docker). The run script, parsing script, and model patch are
uploaded by ``materialize``.
"""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.grading import compute_resolved
from responses_api_agents.swe_env.harness import (
    EvalArtifacts,
    SweEvalReport,
    SweTask,
    SweTaskHarness,
    _ensure_trailing_newline,
)


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


#: nv-internal default working directory.
NV_DEFAULT_WORKDIR = "/app"
#: The generic ``build_task`` default workdir; means "the row didn't set one".
_GENERIC_DEFAULT_WORKDIR = "/testbed"


def _nv_workdir(task: SweTask) -> str:
    """Resolve the working directory for nv-internal hops.

    The generic ``build_task`` defaults ``repo_workdir`` to ``/testbed``, which is
    not the nv-internal convention. A row that explicitly sets a non-default
    ``repo_workdir`` is honored; otherwise the nv-internal default ``/app`` is used.

    Args:
        task: The task whose ``repo_workdir`` is consulted.

    Returns:
        The working directory path (str) to run every nv-internal hop in.
    """
    workdir = task.repo_workdir
    if not workdir or workdir == _GENERIC_DEFAULT_WORKDIR:
        return NV_DEFAULT_WORKDIR
    return workdir


def parse_passed_tests(report: dict[str, Any]) -> list[str]:
    """Extract PASSED test names from a parsing_script ``output.json`` report.

    The report shape is ``{"tests": [{"name": ..., "status": "PASSED"|...}, ...]}``.

    Args:
        report: The parsed ``output.json`` report mapping.

    Returns:
        The list of test names (list[str]) whose status is ``"PASSED"``.
    """
    return [
        test["name"]
        for test in report.get("tests", [])
        if isinstance(test, dict) and test.get("status") == "PASSED" and "name" in test
    ]


class NVInternalHarness(SweTaskHarness):
    """Flat, host-graded harness for the NVIDIA-internal task family.

    Tasks ship their own ``run_script.sh`` and ``parsing_script.py`` that produce
    a structured ``output.json`` report, which is graded entirely host-side. The
    harness runs on any exec-capable provider.
    """

    name = "nv-internal-1"
    grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for an nv-internal task.

        Environment variables parsed from the task's dockerfiles are injected into
        ``spec.env`` so the provider applies them to every exec hop. This is a
        no-op when the dataset does not carry the dockerfiles.

        Args:
            task: The task to build a sandbox spec for.

        Returns:
            A :class:`SandboxSpec` describing the image, workdir, timeouts,
            environment, metadata, resources, and provider options.
        """
        env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_PAGER": "cat"}
        env.update(_parse_dockerfile_env(task))
        return SandboxSpec(
            image=task.image,
            workdir=_nv_workdir(task),
            ttl_s=task.metadata.get("ttl_s", 1800),
            ready_timeout_s=task.metadata.get("ready_timeout_s", 600),
            env=env,
            metadata={
                "instance_id": task.instance_id[:63],
                "benchmark": task.benchmark,
                "harness": self.name,
            },
            resources=SandboxResources.from_mapping(task.metadata.get("resources", {})),
            provider_options=task.metadata.get("provider_options", {}),
        )

    def supports_provider(self, provider_name: str) -> bool:
        """Report whether this harness supports the named provider.

        The family is flat and host-graded, so every exec-capable provider is
        supported.

        Args:
            provider_name: The provider name being checked.

        Returns:
            ``True`` for every provider.
        """
        return True  # flat, host-graded: works on any exec-capable provider

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Upload run_script.sh, parsing_script.py, and the model patch.

        The scripts live in ``task.metadata``. The dataset stores them under
        dotted keys (``"run_script.sh"`` / ``"parsing_script.py"``), which are read
        first, falling back to the extensionless keys only if the dotted ones are
        absent.

        Args:
            env: The environment used to write files into the sandbox.
            task: The task carrying the patch and scripts to upload.
        """
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))
        run_script = task.metadata.get("run_script.sh") or task.metadata.get("run_script", "")
        parsing_script = task.metadata.get("parsing_script.py") or task.metadata.get("parsing_script", "")
        if run_script:
            await env.write_text("/root/run_script.sh", _ensure_trailing_newline(run_script))
        if parsing_script:
            await env.write_text("/root/parsing_script.py", _ensure_trailing_newline(parsing_script))

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Reset the checkout to ``base_commit``.

        Runs ``git reset --hard`` followed by ``git checkout`` of the base commit
        (not ``git clean``) in the nv-internal working directory.

        Args:
            env: The environment used to execute commands in the sandbox.
            task: The task carrying the ``base_commit`` to reset to.
        """
        if task.base_commit:
            await env.execute(
                f"git reset --hard {task.base_commit} && git checkout {task.base_commit}",
                cwd=_nv_workdir(task),
            )

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Run the 3-hop evaluation recipe and collect its artifacts.

        Applies the model patch, runs the optional per-instance repo setup hook,
        then executes the run/parse/read sequence. Sandbox or timeout failures in
        any hop short-circuit and are surfaced via ``raw["error_type"]``.

        Args:
            env: The environment used to execute commands in the sandbox.
            task: The task being evaluated.

        Returns:
            An :class:`EvalArtifacts` holding the report output, return code,
            whether the patch applied cleanly, and any infra error type.
        """
        workdir = _nv_workdir(task)
        # Apply the model patch with rejection to tolerate conflicts:
        # `--reject` writes .rej files instead of failing; `|| true` keeps going.
        patch_applied = True
        if task.model_patch:
            applied = await env.execute(
                "git apply --ignore-space-change --ignore-whitespace --reject -v /root/patch.diff",
                cwd=workdir,
            )
            patch_applied = applied["returncode"] == 0

        # Optional per-instance repo setup hook.
        repo_cmd = task.metadata.get("before_repo_set_cmd", "").strip()
        if repo_cmd:
            repo_cmd = repo_cmd.split("\n")[-1]
            setup = await env.execute(repo_cmd, cwd=workdir, is_eval=True)
            if setup.get("error_type") in {"sandbox", "timeout"}:
                return EvalArtifacts(
                    test_output=setup["output"],
                    return_code=setup["returncode"],
                    patch_applied=patch_applied,
                    raw={"error_type": setup.get("error_type")},
                )

        # Hop 1: run the per-instance script, keeping stdout/stderr separate.
        # The selected test files are passed positionally.
        test_files = _format_test_files(task.metadata.get("selected_test_files_to_run", []))
        run = await env.execute(
            f"bash /root/run_script.sh {test_files} > /root/stdout.log 2> /root/stderr.log || true",
            cwd=workdir,
            is_eval=True,
        )
        if run.get("error_type") in {"sandbox", "timeout"}:
            return EvalArtifacts(
                test_output=run["output"],
                return_code=run["returncode"],
                patch_applied=patch_applied,
                raw={"error_type": run.get("error_type")},
            )

        # Hop 2: parse the logs into a JSON report.
        parse = await env.execute(
            "python /root/parsing_script.py /root/stdout.log /root/stderr.log /root/output.json",
            cwd=workdir,
            is_eval=True,
        )
        if parse.get("error_type") in {"sandbox", "timeout"}:
            return EvalArtifacts(
                test_output=parse["output"],
                return_code=parse["returncode"],
                patch_applied=patch_applied,
                raw={"error_type": parse.get("error_type")},
            )

        # Hop 3: read the report back host-side.
        report = await env.execute("cat /root/output.json", cwd=workdir, is_eval=True)
        return EvalArtifacts(
            test_output=report["output"],
            return_code=report["returncode"],
            patch_applied=patch_applied,
            raw={"error_type": report.get("error_type")},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade the evaluation artifacts into a report.

        Parses the host-side ``output.json`` report, extracts PASSED tests, and
        derives resolution from the required FAIL_TO_PASS / PASS_TO_PASS sets. An
        infra failure (sandbox or timeout) is masked via ``error_kind`` rather than
        scored as unresolved.

        Args:
            task: The task being graded.
            artifacts: The artifacts produced by ``run_eval``.

        Returns:
            A :class:`SweEvalReport` with resolution status, patch flags, and the
            parsed test report.
        """
        # Infra failure → mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )
        try:
            report = json.loads(artifacts.test_output) if artifacts.test_output.strip() else {}
        except (ValueError, TypeError):
            report = {}
        passed = parse_passed_tests(report)
        f2p, p2p = _resolve_required_tests(task)
        # Resolution is derived from tests alone and never gated on patch-apply rc.
        # An empty report or no required tests → unresolved (compute_resolved
        # returns False).
        resolved = compute_resolved(
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            passed=passed,
        )
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status={"passed": passed, "report": report},
        )


def _format_test_files(test_files: Any) -> str:
    """Build the comma-joined test-files argument.

    Accepts a list, or a string that is either a comma-joined value or a
    ``repr``-style list. A stringified list may use single quotes
    (``['a', 'b']``) which ``json.loads`` rejects, so ``ast.literal_eval`` is used
    (handling single-quoted and native lists) with a safe fallback to the raw
    string.

    Args:
        test_files: A list/tuple of names, or a string holding a comma-joined
            value or a stringified list.

    Returns:
        The comma-joined test-files argument (str); empty for unsupported inputs.
    """
    if isinstance(test_files, (list, tuple)):
        return ",".join(str(item) for item in test_files)
    if isinstance(test_files, str):
        stripped = test_files.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return ",".join(str(item) for item in parsed)
            except (ValueError, SyntaxError):
                pass
        return stripped
    return ""


def _resolve_required_tests(task: SweTask) -> tuple[list[str], list[str]]:
    """Resolve the FAIL_TO_PASS / PASS_TO_PASS required-test sets.

    The ``fail_to_pass_select`` / ``pass_to_pass_select`` keys on ``task.metadata``
    take precedence when present; otherwise the plain ``task.fail_to_pass`` /
    ``task.pass_to_pass`` are used. Values may be lists or stringified lists.

    Args:
        task: The task whose required-test sets are resolved.

    Returns:
        A ``(fail_to_pass, pass_to_pass)`` tuple of test-name lists.
    """
    f2p = task.metadata.get("fail_to_pass_select")
    f2p = _coerce_test_list(f2p) if f2p is not None else list(task.fail_to_pass)
    p2p = task.metadata.get("pass_to_pass_select")
    p2p = _coerce_test_list(p2p) if p2p is not None else list(task.pass_to_pass)
    return f2p, p2p


def _coerce_test_list(value: Any) -> list[str]:
    """Coerce a test-list value (list or stringified list) into a list of names.

    Args:
        value: A list/tuple of names, or a string holding a stringified list.

    Returns:
        The list of test names (list[str]); empty for unsupported inputs.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                pass
    return []


def _parse_dockerfile_env(task: SweTask) -> dict[str, str]:
    """Parse ``ENV`` lines from the task's dockerfiles into a name->value mapping.

    Scans ``base_dockerfile + instance_dockerfile`` for ``ENV`` directives and
    converts them to environment variables. Handles both Docker forms:

        ENV KEY=VALUE   (equals)
        ENV KEY VALUE   (space-separated)

    Returns ``{}`` when the dockerfiles are absent from metadata.

    Args:
        task: The task whose dockerfile metadata is scanned.

    Returns:
        A mapping (dict[str, str]) of environment variable names to values.
    """
    base_dockerfile = str(task.metadata.get("base_dockerfile", "") or "")
    instance_dockerfile = str(task.metadata.get("instance_dockerfile", "") or "")
    env: dict[str, str] = {}
    for raw_line in (base_dockerfile + "\n" + instance_dockerfile).split("\n"):
        line = raw_line.strip()
        if not line.startswith("ENV "):
            continue
        body = line[len("ENV ") :].strip()
        if "=" in body:
            # Format: ENV KEY=VALUE -> normalize spaces around the first `=`.
            key, _, value = body.partition("=")
            key = re.sub(r"\s+", "", key)
            value = value.strip()
        else:
            # Format: ENV KEY VALUE -> split into key + remainder value.
            parts = body.split(None, 1)
            if len(parts) < 2:
                continue
            key, value = parts[0], parts[1]
        if key:
            env[key] = value
    return env
