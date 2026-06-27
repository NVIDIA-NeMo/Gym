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

"""swe-rebench harness: a flat, host-graded family with a vendored log parser.

This is a flat host-graded family: reset to base, apply the model patch and test
patch, run the install/test commands, then parse the test log host-side.

Two things distinguish swe-rebench:

* **JAVA env** — SWE-rebench tasks need
  ``_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false``, surfaced via
  ``build_spec.env`` so it is set for the whole sandbox session.
* **Dynamic log parser** — swe-rebench has no single uniform pytest summary; the
  correct per-test PASSED/FAILED status comes from a repo-specific parser keyed
  by ``log_parser`` and shipped in the cloned ``SWE-rebench-V2`` repo
  (``lib/agent/log_parsers.py`` or ``agent/log_parsers.py``). It is imported
  dynamically, guarded by try/except.

The cloned ``SWE-rebench-V2`` directory must be provisioned out-of-band. When it
is absent or the named parser cannot be resolved, ``grade`` masks the sample via
``error_kind`` rather than scoring a misleading ``unresolved``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, SweTaskHarness


if TYPE_CHECKING:
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment


# JAVA flag required for every SWE-rebench task.
_JAVA_OPTIONS = "-Djava.net.preferIPv6Addresses=false"

# Patch-apply flags shared by the model and test patch; non-fatal
# ``git apply --reject`` style so a failed apply still runs the tests.
_APPLY_FLAGS = "--reject --recount --ignore-space-change --whitespace=nowarn"

# Timing/duration suffixes some test runners append to node names; stripped so
# the parser output lines up with the (already-normalized) expected node ids.
_REBENCH_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    """Strip trailing timing annotations from a test node name.

    Args:
        name (str): The raw test node name, possibly carrying a trailing timing
            or duration annotation.

    Returns:
        str: The node name with any timing suffix removed and surrounding
            whitespace stripped.
    """
    for pattern in _REBENCH_TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _load_rebench_log_parsers(rebench_repo_dir: Path):
    """Dynamically import the cloned SWE-rebench-V2 ``log_parsers`` module.

    Prefers ``lib/agent/log_parsers.py`` and falls back to
    ``agent/log_parsers.py``, temporarily prepending the repo (and its ``lib``
    directory) to ``sys.path`` so the module's intra-repo imports resolve.

    Args:
        rebench_repo_dir (Path): Path to the cloned SWE-rebench-V2 repository.

    Returns:
        ModuleType: The imported ``log_parsers`` module.

    Raises:
        FileNotFoundError: If the cloned directory has not been provisioned and
            no ``log_parsers.py`` can be located.
    """
    lp_path = rebench_repo_dir / "lib" / "agent" / "log_parsers.py"
    if not lp_path.exists():
        lp_path = rebench_repo_dir / "agent" / "log_parsers.py"
    if not lp_path.exists():
        raise FileNotFoundError(
            f"SWE-rebench-V2 log_parsers not found under {rebench_repo_dir}; "
            "provision the clone via setup_scripts/swe_rebench.sh"
        )

    extra_paths = [str(rebench_repo_dir), str(rebench_repo_dir / "lib")]
    added: list[str] = []
    for p in extra_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        spec = importlib.util.spec_from_file_location("_rebench_log_parsers", str(lp_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _resolve_parser(log_parsers, log_parser_name: str) -> Callable[[str], dict[str, str]] | None:
    """Resolve a parser callable from the loaded module.

    Looks up the name in the module's ``NAME_TO_PARSER`` mapping first, then
    falls back to a module-level attribute of the same name.

    Args:
        log_parsers: The imported ``log_parsers`` module.
        log_parser_name (str): The name of the parser to resolve.

    Returns:
        Callable[[str], dict[str, str]] | None: The resolved parser callable, or
            ``None`` if no parser matches the name.
    """
    name_to_parser = getattr(log_parsers, "NAME_TO_PARSER", {}) or {}
    return name_to_parser.get(log_parser_name) or getattr(log_parsers, log_parser_name, None)


def _as_list(value: Any) -> list[str]:
    """Coerce a test-command/install/list field to a list of strings.

    Accepts the value as a JSON-encoded string, a bare string, or a list. A
    JSON-encoded string is parsed and coerced recursively; a bare string that
    fails to parse is wrapped in a single-element list.

    Args:
        value (Any): The field value to coerce. May be ``None``, a string, a
            list, a tuple, or any other type.

    Returns:
        list[str]: The value normalized to a list of strings. An empty list is
            returned for ``None`` or an empty string.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return [value]
            return _as_list(parsed)
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


class SweRebenchHarness(SweTaskHarness):
    """Flat, host-graded harness for the swe-rebench benchmark family.

    Applies the model and test patches, runs the install/test commands, then
    parses the test log host-side using a repo-specific parser loaded
    dynamically from the cloned SWE-rebench-V2 repository.
    """

    name = "swe-rebench"
    grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for a swe-rebench task.

        Sets the git and ``_JAVA_OPTIONS`` environment variables, merges any
        task-provided env, and forwards TTL, readiness timeout, resources, and
        provider options from the task metadata.

        Args:
            task (SweTask): The task to build a sandbox specification for.

        Returns:
            SandboxSpec: The sandbox specification for running the task.
        """
        # _JAVA_OPTIONS forces IPv4 for SWE-rebench tasks.
        env = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_PAGER": "cat",
            "_JAVA_OPTIONS": _JAVA_OPTIONS,
        }
        env.update(task.metadata.get("env", {}))
        return SandboxSpec(
            image=task.image,
            workdir=task.repo_workdir,
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
        """Report whether the harness supports a given sandbox provider.

        Being flat and host-graded, it works on any exec-capable provider.

        Args:
            provider_name (str): The name of the sandbox provider.

        Returns:
            bool: Always ``True``.
        """
        return True  # flat, host-graded: works on any exec-capable provider

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Apply patches, run install and test commands, and collect artifacts.

        Applies the model patch then the test patch (both best-effort), runs the
        non-fatal install commands, then runs the test block with the eval
        timeout. Records whether the model patch applied for informational
        purposes only; grading does not gate on it.

        Args:
            env (AsyncSweEnvironment): The environment used to execute commands
                inside the sandbox.
            task (SweTask): The task being evaluated.

        Returns:
            EvalArtifacts: The captured test output, return code, model-patch
                application status, and raw error metadata.
        """
        workdir = task.repo_workdir
        install_config = task.metadata.get("install_config", {}) or {}
        install_cmds = _as_list(install_config.get("install"))
        test_cmds = _as_list(install_config.get("test_cmd")) or ([task.test_command] if task.test_command else [])

        # Apply the model patch first, then the test patch. Both are best-effort:
        # a failed apply still runs the tests; model-patch application is recorded
        # for info only (grading does not gate on it).
        patch_applied = True
        if task.model_patch:
            applied = await env.execute(
                f"git apply {_APPLY_FLAGS} /root/patch.diff",
                cwd=workdir,
            )
            patch_applied = applied["returncode"] == 0
        if task.test_patch:
            await env.execute(f"git apply {_APPLY_FLAGS} /root/test_patch.diff", cwd=workdir)

        # Install commands are non-fatal; failures there should not abort the
        # test run.
        for cmd in install_cmds:
            await env.execute(cmd, cwd=workdir)

        test_block = "\n".join(test_cmds) if test_cmds else "python -m pytest -rA -q"
        # Thread the eval timeout into the test exec, defaulting to 1800s so a
        # stuck swe-rebench run is bounded. A row that explicitly carries a
        # ``tests_timeout`` overrides the default.
        result = await env.execute(
            test_block,
            cwd=workdir,
            is_eval=True,
            timeout_s=task.metadata.get("tests_timeout", 1800),
        )
        return EvalArtifacts(
            test_output=result["output"],
            return_code=result["returncode"],
            patch_applied=patch_applied,
            raw={"error_type": result.get("error_type")},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade a swe-rebench task from its evaluation artifacts.

        Masks infra failures (sandbox/timeout) and grading errors (missing clone,
        unknown parser, parser crash) via ``error_kind`` rather than scoring them.
        Otherwise parses the test output with the resolved repo-specific parser
        and marks the task resolved when every FAIL_TO_PASS and PASS_TO_PASS test
        is in the passed set.

        Args:
            task (SweTask): The task being graded.
            artifacts (EvalArtifacts): The artifacts captured during evaluation.

        Returns:
            SweEvalReport: The grading report, with ``resolved`` set on success
                or ``error_kind`` set when the sample is masked.
        """
        # Infra failure -> mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )

        install_config = task.metadata.get("install_config", {}) or {}
        log_parser_name = install_config.get("log_parser", "")
        # The cloned SWE-rebench-V2 dir is provisioned out-of-band; its absence,
        # an unknown parser name, or a parser crash all mask the sample via
        # ``error_kind`` rather than mis-scoring it.
        rebench_repo_dir = task.metadata.get("rebench_repo_dir")
        if not rebench_repo_dir:
            return self._masked(task, artifacts, "eval_error")
        try:
            log_parsers = _load_rebench_log_parsers(Path(rebench_repo_dir))
            parser = _resolve_parser(log_parsers, log_parser_name)
            if parser is None:
                return self._masked(task, artifacts, "eval_error")
            results = parser(artifacts.test_output)
        except Exception:
            return self._masked(task, artifacts, "eval_error")

        results = {_normalize_test_name(k): v for k, v in (results or {}).items()}
        passed_set = {k for k, v in results.items() if v == "PASSED"}
        fail_to_pass_set = {_normalize_test_name(n) for n in task.fail_to_pass}
        pass_to_pass_set = {_normalize_test_name(n) for n in task.pass_to_pass}

        # Resolution rule: every FAIL_TO_PASS and PASS_TO_PASS test must be in the passed set,
        # AND the required set must be non-empty. The empty-required guard (matching
        # compute_resolved's ``if not required: return False``) keeps swe-rebench consistent with
        # the other families: a degenerate row with no required tests does NOT resolve, instead of
        # inflating to reward 1.0 (an empty set is a subset of anything). Not gated on patch apply.
        required = fail_to_pass_set | pass_to_pass_set
        resolved = bool(required) and fail_to_pass_set <= passed_set and pass_to_pass_set <= passed_set
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status={"passed": sorted(passed_set), "all": results},
        )

    @staticmethod
    def _masked(task: SweTask, artifacts: EvalArtifacts, kind: str) -> SweEvalReport:
        """Build a masked report that records a grading error instead of a score.

        Args:
            task (SweTask): The task being graded.
            artifacts (EvalArtifacts): The artifacts captured during evaluation.
            kind (str): The error kind to record on the report.

        Returns:
            SweEvalReport: A report with ``error_kind`` set and no resolution.
        """
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            patch_applied=artifacts.patch_applied,
            error_kind=kind,
        )
