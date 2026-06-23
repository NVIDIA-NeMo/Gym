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

"""swe-bench / swe-bench-multilingual harness with nested, in-container grading.

A single parametrized class serves both families. Both run the upstream
SWE-bench ``run_local_evaluation`` harness inside the sandbox (the pre-built
venv is bind-mounted from the host setup dir), then read the harness's
``report.json`` to decide ``resolved``.

Because the nested harness shells out to its own Docker/Apptainer runtime to
spin up the per-instance image, these families are gated to the ``apptainer``
provider via ``supports_provider`` (fail-fast on exec-only providers).
``run_eval`` builds and issues the in-container eval command, and ``grade``
parses the emitted ``report.json``.
"""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import (
    EvalArtifacts,
    SweEvalReport,
    SweTask,
    SweTaskHarness,
    _ensure_trailing_newline,
)
from responses_api_agents.swe_env.harnesses import flat_eval


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# Where the nested harness reads predictions / dataset and writes its report.
_DATASET_PATH = "/root/dataset/data.jsonl"
_PREDICTIONS_PATH = "/root/predictions.jsonl"
_REPORT_PATH = "/root/report.json"

# Per-family in-container setup dir and harness subdir used to invoke the
# upstream ``run_local_evaluation`` module. Keyed by harness/dataset name.
#   * swe-bench: harness mounted at /swebench_setup
#   * swe-bench-multilingual: harness mounted at /swebench_multilingual_setup
_FAMILY_CONFIG: dict[str, dict[str, str]] = {
    "swe-bench": {
        "setup_dir": "/swebench_setup",
        "harness_subdir": "SWE-bench",
    },
    "swe-bench-multilingual": {
        "setup_dir": "/swebench_multilingual_setup",
        "harness_subdir": "SWE-bench_Multilingual",
    },
}


class SweBenchHarness(SweTaskHarness):
    """Nested SWE-bench (and multilingual) harness.

    A single class serves both registry keys; construct one instance per family
    (``SweBenchHarness("swe-bench")`` / ``SweBenchHarness("swe-bench-multilingual")``)
    or let ``grade`` fall back to ``task.benchmark`` for family-specific config.
    """

    grade_strategy = "nested-harness"

    def __init__(self, name: str = "swe-bench", *, flat_eval: bool = False) -> None:
        """Initialize the harness for a given swe-bench family.

        Args:
            name: The swe-bench family to serve, one of the keys in
                ``_FAMILY_CONFIG`` (``"swe-bench"`` or ``"swe-bench-multilingual"``).
            flat_eval: When True, the harness runs the instance's eval script
                directly in the sandbox and parses the log host-side, allowing it
                to run on any exec-capable provider. When False, the nested
                in-container grading path is used.

        Raises:
            ValueError: If ``name`` is not a known swe-bench family.
        """
        if name not in _FAMILY_CONFIG:
            raise ValueError(f"Unknown swe-bench family: {name!r} (expected one of {sorted(_FAMILY_CONFIG)})")
        self.name = name
        # Opt-in flat (host-graded) mode. When True the harness runs the
        # instance's eval script directly in the sandbox and parses the log
        # host-side, lifting the apptainer-only gate so it can run on
        # docker/opensandbox. Default False keeps the nested behavior.
        self.flat_eval = flat_eval
        if flat_eval:
            self.grade_strategy = "flat-host-grade"

    # --- provisioning --------------------------------------------------------

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for a task, including family-specific mounts.

        Args:
            task: The task to provision a sandbox for.

        Returns:
            A ``SandboxSpec`` describing the image, workdir, environment, and
            provider options (including the dataset and harness venv bind mounts).
        """
        # Bind-mount the dataset JSONL and the host-built SWE-bench harness venv
        # at both its canonical path and the in-container alias (uv hardcodes
        # absolute paths). Surfaced via ``provider_options['mounts']``, the
        # channel the apptainer provider consumes.
        provider_options = dict(task.metadata.get("provider_options", {}))
        provider_options.setdefault("mounts", self._family_mounts(task))
        return SandboxSpec(
            image=task.image,
            workdir=task.repo_workdir,
            ttl_s=task.metadata.get("ttl_s", 1800),
            ready_timeout_s=task.metadata.get("ready_timeout_s", 600),
            env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_PAGER": "cat"},
            metadata={
                "instance_id": task.instance_id[:63],
                "benchmark": task.benchmark,
                "harness": self.name,
            },
            resources=SandboxResources.from_mapping(task.metadata.get("resources", {})),
            provider_options=provider_options,
        )

    def supports_provider(self, provider_name: str) -> bool:
        """Report whether this harness can run on the named sandbox provider.

        Args:
            provider_name: The name of the sandbox provider (e.g. ``"apptainer"``,
                ``"docker"``).

        Returns:
            True if the provider is supported. Flat mode runs on any
            exec-capable provider; nested mode requires ``apptainer``.
        """
        # Flat mode is host-graded (no nested container), so it runs on any
        # exec-capable provider.
        if self.flat_eval:
            return True
        # Nested family: the upstream harness manages its own container runtime.
        # Reject exec-only providers (docker/fake) and require apptainer.
        return provider_name == "apptainer"

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Write the predictions JSONL the nested harness consumes.

        Args:
            env: The environment used to write files into the sandbox.
            task: The task whose model patch is embedded as the prediction.
        """
        # The nested harness consumes a predictions JSONL keyed by instance_id
        # rather than a bare patch.diff. Normalize the patch trailing newline
        # before embedding it: a non-empty patch missing its trailing newline
        # gets one appended; an empty/absent patch stays empty. The upstream
        # ``git apply`` is newline-sensitive, so an unnormalized patch can fail
        # to apply and silently flip ``resolved`` to False.
        await env.write_text(_PREDICTIONS_PATH, json.dumps(self._prediction(task)) + "\n")

    @staticmethod
    def _prediction(task: SweTask) -> dict[str, str]:
        """Build the prediction record for a task's model patch.

        Args:
            task: The task whose model patch and metadata populate the record.

        Returns:
            A dict with the instance id, model name, and (newline-normalized)
            model patch.
        """
        patch = task.model_patch or ""
        return {
            "instance_id": task.instance_id,
            "model_name_or_path": task.metadata.get("model_name_or_path", "nemo-gym"),
            # Only normalize a non-empty patch; an empty patch stays "".
            "model_patch": _ensure_trailing_newline(patch) if patch else "",
        }

    # --- server-private grading ----------------------------------------------

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Run the SWE-bench evaluation for a task and collect its artifacts.

        In flat mode, runs the instance's eval script in-sandbox and grades the
        log host-side. Otherwise, runs the nested ``run_local_evaluation``
        harness in-container and reads back its ``report.json``.

        Args:
            env: The environment used to execute commands in the sandbox.
            task: The task to evaluate.

        Returns:
            An ``EvalArtifacts`` carrying the test output, return code, whether a
            patch was applied, and the raw report JSON / error type.
        """
        # Opt-in flat mode: run the instance's eval script in-sandbox and grade
        # the log host-side. Default path below is the nested
        # run_local_evaluation harness (apptainer-only).
        if flat_eval.flat_eval_enabled(self.flat_eval, task):
            return await flat_eval.flat_run_eval(env, task)

        cfg = self._family_config(task)
        # A single host-setup key serves both halves of the harness so the bind
        # source and the cd/UV/venv path can never disagree. Default to the
        # family alias (``cfg["setup_dir"]``), which equals the in-container
        # mount alias and the SWE-bench-Verified default; a verifier-provisioned
        # real host dir overrides via the same key.
        setup_dir = self._setup_dir(task)
        harness_subdir = cfg["harness_subdir"]
        venv_python = f"{setup_dir}/{harness_subdir}/venv/bin/python"
        timeout = int(task.metadata.get("tests_timeout", 1800))
        run_id = task.metadata.get("run_id", task.instance_id)
        split = task.split or "test"

        # Build the in-container eval command: run the upstream harness against
        # the materialized predictions and redirect its report.json to a known
        # path. The UV_* exports and PATH point uv/python at the mounted portable
        # dirs so the pre-built venv resolves its hardcoded toolchain.
        eval_cmd = (
            f"cd {setup_dir}/{harness_subdir} && "
            f'export UV_INSTALL_DIR="{setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{setup_dir}/python" && '
            f'export PATH="{setup_dir}/uv/bin:$PATH" && '
            f"env -u VIRTUAL_ENV {shlex.quote(venv_python)} -m swebench.harness.run_local_evaluation "
            f"--predictions_path {shlex.quote(_PREDICTIONS_PATH)} "
            f"--instance_ids {shlex.quote(task.instance_id)} "
            f"--timeout {timeout} "
            f"--dataset_name {shlex.quote(_DATASET_PATH)} "
            f"--split {shlex.quote(split)} "
            f"--run_id {shlex.quote(str(run_id))}"
        )
        # The upstream harness writes logs/run_evaluation/<run_id>/<model>/<instance>/report.json;
        # locate it and copy to a stable path so grade() can read a single file.
        collect_cmd = (
            f"REPORT=$(find logs/run_evaluation/{shlex.quote(str(run_id))} -name report.json | head -n1); "
            f'if [ -n "$REPORT" ]; then cp "$REPORT" {shlex.quote(_REPORT_PATH)}; fi'
        )
        # Thread the eval timeout (tests_timeout + 120s headroom) so a stuck
        # nested harness is killed and masked via error_kind rather than hanging
        # the verifier.
        result = await env.execute(
            f"{eval_cmd} && {collect_cmd}", cwd=task.repo_workdir, is_eval=True, timeout_s=timeout + 120
        )

        # Read the emitted report.json back out of the sandbox for host-side grading.
        report_text = ""
        if result.get("error_type") not in {"sandbox", "timeout"}:
            cat = await env.execute(f"cat {shlex.quote(_REPORT_PATH)}", cwd=task.repo_workdir)
            if cat["returncode"] == 0:
                report_text = cat["output"]

        return EvalArtifacts(
            test_output=result["output"],
            return_code=result["returncode"],
            patch_applied=bool(task.model_patch),
            raw={"error_type": result.get("error_type"), "report_json": report_text},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade a task from its evaluation artifacts.

        Args:
            task: The task being graded.
            artifacts: The evaluation artifacts produced by ``run_eval``.

        Returns:
            A ``SweEvalReport`` recording resolution, patch state, and any error
            kind. Infrastructure failures are masked via ``error_kind`` rather
            than scored as unresolved.
        """
        # Flat mode: host-side parse of the eval-script log. Detected from either
        # the harness flag/task opt-in OR the artifacts produced by flat_run_eval
        # (so a flat run_eval is always graded flat, even on a shared instance).
        if flat_eval.flat_eval_enabled(self.flat_eval, task) or artifacts.raw.get("flat"):
            return flat_eval.flat_grade(task, artifacts)

        # Infra failure -> mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )

        report_text = artifacts.raw.get("report_json") or ""
        resolved = False
        try:
            report = json.loads(report_text)
            # Upstream harness keys report.json by instance_id.
            entry = report.get(task.instance_id, {}) if isinstance(report, dict) else {}
            resolved = bool(entry.get("resolved", False))
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Missing / malformed report -> eval failure; mask rather than score 0.
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind="eval_error",
            )

        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status={"report": report_text},
        )

    # --- helpers -------------------------------------------------------------

    def _family_config(self, task: SweTask) -> dict[str, str]:
        """Resolve the per-family setup config for a task.

        Args:
            task: The task whose family config is resolved.

        Returns:
            The ``_FAMILY_CONFIG`` entry for this instance's name, falling back
            to ``task.benchmark`` and then to the ``swe-bench`` default.
        """
        # Prefer the instance's own name; fall back to task.benchmark so a single
        # shared instance can still serve either family.
        name = self.name if self.name in _FAMILY_CONFIG else task.benchmark
        return _FAMILY_CONFIG.get(name, _FAMILY_CONFIG["swe-bench"])

    def _setup_dir(self, task: SweTask) -> str:
        """Resolve the host setup dir used for both mounting and evaluation.

        Args:
            task: The task whose metadata may carry a setup dir override.

        Returns:
            The setup dir path, preferring ``setup_dir`` then ``host_setup_dir``
            from task metadata, then the family in-container alias.
        """
        # A single host-setup key is consumed by both ``build_spec`` (mount
        # source) and ``run_eval`` (cd/UV/venv path). Accept either key
        # (``setup_dir`` first, then ``host_setup_dir``), then fall back to the
        # family in-container alias (which is also the SWE-bench-Verified default
        # and the canonical mount target).
        return (
            task.metadata.get("setup_dir")
            or task.metadata.get("host_setup_dir")
            or self._family_config(task)["setup_dir"]
        )

    def _family_mounts(self, task: SweTask) -> list[dict[str, str]]:
        """Build the bind mounts for a task's dataset and harness setup dir.

        Args:
            task: The task whose dataset path and setup dir define the mounts.

        Returns:
            A list of ``{"src", "dst"}`` mount entries for the dataset and the
            host setup dir (bound at both the alias and its canonical path).
        """
        cfg = self._family_config(task)
        setup_dir = self._setup_dir(task)
        mounts: list[dict[str, str]] = [
            # Dataset mounted at the fixed in-container path the harness reads.
            {"src": task.metadata.get("dataset_path", _DATASET_PATH), "dst": _DATASET_PATH},
        ]
        # Bind the host setup dir at both the alias and its canonical path (uv
        # venvs hardcode absolute paths). When ``setup_dir`` already equals the
        # alias (the default / no real host dir provisioned), the two binds
        # collapse to one. ``run_eval`` reads the same key, so the bind source
        # and the cd/UV/venv path can never disagree.
        mounts.append({"src": setup_dir, "dst": cfg["setup_dir"]})
        if setup_dir != cfg["setup_dir"]:
            mounts.append({"src": setup_dir, "dst": setup_dir})
        return mounts
