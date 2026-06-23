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

"""r2e-gym harness: nested, in-container-graded family.

Unlike the flat ``swe-bench-ext`` family, r2e-gym does NOT grade host-side: the
per-instance ``report.json`` is produced by the vendored r2e-gym evaluation
harness (``run_local_evaluation.py``) running inside the container. ``grade()``
therefore only parses that report's already-computed ``resolved`` verdict rather
than reconstructing it from per-test status.

Two r2e-gym-specific wrinkles:

* **Test hiding during the agent phase.** ``/r2e_tests`` holds the held-out
  evaluation tests, and ``run_tests.sh`` launches them, so both are removed from
  the agent's checkout (root, ``/root``, ``/testbed``). During grading (the
  verifier) these are present, because the nested harness re-materializes them —
  ``hide_eval_tests_commands`` is exposed for the agent adapter to run after
  ``materialize`` and is intentionally NOT invoked by ``run_eval``.
* **r2egym_setup mount.** The prebuilt R2E-Gym venv has hardcoded absolute paths
  in its uv wrappers, so the setup dir is bind-mounted at both ``/r2egym_setup``
  and its original absolute path. These mounts are surfaced via
  ``provider_options["mounts"]`` for the apptainer provider.

This family requires apptainer and a real ``.sif`` container; it cannot run on
exec-only / docker providers. ``supports_provider`` fails fast on any
non-apptainer provider.
"""

from __future__ import annotations

import json
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


# Location the nested r2e-gym harness writes its per-instance report to inside
# the container. ``run_eval`` redirects ``run_local_evaluation.py`` here and
# then reads it back host-side for parsing.
_REPORT_PATH = "/root/r2egym_report.json"

# In-container predictions JSONL the nested harness reads via
# ``--predictions_path``. Holds the SWE-bench predictions shape
# ({instance_id, model_patch, ...}).
_PREDICTIONS_PATH = "/root/predictions.jsonl"


class R2EGymHarness(SweTaskHarness):
    """Harness for the r2e-gym family of SWE tasks.

    Grades by parsing the report produced by the nested r2e-gym evaluation
    harness running inside the container, or, in opt-in flat mode, by running
    the instance's eval script in-sandbox and parsing the log host-side.
    """

    name = "r2e-gym"
    grade_strategy = "nested-harness"

    def __init__(self, *, flat_eval: bool = False) -> None:
        """Initialize the harness.

        Args:
            flat_eval: When True, opt into flat (host-graded) mode: run the
                instance's eval script directly in the sandbox and parse the log
                host-side, lifting the apptainer-only gate so it can run on any
                exec-capable provider. When False (default), keep the nested
                in-container grading behavior.
        """
        self.flat_eval = flat_eval
        if flat_eval:
            self.grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for an r2e-gym task.

        Bind-mounts the r2e-gym setup dir at both ``/r2egym_setup`` and its
        original absolute path so the prebuilt uv venv (which has hardcoded
        absolute toolchain paths) resolves correctly, surfacing the mounts via
        ``provider_options["mounts"]`` for the apptainer provider.

        Args:
            task: The SWE task whose metadata, image, and workdir describe the
                sandbox to construct.

        Returns:
            SandboxSpec: The fully populated sandbox spec, including image,
            workdir, TTL, environment, metadata, resources, and provider
            options with the r2e-gym setup mounts.
        """
        setup_dir = task.metadata.get("r2egym_setup_dir", "/r2egym_setup")
        # The prebuilt uv venv has hardcoded absolute paths, so the setup dir is
        # bind-mounted at both ``/r2egym_setup`` and its original absolute path.
        # These are surfaced via ``provider_options['mounts']`` so the prebuilt
        # ``{setup}/R2E-Gym/venv`` is bound in and resolves its hardcoded
        # toolchain paths.
        mounts = [
            {"src": setup_dir, "dst": "/r2egym_setup"},
            {"src": setup_dir, "dst": setup_dir},
        ]
        provider_options = dict(task.metadata.get("provider_options", {}))
        provider_options.setdefault("mounts", mounts)
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
            provider_name: Name of the sandbox provider (e.g. ``"apptainer"``,
                ``"docker"``, ``"local"``).

        Returns:
            bool: True for any provider in flat mode (host-graded, no nested
            container); otherwise True only for ``"apptainer"``, since the
            nested vendored harness requires apptainer with a real ``.sif``.
        """
        # Flat mode is host-graded (no nested container), so it runs on any
        # exec-capable provider.
        if self.flat_eval:
            return True
        # Nested family: the vendored harness only runs under apptainer with a
        # real .sif. Fail fast on exec-only providers (docker/local).
        return provider_name == "apptainer"

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Write the model patch into the sandbox for the nested grader.

        The nested harness reads the model patch from a predictions JSONL keyed
        by instance_id (``run_local_evaluation.py --predictions_path``), in the
        SWE-bench predictions shape. The patch's trailing newline is normalized
        before embedding: a non-empty patch missing its trailing newline gets one
        appended; an empty/absent patch stays empty. ``git apply`` is
        newline-sensitive, so an unnormalized patch can fail to apply and silently
        flip ``resolved`` to False.

        Args:
            env: The active SWE environment used to write files into the sandbox.
            task: The SWE task supplying the model patch, instance id, and
                metadata.
        """
        patch = task.model_patch or ""
        prediction = {
            "instance_id": task.instance_id,
            "model_name_or_path": task.metadata.get("model_name_or_path", "nemo-gym"),
            "model_patch": _ensure_trailing_newline(patch) if patch else "",
        }
        await env.write_text(_PREDICTIONS_PATH, json.dumps(prediction) + "\n")

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Reset the repository checkout (no-op for r2e-gym).

        No host-orchestrated reset is performed for this family: the nested
        ``run_local_evaluation`` resets the checkout inside its own container.
        Running a host-side ``git reset --hard <base_commit>`` here would mutate
        state the nested grader owns.

        Args:
            env: The active SWE environment (unused).
            task: The SWE task (unused).
        """
        return None

    def hide_eval_tests_commands(self) -> list[str]:
        """Build shell commands that strip the held-out eval tests from the agent's checkout.

        ``/r2e_tests`` holds the evaluation tests the agent must not see;
        ``run_tests.sh`` launches them. ``run_tests.sh`` is deleted only when it
        references ``r2e_tests`` (substring guard) to avoid clobbering an
        unrelated file with that name. The agent adapter runs these after
        ``materialize``; the verifier does NOT (the nested harness needs the
        tests back for grading).

        Returns:
            list[str]: One shell command per checkout root (``""``, ``/root``,
            ``/testbed``) that removes the eval tests and the launcher script.
        """
        commands: list[str] = []
        for root_dir in ["", "/root", "/testbed"]:
            commands.append(
                f"rm -rf {root_dir}/r2e_tests && "
                f"if grep -qs r2e_tests {root_dir}/run_tests.sh; then rm -rf {root_dir}/run_tests.sh; fi"
            )
        return commands

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Run evaluation for an r2e-gym task and collect its artifacts.

        In opt-in flat mode, runs the instance's eval script in-sandbox and
        defers to the flat eval path. Otherwise runs the nested
        ``run_local_evaluation`` harness in-container: it reads the model patch
        from the predictions file, applies it, runs the held-out tests, and
        writes ``report.json``, which is copied to a stable path and read back
        host-side for grading.

        Args:
            env: The active SWE environment used to execute commands in the
                sandbox.
            task: The SWE task supplying metadata such as setup dir, predictions
                path, dataset path, timeout, and output dir.

        Returns:
            EvalArtifacts: The captured report text (or command output), return
            code, whether the patch was treated as applied, and raw fields
            including the error type and report JSON.
        """
        # Opt-in flat mode: run the instance's eval script in-sandbox and grade
        # the log host-side. Default path below is the nested
        # run_local_evaluation harness (apptainer-only).
        if flat_eval.flat_eval_enabled(self.flat_eval, task):
            return await flat_eval.flat_run_eval(env, task)

        # The nested r2e-gym harness reads the model patch from the predictions
        # file, applies it, runs the held-out tests, and writes ``report.json``.
        # We build the in-container command and redirect its report to
        # ``_REPORT_PATH``, then read it back host-side for grading.
        setup_dir = task.metadata.get("r2egym_setup_dir", "/r2egym_setup")
        predictions_path = task.metadata.get("predictions_path", _PREDICTIONS_PATH)
        dataset_path = task.metadata.get("dataset_path", "/root/dataset/data.jsonl")
        timeout = task.metadata.get("tests_timeout", 1800)
        output_dir = task.metadata.get("eval_output_dir", "/root/eval-outputs")
        eval_cmd = (
            "cd /r2egym_setup/R2E-Gym && "
            f'export UV_INSTALL_DIR="{setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{setup_dir}/python" && '
            f'export PATH="{setup_dir}/uv/bin:$PATH" && '
            f"env -u VIRTUAL_ENV {setup_dir}/R2E-Gym/venv/bin/python "
            "src/r2egym/agenthub/run/run_local_evaluation.py "
            f"--predictions_path {predictions_path} "
            f"--instance_id {task.instance_id} "
            f"--timeout {timeout} "
            f"--dataset {dataset_path} "
            f"--output_dir {output_dir} && "
            # Surface the per-instance report at a stable, well-known path.
            f"cp {output_dir}/report.json {_REPORT_PATH}"
        )
        result = await env.execute(eval_cmd, cwd=task.repo_workdir, is_eval=True, timeout_s=timeout + 120)
        report_text = ""
        if result["returncode"] == 0:
            report = await env.execute(f"cat {_REPORT_PATH}", cwd=task.repo_workdir, is_eval=True)
            if report["returncode"] == 0:
                report_text = report["output"]
        return EvalArtifacts(
            test_output=report_text or result["output"],
            return_code=result["returncode"],
            # The nested harness applies the patch itself; absent a host apply
            # step we treat a clean eval as "applied" and let grade() mask
            # infra failures via error_kind.
            patch_applied=result["returncode"] == 0,
            raw={"error_type": result.get("error_type"), "report_json": report_text},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade an r2e-gym task from its evaluation artifacts.

        In flat mode (from the harness flag/task opt-in or flat artifacts),
        parses the eval-script log host-side. Otherwise masks infra failures via
        ``error_kind`` and parses the nested harness's ``report.json``, trusting
        its already-computed ``resolved`` verdict.

        Args:
            task: The SWE task being graded, supplying the instance id and model
                patch.
            artifacts: The evaluation artifacts produced by ``run_eval``,
                including the report JSON and raw error/flat markers.

        Returns:
            SweEvalReport: The resolved/unresolved verdict with patch existence,
            patch-applied status, per-test status, and any error kind.
        """
        # Flat mode: host-side parse of the eval-script log. Detected from either
        # the harness flag/task opt-in OR the artifacts produced by flat_run_eval
        # (so a flat run_eval is always graded flat, even on a shared instance).
        if flat_eval.flat_eval_enabled(self.flat_eval, task) or artifacts.raw.get("flat"):
            return flat_eval.flat_grade(task, artifacts)

        # Infra failure → mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )
        report_text = artifacts.raw.get("report_json") or artifacts.test_output
        try:
            report = json.loads(report_text)
        except (json.JSONDecodeError, TypeError):
            # The nested harness never produced a parseable report → eval error.
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind="eval_error",
            )
        # report.json is keyed by instance_id (standard SWE-bench shape); fall
        # back to the sole entry if the key was rewritten.
        entry = report.get(task.instance_id)
        if entry is None and len(report) == 1:
            entry = next(iter(report.values()))
        if not isinstance(entry, dict):
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind="eval_error",
            )
        # The nested harness has already computed ``resolved``; trust it.
        resolved = bool(entry.get("resolved", False))
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status=entry.get("tests_status", {}),
        )
