# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade one converted SWE task inside Harbor's verifier sandbox.

``convert.py`` copies this file into every task as ``tests/nemo_gym/grade.py``, next to the source resources
server's own grading modules (vendored under the same relative paths they have in Gym), and ``tests/test.sh`` runs
it with the image's Python (3.9 or newer). It repeats what the source server's ``/verify`` does once it holds the
agent's patch, calling the server's own functions:

1. drop the files that were untracked before the agent started, as ``capture_model_patch`` does with the
   server's ``drop_patch_sections``;
2. build the server's ``VerificationInputs`` the way its ``_inputs`` does (the test files dropped from the patch);
3. write the server's ``verification_files`` into this fresh sandbox;
4. run the server's ``run_verification``, whose one ``sandbox.exec`` is served here by a local process;
5. write the verdict to ``reward.json``: ``reward`` is 1.0 when the instance resolves and 0.0 otherwise, as the
   server scores it, and the other keys are metrics.

The agent's patch arrives from the agent's sandbox through ``/logs/artifacts/nemo_gym`` (the task's
``[[verifier.collect]]`` hook writes it there). A missing patch file means the capture failed; the server grades
an empty patch then, and so does this script.
"""

# The task image's Python runs this file, so it must parse on 3.9.
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
PATCH_FILE = "model_patch.diff"
PRISTINE_UNTRACKED_FILE = "pristine_untracked.txt"
# Where the vendored copy of SWE-rebench's upstream parsers lives, laid out as the server's clone cache.
SWE_REBENCH_PARSER_CACHE = "swe_rebench_parsers"


def _load(name: str, relative: str) -> ModuleType:
    """Import a vendored module by path. Registered before execution, as ``dataclasses`` requires."""
    spec = importlib.util.spec_from_file_location(name, HERE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _text(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="replace")


class _ExecResult:
    def __init__(self, stdout: str, stderr: str, return_code: int, error_type: str | None = None) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.error_type = error_type


class LocalSandbox:
    """Serves ``run_verification``'s ``sandbox.exec`` in this sandbox.

    The command runs in its own process group so that a timeout stops the whole test run, not only the shell:
    a test process left behind would keep the output pipes open and hang the read.
    """

    async def exec(self, command: str, timeout_s: float | None = None, **_: Any) -> _ExecResult:
        process = subprocess.Popen(
            ["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return _ExecResult(_text(stdout), _text(stderr), 124, "timeout")
        return _ExecResult(_text(stdout), _text(stderr), process.returncode)


def read_agent_patch(artifacts: Path) -> tuple[str, list[str], bool]:
    """The captured patch, the files untracked before the agent started, and whether the capture failed."""
    patch_path = artifacts / PATCH_FILE
    pristine_path = artifacts / PRISTINE_UNTRACKED_FILE
    capture_failed = not patch_path.is_file()
    patch = "" if capture_failed else patch_path.read_text(errors="replace")
    pristine = []
    if pristine_path.is_file():
        pristine = [line.strip() for line in pristine_path.read_text(errors="replace").splitlines() if line.strip()]
    return patch, pristine, capture_failed


def write_verification_files(files: dict[str, str], root: Path) -> None:
    for sandbox_path, content in files.items():
        target = root / sandbox_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def grade_scale_swe(task: dict[str, Any], patch: str, pristine: list[str], root: Path, logs: Path):
    """``resources_servers/scale_swe/app.py``: ``_extract_model_patch``, ``_inputs`` and ``run_verification``."""
    v = _load("nemo_gym_scale_swe_verification", "resources_servers/scale_swe/verification.py")
    row = task["row"]
    if pristine:
        patch = v.drop_patch_sections(patch, pristine)
    inputs = v.VerificationInputs(
        instance_id=row["instance_id"],
        workdir=row["workdir"],
        patch=v.drop_test_patch_files(patch, row.get("f2p_patch") or ""),
        pre_commands=row.get("pre_commands") or "",
        f2p_patch=row.get("f2p_patch") or "",
        f2p_script=row.get("f2p_script") or "",
        fail_to_pass=v.as_id_list(row.get("FAIL_TO_PASS")),
        pass_to_pass=v.as_id_list(row.get("PASS_TO_PASS")),
    )
    write_verification_files(v.verification_files(inputs), root)
    result = asyncio.run(
        v.run_verification(sandbox=LocalSandbox(), inputs=inputs, timeout_s=task["evaluation_timeout"], log_dir=logs)
    )
    return v, result, patch


def grade_swe_rebench(task: dict[str, Any], patch: str, pristine: list[str], root: Path, logs: Path):
    """``resources_servers/swe_rebench/app.py``: ``_extract_model_patch``, ``_inputs``, the parser lookup and
    ``run_verification``."""
    v = _load("nemo_gym_swe_rebench_verification", "resources_servers/swe_rebench/verification.py")
    parsers = _load("nemo_gym_swe_rebench_log_parsers", "resources_servers/swe_rebench/log_parsers.py")
    row = task["row"]
    install_config = row.get("install_config") or {}
    if pristine:
        patch = v.drop_patch_sections(patch, pristine)
    inputs = v.VerificationInputs(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        patch=v.drop_test_patch_files(patch, row.get("test_patch") or ""),
        test_patch=row.get("test_patch") or "",
        install=v.as_command_list(install_config.get("install")),
        test_cmd=v.as_command_list(install_config.get("test_cmd")),
        log_parser=str(install_config.get("log_parser") or ""),
        fail_to_pass=list(row.get("FAIL_TO_PASS") or []),
        pass_to_pass=list(row.get("PASS_TO_PASS") or []),
    )
    try:
        parser = parsers.resolve_parser(inputs.log_parser, HERE / SWE_REBENCH_PARSER_CACHE)
    except Exception as exc:
        # The server answers before spending a sandbox: an unknown parser is a 0 without a verdict.
        result = v.VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output="",
            error=f"log parser {inputs.log_parser!r} unavailable: {exc}",
        )
        return v, result, patch
    write_verification_files(
        v.verification_files(inputs, overwrite_maven_settings=task.get("overwrite_maven_settings", True)), root
    )
    result = asyncio.run(
        v.run_verification(LocalSandbox(), inputs, parser, timeout_s=task["evaluation_timeout"], log_dir=logs)
    )
    return v, result, patch


SOURCES = {"scale_swe": grade_scale_swe, "swe_rebench": grade_swe_rebench}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=Path("/logs/artifacts/nemo_gym"))
    parser.add_argument("--logs", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--root", type=Path, default=Path("/"), help="Where the verification files are written.")
    args = parser.parse_args(argv)

    task = json.loads((HERE / "task.json").read_text())
    args.logs.mkdir(parents=True, exist_ok=True)
    patch, pristine, capture_failed = read_agent_patch(args.artifacts)
    module, result, cleaned_patch = SOURCES[task["source"]](task, patch, pristine, args.root, args.logs)

    # Newer servers also tell an infrastructure failure from a verdict; report it where the module can say.
    infrastructure_failure = getattr(module, "infrastructure_failure", None)
    infrastructure_reason = infrastructure_failure(result) if callable(infrastructure_failure) else None

    verdict = {
        "reward": 1.0 if result.resolved else 0.0,
        "evaluation_completed": float(bool(result.completed)),
        "resolved": float(bool(result.resolved)),
        "patch_applied": float(bool(result.patch_applied)),
        "test_patch_failed": float(bool(getattr(result, "test_patch_failed", False))),
        "patch_capture_failed": float(capture_failed),
        "infrastructure_failure": float(infrastructure_reason is not None),
        "model_patch_bytes": float(len(cleaned_patch.encode("utf-8", errors="replace"))),
    }
    details = {
        "source": task["source"],
        "instance_id": task["row"]["instance_id"],
        "error": result.error,
        "infrastructure_failure": infrastructure_reason,
        "test_results": result.test_results,
    }
    (args.logs / "grading.json").write_text(json.dumps(details, indent=2, default=str))
    (args.logs / "reward.json").write_text(json.dumps(verdict))
    print(json.dumps({"reward": verdict["reward"], "error": result.error}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
