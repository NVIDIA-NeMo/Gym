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

"""SWE-bench Pro patch verification using a NeMo Gym sandbox.

Upstream source:
https://github.com/scaleapi/SWE-bench_Pro-os/blob/ca10a60a5fcae51e6948ffe1485d4153d421e6c5/swe_bench_pro_eval.py

Copied functions retain their upstream names and structure:

* ``strip_binary_hunks`` is copied verbatim.
* ``create_entryscript`` is copied with each changed statement marked
  ``NeMo Gym change``.
* ``assemble_workspace_files`` is copied with embedded-asset and path changes
  marked ``NeMo Gym change``.
* ``grade_output`` extracts the grading statements from upstream ``main``;
  its safe parsing change is marked inline.

``run_verification`` and the dataclasses are NeMo Gym additions. They replace
Modal/local-Docker orchestration with ``AsyncSandbox`` and persist per-request
logs. Dataset preparation supplies the upstream scripts and Dockerfiles in each
JSONL row, avoiding a runtime checkout of the standalone upstream repository.
"""

import ast
import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox


WORKSPACE_DIR = "/workspace"
REPOSITORY_DIR = "/app"
PATCH_PATH = f"{WORKSPACE_DIR}/patch.diff"
RUN_SCRIPT_PATH = f"{WORKSPACE_DIR}/run_script.sh"
PARSER_PATH = f"{WORKSPACE_DIR}/parser.py"
ENTRY_SCRIPT_PATH = f"{WORKSPACE_DIR}/entryscript.sh"
STDOUT_PATH = f"{WORKSPACE_DIR}/stdout.log"
STDERR_PATH = f"{WORKSPACE_DIR}/stderr.log"
OUTPUT_PATH = f"{WORKSPACE_DIR}/output.json"
RESET_STATUS_PATH = f"{WORKSPACE_DIR}/reset_status"
CHECKOUT_STATUS_PATH = f"{WORKSPACE_DIR}/checkout_status"
PATCH_STATUS_PATH = f"{WORKSPACE_DIR}/patch_apply_status"
RUNTIME_SETUP_STATUS_PATH = f"{WORKSPACE_DIR}/runtime_setup_status"
TEST_SETUP_STATUS_PATH = f"{WORKSPACE_DIR}/test_setup_status"
PREFETCH_STATUS_PATH = f"{WORKSPACE_DIR}/prefetch_status"
TEST_STATUS_PATH = f"{WORKSPACE_DIR}/test_status"
PARSER_STATUS_PATH = f"{WORKSPACE_DIR}/parser_status"

# NeMo Gym change: runtime parity shims are limited to the pinned dataset rows
# where the corresponding OpenSandbox divergence was reproduced.
FLIPT_OTEL_INSTANCE_ID = "instance_flipt-io__flipt-690672523398c2b6f6e4562f0bf9868664ab894f"
NODEBB_IPV4_INSTANCE_ID = "instance_nodebb__nodebb-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
NODEBB_REDIS_INSTANCE_ID = "instance_nodebb__nodebb-eb49a64974ca844bca061744fb3383f5d13b02ad-vnan"
QUTEBROWSER_XVFB_INSTANCE_ID = (
    "instance_qutebrowser__qutebrowser-f631cd4422744160d9dcf7a0455da532ce973315"
    "-v35616345bb8052ea303186706cec663146f0f184"
)


@dataclass(frozen=True)
class VerificationInputs:
    instance_id: str
    base_commit: str
    patch: str
    run_script: str
    parser_script: str
    selected_test_files_to_run: str | list[str]
    fail_to_pass: str | list[str]
    pass_to_pass: str | list[str]
    before_repo_set_cmd: str = ""
    base_dockerfile: str = ""
    instance_dockerfile: str = ""
    repo_language: str = ""
    prefetch_go_modules: bool = False
    runtime_parity_adaptations: bool = False


@dataclass(frozen=True)
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str = ""
    error: str | None = None
    reset_exit_code: int | None = None
    checkout_exit_code: int | None = None
    patch_exit_code: int | None = None
    runtime_setup_exit_code: int | None = None
    test_setup_exit_code: int | None = None
    prefetch_exit_code: int | None = None
    test_exit_code: int | None = None
    parser_exit_code: int | None = None


def parse_string_list(value: str | list[str]) -> list[str]:
    """Parse JSON/Python list strings without executing dataset content."""
    if isinstance(value, list):
        parsed = value
    elif not value.strip():
        parsed = []
    else:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)

    if not isinstance(parsed, (list, tuple)) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Expected a list of strings, got {value!r}")
    return list(parsed)


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)

    return "".join(kept)


def create_entryscript(sample: dict[str, Any]) -> str:
    """Create the upstream in-container evaluation script."""
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1] or ":"
    # NeMo Gym change: parse untrusted dataset content safely instead of using eval().
    selected_test_files_to_run = ",".join(parse_string_list(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    # NeMo Gym change: optionally warm Go's module cache before the upstream test command.
    should_prefetch_go_modules = sample.get("prefetch_go_modules", False) and (
        str(sample.get("repo_language", "")).lower() == "go" or "go test " in sample["run_script"]
    )
    go_module_prefetch_cmd = ":"
    if should_prefetch_go_modules:
        go_module_prefetch_cmd = """# NeMo Gym change: prefetch modules without modifying the upstream test script.
if [ -f go.mod ]; then
  go mod download
fi"""
    runtime_setup_cmd = ":"
    if sample.get("runtime_parity_adaptations"):
        runtime_setup_cmd = _runtime_environment_command(sample["instance_id"])
    # NeMo Gym change: Dockerfiles are embedded in the prepared row instead of read from an upstream checkout.
    base_dockerfile = sample["base_dockerfile"]
    instance_dockerfile = sample["instance_dockerfile"]

    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                # Convert ENV commands to export statements
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)

    env_cmds = "\n".join(env_cmds)

    entry_script = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
RESET_STATUS=$?
git checkout {base_commit}
CHECKOUT_STATUS=$?
git apply -v /workspace/patch.diff
# NeMo Gym change: retain patch application status for the structured verification response.
PATCH_APPLY_STATUS=$?
{runtime_setup_cmd}
RUNTIME_SETUP_STATUS=$?
{before_repo_set_cmd}
TEST_SETUP_STATUS=$?
{go_module_prefetch_cmd}
PREFETCH_STATUS=$?
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
TEST_STATUS=$?
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
PARSER_STATUS=$?
# NeMo Gym change: persist the status after running the upstream script sequence.
printf '%s\\n' "$RESET_STATUS" > /workspace/reset_status
printf '%s\\n' "$CHECKOUT_STATUS" > /workspace/checkout_status
printf '%s\\n' "$PATCH_APPLY_STATUS" > /workspace/patch_apply_status
printf '%s\\n' "$RUNTIME_SETUP_STATUS" > /workspace/runtime_setup_status
printf '%s\\n' "$TEST_SETUP_STATUS" > /workspace/test_setup_status
printf '%s\\n' "$PREFETCH_STATUS" > /workspace/prefetch_status
printf '%s\\n' "$TEST_STATUS" > /workspace/test_status
printf '%s\\n' "$PARSER_STATUS" > /workspace/parser_status
exit "$PARSER_STATUS"
"""
    return entry_script


def _runtime_environment_command(instance_id: str) -> str:
    """Return environment-only compatibility setup for known OpenSandbox differences."""
    commands = [":"]
    normalized_id = instance_id.lower()
    if normalized_id == FLIPT_OTEL_INSTANCE_ID:
        commands.append(
            """# NeMo Gym change: do not expose execd's telemetry configuration to application tests.
while IFS='=' read -r variable _; do
  case "$variable" in
    OTEL_*) unset "$variable" ;;
  esac
done < <(env)"""
        )
    if normalized_id == NODEBB_IPV4_INSTANCE_ID:
        commands.append(
            """# NeMo Gym change: NodeBB binds IPv4 while OpenSandbox localhost may prefer ::1.
export NODE_OPTIONS="--dns-result-order=ipv4first${NODE_OPTIONS:+ $NODE_OPTIONS}\""""
        )
    return "\n".join(commands)


def _fix_nodebb_redis_readiness(run_script: str) -> str:
    old = """  while ! redis-cli ping; do
    echo "Waiting for Redis to start..."
    sleep 1
  done"""
    new = """  # NeMo Gym change: redis-cli can exit zero while Redis still reports LOADING.
  until [ "$(redis-cli ping 2>/dev/null)" = "PONG" ]; do
    echo "Waiting for Redis to become ready..."
    sleep 1
  done"""
    return run_script.replace(old, new)


def _fix_qutebrowser_xvfb_startup(run_script: str) -> str:
    if "pytest " not in run_script or "PYTEST_QT_API" not in run_script:
        return run_script

    helper = """# NeMo Gym change: pytest-xvfb's -displayfd startup times out in OpenSandbox.
start_opensandbox_xvfb() {
  export DISPLAY=:99
  rm -f /tmp/.X99-lock
  mkdir -p /tmp/.X11-unix
  Xvfb :99 -br -nolisten tcp -screen 0 800x600x16 >/tmp/xvfb.log 2>&1 &
  for _ in $(seq 1 30); do
    if [ -S /tmp/.X11-unix/X99 ]; then
      return 0
    fi
    sleep 1
  done
  cat /tmp/xvfb.log >&2
  return 1
}
start_opensandbox_xvfb

"""
    marker = "set -e\n"
    if marker not in run_script:
        return run_script
    return run_script.replace(marker, marker + "\n" + helper, 1).replace("pytest ", "pytest -p no:xvfb ")


def apply_opensandbox_runtime_parity(instance_id: str, run_script: str) -> str:
    """Apply narrowly scoped runtime shims while keeping upstream scripts recognizable."""
    normalized_id = instance_id.lower()
    if normalized_id == NODEBB_REDIS_INSTANCE_ID:
        run_script = _fix_nodebb_redis_readiness(run_script)
    if normalized_id == QUTEBROWSER_XVFB_INSTANCE_ID:
        run_script = _fix_qutebrowser_xvfb_startup(run_script)
    return run_script


def assemble_workspace_files(
    uid: str,
    scripts_dir: str | None,
    patch: str,
    sample: dict[str, Any],
) -> tuple[dict[str, str], str]:
    """Assemble the files expected by the upstream evaluator workspace."""
    # NeMo Gym change: scripts are embedded in the prepared row; scripts_dir is retained to match upstream's signature.
    del uid, scripts_dir
    run_script = sample["run_script"]
    if sample.get("runtime_parity_adaptations"):
        run_script = apply_opensandbox_runtime_parity(sample["instance_id"], run_script)
    parser_script = sample["parser_script"]
    entryscript_content = create_entryscript(sample)

    cleaned_patch = strip_binary_hunks(patch)

    files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content


def grade_output(output: dict[str, Any], sample: dict[str, Any]) -> bool:
    """Grade one parser output using the statements extracted from upstream main()."""
    passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}
    # NeMo Gym change: parse untrusted dataset content safely instead of using eval().
    f2p = set(parse_string_list(sample["fail_to_pass"]))
    p2p = set(parse_string_list(sample["pass_to_pass"]))
    return (f2p | p2p) <= passed_tests


def _parse_exit_code(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


async def run_verification(
    sandbox: AsyncSandbox,
    inputs: VerificationInputs,
    log_dir: Path,
    timeout_s: int | None,
) -> VerificationResult:
    """Apply one patch, execute Pro's task scripts, and grade their JSON output."""
    sample = asdict(inputs)
    files, entry_script = assemble_workspace_files(inputs.instance_id, None, inputs.patch, sample)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "patch.diff").write_text(files["patch.diff"], encoding="utf-8")
    (log_dir / "eval.sh").write_text(entry_script, encoding="utf-8")

    await sandbox.exec(f"chmod +x {shlex.quote(RUN_SCRIPT_PATH)} {shlex.quote(ENTRY_SCRIPT_PATH)}")
    execution = None
    execution_error = None
    try:
        execution = await sandbox.exec(
            f"bash {shlex.quote(ENTRY_SCRIPT_PATH)}",
            timeout_s=timeout_s,
        )
    except Exception as exc:
        execution_error = exc

    async def capture(remote_path: str, local_name: str) -> str:
        try:
            result = await sandbox.exec(f"cat {shlex.quote(remote_path)}")
            if result.return_code != 0:
                raise RuntimeError(result.stderr or f"cat exited with {result.return_code}")
            contents = result.stdout or ""
        except Exception as exc:
            contents = f"Failed to read {remote_path}: {exc}\n"
        (log_dir / local_name).write_text(contents, encoding="utf-8")
        return contents

    test_stdout = await capture(STDOUT_PATH, "test_stdout.log")
    test_stderr = await capture(STDERR_PATH, "test_stderr.log")
    test_output = f"STDOUT:\n{test_stdout}\n\nSTDERR:\n{test_stderr}"
    reset_status = await capture(RESET_STATUS_PATH, "reset_status")
    checkout_status = await capture(CHECKOUT_STATUS_PATH, "checkout_status")
    patch_status = await capture(PATCH_STATUS_PATH, "patch_apply_status")
    runtime_setup_status = await capture(RUNTIME_SETUP_STATUS_PATH, "runtime_setup_status")
    test_setup_status = await capture(TEST_SETUP_STATUS_PATH, "test_setup_status")
    prefetch_status = await capture(PREFETCH_STATUS_PATH, "prefetch_status")
    test_status = await capture(TEST_STATUS_PATH, "test_status")
    parser_status = await capture(PARSER_STATUS_PATH, "parser_status")
    output_text = await capture(OUTPUT_PATH, "output.json")
    patch_applied = patch_status.strip() == "0"
    reset_exit_code = _parse_exit_code(reset_status)
    checkout_exit_code = _parse_exit_code(checkout_status)
    patch_exit_code = _parse_exit_code(patch_status)
    runtime_setup_exit_code = _parse_exit_code(runtime_setup_status)
    test_setup_exit_code = _parse_exit_code(test_setup_status)
    prefetch_exit_code = _parse_exit_code(prefetch_status)
    test_exit_code = _parse_exit_code(test_status)
    parser_exit_code = _parse_exit_code(parser_status)
    phase_exit_codes = {
        "reset": reset_exit_code,
        "checkout": checkout_exit_code,
        "patch": patch_exit_code,
        "runtime setup": runtime_setup_exit_code,
        "test setup": test_setup_exit_code,
        "parser": parser_exit_code,
    }
    (log_dir / "execution_stdout.log").write_text(
        (execution.stdout if execution is not None else "") or "", encoding="utf-8"
    )
    (log_dir / "execution_stderr.log").write_text(
        (execution.stderr if execution is not None else "") or "", encoding="utf-8"
    )

    if execution_error is not None:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            test_output=test_output,
            error=f"Evaluation execution failed: {execution_error}",
            reset_exit_code=reset_exit_code,
            checkout_exit_code=checkout_exit_code,
            patch_exit_code=patch_exit_code,
            runtime_setup_exit_code=runtime_setup_exit_code,
            test_setup_exit_code=test_setup_exit_code,
            prefetch_exit_code=prefetch_exit_code,
            test_exit_code=test_exit_code,
            parser_exit_code=parser_exit_code,
        )

    try:
        test_results = json.loads(output_text)
    except json.JSONDecodeError as exc:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            test_output=test_output,
            error=f"Parser produced invalid JSON: {exc}",
            reset_exit_code=reset_exit_code,
            checkout_exit_code=checkout_exit_code,
            patch_exit_code=patch_exit_code,
            runtime_setup_exit_code=runtime_setup_exit_code,
            test_setup_exit_code=test_setup_exit_code,
            prefetch_exit_code=prefetch_exit_code,
            test_exit_code=test_exit_code,
            parser_exit_code=parser_exit_code,
        )
    if not isinstance(test_results, dict):
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            test_output=test_output,
            error="Parser output must be a JSON object",
            reset_exit_code=reset_exit_code,
            checkout_exit_code=checkout_exit_code,
            patch_exit_code=patch_exit_code,
            runtime_setup_exit_code=runtime_setup_exit_code,
            test_setup_exit_code=test_setup_exit_code,
            prefetch_exit_code=prefetch_exit_code,
            test_exit_code=test_exit_code,
            parser_exit_code=parser_exit_code,
        )

    failed_infrastructure_phases = [name for name, code in phase_exit_codes.items() if code != 0]
    completed = not failed_infrastructure_phases
    resolved = completed and patch_applied and grade_output(test_results, sample)
    error = None
    if failed_infrastructure_phases:
        error = f"Failed verification phase(s): {', '.join(failed_infrastructure_phases)}"
    elif prefetch_exit_code not in (None, 0):
        error = f"Go module prefetch exited with {prefetch_exit_code}; tests continued"
    elif test_exit_code not in (None, 0):
        error = f"Test command exited with {test_exit_code}"
    elif execution.return_code != 0:
        error = f"Evaluation script exited with {execution.return_code}"
    return VerificationResult(
        completed=completed,
        resolved=resolved,
        patch_applied=patch_applied,
        test_results=test_results,
        test_output=test_output,
        error=error,
        reset_exit_code=reset_exit_code,
        checkout_exit_code=checkout_exit_code,
        patch_exit_code=patch_exit_code,
        runtime_setup_exit_code=runtime_setup_exit_code,
        test_setup_exit_code=test_setup_exit_code,
        prefetch_exit_code=prefetch_exit_code,
        test_exit_code=test_exit_code,
        parser_exit_code=parser_exit_code,
    )
