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
from collections.abc import Iterable
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
    # None means the default set; an explicit tuple (even empty) is honoured.
    environment_repairs: tuple[str, ...] | None = None


@dataclass(frozen=True)
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str = ""
    error: str | None = None


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


# NeMo Gym additions below: patch extraction, resilient application, container repairs
# and verdict classification. Measurements and rejected alternatives: tools/README.md.


def patch_section_path(section: str) -> str | None:
    """Return the repository-relative path one ``diff --git`` section targets."""
    a_path: str | None = None
    b_path: str | None = None
    for line in section.splitlines():
        if line.startswith("@@"):
            break
        if line.startswith("--- ") and a_path is None:
            value = line[4:].strip()
            a_path = None if value == "/dev/null" else value.removeprefix("a/")
        elif line.startswith("+++ ") and b_path is None:
            value = line[4:].strip()
            b_path = None if value == "/dev/null" else value.removeprefix("b/")

    if b_path or a_path:
        return b_path or a_path

    header = re.match(r"^diff --git a/(.+?) b/(.+)$", section.splitlines()[0] if section else "")
    return header.group(2) if header else None


def drop_patch_sections(patch: str, paths: Iterable[str]) -> str:
    """Drop the diff sections targeting ``paths``; see the notes above."""
    dropped = set(paths)
    if not patch or not dropped:
        return patch

    kept: list[str] = []
    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        if not section.strip():
            continue
        path = patch_section_path(section)
        if path is not None and path in dropped:
            continue
        kept.append(section)

    return "".join(kept)


# The sandbox daemon exports its own OTEL_* config into every command; tests read it.
SCRUB_HARNESS_ENV = """# NeMo Gym change: drop the sandbox daemon's telemetry env, which leaks into tasks.
for ng_env_var in $(env | sed -n 's/^\\(OTEL_[A-Za-z0-9_]*\\)=.*/\\1/p'); do
  unset "$ng_env_var"
done"""


WARM_XVFB_CACHE = """if command -v Xvfb >/dev/null 2>&1; then
  Xvfb :121 -screen 0 800x600x16 -nolisten tcp >/tmp/ng_xvfb_warmup.log 2>&1 &
  ng_xvfb_pid=$!
  for _ in $(seq 1 60); do
    if [ -e /tmp/.X121-lock ]; then break; fi
    sleep 0.5
  done
  kill "$ng_xvfb_pid" 2>/dev/null || true
  wait "$ng_xvfb_pid" 2>/dev/null || true
  rm -f /tmp/.X121-lock
fi"""


DROP_STALE_DATABASE_STATE = """# NeMo Gym change: drop stale redis state the task image shipped.
for ng_stale in dump.rdb appendonly.aof appendonlydir; do
  if [ -e "$ng_stale" ] && [ -z "$(git ls-files -- "$ng_stale")" ]; then
    rm -rf "$ng_stale"
  fi
done"""


DISABLE_CORE_DUMPS = """ulimit -c 0 2>/dev/null || true"""


WARM_DEPENDENCY_CACHE = """ng_warm=""
for ng_dir in node_modules vendor .venv venv third_party node_modules/.bin; do
  if [ -d "$ng_dir" ]; then ng_warm="$ng_warm $ng_dir"; fi
done
if [ -n "$ng_warm" ]; then
  timeout 180 sh -c "find $ng_warm -type f -print0 2>/dev/null | xargs -0 -r -P 4 -n 256 cat" >/dev/null 2>&1 || true
fi"""


# Container repairs, keyed so any can be dropped and re-measured; see tools/README.md.
ENVIRONMENT_REPAIRS: dict[str, str] = {
    "stale_database_state": DROP_STALE_DATABASE_STATE,
    "core_dump_limit": DISABLE_CORE_DUMPS,
    "dependency_cache": WARM_DEPENDENCY_CACHE,
    "xvfb_cache": WARM_XVFB_CACHE,
}
DEFAULT_ENVIRONMENT_REPAIRS: tuple[str, ...] = tuple(ENVIRONMENT_REPAIRS)
# Only repairs that outlive a single command are useful in the agent container.
AGENT_ENVIRONMENT_REPAIRS: tuple[str, ...] = ("stale_database_state", "dependency_cache", "xvfb_cache")


def build_environment_repairs(names: Iterable[str]) -> str:
    """Concatenate the named repairs in their canonical order."""
    requested = set(names)
    unknown = requested - set(ENVIRONMENT_REPAIRS)
    if unknown:
        raise ValueError(f"Unknown environment repairs: {sorted(unknown)}")
    selected = [body for key, body in ENVIRONMENT_REPAIRS.items() if key in requested]
    if not selected:
        return "# NeMo Gym change: all environment repairs disabled for this run."
    header = "# NeMo Gym change: repair host properties that break tests; see verification.py."
    return "\n".join([header, *selected])


# The agent runs the same suites, so its container needs the same repairs.
def build_seed_normalization(names: Iterable[str]) -> str:
    """The repairs worth applying to the agent container, as one script."""
    selected = [n for n in names if n in AGENT_ENVIRONMENT_REPAIRS]
    return "cd /app 2>/dev/null || exit 0\n" + build_environment_repairs(selected)


# `git apply` is all-or-nothing, so retry excluding the paths it reported, then GNU patch.
APPLY_PATCH_FUNCTION = """# NeMo Gym change: apply the patch resiliently instead of `git apply -v <patch>`.
apply_patch() {
  if [ ! -s /workspace/patch.diff ]; then
    echo "NG_APPLY: patch is empty, nothing to apply"
    return 0
  fi
  if git apply -v --whitespace=nowarn /workspace/patch.diff 2>/workspace/apply_strict.log; then
    cat /workspace/apply_strict.log
    echo "NG_APPLY: applied cleanly"
    return 0
  fi
  cat /workspace/apply_strict.log
  ng_exclude_args=()
  while IFS= read -r ng_path; do
    if [ -n "$ng_path" ]; then
      ng_exclude_args+=("--exclude=$ng_path")
    fi
  done < <(sed -n 's/^error: \\(.*\\): already exists in working directory$/\\1/p' /workspace/apply_strict.log)
  if [ ${#ng_exclude_args[@]} -gt 0 ]; then
    if git apply -v --whitespace=nowarn "${ng_exclude_args[@]}" /workspace/patch.diff 2>/workspace/apply_excluded.log; then
      cat /workspace/apply_excluded.log
      echo "NG_APPLY: applied after excluding pre-existing container artifacts: ${ng_exclude_args[*]}"
      return 0
    fi
    cat /workspace/apply_excluded.log
  fi
  if command -v patch >/dev/null 2>&1; then
    if patch -p1 --batch --fuzz=5 -i /workspace/patch.diff >/workspace/apply_fuzz.log 2>&1; then
      cat /workspace/apply_fuzz.log
      echo "NG_APPLY: applied with GNU patch fallback"
      return 0
    fi
    cat /workspace/apply_fuzz.log
  fi
  echo "NG_APPLY: failed to apply patch"
  return 1
}"""


def inconclusive_reason(result: VerificationResult, sample: dict[str, Any]) -> str | None:
    """Say why a verification produced no verdict, or ``None`` if it produced one."""
    if not result.completed:
        return f"evaluation did not complete ({result.error or 'unknown error'})"
    if result.test_results is None:
        return "parser produced no usable output"

    reported = {test["name"] for test in result.test_results.get("tests") or []}
    required = set(parse_string_list(sample["fail_to_pass"])) | set(parse_string_list(sample["pass_to_pass"]))
    if not required:
        return None
    if not reported:
        return "parser reported no tests at all"

    unobserved = required - reported
    if unobserved:
        return f"{len(unobserved)} of {len(required)} graded tests never reported an outcome"
    return None


def create_entryscript(sample: dict[str, Any]) -> str:
    """Create the upstream in-container evaluation script."""
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    # NeMo Gym change: parse untrusted dataset content safely instead of using eval().
    selected_test_files_to_run = ",".join(parse_string_list(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    # NeMo Gym change: optionally warm Go's module cache before the upstream test command.
    should_prefetch_go_modules = sample.get("prefetch_go_modules", False) and (
        str(sample.get("repo_language", "")).lower() == "go" or "go test " in sample["run_script"]
    )
    go_module_prefetch_cmd = ""
    if should_prefetch_go_modules:
        go_module_prefetch_cmd = """# NeMo Gym change: prefetch modules without modifying the upstream test script.
if [ -f go.mod ]; then
  go mod download
fi"""
    requested_repairs = sample.get("environment_repairs")
    environment_repairs = build_environment_repairs(
        DEFAULT_ENVIRONMENT_REPAIRS if requested_repairs is None else requested_repairs
    )
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
{SCRUB_HARNESS_ENV}
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
{APPLY_PATCH_FUNCTION}
apply_patch
# NeMo Gym change: retain patch application status for the structured verification response.
PATCH_APPLY_STATUS=$?
{before_repo_set_cmd}
{environment_repairs}
{go_module_prefetch_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
# NeMo Gym change: persist the status after running the upstream script sequence.
printf '%s\\n' "$PATCH_APPLY_STATUS" > /workspace/patch_apply_status
"""
    return entry_script


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
            contents = result.stdout or ""
        except Exception as exc:
            contents = f"Failed to read {remote_path}: {exc}\n"
        (log_dir / local_name).write_text(contents, encoding="utf-8")
        return contents

    test_stdout = await capture(STDOUT_PATH, "test_stdout.log")
    test_stderr = await capture(STDERR_PATH, "test_stderr.log")
    test_output = f"STDOUT:\n{test_stdout}\n\nSTDERR:\n{test_stderr}"
    patch_status = await capture(f"{WORKSPACE_DIR}/patch_apply_status", "patch_apply_status")
    output_text = await capture(OUTPUT_PATH, "output.json")
    patch_applied = patch_status.strip() == "0"
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
        )
    if not isinstance(test_results, dict):
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            test_output=test_output,
            error="Parser output must be a JSON object",
        )

    resolved = patch_applied and grade_output(test_results, sample)
    return VerificationResult(
        completed=execution.return_code == 0,
        resolved=resolved,
        patch_applied=patch_applied,
        test_results=test_results,
        test_output=test_output,
        error=None if execution.return_code == 0 else f"Evaluation script exited with {execution.return_code}",
    )
