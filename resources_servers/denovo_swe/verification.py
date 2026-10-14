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
"""Run one AweAI-Team/DeNovoSWE task's tests in a sandbox and grade the result.

DeNovoSWE is a "document-to-repository" benchmark, not a bug-fix-from-a-failing-test benchmark
like the other SWE resources servers: each image ships the ORIGINAL package source at
``parent_commit`` plus a ``document`` (a README/spec). The agent's job is to regenerate the
package from scratch given only the document -- so unlike every other SWE resources server here,
there is no golden ``patch`` to apply for golden-patch validation: the image's pre-existing source
IS the golden answer, and ``patch`` is legitimately empty on that path.

Grading itself is delegated entirely to ``_denovoswe_eval.py``, a verbatim local copy of the
already-battle-tested in-container evaluator from
``responses_api_agents/swe_agents/_denovoswe_eval.py`` (used by the SIF/Apptainer-based swe_agents
harness, where this dataset has already been golden-patch validated: 3034/3668, 82.69%). It
collects each ``passed_ptp`` test file with a ``pytest --collect-only`` pre-flight (mitigating
parametrize-label / hypothesis-seed drift that would otherwise abort an entire file's batch on one
stale id), runs only the resolvable ids, and writes its own verdict (``reward``: "1" iff every
``passed_ptp`` test passes) -- so this file's ``grade()`` just reads that verdict back rather than
reimplementing pytest-output parsing, mirroring how swe_rebench reuses its own upstream
``log_parsers.py`` instead of a hand-rolled parser.

``_denovoswe_clean.sh`` (also a verbatim local copy) wipes the image's source, preserving the
installed environment and config files, so an agent regenerating the package can't just read the
pre-existing implementation it's supposed to reproduce. It is run twice per task: once in
``seed_session`` (before the agent starts), and again inside the verification eval script (since
verification always gets a FRESH sandbox, not the agent's own session) -- both immediately followed
by re-injecting ``document`` as ``README.md`` and folding it into the git history via
``commit --amend``, so the eventual ``git diff`` the agent produces doesn't include a spurious
README addition. Golden-patch validation skips both the wipe and the amend: the image's
pre-existing source is exactly what's being graded.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


REPORT_BEGIN = "___NEMO_GYM_DENOVO_SWE_REPORT_BEGIN___"
REPORT_END = "___NEMO_GYM_DENOVO_SWE_REPORT_END___"

CLEAN_SH_PATH = "/root/_denovoswe_clean.sh"
EVAL_PY_PATH = "/root/_denovoswe_eval.py"
DOCUMENT_PATH = "/root/denovoswe_document.md"
META_PATH = "/root/denovoswe_meta.json"
TEST_PATCH_PATH = "/root/denovoswe_test_patch.diff"
TEST_BINARY_PATH = "/root/denovoswe_test_binary.b64"
REPORT_JSON_PATH = "/trajectories_mount/eval_results/report.json"


@dataclass
class VerificationInputs:
    instance_id: str
    workdir: str
    base_commit: str
    patch: str
    test_patch: str
    document: str
    pypi_name: str = ""
    passed_ptp: Sequence[str] = field(default_factory=tuple)
    failed_ptp: Sequence[str] = field(default_factory=tuple)
    test_binary_archive_b64: str = ""
    expected_coverage_percent: float = 0.0


@dataclass
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None = None


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
    """Drop the diff sections targeting ``paths``.

    ``git add -N . && git diff`` picks up every untracked file, including ones that were already
    untracked before the agent touched anything. Excluding those paths keeps the extracted patch
    to what the agent actually changed.
    """
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


def seed_prep_script(workdir: str) -> str:
    """Wipe the image's source and re-inject ``document`` as ``README.md``, run BEFORE an agent
    starts working (in ``seed_session``) so it cannot just read the implementation it is supposed
    to regenerate. Mirrors AweAgent's ``DeNovoSWETask.prepare_session`` and the identical step the
    SIF-based swe_agents harness runs for this same dataset (see module docstring).
    """
    wp = shlex.quote(workdir)
    return f"""#!/bin/bash
set +e
bash {CLEAN_SH_PATH} {wp} > /tmp/denovoswe_clean.log 2>&1 || echo "WARN: clean.sh exited non-zero (continuing)" >> /tmp/denovoswe_clean.log
if [ -s {DOCUMENT_PATH} ]; then
    cp {DOCUMENT_PATH} {wp}/README.md
    ( cd {wp} && git add README.md 2>/dev/null && git commit --amend --no-edit -q 2>/dev/null ) || true
fi
"""


def build_eval_script(inputs: VerificationInputs, is_golden: bool) -> str:
    """The verification-sandbox script. Faithfully replicates
    ``DeNovoSWEDatasetProcessor.get_run_command`` (the already-validated SIF-harness eval script,
    see module docstring) step for step, adapted to nemo_gym.sandbox's file-upload model instead
    of Apptainer bind mounts.

    Verification always gets a FRESH sandbox (not the agent's own seed_session one), so unlike
    that live session -- which was already wiped + README-injected once -- this script repeats the
    wipe (non-golden only) so the agent's patch, diffed against that same wiped+amended baseline,
    applies cleanly onto a matching tree here too.
    """
    wp = shlex.quote(inputs.workdir)
    apply_patch = (
        "git apply --whitespace=nowarn /tmp/nemo_gym_patch.diff "
        "|| git apply -3 --whitespace=nowarn /tmp/nemo_gym_patch.diff "
        "|| git apply --reject --recount --ignore-space-change --whitespace=nowarn /tmp/nemo_gym_patch.diff "
        "|| true"
    )
    wipe_and_reinject = f"""
bash {CLEAN_SH_PATH} {wp} > /trajectories_mount/eval_results/clean.log 2>&1 \\
    || echo "WARN: clean.sh exited non-zero (continuing)" >> /trajectories_mount/eval_results/clean.log
if [ -s {DOCUMENT_PATH} ]; then
    cp {DOCUMENT_PATH} {wp}/README.md
    ( cd {wp} && git add README.md 2>/dev/null && git commit --amend --no-edit -q 2>/dev/null ) || true
fi
"""
    return f"""#!/bin/bash
set -o pipefail
mkdir -p /trajectories_mount/eval_results
cd {wp} || {{ echo '{{"reward": "0", "error": "workdir_missing"}}' > {REPORT_JSON_PATH}; exit 97; }}
git config --global --add safe.directory {wp} 2>/dev/null || true

# 1. Hard-reset to parent_commit so the image baseline is deterministic.
BASE_COMMIT={shlex.quote(inputs.base_commit)}
if [ -n "$BASE_COMMIT" ]; then
    git checkout -f "$BASE_COMMIT" 2>/dev/null || git reset --hard HEAD 2>/dev/null || true
fi

# 1b. AGENT-RUN ONLY: wipe the source again (fresh sandbox, not the agent's own seeded one) and
#     re-inject the spec, matching the baseline the agent's patch was diffed against. Skipped on
#     the golden path, where the image's pre-existing source is what's graded.
{"" if is_golden else wipe_and_reinject}

# 2. Apply the candidate patch. Golden path: intentionally empty, nothing to apply.
{"" if is_golden else (apply_patch if inputs.patch.strip() else "")}

# 3. Delete EVERY pre-existing test file so test_patch can lay the canonical suite from scratch.
find {wp} -type d \\( -iname tests -o -iname testsuite -o -iname testsuites \\
    -o -iname testing -o -iname test_suite -o -iname test \\) -exec rm -rf {{}} + 2>/dev/null || true
find {wp} -type f \\( -iname 'test_*.py' -o -iname '*_test.py' \\
    -o -iname '*_tests.py' -o -iname 'conftest.py' \\) -delete 2>/dev/null || true
find {wp} -type f -name '.coveragerc' -delete 2>/dev/null || true

# 4. Pre-create parent dirs for files test_patch will add, and pre-clean add-only target paths so
#    the patch's ``--- /dev/null`` hunks don't reject.
cd {wp} && python3 - <<'PY'
import os, re, sys
try:
    p = open({TEST_PATCH_PATH!r}).read()
except OSError:
    sys.exit(0)
add_only = set()
all_dirs = set()
for block in re.split(r'^diff --git ', p, flags=re.MULTILINE)[1:]:
    m = re.search(r'^\\+\\+\\+ b/(.+)$', block, re.MULTILINE)
    if not m:
        continue
    path = m.group(1).strip()
    if not path or path == '/dev/null':
        continue
    d = os.path.dirname(path)
    if d:
        all_dirs.add(d)
    if '--- /dev/null' in block:
        add_only.add(path)
for d in all_dirs:
    os.makedirs(d, exist_ok=True)
for path in add_only:
    try:
        os.unlink(path)
    except (FileNotFoundError, IsADirectoryError):
        pass
PY

# 5. Apply test_patch.
git apply --whitespace=nowarn {TEST_PATCH_PATH} \\
    || git apply -3 --whitespace=nowarn {TEST_PATCH_PATH} \\
    || git apply --reject --recount --ignore-space-change --whitespace=nowarn {TEST_PATCH_PATH} \\
    || true

# 6. Extract binary fixtures (base64-encoded tar.gz), if any.
if [ -s {TEST_BINARY_PATH} ] && [ "$(wc -c <{TEST_BINARY_PATH})" -gt 4 ]; then
    base64 -d {TEST_BINARY_PATH} 2>/dev/null | tar -xzf - -C {wp} 2>> /trajectories_mount/eval_results/binary_extract.log || true
fi

# 7. Uninstall the package (every interpreter we can find) then re-install in editable mode so the
#    on-disk source becomes the one that gets imported.
PYPI_NAME={shlex.quote(inputs.pypi_name)}
if [ -n "$PYPI_NAME" ]; then
    for pyx in $(command -v python) $(command -v python3) /opt/conda/envs/*/bin/python /opt/conda/bin/python /usr/bin/python /usr/bin/python3 /usr/local/bin/python /usr/local/bin/python3; do
        [ -x "$pyx" ] || continue
        "$pyx" -m pip uninstall -y "$PYPI_NAME" >/dev/null 2>&1 || true
    done
fi
pip install -e . > /trajectories_mount/eval_results/pip_install.log 2>&1 || true

# 8. Per-file pytest evaluator -> writes report.json + reward.txt.
python3 {EVAL_PY_PATH} > /trajectories_mount/eval_results/eval_stdout.log 2>&1 || true

# 9. Fallback report if the eval script never wrote one.
if [ ! -s {REPORT_JSON_PATH} ]; then
    echo '{{"_test_completed": true, "reward": "0", "error": "eval_script_no_report"}}' > {REPORT_JSON_PATH}
fi

echo "{REPORT_BEGIN}"
cat {REPORT_JSON_PATH} 2>/dev/null
echo "{REPORT_END}"
"""


def _slice(log: str, begin: str, end: str) -> str:
    start = log.find(begin)
    if start == -1:
        return ""
    start += len(begin)
    stop = log.find(end, start)
    return log[start:stop] if stop != -1 else log[start:]


def _local_file(name: str) -> str:
    return (Path(__file__).resolve().parent / name).read_text()


def verification_files(inputs: VerificationInputs, is_golden: bool) -> dict[str, str]:
    import json

    files = {
        "/tmp/nemo_gym_eval.sh": build_eval_script(inputs, is_golden),
        CLEAN_SH_PATH: _local_file("_denovoswe_clean.sh"),
        EVAL_PY_PATH: _local_file("_denovoswe_eval.py"),
        DOCUMENT_PATH: inputs.document,
        TEST_PATCH_PATH: inputs.test_patch,
        TEST_BINARY_PATH: inputs.test_binary_archive_b64,
        META_PATH: json.dumps(
            {
                "instance_id": inputs.instance_id,
                "workdir": inputs.workdir,
                "passed_ptp": list(inputs.passed_ptp),
                "failed_ptp": list(inputs.failed_ptp),
                "pypi_name": inputs.pypi_name,
                "expected_coverage_percent": inputs.expected_coverage_percent,
            }
        ),
    }
    if not is_golden and inputs.patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = inputs.patch
    return files


async def run_verification(
    sandbox: Any,
    inputs: VerificationInputs,
    is_golden: bool,
    timeout_s: float | None = None,
    log_dir: Path | None = None,
) -> VerificationResult:
    import asyncio
    import json

    result = await sandbox.exec("bash /tmp/nemo_gym_eval.sh", timeout_s=timeout_s)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

    if log_dir is not None:

        def _persist() -> None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "test_output.log").write_text(output, errors="replace")
            except OSError:
                pass

        await asyncio.to_thread(_persist)

    if result.return_code == 97:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error=f"workdir {inputs.workdir} not present in the image",
        )

    report_text = _slice(output, REPORT_BEGIN, REPORT_END).strip()
    if not report_text:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=True,
            test_results=None,
            test_output=output,
            error="no report.json produced (eval script crashed before writing one)",
        )

    try:
        report = await asyncio.to_thread(json.loads, report_text)
    except json.JSONDecodeError as exc:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=True,
            test_results=None,
            test_output=output,
            error=f"malformed report.json: {exc}",
        )

    # _denovoswe_eval.py's own verdict: reward "1" iff every passed_ptp test passes. We trust it
    # rather than re-deriving resolved from pass/fail counts -- it already encodes the
    # collect-then-intersect-then-run mitigation for parametrize/hypothesis-seed drift (see module
    # docstring), which a from-scratch re-derivation here would not have.
    resolved = report.get("reward") == "1"
    completed = bool(report.get("_test_completed", True)) and report.get("error") not in (
        "workdir_missing",
        "eval_script_no_report",
    )
    return VerificationResult(
        completed=completed,
        resolved=resolved,
        patch_applied=True,
        test_results=report,
        test_output=output,
        error=report.get("error"),
    )
