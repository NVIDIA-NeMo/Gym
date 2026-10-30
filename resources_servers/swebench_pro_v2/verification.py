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

"""SWE-bench Pro v2 verification: run each task's own upstream ``tests/test.sh``.

Upstream source:
https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2

v2 grades with a per-task ``test.sh`` (reset test files, apply the test patch, run the
task's ``run_script.sh``, parse, check fail-to-pass and pass-to-pass), replacing v1's
``before_repo_set_cmd`` entryscript. Everything else is reused from the v1 verifier:
``run_verification`` runs the entryscript below and grades the parser output with the same
required-pass rule that ``test.sh`` applies.
"""

from dataclasses import dataclass
from typing import Any

from resources_servers.swebench_pro.verification import (
    APPLY_PATCH_FUNCTION,
    SCRUB_HARNESS_ENV,
    VerificationInputs,
    strip_binary_hunks,
)


# test.sh reads these paths, so they are copied from the uploaded workspace to /tests.
TESTS_DIR = "/tests"
TEST_FILES = ("test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch")


@dataclass(frozen=True)
class VerificationInputsV2(VerificationInputs):
    test_sh: str = ""
    test_config: str = ""
    test_patch: str = ""


def create_entryscript(sample: dict[str, Any]) -> str:
    """Apply the candidate patch, run upstream ``test.sh``, and expose its outputs where v1 looks.

    Harbor pre-creates /logs/verifier for ``test.sh``; it only mkdirs it in its exit trap.
    """
    return f"""
{SCRUB_HARNESS_ENV}
cd /app 2>/dev/null || cd /testbed
{APPLY_PATCH_FUNCTION}
apply_patch
printf '%s\\n' "$?" > /workspace/patch_apply_status
mkdir -p {TESTS_DIR} /logs/verifier
cp {" ".join(f"/workspace/{name}" for name in TEST_FILES)} {TESTS_DIR}/
bash {TESTS_DIR}/test.sh > /workspace/test_sh.log 2>&1
cp /logs/verifier/output.json /workspace/output.json
cp /logs/verifier/run-script-stdout.txt /workspace/stdout.log
cp /logs/verifier/run-script-stderr.txt /workspace/stderr.log
"""


def assemble_workspace_files(
    uid: str,
    scripts_dir: str | None,
    patch: str,
    sample: dict[str, Any],
) -> tuple[dict[str, str], str]:
    """Same contract as v1's ``assemble_workspace_files``, with the v2 verifier files."""
    del uid, scripts_dir
    entryscript_content = create_entryscript(sample)
    files = {
        "patch.diff": strip_binary_hunks(patch),
        "entryscript.sh": entryscript_content,
        "test.sh": sample["test_sh"],
        "run_script.sh": sample["run_script"],
        "parser.py": sample["parser_script"],
        "config.json": sample["test_config"],
        "test_patch.patch": sample["test_patch"],
    }
    return files, entryscript_content
