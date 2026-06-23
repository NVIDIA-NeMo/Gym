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

"""Real apptainer-provider end-to-end (env-gated; never runs in CI).

Builds a ``.sif`` from the docker itest image (the calc-bug git repo) and runs
``verify_task`` through the ApptainerSandboxProvider on the flat swe-bench-ext
path. Enable with ``SWE_ENV_APPTAINER_ITEST=1`` on a box with apptainer + docker.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys

import pytest

import responses_api_agents.swe_env.harnesses  # noqa: F401  (register harnesses)
from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import SweTask


_RUN = os.environ.get("SWE_ENV_APPTAINER_ITEST") == "1" and shutil.which("apptainer") is not None
_SIF = "/tmp/swe-env-itest.sif"
_GOLD = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"


@pytest.mark.skipif(not _RUN, reason="set SWE_ENV_APPTAINER_ITEST=1 and install apptainer")
def test_apptainer_real_end_to_end():
    """Build the .sif if absent, then verify the gold patch resolves via the apptainer provider."""
    if not os.path.exists(_SIF):
        build = subprocess.run(
            ["apptainer", "build", "--force", _SIF, "docker-daemon://swe-env-itest:local"],
            capture_output=True,
        )
        assert build.returncode == 0, build.stderr.decode(errors="replace")[-3000:]

    task = SweTask(
        instance_id="calc-apptainer",
        image="swe-env-itest",
        base_commit="HEAD",
        repo_workdir="/testbed",
        test_command="python -m pytest -rA -q",
        model_patch=_GOLD,
        fail_to_pass=["test_calc.py::test_add"],
        benchmark="swe-bench-ext",
        metadata={"provider_options": {"sif_path": _SIF}},
    )
    report = asyncio.run(verify_task({"apptainer": {}}, task))
    sys.stderr.write(f"\n[apptainer itest] {report}\n")
    assert report.patch_applied is True
    assert report.resolved is True
    assert reward_from_report(report) == 1.0
