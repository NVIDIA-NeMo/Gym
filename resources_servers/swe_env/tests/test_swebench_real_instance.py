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

"""Real SWE-bench Verified instance end-to-end (env-gated; never runs in CI).

Pulls the public SWE-bench docker image for one instance and grades its GOLD
patch through the verifier (docker provider) — proving the verifier works on
REAL benchmark data, not synthetic. Validated locally on both providers:
``astropy__astropy-13453`` -> resolved=True, reward=1.0 (docker AND the
docker->.sif apptainer path; build the .sif with
``apptainer build x.sif docker-daemon://swebench/sweb.eval.x86_64.<id>``).

Enable with ``SWE_ENV_REAL_SWEBENCH=1`` (needs docker + network; pulls ~2.7GB).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import urllib.request

import pytest

from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import SweTask


_RUN = os.environ.get("SWE_ENV_REAL_SWEBENCH") == "1" and shutil.which("docker") is not None
_INSTANCE = "astropy__astropy-13453"
_OFFSET = 4  # index of _INSTANCE in SWE-bench_Verified test split
_DATASET_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset=princeton-nlp/SWE-bench_Verified&config=default&split=test&offset={_OFFSET}&length=1"
)


def _image_for(instance_id: str) -> str:
    """Map a SWE-bench instance id to its public Docker Hub image name.

    Args:
        instance_id: The SWE-bench instance id (``__`` separator).

    Returns:
        str: The Docker Hub image name for that instance.
    """
    # SWE-bench Docker Hub naming: __ -> _1776_, lowercased.
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_").lower()


def _as_list(value):
    """Normalize a value into a list, parsing JSON strings when possible.

    Args:
        value: A list, a JSON-encoded string, a bare string, or None.

    Returns:
        list: The parsed/normalized list (empty for None).
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [value]
    return value or []


@pytest.mark.skipif(not _RUN, reason="set SWE_ENV_REAL_SWEBENCH=1 (needs docker + network, pulls ~2.7GB)")
def test_real_swebench_gold_patch_resolves():
    """Pull a real SWE-bench Verified instance and verify its gold patch resolves to reward 1.0."""
    with urllib.request.urlopen(_DATASET_URL, timeout=60) as resp:
        row = json.load(resp)["rows"][0]["row"]
    assert row["instance_id"] == _INSTANCE

    f2p = _as_list(row.get("FAIL_TO_PASS"))
    p2p = _as_list(row.get("PASS_TO_PASS"))
    nodeids = " ".join("'" + n + "'" for n in f2p + p2p)
    test_command = (
        f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && python -m pytest -rA {nodeids}"
    )
    task = SweTask(
        instance_id=_INSTANCE,
        image=_image_for(_INSTANCE),
        base_commit=row["base_commit"],
        repo_workdir="/testbed",
        test_command=test_command,
        model_patch=row["patch"],
        test_patch=row.get("test_patch", ""),
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        benchmark="swe-bench-ext",
        metadata={"ttl_s": 3600, "ready_timeout_s": 900},
    )

    report = asyncio.run(verify_task({"docker": {}}, task))
    assert report.patch_applied is True
    assert report.resolved is True
    assert reward_from_report(report) == 1.0
