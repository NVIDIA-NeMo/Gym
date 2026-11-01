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
"""The remote collector reads each large source a bounded number of times, and still sees a change during collection.

Every check that binds a source to a recorded digest (quality binding, export receipt, health sources) uses the
opening snapshot; one closing snapshot then proves nothing changed in between. Each snapshot hash keeps its own two
passes. The real ``COLLECT_SCRIPT`` runs with ``open`` instrumented, as on a login node.
"""

import json
import subprocess
import sys
from pathlib import Path

from nemo_gym.orchestration.recovery import COLLECT_SCRIPT, CollectedEvidence
from tests.unit_tests.test_eval_resubmit_decision import complete_job, produce_quality


# Counts binary opens of artifacts/rollouts.jsonl; with MUTATE set, the parse pass appends a row before it reads.
_INSTRUMENT = r"""
import builtins, os, sys
_open, _opens = builtins.open, []
def _counting_open(path, mode="r", *args, **kwargs):
    if str(path).endswith(os.path.join("artifacts", "rollouts.jsonl")) and "b" in mode:
        _opens.append(sys._getframe(1).f_code.co_name)
        if os.environ.get("MUTATE") and _opens[-1] == "rows":
            with _open(path, "ab") as handle:
                handle.write(b'{"_ng_task_index": 99, "_ng_rollout_index": 0, "reward": 0.0}\n')
    return _open(path, mode, *args, **kwargs)
builtins.open = _counting_open
import atexit
atexit.register(lambda: print("__OPENS:" + ",".join(_opens), file=sys.stderr))
"""


def run_collector(job: Path, *, mutate: bool = False) -> tuple[CollectedEvidence, list[str]]:
    env = {"MUTATE": "1"} if mutate else {}
    out = subprocess.run(
        [sys.executable, "-c", _INSTRUMENT + COLLECT_SCRIPT, str(job), json.dumps(["agent", "mean/reward"])],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "", **env},
    )
    opens = next(line for line in out.stderr.splitlines() if line.startswith("__OPENS:"))
    return CollectedEvidence.model_validate_json(out.stdout), opens.removeprefix("__OPENS:").split(",")


def finished_job(tmp_path: Path) -> Path:
    """A finished run with both digest-bound records the collector checks: a quality summary and a post-run receipt."""
    job = complete_job(tmp_path)
    produce_quality(job)
    receipt_path = job / "artifacts" / "rollouts_mlflow_export_done"
    assert json.loads(receipt_path.read_text())["version"] == 1
    assert (job / "artifacts" / "quality_summary.json").exists()
    return job


def test_rollouts_are_read_by_two_snapshots_and_one_parse(tmp_path) -> None:
    job = finished_job(tmp_path)
    collected, opens = run_collector(job)
    # Opening snapshot (2 passes), one parse, closing snapshot (2 passes). Before, the quality binding and the receipt
    # check each hashed the file again, and the closing snapshot was taken twice: 11 reads of the largest file.
    assert opens == ["sha256", "sha256", "rows", "sha256", "sha256"]
    assert collected.source_digests is not None
    assert collected.quality is not None and collected.mlflow["receipt"]["run_id"] == "postrun-run"


def test_the_bound_checks_use_the_opening_snapshot_digests(tmp_path) -> None:
    job = finished_job(tmp_path)
    collected, _ = run_collector(job)
    digests = collected.source_digests
    binding = collected.quality["provenance"]["rollout_artifact_digests"][0]
    assert binding["rollouts_jsonl_fpath"] == digests["rollouts.jsonl"]
    assert collected.quality["source_matches"] is True


def test_a_change_during_collection_still_marks_the_sources_unstable(tmp_path) -> None:
    job = finished_job(tmp_path)
    collected, opens = run_collector(job, mutate=True)
    assert "rows" in opens
    assert collected.source_digests is None
    assert collected.quality["source_matches"] is False
