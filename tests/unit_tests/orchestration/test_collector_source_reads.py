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
opening snapshot, and one closing snapshot proves the sources ended as they started. Each snapshot hash is one pass;
the parse hashes the bytes it reads and must match the opening snapshot, which also catches a rewrite during the
parse that is undone before the closing snapshot. The real ``COLLECT_SCRIPT`` runs with ``open`` instrumented, as on
a login node.
"""

import json
import subprocess
import sys
from pathlib import Path

from nemo_gym.orchestration.recovery import COLLECT_SCRIPT, CollectedEvidence
from tests.unit_tests.orchestration.test_eval_resubmit_decision import (
    complete_job,
    ident,
    produce_quality,
    write_job_dir,
)


# Counts binary reads of artifacts/rollouts.jsonl. MUTATE=append adds a row just before the parse opens the file.
# MUTATE=aba rewrites the rows after the first in place (same size) once the parse has read the first one, and
# restores the bytes and modification time when the parse is done, so the closing snapshot matches the opening one.
# MUTATE=opening rewrites the block the opening snapshot has just hashed (same size, in place),
# with every stat of the file frozen as on a filesystem with coarse timestamps: stat identity cannot see it, and the
# opening digest mixes old and new bytes.
_INSTRUMENT = r"""
import builtins, os, sys
_open, _opens, _mode = builtins.open, [], os.environ.get("MUTATE")
class _ABAReader:
    def __init__(self, path, handle):
        self._path, self._handle = path, handle
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self._handle.close()
    def __iter__(self):
        with _open(self._path, "rb") as handle:
            original = handle.read()
        stat = os.stat(self._path)
        first = True
        for line in self._handle:
            yield line
            if first:
                first = False
                with _open(self._path, "r+b") as handle:
                    handle.seek(len(line))
                    handle.write(original[len(line):].replace(b"x", b"y"))
        with _open(self._path, "wb") as handle:
            handle.write(original)
        os.utime(self._path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
_frozen, _real_stat, _real_fstat = [], os.stat, os.fstat
def _is_rollouts(path):
    return str(path).endswith(os.path.join("artifacts", "rollouts.jsonl"))
def _frozen_stat(path, *args, **kwargs):
    return _frozen[0] if _frozen and _is_rollouts(path) else _real_stat(path, *args, **kwargs)
def _frozen_fstat(fd):
    return _frozen[0] if _frozen and _is_rollouts(os.readlink(f"/proc/self/fd/{fd}")) else _real_fstat(fd)
def _rewrite_after_first_block(path, source):
    _frozen.append(_real_stat(path))
    os.stat, os.fstat = _frozen_stat, _frozen_fstat
    read, done = source.read, []
    def wrapped(size=-1):
        block = read(size)
        if block and not done:
            done.append(True)
            with _open(path, "r+b") as handle:
                handle.write(block.replace(b"x", b"y"))
        return block
    source.read = wrapped
    return source
def _counting_open(path, mode="r", *args, **kwargs):
    if str(path).endswith(os.path.join("artifacts", "rollouts.jsonl")) and mode == "rb":
        _opens.append(sys._getframe(1).f_code.co_name)
        if _mode == "opening" and _opens == ["sha256"]:
            return _rewrite_after_first_block(path, _open(path, mode, *args, **kwargs))
        if _mode == "append" and _opens[-1] == "rows":
            with _open(path, "ab") as handle:
                handle.write(b'{"_ng_task_index": 99, "_ng_rollout_index": 0, "reward": 0.0}\n')
        if _mode == "aba" and _opens[-1] == "rows":
            return _ABAReader(path, _open(path, mode, *args, **kwargs))
    return _open(path, mode, *args, **kwargs)
builtins.open = _counting_open
import atexit
atexit.register(lambda: print("__OPENS:" + ",".join(_opens), file=sys.stderr))
"""


def run_collector_process(job: Path, *, mutate: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _INSTRUMENT + COLLECT_SCRIPT, str(job), json.dumps(["agent", "mean/reward"])],
        capture_output=True,
        text=True,
        env={"PATH": "", **({"MUTATE": mutate} if mutate else {})},
    )


def run_collector(job: Path, *, mutate: str | None = None) -> tuple[CollectedEvidence, list[str]]:
    out = run_collector_process(job, mutate=mutate)
    assert out.returncode == 0, out.stderr
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


def test_rollouts_are_read_by_one_opening_hash_one_parse_and_one_closing_hash(tmp_path) -> None:
    job = finished_job(tmp_path)
    collected, opens = run_collector(job)
    # Before: two passes per snapshot hash, the quality binding and the receipt check each hashing again, and the
    # closing snapshot taken twice, so up to 11 reads of the largest file. Now 3, and the parse runs once.
    assert opens == ["sha256", "rows", "sha256"]
    assert collected.source_digests is not None
    assert collected.quality is not None and collected.mlflow["receipt"]["run_id"] == "postrun-run"


def test_the_bound_checks_use_the_opening_snapshot_digests(tmp_path) -> None:
    job = finished_job(tmp_path)
    collected, _ = run_collector(job)
    binding = collected.quality["provenance"]["rollout_artifact_digests"][0]
    assert binding["rollouts_jsonl_fpath"] == collected.source_digests["rollouts.jsonl"]
    assert collected.quality["source_matches"] is True


def padded_job(tmp_path: Path) -> Path:
    """Rows large enough that a pass reads them in several blocks, so a rewrite mid-pass is visible to it."""
    rows = [ident(t, reward=1.0, padding="x" * 300_000) for t in range(4)]
    return write_job_dir(tmp_path / "job", n=4, results=rows, failures=[])


def test_a_change_between_the_opening_snapshot_and_the_parse_is_unstable(tmp_path) -> None:
    collected, _ = run_collector(finished_job(tmp_path), mutate="append")
    assert collected.source_digests is None
    assert collected.quality["source_matches"] is False


def test_a_same_size_rewrite_during_the_parse_is_caught_even_when_undone_before_closing(tmp_path) -> None:
    job = padded_job(tmp_path)
    saved = job / "artifacts" / "rollouts.jsonl"
    original, mtime = saved.read_bytes(), saved.stat().st_mtime_ns

    collected, opens = run_collector(job, mutate="aba")

    # The closing snapshot alone cannot see it: the bytes and modification time are back. Only the parse digest can.
    assert (saved.read_bytes(), saved.stat().st_mtime_ns) == (original, mtime)
    assert opens == ["sha256", "rows", "sha256"]
    assert collected.source_digests is None


def test_a_same_tick_rewrite_during_the_opening_hash_leaves_the_sources_unstable(tmp_path) -> None:
    # The per-call second pass used to catch this. The opening digest now mixes old and new bytes, so it matches
    # neither the parse digest nor the closing snapshot.
    job = padded_job(tmp_path)

    collected, _ = run_collector(job, mutate="opening")

    assert b"y" in (job / "artifacts" / "rollouts.jsonl").read_bytes()  # the rewrite happened
    assert collected.source_digests is None
