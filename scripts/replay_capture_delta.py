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
"""Replay a full-format model-call capture through the delta format and check that nothing is lost.

``convert`` streams a ``<rollout_id>.capture.jsonl`` file one line at a time into a delta-format store under
``--out-dir`` (``<rollout_id>.capture.jsonl`` and ``blobs/<rollout_id>/``), then reads the delta capture back
and checks that every rebuilt exchange equals its original line, byte for byte once serialized. The source is
only read. It prints one JSON report: sizes, line counts, timings and the process's peak RSS after each phase.

``merge`` times the finished-rollout merge (``merge_model_call_capture_into_record``) on one capture, in one
``model_call_payloads`` mode, and prints its time, the merged record's size and peak RSS.

Run each measurement on a large capture under a memory cap, in its own process, for example::

    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --unit=capture-replay \\
        python scripts/replay_capture_delta.py convert /runs/x/model-calls/9-0.capture.jsonl --out-dir /tmp/delta
    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --unit=capture-merge \\
        python scripts/replay_capture_delta.py merge /tmp/delta 9-0 --payloads reference
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import orjson

from nemo_gym.models import CaptureStore
from nemo_gym.rollouts import merge_model_call_capture_into_record


_CAPTURE_SUFFIX = ".capture.jsonl"
# The key only a delta-format line has (``CAPTURE_FORMAT_KEY`` in ``nemo_gym/models/capture_format.py``).
_DELTA_LINE_KEY = "capture_format"


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _tree_bytes(path: Path) -> tuple[int, int]:
    """Total bytes and number of files under ``path``."""
    files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else []
    return sum(item.stat().st_size for item in files), len(files)


def convert(source: Path, out_dir: Path) -> dict[str, Any]:
    """Write ``source`` as a delta capture under ``out_dir`` and verify every exchange; return the report."""
    if not source.name.endswith(_CAPTURE_SUFFIX):
        raise ValueError(f"{source} is not a <rollout_id>{_CAPTURE_SUFFIX} file")
    rollout_id = source.name.removesuffix(_CAPTURE_SUFFIX)
    store = CaptureStore(out_dir, capture_format="delta")
    if store.path_for(rollout_id).exists():
        raise FileExistsError(f"{store.path_for(rollout_id)} exists; convert into an empty directory")

    started = time.monotonic()
    lines = skipped = 0
    with source.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                exchange = orjson.loads(stripped)
            except orjson.JSONDecodeError:
                exchange = None
            if not isinstance(exchange, dict):
                # A damaged line cannot be re-encoded; copied as it is, it keeps every later line's capture index.
                skipped += 1
                with store.path_for(rollout_id).open("ab") as output:
                    output.write(stripped + b"\n")
            elif _DELTA_LINE_KEY in exchange:
                # Its rebuilt exchange could never equal the stored line, so every line would read as a mismatch.
                raise ValueError(
                    f"{source} line {lines} is already in the delta format; convert a full-format capture"
                )
            else:
                store.record(rollout_id, exchange)
            lines += 1
    convert_s = time.monotonic() - started
    convert_peak = _peak_rss_bytes()

    started = time.monotonic()
    equal = identical = 0
    mismatches: list[int] = []
    with source.open("rb") as handle:
        originals = (line.strip() for line in handle if line.strip())
        for (index, rebuilt), original in zip(store.iter_available(rollout_id), originals, strict=True):
            if rebuilt is None:
                if _parses_to_dict(original):
                    mismatches.append(index)
                continue
            if rebuilt == orjson.loads(original):
                equal += 1
                identical += orjson.dumps(rebuilt) == original
            else:
                mismatches.append(index)
    verify_s = time.monotonic() - started

    blob_bytes, blob_files = _tree_bytes(store.blob_dir_for(rollout_id))
    line_bytes = store.path_for(rollout_id).stat().st_size
    source_bytes = source.stat().st_size
    return {
        "rollout_id": rollout_id,
        "lines": lines,
        "unreadable_source_lines": skipped,
        "verified_equal": equal,
        "verified_byte_identical": identical,
        "mismatched_lines": mismatches[:20],
        "mismatch_count": len(mismatches),
        "full_bytes": source_bytes,
        "delta_line_bytes": line_bytes,
        "delta_blob_bytes": blob_bytes,
        "delta_blob_files": blob_files,
        "delta_total_bytes": line_bytes + blob_bytes,
        "ratio_full_to_delta": round(source_bytes / max(1, line_bytes + blob_bytes), 2),
        "convert_seconds": round(convert_s, 1),
        "verify_seconds": round(verify_s, 1),
        "peak_rss_after_convert_bytes": convert_peak,
        "peak_rss_after_verify_bytes": _peak_rss_bytes(),
    }


def _parses_to_dict(line: bytes) -> bool:
    try:
        return isinstance(orjson.loads(line), dict)
    except orjson.JSONDecodeError:
        return False


def merge(capture_dir: Path, rollout_id: str, payloads: str) -> dict[str, Any]:
    """Time the finished-rollout merge of one capture; return the report."""
    record = {"_ng_rollout_id": rollout_id, "_ng_task_index": 0, "_ng_rollout_index": 0}
    started = time.monotonic()
    merge_model_call_capture_into_record(record, [capture_dir], payloads=payloads)
    elapsed = time.monotonic() - started
    capture = record.get("ng_model_call_capture") or {}
    return {
        "rollout_id": rollout_id,
        "payloads": payloads,
        "calls": len(capture.get("calls") or []),
        "gaps": capture.get("gaps") or [],
        "record_bytes": len(orjson.dumps(record, default=str)),
        "merge_seconds": round(elapsed, 1),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert", help="Write a full-format capture as delta and verify it.")
    convert_parser.add_argument("source", type=Path, help="A <rollout_id>.capture.jsonl file in the full format.")
    convert_parser.add_argument("--out-dir", type=Path, required=True, help="Where to write the delta capture.")
    merge_parser = commands.add_parser("merge", help="Time the finished-rollout merge of one capture.")
    merge_parser.add_argument("capture_dir", type=Path, help="The capture directory, in either format.")
    merge_parser.add_argument("rollout_id")
    merge_parser.add_argument("--payloads", choices=("inline", "reference", "none"), default="inline")
    args = parser.parse_args(argv)
    if args.command == "convert":
        report = convert(args.source, args.out_dir)
        ok = report["mismatch_count"] == 0
    else:
        report = merge(args.capture_dir, args.rollout_id, args.payloads)
        ok = True
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
