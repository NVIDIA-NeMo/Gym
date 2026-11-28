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
"""Replay finished rollout rows through ``row_media: reference`` and check that each comes back exactly.

``convert`` reads one row of a ``rollouts.jsonl`` (by ``--offset``/``--length``, or by ``--line``, streaming to it),
references its media into a blob directory under ``--out-dir`` as collection would, appends the new row to
``--out-dir``/rollouts.jsonl, restores it from the blobs and checks that the restored row serializes to the same
bytes as the original. The source is only read. With ``--capture-dir`` it also counts how many of the row's images
the run's delta capture already holds under ``blobs/<rollout_id>/``, which a row written with that capture dir
would not store again. It prints one JSON report: sizes, image counts, timings, and the process's current and peak
RSS after each phase.

Run it under a memory cap; rows of long sessions reach gigabytes::

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --unit=row-media-replay \\
        python scripts/replay_row_media.py convert /runs/x/rollouts.jsonl --line 3 --out-dir /tmp/row-media
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Optional

import orjson

from nemo_gym.core import maybe_rollout_id_from_run_body
from nemo_gym.trajectory import NG_ROW_MEDIA_KEY, reference_row_media, restore_row_media, rollouts_media_location


_CHUNK = 64 << 20


def _current_rss_bytes() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _high_water_rss_bytes() -> int:
    """The process's peak RSS since the last reset (``VmHWM``)."""
    with open("/proc/self/status") as handle:
        for entry in handle:
            if entry.startswith("VmHWM:"):
                return int(entry.split()[1]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _start() -> tuple[float, int]:
    """Begin a phase: reset the peak RSS counter, so the phase's own peak can be read at its end."""
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
    except OSError:
        pass
    return time.monotonic(), _current_rss_bytes()


def _phase(report: dict[str, Any], name: str, started: tuple[float, int]) -> None:
    peak = _high_water_rss_bytes()
    report["peak_rss_bytes"] = max(report.get("peak_rss_bytes", 0), peak)
    report.setdefault("phases", {})[name] = {
        "seconds": round(time.monotonic() - started[0], 3),
        "rss_before_bytes": started[1],
        "rss_after_bytes": _current_rss_bytes(),
        "peak_rss_bytes": peak,
    }


def _read_line(source: Path, offset: Optional[int], length: Optional[int], line: Optional[int]) -> bytearray:
    """One line's bytes without its newline; pages read are dropped from the page cache."""
    fd = os.open(source, os.O_RDONLY)
    try:
        if offset is None or length is None:
            if line is None:
                raise ValueError("give --offset and --length, or --line")
            offset, length = _locate_line(fd, line)
        # One read returns at most about 2 GiB, so fill the buffer in chunks.
        raw = bytearray(length)
        view = memoryview(raw)
        done = 0
        while done < length:
            count = os.preadv(fd, [view[done : done + _CHUNK]], offset + done)
            if count == 0:
                raise ValueError(f"read {done} of {length} bytes at offset {offset}")
            done += count
        view.release()
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    if raw.endswith(b"\n"):
        del raw[-1:]
    return raw


def _locate_line(fd: int, wanted: int) -> tuple[int, int]:
    size = os.fstat(fd).st_size
    index = start = position = 0
    while position < size:
        buffer = os.pread(fd, min(_CHUNK, size - position), position)
        newline = buffer.find(b"\n")
        while newline != -1:
            end = position + newline + 1
            if index == wanted:
                return start, end - start
            index += 1
            start = end
            newline = buffer.find(b"\n", newline + 1)
        os.posix_fadvise(fd, position, len(buffer), os.POSIX_FADV_DONTNEED)
        position += len(buffer)
    raise ValueError(f"the file has no complete line {wanted}")


def convert(
    source: Path,
    out_dir: Path,
    *,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    line: Optional[int] = None,
    capture_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Reference one row's media under ``out_dir``, restore it, and verify it; return the report."""
    report: dict[str, Any] = {"source": str(source), "line": line, "offset": offset}
    started = _start()
    raw = _read_line(source, offset, length, line)
    original_bytes = len(raw)
    original_sha = hashlib.sha256(raw).hexdigest()
    row = orjson.loads(raw)
    del raw
    # Rows are written by orjson, so a parsed row serializes back to its line; the check below relies on it.
    report["orjson_round_trip_identical"] = hashlib.sha256(orjson.dumps(row)).hexdigest() == original_sha
    _phase(report, "read_and_parse", started)

    rollout_id = maybe_rollout_id_from_run_body(row)
    if rollout_id is None:
        raise ValueError("the row names no rollout id")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = out_dir / "rollouts.jsonl"
    location = rollouts_media_location(rows_out, rollout_id)

    started = _start()
    stats = reference_row_media(row, location)
    _phase(report, "reference", started)

    started = _start()
    new_line = orjson.dumps(row)
    with rows_out.open("ab") as handle:
        handle.write(new_line + b"\n")
    _phase(report, "serialize_referenced", started)

    blob_files = (
        [path for path in location.directory.iterdir() if path.is_file()] if location.directory.is_dir() else []
    )
    references = row.get(NG_ROW_MEDIA_KEY) or {}
    digests = {digest for _, digest in references.get("blobs", [])}
    report.update(
        rollout_id=rollout_id,
        original_row_bytes=original_bytes,
        referenced_row_bytes=len(new_line),
        row_media_listing_bytes=len(orjson.dumps(references)) if references else 0,
        image_references=stats.references,
        unique_images=stats.unique,
        referenced_chars=stats.referenced_chars,
        blobs_written=stats.written_blobs,
        blob_bytes_written=stats.written_bytes,
        blob_dir_bytes=sum(path.stat().st_size for path in blob_files if path.name in digests),
        reference_paths_by_key=_paths_by_key(references.get("blobs", [])),
    )
    if capture_dir is not None:
        held = [digest for digest in digests if (capture_dir / "blobs" / rollout_id / digest).is_file()]
        missing_bytes = sum(
            (location.directory / digest).stat().st_size for digest in digests if digest not in set(held)
        )
        report["capture_blobs_already_held"] = {
            "capture_dir": str(capture_dir),
            "unique_images": len(digests),
            "held_by_capture": len(held),
            "bytes_a_capture_root_row_would_add": missing_bytes,
        }
    del new_line

    started = _start()
    restored = restore_row_media(row, rollouts_dir=out_dir, in_place=True)
    _phase(report, "restore", started)

    started = _start()
    restored_line = orjson.dumps(restored)
    report["restored_bytes"] = len(restored_line)
    report["restored_identical"] = hashlib.sha256(restored_line).hexdigest() == original_sha
    del restored_line, restored, row
    _phase(report, "verify", started)
    return report


def _paths_by_key(entries: list[Any]) -> dict[str, int]:
    """How many references fall under each place, with list indexes folded: where the copies were."""
    counts: dict[str, int] = {}
    for path, _ in entries:
        key = "/".join("[]" if isinstance(step, int) else step for step in path)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert", help="reference one row's media, restore it and verify it")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("--out-dir", type=Path, required=True)
    convert_parser.add_argument("--offset", type=int)
    convert_parser.add_argument("--length", type=int)
    convert_parser.add_argument("--line", type=int)
    convert_parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args(argv)
    report = convert(
        args.source,
        args.out_dir,
        offset=args.offset,
        length=args.length,
        line=args.line,
        capture_dir=args.capture_dir,
    )
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if report["restored_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
