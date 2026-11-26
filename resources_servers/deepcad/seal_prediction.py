# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run as root with isolated Python in the grading sandbox, after generated code exits."""

import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path

import cadquery as cq


INPUT_PATH = "/sandbox/prediction.step"


def main() -> None:
    """Snapshot a bounded regular STEP, fuse its solids, and seal the output from model writes."""
    maximum = int(sys.argv[1])
    # The generated process can have left background writers or replaced its output with a symlink.
    with os.fdopen(os.open(INPUT_PATH, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise ValueError("Missing or oversized regular STEP artifact")
        data = source.read(maximum + 1)
    if not 0 < len(data) <= maximum:
        raise ValueError("STEP artifact grew beyond max_step_bytes")
    directory = Path(tempfile.mkdtemp(prefix="deepcad-sealed-", dir="/tmp"))
    snapshot = directory / "input.step"
    snapshot.write_bytes(data)
    imported = cq.importers.importStep(str(snapshot))
    solids = [solid for value in imported.vals() for solid in value.Solids()]
    if not solids:
        raise ValueError("STEP artifact must contain solids")
    result = solids[0].fuse(*solids[1:]) if len(solids) > 1 else solids[0]
    volume = result.Volume()
    if not result.isValid() or not math.isfinite(volume) or volume <= 0:
        raise ValueError("STEP artifact must have a valid positive-volume solid union")
    output = directory / "prediction.step"
    cq.exporters.export(result, str(output))
    if not 0 < output.stat().st_size <= maximum:
        raise ValueError("Fused STEP artifact exceeds max_step_bytes")
    print(json.dumps({"path": str(output)}))


if __name__ == "__main__":
    main()
