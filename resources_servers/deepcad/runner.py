# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run only inside a disposable sandbox. No target CAD is available here."""

import math
import sys
from pathlib import Path

import cadquery as cq


def main() -> None:
    # Strict imports remain the default; opt in to the original training behavior.
    namespace = {"__name__": "__main__"}
    if "--preload-cq" in sys.argv:
        namespace["cq"] = cq
    code = Path("/sandbox/model.py").read_text()
    exec(compile(code, "model.py", "exec"), namespace)
    result = namespace.get("result")
    if isinstance(result, cq.Workplane):
        solids = [solid for value in result.vals() if isinstance(value, cq.Shape) for solid in value.Solids()]
        result = cq.Compound.makeCompound(solids)
    if not isinstance(result, cq.Shape) or not result.Solids() or not result.isValid():
        raise ValueError("Assign valid CadQuery solid geometry to result")
    volume = result.Volume()
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("result must have positive finite volume")
    cq.exporters.export(result, "/sandbox/prediction.step")


if __name__ == "__main__":
    main()
