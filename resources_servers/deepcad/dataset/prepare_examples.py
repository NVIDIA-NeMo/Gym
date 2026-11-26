# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate five synthetic CAD examples and reference solids, without a corpus download."""

import json
import tempfile
from pathlib import Path

import cadquery as cq

from resources_servers.deepcad.dataset.prepare_gym import dimension_data, render_views, task_row


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "data"
    references = directory / "references"
    references.mkdir(parents=True, exist_ok=True)
    # Reference lengths are metres, matching the preparation pipeline. Prompts use millimetres.
    shapes = {
        "box": cq.Workplane("XY").box(0.02, 0.03, 0.01),
        "cylinder": cq.Workplane("XY").circle(0.005).extrude(0.02),
        "sphere": cq.Workplane("XY").sphere(0.01),
        "tube": cq.Workplane("XY").circle(0.01).circle(0.006).extrude(0.02),
        "drilled-plate": cq.Workplane("XY").box(0.02, 0.03, 0.004).faces(">Z").workplane().hole(0.006),
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix="cad-examples-") as temporary:
        for name, workplane in shapes.items():
            shape = workplane.val()
            reference = references / f"{name}.step"
            cq.exporters.export(shape, str(reference))
            reference.write_text("".join(line.rstrip() + "\n" for line in reference.read_text().splitlines()))
            rows.append(
                task_row(
                    f"synthetic-{name}",
                    f"{name}.step",
                    render_views(shape, Path(temporary) / name),
                    metrics={"difficulty": 0},
                    dimensions_mm=dimension_data(shape)["size_mm"].tolist(),
                )
            )
    (directory / "example.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
