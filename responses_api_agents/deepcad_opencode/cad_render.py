# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-only render helper; users may also import render_model from their scripts."""

import argparse
import json
import math
import runpy
from pathlib import Path


def render_model(model: Path, output: Path, *, view_mode: str, color: str) -> list[Path]:
    import cadquery as cq
    from PIL import Image

    from resources_servers.deepcad.dataset.augment_inputs import colorize_image
    from resources_servers.deepcad.dataset.prepare_gym import render_diagonal_views, render_views

    # run_path has its own globals: importing cq here does not supply it to model.py.
    result = runpy.run_path(str(model), run_name="__main__").get("result")
    if isinstance(result, cq.Workplane):
        result = cq.Compound.makeCompound([s for v in result.vals() if isinstance(v, cq.Shape) for s in v.Solids()])
    if not isinstance(result, cq.Shape) or not result.Solids() or not result.isValid():
        raise ValueError("model.py must import cadquery and assign a valid solid to result")
    if not math.isfinite(result.Volume()) or result.Volume() <= 0:
        raise ValueError("result must have positive finite volume")
    images = (
        render_diagonal_views(result, output)
        if view_mode == "diagonal"
        else render_views(result, output, engineering_annotations=False)
    )
    for path in images.values():
        with Image.open(path) as image:
            colored = colorize_image(image, color=color)
        colored.save(path)
    return list(images.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/workspace/previews"))
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("settings.json").read_text())
    print(
        json.dumps(
            [
                str(p)
                for p in render_model(
                    args.model, args.output, view_mode=settings["view_mode"], color=settings["color"]
                )
            ]
        )
    )
