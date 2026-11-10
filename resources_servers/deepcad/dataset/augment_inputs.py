# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare fixed view-specific instructions and foreground colors for GRPO inputs."""

import argparse
import base64
import hashlib
import io
import json
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image


FOREGROUND_COLORS = {
    "blue": (45, 105, 170),
    "green": (45, 135, 80),
    "teal": (25, 135, 140),
    "steel_blue": (75, 105, 140),
    "violet": (115, 80, 155),
    "amber": (165, 105, 40),
}


def sample_foreground_color(task_id: str, *, seed: int = 42) -> str:
    """Choose one color independently of view choice, task order and worker count."""
    digest = hashlib.sha256(f"deepcad-colors:{seed}:{task_id}".encode()).digest()
    return tuple(FOREGROUND_COLORS)[int.from_bytes(digest[:8], "big") % len(FOREGROUND_COLORS)]


def sample_annotations(task_id: str, *, no_annotations_probability: float = 0.75, seed: int = 42) -> bool:
    """Fix annotation visibility once per example, independently of views and color."""
    if not 0 <= no_annotations_probability <= 1:
        raise ValueError("no_annotations_probability must be between 0 and 1")
    digest = hashlib.sha256(f"deepcad-annotations:{seed}:{task_id}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return no_annotations_probability != 1 and draw >= no_annotations_probability


def colorize_image(image: Image.Image, *, color: str) -> Image.Image:
    """Tint shading and annotations, keeping black edges and white background.

    The minimum RGB channel removes any previous surface hue. A continuous
    black/color/white ramp preserves shading and antialiased white boundaries.
    Composite gutters and background pixels stay exactly white.
    """
    rgba = image.convert("RGBA")
    opaque = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    rgb = np.asarray(opaque.convert("RGB"), dtype=float)
    gray = rgb.min(axis=2, keepdims=True)
    foreground = np.array(FOREGROUND_COLORS[color], dtype=float)
    tinted = np.where(
        gray <= 128,
        foreground * gray / 128,
        foreground + (255 - foreground) * (gray - 128) / 127,
    )
    return Image.fromarray(np.rint(tinted).astype(np.uint8))


@cache
def view_prompt(mode: str, *, engineering_annotations: bool = True) -> str:
    """Describe the actual three-image or four-view composite coordinate convention."""
    if mode == "diagonal":
        layout = "a 2x2 composite of 4 diagonal orthographic views"
        cameras = (
            "Views (camera directions relative to the part center, all looking at [0.5, 0.5, 0.5]):\n"
            "- Top-left:     [-1, -1, -1]\n"
            "- Top-right:    [ 1,  1,  1]\n"
            "- Bottom-left:  [ 1, -1,  1]\n"
            "- Bottom-right: [-1,  1, -1]"
            "\nAll diagonal cameras use world Z as up."
        )
        convention = (
            "Renders are normalized: bbox centered at [0.5, 0.5, 0.5], longest side maps to [0,1]. "
            "Use this normalized coordinate convention in your code."
        )
        if engineering_annotations:
            convention += " Supplied bounding-box dimensions in millimetres describe the physical proportions."
    elif mode == "axis":
        layout = "3 separate front, side, and top orthographic views"
        if engineering_annotations:
            layout = "3 separate dimensioned front, side, and top orthographic views"
        cameras = (
            "Views (all looking at the part center):\n"
            "- Front: camera on -Y, looking along +Y; X right, Z up\n"
            "- Side: camera on +X, looking along -X; Y left, Z up\n"
            "- Top: camera on +Z, looking along -Z; X right, Y up"
        )
        convention = "All views have the same orthographic scale and are centered on the bounding box."
        if engineering_annotations:
            convention += (
                " Use millimetres in your code. Dimension annotations are in millimetres: "
                "R denotes radius and depth denotes rectangular additive extrusion depth."
            )
        else:
            convention += " Infer proportions from the images and use any consistent length unit in your code."
    else:
        raise ValueError(f"Unknown view mode: {mode}")
    prompt = (
        (Path(__file__).parent / "gym-prompt.txt")
        .read_text()
        .strip()
        .format(layout=layout, cameras=cameras, convention=convention)
    )
    return prompt


def augment_inputs(row: dict, *, seed: int = 42) -> dict:
    """Bake a shared color and matching prompt into a row; never sample at runtime.

    Reapplying the same seed preserves the image bytes. Changing a saved color
    requires an uncolored source so repeated tinting cannot degrade the renders.
    """
    metadata = row["verifier_metadata"]
    mode = metadata.get("view_mode", "axis")
    color = sample_foreground_color(metadata["task_id"], seed=seed)
    previous_color = metadata.get("foreground_color")
    if previous_color is not None and previous_color != color:
        raise ValueError("Use an original uncolored dataset to change the foreground color seed")
    messages = row["responses_create_params"]["input"]
    system = next(message for message in messages if message["role"] == "system")
    user = next(message for message in messages if message["role"] == "user")
    positions = [index for index, item in enumerate(user["content"]) if item["type"] == "input_image"]
    if len(positions) != (1 if mode == "diagonal" else 3):
        raise ValueError(f"Image count does not match {mode} view mode")
    annotations = metadata.get("engineering_annotations", True)
    system["content"] = view_prompt(mode, engineering_annotations=annotations)
    description = user["content"][0]
    dimensions = description["text"].partition("Overall bounding-box dimensions in mm (X,Y,Z): ")[2]
    description["text"] = "Reproduce the geometry shown in the following views."
    if annotations and dimensions:
        description["text"] += " Overall bounding-box dimensions in mm (X,Y,Z): " + dimensions
    labels = (
        ["Four diagonal views (2x2 composite):"] if mode == "diagonal" else ["Front view:", "Side view:", "Top view:"]
    )
    for index, label in zip(positions, labels, strict=True):
        item = user["content"][index]
        user["content"][index - 1]["text"] = label
        if previous_color is not None:
            continue
        with Image.open(io.BytesIO(base64.b64decode(item["image_url"].split(",", 1)[1]))) as image:
            output = io.BytesIO()
            colorize_image(image, color=color).save(output, format="PNG")
        item["image_url"] = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    metadata["foreground_color"] = color
    metadata["input_augmentation_seed"] = seed
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output to preserve the original input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    colors = Counter()
    views = Counter()
    with args.input.open() as source, temporary.open("w") as output:
        for line in source:
            if not line.strip():
                continue
            row = augment_inputs(json.loads(line), seed=args.seed)
            output.write(json.dumps(row) + "\n")
            colors[row["verifier_metadata"]["foreground_color"]] += 1
            views[row["verifier_metadata"].get("view_mode", "axis")] += 1
    temporary.replace(args.output)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_augmentation_seed": args.seed,
        "foreground_colors": FOREGROUND_COLORS,
        "color_counts": dict(colors),
        "view_counts": dict(views),
        "written": sum(colors.values()),
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
