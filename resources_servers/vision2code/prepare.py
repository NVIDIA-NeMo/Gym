# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert an id/image_path/rubric JSONL manifest into single-turn Gym tasks."""

import argparse
import base64
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from time import monotonic

from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config import GlobalConfigDictParser
from resources_servers.vision2code.evaluation import RubricName
from resources_servers.vision2code.prompts import SYSTEM_PROMPT, USER_PROMPT


class PreparationConfig(BaseModel):
    """Optional additions to the base generation prompt, applied during preparation."""

    model_config = ConfigDict(extra="forbid")
    include_canvas_dimensions: bool = False
    include_rendering_instructions: bool = False
    include_library_instructions: bool = False


def load_preparation_config(path: Path) -> PreparationConfig:
    """Read prompt options from a composed environment config without resolving credentials."""
    _, configs = GlobalConfigDictParser().load_extra_config_paths([str(path)])
    config = OmegaConf.merge(*configs)
    options = OmegaConf.select(config, "vision2code.resources_servers.vision2code.preparation")
    return PreparationConfig.model_validate(OmegaConf.to_container(options, resolve=True) if options else {})


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    image_path: str
    rubric: RubricName
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def read_reference(path: Path) -> tuple[bytes, str, int, int]:
    """Validate a supported, single-frame reference without changing its pixels."""
    raw = path.read_bytes()
    with Image.open(BytesIO(raw)) as im:
        if im.format not in {"PNG", "JPEG", "WEBP", "GIF"} or getattr(im, "n_frames", 1) != 1:
            raise ValueError(f"Unsupported reference image: {path}")
        mime = Image.MIME[im.format]
        width, height = im.size
        im.verify()
    with Image.open(BytesIO(raw)) as im:
        im.load()
    return raw, mime, width, height


def prepare(
    manifest: Path,
    output: Path,
    *,
    include_canvas_dimensions: bool = False,
    include_rendering_instructions: bool = False,
    include_library_instructions: bool = False,
) -> int:
    """Validate the complete manifest before replacing the destination JSONL."""
    started = last_report = monotonic()
    print(f"Reading manifest: {manifest}", file=sys.stderr, flush=True)
    lines = manifest.read_text().splitlines()
    total = sum(bool(line.strip()) for line in lines)
    print(f"Validating and encoding {total} images...", file=sys.stderr, flush=True)
    rows = []
    seen = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        item = json.loads(line)
        sample_id = str(item["id"])
        if not sample_id or sample_id in seen:
            raise ValueError(f"Empty or duplicate id at line {number}: {sample_id!r}")
        seen.add(sample_id)
        path = Path(item["image_path"])
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve(strict=True)
        raw, mime, width, height = read_reference(path)
        ref = Reference(
            id=sample_id,
            image_path=str(path),
            rubric=item["rubric"],
            width=width,
            height=height,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        instruction = USER_PROMPT
        if include_canvas_dimensions:
            instruction += f" Target canvas: {width} x {height} pixels."
        if include_rendering_instructions:
            instruction += (
                f"\nThe target image is {width} pixels wide and {height} pixels tall. "
                "The launcher provides OUTPUT_PATH='/workspace/output.png', TARGET_WIDTH, and TARGET_HEIGHT "
                "as Python globals and environment variables. Save a PNG at OUTPUT_PATH with exactly these dimensions. "
                "The renderer has no network access and does not contain the reference image."
            )
        if include_library_instructions:
            instruction += (
                "\nAvailable libraries: matplotlib, NumPy, Pillow, pandas, NetworkX, RDKit, Schemdraw, SciPy, "
                "Seaborn, Shapely, svgwrite, CairoSVG, ReportLab; DejaVu and Liberation fonts."
            )
        rows.append(
            {
                "id": sample_id,
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "detail": "auto",
                                    "image_url": f"data:{mime};base64,{base64.b64encode(raw).decode()}",
                                },
                                {"type": "input_text", "text": instruction},
                            ],
                        },
                    ],
                    "tools": [],
                },
                "verifier_metadata": ref.model_dump(),
            }
        )
        completed = len(rows)
        now = monotonic()
        if completed == 1 or completed % 100 == 0 or completed == total or now - last_report >= 5:
            elapsed = now - started
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / rate
            print(
                f"Images {completed}/{total} ({completed / total:.1%}) | "
                f"elapsed {elapsed:.1f}s | {rate:.2f} images/s | ETA {eta:.0f}s | last: {path.name}",
                file=sys.stderr,
                flush=True,
            )
            last_report = now
    if not rows:
        raise ValueError("Manifest is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Serializing and writing {len(rows)} tasks to {output}...", file=sys.stderr, flush=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Finished in {monotonic() - started:.1f}s", file=sys.stderr, flush=True)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, help="Environment YAML containing preparation prompt options.")
    parser.add_argument(
        "--include-canvas-dimensions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append only the target canvas dimensions.",
    )
    parser.add_argument(
        "--include-rendering-instructions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append dimensions and sandbox/output instructions.",
    )
    parser.add_argument(
        "--include-library-instructions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append the available rendering libraries and fonts.",
    )
    args = parser.parse_args()
    options = load_preparation_config(args.config) if args.config else PreparationConfig()
    overrides = {name: value for name in PreparationConfig.model_fields if (value := getattr(args, name)) is not None}
    options = options.model_copy(update=overrides)
    count = prepare(
        args.manifest,
        args.output,
        **options.model_dump(),
    )
    print(f"Prepared {count} tasks in {args.output}")


if __name__ == "__main__":
    main()
