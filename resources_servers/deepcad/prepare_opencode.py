# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare existing CAD image tasks for stock session-based coding harnesses."""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.deepcad.dataset.augment_inputs import view_prompt
from resources_servers.deepcad.task_data import TaskData


INSTRUCTIONS = """Reconstruct the target CAD geometry by writing and editing /workspace/model.py.
The final model.py is the ONLY artifact graded; prose, code fences and preview geometry are not graded.
Explicitly import cadquery as cq and assign the final valid solid to result.
Inspect all target images. Matching target files are also available as /workspace/target-0.png, target-1.png, etc.
Render your current code with bash:
python /workspace/.cad/cad_render.py /workspace/model.py --output /workspace/previews
Open /workspace/previews/preview.png with the read tool to inspect the rendered image, then revise model.py as needed.
Render and inspect your work before finishing. A renderer error is feedback for repairing your code.
No reference STEP or reference source is provided. Leave your final complete program in model.py.
Do not delegate or compact the conversation.
"""


def prepare_row(row: dict) -> dict:
    """Convert one materialized image task without changing the caller's sampling fields."""
    row = deepcopy(row)
    metadata = TaskData.model_validate(row["verifier_metadata"])
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    system, content = [], []
    for message in params.input:
        role = getattr(message, "role", None)
        if role not in ("system", "developer", "user"):
            raise ValueError("Prepared CAD rows require system/developer/user messages")
        parts = (
            [{"type": "input_text", "text": message.content}]
            if isinstance(message.content, str)
            else [
                part.model_dump(exclude_none=True) if hasattr(part, "model_dump") else dict(part)
                for part in message.content
            ]
        )
        for part in parts:
            if part["type"] == "input_text":
                text = re.sub(r"Output ONLY[^\n]*(?:\n-[^\n]*)*", "", part["text"]).strip()
                if role == "user":
                    content.append({"type": "input_text", "text": text})
                elif text:
                    system.append(text)
            elif role == "user" and part["type"] == "input_image":
                content.append(part)
            else:
                raise ValueError("CAD tasks support user images and text only")
    if not any(part["type"] == "input_image" for part in content):
        raise ValueError("Prepared CAD tasks require target images")
    views = view_prompt(metadata.view_mode, engineering_annotations=metadata.engineering_annotations)
    row["responses_create_params"]["input"] = [
        {"role": "system", "content": "\n".join([INSTRUCTIONS, views.partition("Output ONLY")[0], *system])},
        {"role": "user", "content": content},
    ]
    row["target_images"] = [part["image_url"] for part in content if part["type"] == "input_image"]
    row["task_source"] = "deepcad_opencode"
    return row


def prepare(source: Path, output: Path) -> int:
    """Validate prepared image rows, add workspace instructions, and route to the CAD resources instance."""
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output manifest")
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as src, output.open("x") as dst:
        for line in src:
            row = deepcopy(json.loads(line))
            if "deepcad_lazy" in row:
                from resources_servers.deepcad.dataset.lazy_inputs import LazyRenderConfig

                metadata = TaskData.model_validate(row["verifier_metadata"])
                config = LazyRenderConfig.model_validate(row["deepcad_lazy"]["config"])
                record = row["deepcad_lazy"]["record"]
                if (
                    record["task_id"] != metadata.task_id
                    or Path(record["reference_step"]).name != metadata.reference_step
                    or not record["source_sha256"]
                ):
                    raise ValueError("Lazy CAD record does not match its task metadata")
                config.harness = "opencode"
                row["deepcad_lazy"]["config"] = config.model_dump()
                row["task_source"] = "deepcad_opencode"
            else:
                row = prepare_row(row)
            dst.write(json.dumps(row) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Prepared {prepare(args.source, args.output)} CAD tasks")
