# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Original visual-prefix and final-answer instructions from https://github.com/katerynaCh/Gym,
# 168e95ee80780737273dda011d31d1d5c58f35ce, Apache-2.0,
# responses_api_agents/opencode_multimodal_agent/{app.py,prompts/visual_prefix.txt}.
"""Explicit original AGA training subsets; no public benchmark or gold substitution."""

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import tempfile
from pathlib import Path, PurePosixPath

from PIL import Image

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.aga_visual_tasks.media import stage_inputs


HARNESSES = ("opencode", "pi", "codex", "claude_code", "openclaw")


def native_row(
    row: dict, family: str, image_root: Path, *, include_image_paths: bool = False, file_input: bool = False
) -> dict:
    """Keep original task/scoring fields and original image bytes, stage public files only."""
    if family == "code_assisted_vqa":
        if row.get("source_task") != "opencv_visual_question_answering":
            raise ValueError("Not an explicit original Code-assisted VQA row")
    elif family == "agent_x":
        if row.get("metadata", {}).get("category") != "agentx":
            raise ValueError("Not an explicit original Agent-X row")
    else:
        raise ValueError("Unknown AGA training family")
    result = copy.deepcopy(row)
    params = result["responses_create_params"]
    messages = params["input"]
    if [message.get("role") for message in messages] not in (["user"], ["system", "user"]):
        raise ValueError("Native AGA preparation requires original single-user task input")
    if params.get("tools"):
        raise ValueError("Do not silently replace original declared tools")
    params["model"] = None
    result["aga_family"] = family
    images = {}
    root = image_root.resolve()
    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "input_text", "text": message["content"]}]
        for item in message["content"]:
            if item["type"] != "input_image":
                continue
            original = item["image_url"]
            relative = PurePosixPath(original).relative_to("/images")
            source = (root / str(relative)).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Original image path escapes image root")
            data = source.read_bytes()
            mime = mimetypes.guess_type(source.name)[0]
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError("Unsupported original image MIME")
            images[original] = hashlib.sha256(data).hexdigest()
            item["image_url"] = f"data:{mime};base64," + base64.b64encode(data).decode()
    if len(images) > 33:
        raise ValueError("Original 33-image input limit exceeded")
    with tempfile.TemporaryDirectory(prefix="aga-original-images-") as directory:
        _, _, staged = stage_inputs(
            NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
            Path(directory),
            max_source_pixels=150_000_000,
            max_images=33,
        )
        staged_iter = iter(staged)
        for message in messages:
            for item in message["content"]:
                if item["type"] == "input_image":
                    staged_path = next(staged_iter)
                    data = staged_path.read_bytes()
                    encoded = base64.b64encode(data).decode()
                    with Image.open(staged_path) as image:
                        if max(image.size) > 8000 or len(encoded) > 1073741824:
                            raise ValueError(
                                "Original Photon image resizing is required; do not substitute different pixels"
                            )
                    item["image_url"] = "data:image/png;base64," + encoded
    messages[-1]["content"].insert(
        0, {"type": "input_text", "text": Path(__file__).with_name("visual_prefix.txt").read_text()}
    )
    result["source_images"] = images
    if include_image_paths:
        messages[-1]["content"].append(
            {
                "type": "input_text",
                "text": "Original input image files, in first appearance order:\n"
                + "\n".join(f"{index}. {path}" for index, path in enumerate(images, start=1)),
            }
        )
    if file_input:
        # A harness whose prompt takes only text (Claude Code) reads the seeded /images files named by the note above.
        if not include_image_paths:
            raise ValueError("file_input needs include_image_paths, so the prompt names the image files")
        for message in messages:
            message["content"] = [item for item in message["content"] if item["type"] != "input_image"]
    result["_ng_model_sampling"] = {"max_output_tokens": 49152}
    for key in ("temperature", "top_p"):
        if params.get(key) is not None:
            result["_ng_model_sampling"][key] = params.pop(key)
    if params.get("max_output_tokens") not in (None, 49152):
        raise ValueError("Unexpected original output cap; do not replace it")
    params.pop("max_output_tokens", None)
    # The original OpenCode adapter only echoed this field in its response; it
    # never controlled CLI execution. Each native harness owns tool concurrency.
    params.pop("parallel_tool_calls", None)
    return result


def prepare(
    source: Path,
    output: Path,
    family: str,
    image_root: Path,
    harness: str,
    *,
    include_image_paths: bool = False,
    file_input: bool = False,
) -> Path:
    """Write the original rows, routed to ``aga_<harness>_agent``; ``configs/multiharness.yaml`` holds the servers."""
    if harness not in HARNESSES:
        raise ValueError("Unknown native harness")
    if harness == "claude_code" and not file_input:
        raise ValueError("Claude Code takes only text in its prompt; use --file-input")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = native_row(
            json.loads(line),
            family,
            image_root,
            include_image_paths=include_image_paths or file_input,
            file_input=file_input,
        )
        row["agent_ref"] = {"type": "responses_api_agents", "name": f"aga_{harness}_agent"}
        row["_ng_environment_server"] = f"aga_{harness}_environment_server"
        rows.append(row)
    if not rows:
        raise ValueError("Original training subset is empty")
    data = output / "train.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "family": family,
                "rows": len(rows),
                "harness": harness,
                **({"input_delivery": "inline_images_with_original_file_paths"} if include_image_paths else {}),
            },
            indent=2,
        )
        + "\n"
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("code_assisted_vqa", "agent_x"), required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--harness", choices=HARNESSES, required=True)
    parser.add_argument(
        "--file-input",
        action="store_true",
        help="Replace the inline images with the original /images paths, for a harness whose prompt takes only text.",
    )
    parser.add_argument(
        "--include-image-paths",
        action="store_true",
        help="Append public source image paths to the user message; records a separate input-delivery protocol.",
    )
    prepare(**vars(parser.parse_args()))
