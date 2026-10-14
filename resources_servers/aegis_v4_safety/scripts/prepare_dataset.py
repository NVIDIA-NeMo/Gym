# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert simple prompt/image/response JSONL files to Aegis v4 Gym inputs."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Optional


DEFAULT_AGENT_NAME = "aegis_v4_safety_simple_agent"


def _value_at(row: dict[str, Any], field: Optional[str]) -> Any:
    if not field:
        return None
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _image_url(value: str, *, image_root: Optional[Path], embed_images: bool) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value)
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    if not embed_images:
        return path.as_uri()

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def make_task_row(
    row: dict[str, Any],
    *,
    row_index: int,
    prompt_field: str,
    image_field: Optional[str],
    id_field: Optional[str],
    dataset_name: Optional[str],
    system_prompt: Optional[str],
    image_root: Optional[Path],
    embed_images: bool,
    agent_name: str,
    materialized: bool,
) -> dict[str, Any]:
    prompt = _value_at(row, prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"row {row_index}: {prompt_field!r} must contain non-empty text")

    content: str | list[dict[str, Any]] = prompt.strip()
    image = _value_at(row, image_field)
    if image is not None:
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"row {row_index}: {image_field!r} must contain an image URL or path")
        content = [
            {
                "type": "input_image",
                "image_url": _image_url(image.strip(), image_root=image_root, embed_images=embed_images),
                "detail": "auto",
            },
            {"type": "input_text", "text": prompt.strip()},
        ]

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    sample_id = _value_at(row, id_field)
    task: dict[str, Any] = {
        "sample_id": row_index if sample_id is None else sample_id,
        "dataset_name": dataset_name,
        "metadata": {"source_row": row_index},
        "responses_create_params": {"input": messages},
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
    }
    if materialized:
        task["_ng_task_index"] = row_index
        task["_ng_rollout_index"] = 0
    return task


def make_rollout_row(task: dict[str, Any], response_text: str, *, model_name: str) -> dict[str, Any]:
    return {
        "_ng_task_index": task["_ng_task_index"],
        "_ng_rollout_index": task["_ng_rollout_index"],
        "agent_ref": task["agent_ref"],
        "response": {
            "id": f"existing_response_{task['_ng_task_index']}",
            "created_at": 0.0,
            "model": model_name,
            "object": "response",
            "output": [
                {
                    "id": f"existing_message_{task['_ng_task_index']}",
                    "content": [{"annotations": [], "text": response_text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def convert(args: argparse.Namespace) -> tuple[int, int]:
    source = Path(args.input)
    tasks_output = Path(args.output)
    rollouts_output = Path(args.rollouts_output) if args.rollouts_output else None
    if args.response_field and rollouts_output is None:
        raise ValueError("--rollouts-output is required when --response-field is set")
    if rollouts_output is not None and not args.response_field:
        raise ValueError("--response-field is required when --rollouts-output is set")

    tasks_output.parent.mkdir(parents=True, exist_ok=True)
    if rollouts_output is not None:
        rollouts_output.parent.mkdir(parents=True, exist_ok=True)

    task_count = 0
    rollout_count = 0
    with source.open(encoding="utf-8") as source_file, tasks_output.open("w", encoding="utf-8") as task_file:
        rollout_file = rollouts_output.open("w", encoding="utf-8") if rollouts_output is not None else None
        try:
            for source_line, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {source_line}: expected a JSON object")
                task = make_task_row(
                    row,
                    row_index=task_count,
                    prompt_field=args.prompt_field,
                    image_field=args.image_field,
                    id_field=args.id_field,
                    dataset_name=args.dataset_name,
                    system_prompt=args.system_prompt,
                    image_root=Path(args.image_root) if args.image_root else source.parent,
                    embed_images=args.embed_images,
                    agent_name=args.agent_name,
                    materialized=rollout_file is not None,
                )
                task_file.write(json.dumps(task, ensure_ascii=False) + "\n")
                task_count += 1

                if rollout_file is None:
                    continue
                response = _value_at(row, args.response_field)
                if not isinstance(response, str):
                    raise ValueError(f"line {source_line}: {args.response_field!r} must contain response text")
                rollout = make_rollout_row(task, response, model_name=args.existing_model_name)
                rollout_file.write(json.dumps(rollout, ensure_ascii=False) + "\n")
                rollout_count += 1
        finally:
            if rollout_file is not None:
                rollout_file.close()

    return task_count, rollout_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source JSONL file")
    parser.add_argument("--output", required=True, help="Gym task JSONL output")
    parser.add_argument("--prompt-field", default="prompt", help="Prompt field, including dotted paths")
    parser.add_argument("--image-field", help="Optional image URL/path field")
    parser.add_argument("--id-field", help="Optional source ID field")
    parser.add_argument("--dataset-name", help="Dataset name stored in each row")
    parser.add_argument("--system-prompt", help="Optional system message for the target model")
    parser.add_argument("--image-root", help="Base directory for relative image paths; defaults to source directory")
    parser.add_argument("--embed-images", action="store_true", help="Embed local images as data URLs")
    parser.add_argument("--response-field", help="Existing response field; enables score-only preparation")
    parser.add_argument("--rollouts-output", help="Gym rollouts JSONL for existing responses")
    parser.add_argument("--existing-model-name", default="existing-response", help="Model name stored in rollouts")
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME, help="Gym agent instance name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_count, rollout_count = convert(args)
    print(f"Wrote {task_count} task rows to {args.output}")
    if args.rollouts_output:
        print(f"Wrote {rollout_count} existing-response rows to {args.rollouts_output}")


if __name__ == "__main__":
    main()
