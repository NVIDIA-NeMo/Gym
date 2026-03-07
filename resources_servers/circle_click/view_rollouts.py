# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

def _decode_image(data_url: str) -> Image.Image:
    _, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _to_int(val) -> int | None:
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, list) and len(val) > 0:
        return int(val[0])
    return None


def _extract_click(row: dict) -> tuple[int | None, int | None]:
    if row.get("clicked_x") is not None:
        return row["clicked_x"], row["clicked_y"]
    for item in row.get("response", {}).get("output", []):
        if item.get("type") == "function_call" and item.get("name") == "click":
            try:
                args = json.loads(item["arguments"])
                return _to_int(args.get("x")), _to_int(args.get("y"))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
    return None, None


def _extract_image(row: dict) -> str | None:
    for msg in row.get("responses_create_params", {}).get("input", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "input_image":
                    return part.get("image_url")
    return None


def _load_inputs(materialized_path: Path) -> dict[int, dict]:
    inputs = {}
    with materialized_path.open() as f:
        for line in f:
            row = json.loads(line)
            inputs[row["_ng_task_index"]] = row
    return inputs


def _draw(rollout: dict, task_input: dict, idx: int, out_dir: Path) -> None:
    image_url = _extract_image(task_input) or _extract_image(rollout)
    if not image_url:
        return

    img = _decode_image(image_url)
    draw = ImageDraw.Draw(img)

    target_color = task_input.get("target_color", "?")
    circles = task_input.get("circles", [])
    clicked_x, clicked_y = _extract_click(rollout)
    reward = rollout.get("reward", 0.0)

    for circle in circles:
        if circle["color"] == target_color:
            cx, cy, r = circle["x"], circle["y"], circle["radius"]
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="white", width=3)

    if clicked_x is not None and clicked_y is not None:
        size = 8
        color = "lime" if reward > 0 else "red"
        draw.line([(clicked_x - size, clicked_y), (clicked_x + size, clicked_y)], fill=color, width=3)
        draw.line([(clicked_x, clicked_y - size), (clicked_x, clicked_y + size)], fill=color, width=3)

    label = f"target={target_color}  click=({clicked_x},{clicked_y})  reward={reward}"
    draw.text((5, 5), label, fill="black")

    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"rollout_{idx:04d}_r{reward:.0f}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts_jsonl", type=str)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=str, default=".")
    args = parser.parse_args()

    rollouts_path = Path(args.rollouts_jsonl)
    materialized_path = rollouts_path.parent / (rollouts_path.stem + "_materialized_inputs.jsonl")

    inputs = {}
    if materialized_path.exists():
        inputs = _load_inputs(materialized_path)

    out_dir = Path(args.out)
    count = 0
    with rollouts_path.open() as f:
        for i, line in enumerate(f):
            if count >= args.n:
                break
            rollout = json.loads(line)
            task_idx = rollout.get("_ng_task_index", i)
            task_input = inputs.get(task_idx, rollout)
            _draw(rollout, task_input, i, out_dir)
            count += 1

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
