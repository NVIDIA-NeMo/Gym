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
import random
from pathlib import Path

from PIL import Image, ImageDraw


COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (133, 153, 0),
    "yellow": (181, 137, 0),
    "purple": (108, 113, 196),
    "orange": (203, 75, 22),
    "cyan": (42, 161, 152),
    "pink": (211, 54, 130),
}

SYSTEM_PROMPT = (
    "You are a visual assistant. Count the number of circles of the specified color in the image. "
    "Output your final answer in \\boxed{} format, e.g. \\boxed{3}."
)


def _place_circles(n: int, img_size: int, radius: int, rng: random.Random) -> list[dict]:
    margin = radius + 10
    circles = []
    for _ in range(n):
        for _ in range(500):
            x = rng.randint(margin, img_size - margin)
            y = rng.randint(margin, img_size - margin)
            if all(((x - c["x"]) ** 2 + (y - c["y"]) ** 2) ** 0.5 > 2 * radius + 15 for c in circles):
                circles.append({"x": x, "y": y})
                break
    return circles


def _generate_image(circles: list[dict], img_size: int, radius: int) -> str:
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for c in circles:
        r, g, b = COLORS[c["color"]]
        draw.ellipse([c["x"] - radius, c["y"] - radius, c["x"] + radius, c["y"] + radius], fill=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_example(
    seed: int,
    img_size_range: tuple[int, int] = (1000, 1000),
    circle_radius_range: tuple[int, int] = (30, 60),
    num_circles_range: tuple[int, int] = (5, 20),
    num_colors_range: tuple[int, int] = (2, 4),
) -> dict:
    rng = random.Random(seed)
    img_size = rng.randint(*img_size_range)
    radius = rng.randint(*circle_radius_range)
    num_circles = rng.randint(*num_circles_range)

    num_colors = rng.randint(*num_colors_range)
    palette = rng.sample(list(COLORS.keys()), min(num_colors, len(COLORS)))
    color_names = [rng.choice(palette) for _ in range(num_circles)]
    target_color = rng.choice(palette)

    positions = _place_circles(num_circles, img_size, radius, rng)
    circles = [{"x": p["x"], "y": p["y"], "radius": radius, "color": color_names[i]} for i, p in enumerate(positions)]

    image_url = _generate_image(circles, img_size, radius)

    user_text = f"How many {target_color} circles are in the image?"

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url, "detail": "auto"},
                        {"type": "input_text", "text": user_text},
                    ],
                },
            ],
        },
        "circles": circles,
        "target_color": target_color,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate circle count dataset.")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "data" / "example.jsonl"))
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--img-size-min", type=int, default=1000)
    parser.add_argument("--img-size-max", type=int, default=1000)
    parser.add_argument("--radius-min", type=int, default=30)
    parser.add_argument("--radius-max", type=int, default=60)
    parser.add_argument("--num-circles-min", type=int, default=5)
    parser.add_argument("--num-circles-max", type=int, default=20)
    parser.add_argument("--num-colors-min", type=int, default=2)
    parser.add_argument("--num-colors-max", type=int, default=4)
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for i in range(args.n):
            example = make_example(
                args.seed_offset + i,
                img_size_range=(args.img_size_min, args.img_size_max),
                circle_radius_range=(args.radius_min, args.radius_max),
                num_circles_range=(args.num_circles_min, args.num_circles_max),
                num_colors_range=(args.num_colors_min, args.num_colors_max),
            )
            f.write(json.dumps(example) + "\n")

    print(f"Generated {args.n} examples: {output_path}")


if __name__ == "__main__":
    main()
