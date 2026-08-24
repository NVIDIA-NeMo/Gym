#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create deterministic assets and RL manifests for the requested VisGym tasks."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# Must match VisGymResourcesServer.DEFAULT_SHAPING_WEIGHT in ../app.py --
# duplicated rather than imported so this script stays free of the resources
# server's FastAPI/aiohttp dependencies. test_requested_env_data.py asserts
# the two stay equal.
DEFAULT_SHAPING_WEIGHT = 0.3

VISGYM_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = VISGYM_ROOT / "data"
ASSET_ROOT = DATA_ROOT / "requested_env_assets"
# Asset paths go into the manifests repo-relative, not as an absolute path
# into one deployment's container. The resource server resolves them at reset
# against NeMo-Gym's component search roots, so the same row works from a
# checkout, a code snapshot or a container mount.
CONTAINER_ASSET_ROOT = Path("resources_servers/visgym/data/requested_env_assets")


@dataclass(frozen=True)
class EnvironmentSpec:
    env_id: str
    grammar: str
    horizon_cap: int
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    seed_key: str | None = "seed"
    reward_shaping: dict[str, Any] | None = None
    seed_cycle: tuple[int, ...] = ()


MATCHSTICK_EASY_SEEDS = (
    1259,
    1289,
    1310,
    1319,
    1327,
    1341,
    1347,
    1368,
    1369,
    1396,
    1398,
    1420,
    1434,
    1455,
    1506,
    1538,
    1577,
    1600,
    1662,
    1680,
    1716,
    1720,
    1729,
    1761,
    1824,
    1841,
    1853,
    1868,
    1906,
    1911,
    1913,
    2023,
    2034,
    2070,
    2072,
    2086,
    2149,
    2184,
    2255,
    2260,
    2290,
    2323,
    2413,
    2419,
    2438,
    2444,
    2472,
    2510,
    2521,
    2563,
    2575,
    2620,
    2648,
    2664,
    2670,
    2687,
    2693,
    2963,
    2980,
    2994,
    3002,
    3015,
    3063,
    3072,
)


ENV_SPECS = {
    "matchstick_equation": EnvironmentSpec(
        env_id="matchstick_equation/easy",
        grammar=r"^(?:\('move',\s*\[[0-9]+,\s*[0-9]+,\s*[0-9]+,\s*[0-9]+\]\)|\('undo',\s*'undo'\)|\('stop',\s*'stop'\))$",
        horizon_cap=4,
        env_kwargs={"break_moves": 1, "enforce_min_distance": True},
        reward_shaping={"type": "distance_delta", "info_key": "matchstick_distance", "weight": DEFAULT_SHAPING_WEIGHT},
        seed_cycle=MATCHSTICK_EASY_SEEDS,
    ),
    "maze_3d": EnvironmentSpec(
        env_id="maze_3d/easy",
        grammar=r"^(?:\('move',\s*0\)|\('turn',\s*[1-3]\)|\('stop',\s*'stop'\))$",
        horizon_cap=8,
        env_kwargs={"maze_width": 5, "maze_height": 5, "render_size": [128, 128]},
        reward_shaping={"type": "distance_delta", "info_key": "distance", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "jigsaw": EnvironmentSpec(
        env_id="jigsaw/easy",
        grammar=r"^(?:\('swap',\s*\(\([0-1],\s*[0-1]\),\s*\([0-1],\s*[0-1]\)\)\)|\('reorder',\s*\[[0-3,\s]+\]\)|\('stop',\s*'stop'\))$",
        horizon_cap=4,
        env_kwargs={
            "num_rows": 2,
            "num_cols": 2,
            "sample_dir": str(CONTAINER_ASSET_ROOT / "images"),
        },
    ),
    "sliding_block": EnvironmentSpec(
        env_id="sliding_block/easy",
        grammar=r"^(?:\('move',\s*\([0-9]+,\s*[0-3]\)\)|\('stop',\s*'stop'\))$",
        horizon_cap=4,
        env_kwargs={"num_shuffle_moves": 4},
        reward_shaping={"type": "distance_delta", "info_key": "sliding_distance", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "colorization": EnvironmentSpec(
        env_id="colorization/easy",
        grammar=r"^(?:\('(?:rotate|saturate)',\s*-?[0-9]+\)|\('stop',\s*'stop'\))$",
        horizon_cap=6,
        env_kwargs={
            "sample_dir": str(CONTAINER_ASSET_ROOT / "colorization"),
            "circle_size": 10,
            "region_radius": 16,
            "accuracy_radius": 6,
            "img_size": 128,
            "max_steps": 25,
            "min_brightness": 25,
            "max_brightness": 230,
        },
        reward_shaping={"type": "distance_delta", "info_key": "color_distance", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "counting": EnvironmentSpec(
        env_id="counting/easy",
        grammar=r"^(?:\('mark',\s*\([0-9.]+,\s*[0-9.]+\)\)|\('undo',\s*'undo'\)|\('guess',\s*[0-9]+\)|\('stop',\s*'stop'\))$",
        horizon_cap=4,
        env_kwargs={
            "annotation_file": str(CONTAINER_ASSET_ROOT / "counting" / "lvis_v1_train.json"),
            "sample_dir": str(CONTAINER_ASSET_ROOT / "counting"),
            "min_count": 2,
            "max_count": 20,
            "radius": 5,
        },
    ),
    "fetch_pick_and_place": EnvironmentSpec(
        env_id="fetch_pick_and_place/easy",
        grammar=r"^(?:\('move',\s*\[-?[0-1],\s*-?[0-1],\s*-?[0-1]\]\)|\('gripper',\s*[0-1]\)|\('stop',\s*'stop'\))$",
        horizon_cap=8,
        env_kwargs={
            "reward_type": "sparse",
            "render_mode": "rgb_array",
            "use_follow_camera": True,
            "follow_cam_azimuth": 200.0,
            "follow_cam_elevation": -65.0,
            "follow_cam_distance": 0.36,
            "camera_default_distance": 1.75,
        },
        seed_key=None,
        reward_shaping={"type": "distance_delta", "info_key": "fetch_pick_distance", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "fetch_reach": EnvironmentSpec(
        env_id="fetch_reach/easy",
        grammar=r"^(?:\('move',\s*\[-?[0-1],\s*-?[0-1],\s*-?[0-1]\]\)|\('stop',\s*'stop'\))$",
        horizon_cap=8,
        env_kwargs={
            "reward_type": "sparse",
            "render_mode": "rgb_array",
            "use_follow_camera": True,
            "follow_cam_azimuth": 200.0,
            "follow_cam_elevation": -65.0,
            "follow_cam_distance": 0.36,
            "camera_default_distance": 1.75,
        },
        seed_key=None,
        reward_shaping={"type": "distance_delta", "info_key": "distance_to_goal", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "mental_rotation_2d": EnvironmentSpec(
        env_id="mental_rotation_2d/easy",
        grammar=r"^(?:\('rotate',\s*-?[0-9]+\)|\('stop',\s*'stop'\))$",
        horizon_cap=5,
        env_kwargs={
            "sample_dir": str(CONTAINER_ASSET_ROOT / "images"),
            "image_size": 128,
            "tolerance": 10.0,
        },
        # rotation_error is in degrees here vs. an already-normalized value in
        # the 3D variant, but reward_shaping normalizes progress to each
        # episode's own starting distance (see VisGymResourcesServer.
        # _training_step_reward), so the same weight is correct regardless.
        reward_shaping={"type": "distance_delta", "info_key": "rotation_error", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "zoom_in_puzzle": EnvironmentSpec(
        env_id="zoom_in_puzzle/easy",
        grammar=r"^(?:\('swap',\s*\([1-4],\s*[1-4]\)\)|\('reorder',\s*\[[1-4,\s]+\]\)|\('stop',\s*'stop'\))$",
        horizon_cap=5,
        env_kwargs={
            "sample_dir": str(CONTAINER_ASSET_ROOT / "images"),
            "min_zoom_level": 0.75,
            "zoom_gap": 0.75,
            "zoom_std": 0.2,
            "nested": True,
            "num_zoom_views": 4,
        },
    ),
    "maze_2d_7x7": EnvironmentSpec(
        env_id="maze_2d/easy",
        grammar=r"^(?:\('move',\s*[0-3]\)|\('stop',\s*'stop'\))$",
        horizon_cap=20,
        env_kwargs={"maze_width": 7, "maze_height": 7},
        reward_shaping={"type": "distance_delta", "info_key": "distance", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "maze_3d_7x7": EnvironmentSpec(
        env_id="maze_3d/easy",
        grammar=r"^(?:\('move',\s*0\)|\('turn',\s*[1-3]\)|\('stop',\s*'stop'\))$",
        horizon_cap=20,
        env_kwargs={"maze_width": 7, "maze_height": 7, "render_size": [128, 128]},
        reward_shaping={"type": "distance_delta", "info_key": "distance", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
    "refcoco_plus": EnvironmentSpec(
        env_id="referring_dot_pointing/easy",
        grammar=r"^(?:\('mark',\s*\([0-9.]+,\s*[0-9.]+\)\)|\('stop',\s*'stop'\))$",
        horizon_cap=4,
        env_kwargs={
            "sample_dir": str(CONTAINER_ASSET_ROOT / "refcoco"),
            "splitBy": "unc",
            "radius": 5,
        },
    ),
    "mental_rotation_3d": EnvironmentSpec(
        env_id="mental_rotation_3d_cube/easy",
        grammar=r"^(?:\('rotate',\s*\[-?[0-9.]+,\s*-?[0-9.]+,\s*-?[0-9.]+\]\)|\('stop',\s*'stop'\))$",
        horizon_cap=8,
        env_kwargs={
            "num_segments": 4,
            "image_size": [128, 128],
            "angle_tol": math.radians(10.0),
            "action_frame": "object",
        },
        reward_shaping={"type": "distance_delta", "info_key": "rotation_error", "weight": DEFAULT_SHAPING_WEIGHT},
    ),
}


def font(size: int) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def create_general_images(count: int = 32) -> None:
    image_dir = ASSET_ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    palette = [
        (39, 125, 161),
        (220, 87, 72),
        (73, 155, 118),
        (230, 178, 63),
        (135, 92, 165),
        (43, 48, 58),
    ]
    for index in range(count):
        image = Image.new("RGB", (512, 512), palette[index % len(palette)])
        draw = ImageDraw.Draw(image)
        for grid in range(0, 513, 64):
            draw.line((grid, 0, grid, 512), fill=(235, 237, 238), width=2)
            draw.line((0, grid, 512, grid), fill=(235, 237, 238), width=2)
        shift = (index * 17) % 90
        draw.polygon(
            [(55 + shift, 90), (205 + shift, 55), (175 + shift, 225), (80 + shift, 270)],
            fill=palette[(index + 2) % len(palette)],
            outline=(15, 18, 20),
            width=6,
        )
        draw.ellipse((280 - shift // 3, 85, 450 - shift // 3, 255), fill=palette[(index + 3) % len(palette)], width=5)
        draw.rectangle(
            (255, 300 - shift // 4, 465, 455 - shift // 4),
            fill=palette[(index + 1) % len(palette)],
            outline=(15, 18, 20),
            width=6,
        )
        draw.line((45, 435, 225, 305), fill=(250, 250, 250), width=22)
        draw.text(
            (28, 20),
            f"SCENE {index + 1:02d}",
            font=font(30),
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        image.save(image_dir / f"scene_{index:03d}.png")


def create_colorization_images() -> None:
    source_dir = ASSET_ROOT / "images"
    output_dir = ASSET_ROOT / "colorization"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("scene_*.png")):
        with Image.open(source) as image:
            image.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS).save(output_dir / source.name)


def circle_polygon(cx: int, cy: int, radius: int, points: int = 16) -> list[float]:
    polygon: list[float] = []
    for step in range(points):
        angle = 2.0 * math.pi * step / points
        polygon.extend([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
    return polygon


def create_counting_dataset(count: int = 32) -> None:
    output_dir = ASSET_ROOT / "counting"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    for image_index in range(count):
        true_count = 2 + image_index % 7
        image = Image.new("RGB", (224, 168), (236, 239, 242))
        draw = ImageDraw.Draw(image)
        centers = []
        for object_index in range(true_count):
            col = object_index % 4
            row = object_index // 4
            cx = 28 + col * 55 + (image_index * 3 % 8)
            cy = 44 + row * 76 + (image_index * 5 % 8)
            centers.append((cx, cy))
            draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=(218, 54, 58), outline=(84, 20, 24), width=2)
            draw.line((cx - 6, cy, cx + 6, cy), fill=(255, 255, 255), width=3)
        filename = f"count_{image_index:03d}.png"
        image.save(output_dir / filename)
        images.append(
            {
                "id": image_index + 1,
                "width": 224,
                "height": 168,
                "file_name": filename,
                "coco_url": "",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            }
        )
        for cx, cy in centers:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index + 1,
                    "category_id": 1,
                    "segmentation": [circle_polygon(cx, cy, 12)],
                    "area": math.pi * 12 * 12,
                    "bbox": [cx - 12, cy - 12, 24, 24],
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": "Deterministic VisGym counting smoke fixtures"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "red token",
                "synonyms": ["red token"],
                "frequency": "f",
                "def": "a red circular token",
                "image_count": count,
                "instance_count": len(annotations),
            }
        ],
    }
    with (output_dir / "lvis_v1_train.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def create_refcoco_plus_dataset(count: int = 32) -> None:
    dataset_dir = ASSET_ROOT / "refcoco" / "refcoco+"
    image_dir = dataset_dir / "images" / "mscoco" / "images" / "train2014"
    image_dir.mkdir(parents=True, exist_ok=True)

    width, height = 224, 168
    colors = (
        ("red", (214, 58, 62)),
        ("blue", (48, 112, 190)),
        ("green", (52, 154, 98)),
        ("yellow", (230, 181, 48)),
    )
    positions = (
        ("upper left", (28, 24, 88, 84)),
        ("upper right", (136, 24, 196, 84)),
        ("lower left", (28, 100, 88, 160)),
        ("lower right", (136, 100, 196, 160)),
    )
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []

    for index in range(count):
        target_index = index % len(positions)
        filename = f"refcoco_plus_{index:012d}.jpg"
        image = Image.new("RGB", (width, height), (237, 240, 242))
        draw = ImageDraw.Draw(image)

        for object_index, ((color_name, color), (position_name, box)) in enumerate(
            zip(colors, positions, strict=True)
        ):
            left, top, right, bottom = box
            offset = (index * (object_index + 3)) % 7 - 3
            left += offset
            right += offset
            draw.rectangle(
                (left, top, right, bottom),
                fill=color,
                outline=(25, 28, 32),
                width=3,
            )
            if object_index != target_index:
                continue

            annotation_id = index + 1
            segmentation = [[left, top, right, top, right, bottom, left, bottom]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": index + 1,
                    "category_id": object_index + 1,
                    "segmentation": segmentation,
                    "area": (right - left) * (bottom - top),
                    "bbox": [left, top, right - left, bottom - top],
                    "iscrowd": 0,
                }
            )
            sentence = f"the {color_name} square in the {position_name}"
            refs.append(
                {
                    "ref_id": annotation_id,
                    "ann_id": annotation_id,
                    "category_id": object_index + 1,
                    "image_id": index + 1,
                    "split": "train",
                    "sentences": [
                        {
                            "sent_id": annotation_id,
                            "sent": sentence,
                            "tokens": sentence.split(),
                        }
                    ],
                }
            )

        image.save(image_dir / filename, quality=95)
        images.append(
            {
                "id": index + 1,
                "width": width,
                "height": height,
                "file_name": filename,
            }
        )

    instances = {
        "info": {"description": "Deterministic RefCOCO+ compatible VisGym fixtures"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": f"{color_name} square", "supercategory": "shape"}
            for index, (color_name, _) in enumerate(colors)
        ],
    }
    with (dataset_dir / "instances.json").open("w", encoding="utf-8") as handle:
        json.dump(instances, handle, separators=(",", ":"))
    with (dataset_dir / "refs(unc).p").open("wb") as handle:
        pickle.dump(refs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def task_row(
    slug: str,
    spec: EnvironmentSpec,
    task_index: int,
    seed_base: int,
    max_output_tokens: int,
    horizon_cap: int | None = None,
) -> dict[str, Any]:
    seed = spec.seed_cycle[task_index % len(spec.seed_cycle)] if spec.seed_cycle else seed_base + task_index
    row: dict[str, Any] = {
        "agent_ref": {"type": "responses_api_agents", "name": "visgym_agent"},
        "env_id": spec.env_id,
        "env_kwargs": spec.env_kwargs,
        "seed": seed,
        "task_id": f"{slug}_easy_seed{seed}_sample{task_index}" if spec.seed_cycle else f"{slug}_easy_seed{seed}",
        "act_grammar_regex": spec.grammar,
        "horizon_cap": horizon_cap if horizon_cap is not None else spec.horizon_cap,
        "task_metadata": {
            "suite": "visgym",
            "difficulty": "easy",
            "manifest_kind": "requested_env_online_rl",
            "environment_slug": slug,
        },
        "responses_create_params": {
            "model": "policy_model",
            "input": [],
            "temperature": 1.0,
            "max_output_tokens": max_output_tokens,
            "tools": [],
        },
        "task_idx": task_index,
    }
    if spec.seed_key is not None:
        row["seed_key"] = spec.seed_key
    if spec.reward_shaping is not None:
        row["task_metadata"]["reward_shaping"] = spec.reward_shaping
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary_path.replace(path)


def create_manifests(
    samples: int,
    smoke_samples: int,
    seed_base: int,
    max_output_tokens: int,
    selected_slug: str | None = None,
    horizon_cap: int | None = None,
    combine_slugs: tuple[str, ...] | None = None,
) -> None:
    manifest_dir = DATA_ROOT / "requested_env_manifests"
    horizon_suffix = f"_h{horizon_cap}" if horizon_cap is not None else ""
    index: dict[str, Any] = {
        "training_steps": 20,
        "prompts_per_step": 64,
        "generations_per_prompt": 16,
        "rows_per_environment": samples,
        "environments": {},
    }
    rows_by_slug: dict[str, list[dict[str, Any]]] = {}
    smoke_rows_by_slug: dict[str, list[dict[str, Any]]] = {}
    for env_offset, (slug, spec) in enumerate(ENV_SPECS.items()):
        env_seed_base = seed_base + env_offset * 100_000
        train_path = manifest_dir / (f"{slug}_easy_train_{samples}{horizon_suffix}_t{max_output_tokens}.jsonl")
        smoke_path = manifest_dir / (f"{slug}_easy_smoke_{smoke_samples}{horizon_suffix}_t{max_output_tokens}.jsonl")
        if selected_slug is None or selected_slug == slug:
            rows = [
                task_row(
                    slug,
                    spec,
                    index,
                    env_seed_base,
                    max_output_tokens,
                    horizon_cap,
                )
                for index in range(samples)
            ]
            smoke_rows = rows[:smoke_samples]
            write_jsonl(train_path, rows)
            write_jsonl(smoke_path, smoke_rows)
        else:
            with train_path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            with smoke_path.open(encoding="utf-8") as handle:
                smoke_rows = [json.loads(line) for line in handle if line.strip()]
        rows_by_slug[slug] = rows
        smoke_rows_by_slug[slug] = smoke_rows
        index["environments"][slug] = {
            "env_id": spec.env_id,
            "train_manifest": str(train_path.relative_to(VISGYM_ROOT)),
            "smoke_manifest": str(smoke_path.relative_to(VISGYM_ROOT)),
            "horizon_cap": horizon_cap if horizon_cap is not None else spec.horizon_cap,
        }

    # Which environments make it into the blended manifest. Defaults to all of
    # them, but several need assets or forked wheels that a given checkout may
    # not have (counting needs lvis, fetch_* need gymnasium_robotics), and a
    # blend containing an environment that cannot start fails the rollout batch
    # rather than that one row. Callers pass the subset they have verified.
    combine = tuple(combine_slugs) if combine_slugs else tuple(ENV_SPECS)
    combined_train = [rows_by_slug[slug][sample_index] for sample_index in range(samples) for slug in combine]
    combined_smoke = [row for slug in combine for row in smoke_rows_by_slug[slug]]
    combined_train_path = manifest_dir / (
        f"requested_envs_combined_train_{len(combined_train)}{horizon_suffix}_t{max_output_tokens}.jsonl"
    )
    write_jsonl(combined_train_path, combined_train)
    write_jsonl(
        manifest_dir
        / f"requested_envs_combined_smoke_{len(combined_smoke)}{horizon_suffix}_t{max_output_tokens}.jsonl",
        combined_smoke,
    )
    index["combined_train_manifest"] = str(combined_train_path.relative_to(VISGYM_ROOT))
    if horizon_cap is None:
        combined_h15 = [{**row, "horizon_cap": 15} for row in combined_train]
        combined_h15_path = manifest_dir / (
            f"requested_envs_combined_train_{len(combined_h15)}_h15_t{max_output_tokens}.jsonl"
        )
        write_jsonl(combined_h15_path, combined_h15)
        index["combined_train_h15_manifest"] = str(combined_h15_path.relative_to(VISGYM_ROOT))
    else:
        index["uniform_horizon_cap"] = horizon_cap
        index[f"combined_train_h{horizon_cap}_manifest"] = str(combined_train_path.relative_to(VISGYM_ROOT))
    index["combined_train_rows"] = len(combined_train)
    index["combined_train_order"] = "round_robin_by_environment"
    index["combined_environments"] = list(combine)
    index_path = manifest_dir / f"requested_env_manifest_index{horizon_suffix}.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1280)
    parser.add_argument("--smoke-samples", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--env", choices=tuple(ENV_SPECS), help="Regenerate only one environment's manifests")
    parser.add_argument("--horizon-cap", type=int, help="Use one horizon for every generated environment")
    parser.add_argument("--skip-assets", action="store_true", help="Do not regenerate deterministic image assets")
    parser.add_argument(
        "--combine-envs",
        help=(
            "Comma-separated environments to include in the blended manifests "
            "(default: all). Use this to blend only the environments whose "
            "dependencies and assets are present in this checkout."
        ),
    )
    args = parser.parse_args()
    if args.samples != 20 * 64:
        raise SystemExit("--samples must equal 20 steps * 64 prompts = 1280")
    if not 1 <= args.smoke_samples <= args.samples:
        raise SystemExit("--smoke-samples must be between 1 and --samples")
    if args.horizon_cap is not None and args.horizon_cap < 1:
        raise SystemExit("--horizon-cap must be positive")
    if args.env is not None and args.horizon_cap is not None:
        raise SystemExit("--env cannot be combined with --horizon-cap; regenerate the uniform suite together")
    if not args.skip_assets and args.env is None:
        create_general_images()
        create_colorization_images()
        create_counting_dataset()
        create_refcoco_plus_dataset()
    elif not args.skip_assets and args.env == "colorization":
        create_colorization_images()
    elif not args.skip_assets and args.env == "refcoco_plus":
        create_refcoco_plus_dataset()
    combine_slugs = None
    if args.combine_envs:
        combine_slugs = tuple(s.strip() for s in args.combine_envs.split(",") if s.strip())
        unknown = [s for s in combine_slugs if s not in ENV_SPECS]
        if unknown:
            raise SystemExit(f"--combine-envs has unknown environments: {', '.join(unknown)}")
    create_manifests(
        args.samples,
        args.smoke_samples,
        args.seed_base,
        args.max_output_tokens,
        args.env,
        args.horizon_cap,
        combine_slugs,
    )


if __name__ == "__main__":
    main()
