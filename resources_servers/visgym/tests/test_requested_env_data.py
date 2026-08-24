# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from resources_servers.visgym import app as visgym_app
from resources_servers.visgym.schemas import VisGymTaskRow


VISGYM_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = VISGYM_ROOT / "data" / "requested_env_manifests"
ASSET_ROOT = VISGYM_ROOT / "data" / "requested_env_assets"

# scripts/ has no __init__.py (it is a collection of standalone CLI entry
# points, not an importable package), so the generator is loaded by path
# instead of a normal import.
_generator_spec = importlib.util.spec_from_file_location(
    "create_fourteen_env_data", VISGYM_ROOT / "scripts" / "create_fourteen_env_data.py"
)
create_fourteen_env_data = importlib.util.module_from_spec(_generator_spec)
# The module's frozen dataclasses look themselves up in sys.modules while
# their class bodies are being evaluated, so it must be registered before
# exec_module runs, not just assigned to this local name.
sys.modules[_generator_spec.name] = create_fourteen_env_data
_generator_spec.loader.exec_module(create_fourteen_env_data)
SLUGS = (
    "matchstick_equation",
    "maze_3d",
    "jigsaw",
    "sliding_block",
    "colorization",
    "counting",
    "fetch_pick_and_place",
    "fetch_reach",
    "mental_rotation_2d",
    "zoom_in_puzzle",
    "maze_2d_7x7",
    "maze_3d_7x7",
    "refcoco_plus",
    "mental_rotation_3d",
)
VALID_ACTIONS = {
    "matchstick_equation": "('move', [0, 6, 2, 0])",
    "maze_3d": "('turn', 1)",
    "jigsaw": "('reorder', [0, 1, 2, 3])",
    "sliding_block": "('move', (10, 1))",
    "colorization": "('rotate', -45)",
    "counting": "('guess', 5)",
    "fetch_pick_and_place": "('gripper', 1)",
    "fetch_reach": "('move', [-1, 0, 1])",
    "mental_rotation_2d": "('rotate', -90)",
    "zoom_in_puzzle": "('reorder', [1, 3, 2, 4])",
    "maze_2d_7x7": "('move', 1)",
    "maze_3d_7x7": "('turn', 1)",
    "refcoco_plus": "('mark', (0.25, 0.25))",
    "mental_rotation_3d": "('rotate', [15.0, 0.0, -10.0])",
}


GENERATE_HINT = (
    "Generate them first:\n"
    "  resources_servers/visgym/scripts/create_fourteen_env_data.py\n"
    "  resources_servers/visgym/scripts/create_fourteen_env_data.py "
    "--horizon-cap 20 --skip-assets"
)


@pytest.fixture(autouse=True)
def requested_env_artifacts_present() -> None:
    """Skip unless the requested-env manifests and assets have been generated.

    The full set is ~66 MB of manifests plus 131 rendered PNGs, so it is
    produced by the committed generator rather than committed itself. maze_2d,
    the environment the launcher trains on by default, needs none of it.
    """
    if not MANIFEST_ROOT.is_dir() or not (ASSET_ROOT / "images").is_dir():
        pytest.skip(f"VisGym requested-env artifacts are not generated. {GENERATE_HINT}")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_requested_training_and_smoke_manifests() -> None:
    for slug in SLUGS:
        train_rows = load_rows(MANIFEST_ROOT / f"{slug}_easy_train_1280_t1024.jsonl")
        smoke_rows = load_rows(MANIFEST_ROOT / f"{slug}_easy_smoke_16_t1024.jsonl")
        assert len(train_rows) == 20 * 64
        assert len(smoke_rows) == 16
        assert smoke_rows == train_rows[:16]
        for row in (train_rows[0], train_rows[-1]):
            validated = VisGymTaskRow.model_validate(row)
            assert validated.responses_create_params.max_output_tokens == 1024
            assert re.fullmatch(validated.act_grammar_regex, VALID_ACTIONS[slug])

    maze_row = load_rows(MANIFEST_ROOT / "maze_3d_easy_train_1280_t1024.jsonl")[0]
    matchstick_rows = load_rows(MANIFEST_ROOT / "matchstick_equation_easy_train_1280_t1024.jsonl")
    assert len({row["seed"] for row in matchstick_rows}) == 64
    assert [row["seed"] for row in matchstick_rows[:64]] == [row["seed"] for row in matchstick_rows[64:128]]
    assert matchstick_rows[0]["task_metadata"]["reward_shaping"] == {
        "type": "distance_delta",
        "info_key": "matchstick_distance",
        "weight": create_fourteen_env_data.DEFAULT_SHAPING_WEIGHT,
    }
    assert maze_row["env_kwargs"]["maze_width"] == 5
    assert maze_row["env_kwargs"]["maze_height"] == 5
    assert maze_row["env_kwargs"]["render_size"] == [128, 128]
    assert maze_row["horizon_cap"] == 8
    assert maze_row["task_metadata"]["reward_shaping"] == {
        "type": "distance_delta",
        "info_key": "distance",
        "weight": create_fourteen_env_data.DEFAULT_SHAPING_WEIGHT,
    }
    sliding_row = load_rows(MANIFEST_ROOT / "sliding_block_easy_train_1280_t1024.jsonl")[0]
    assert sliding_row["env_kwargs"]["num_shuffle_moves"] == 4
    assert sliding_row["horizon_cap"] == 4
    assert sliding_row["task_metadata"]["reward_shaping"]["info_key"] == "sliding_distance"
    color_row = load_rows(MANIFEST_ROOT / "colorization_easy_train_1280_t1024.jsonl")[0]
    assert color_row["task_metadata"]["reward_shaping"]["info_key"] == "color_distance"
    assert color_row["env_kwargs"]["img_size"] == 128
    fetch_pick_row = load_rows(MANIFEST_ROOT / "fetch_pick_and_place_easy_train_1280_t1024.jsonl")[0]
    fetch_reach_row = load_rows(MANIFEST_ROOT / "fetch_reach_easy_train_1280_t1024.jsonl")[0]
    assert fetch_pick_row["task_metadata"]["reward_shaping"]["info_key"] == "fetch_pick_distance"
    assert fetch_reach_row["task_metadata"]["reward_shaping"]["info_key"] == "distance_to_goal"
    assert fetch_pick_row["horizon_cap"] == 8
    assert fetch_reach_row["horizon_cap"] == 8
    counting_row = load_rows(MANIFEST_ROOT / "counting_easy_train_1280_t1024.jsonl")[0]
    assert counting_row["horizon_cap"] == 4
    refcoco_row = load_rows(MANIFEST_ROOT / "refcoco_plus_easy_train_1280_t1024.jsonl")[0]
    assert refcoco_row["env_id"] == "referring_dot_pointing/easy"
    rotation_3d_row = load_rows(MANIFEST_ROOT / "mental_rotation_3d_easy_train_1280_t1024.jsonl")[0]
    assert rotation_3d_row["env_id"] == "mental_rotation_3d_cube/easy"
    assert rotation_3d_row["task_metadata"]["reward_shaping"]["info_key"] == "rotation_error"


def test_requested_combined_training_manifest_is_balanced() -> None:
    combined_path = MANIFEST_ROOT / "requested_envs_combined_train_17920_t1024.jsonl"
    rows = load_rows(combined_path)
    assert len(rows) == len(SLUGS) * 20 * 64
    assert Counter(row["task_metadata"]["environment_slug"] for row in rows) == {slug: 20 * 64 for slug in SLUGS}
    assert [row["task_metadata"]["environment_slug"] for row in rows[: len(SLUGS)]] == list(SLUGS)

    for step_start in range(0, len(rows), 64):
        step_slugs = {row["task_metadata"]["environment_slug"] for row in rows[step_start : step_start + 64]}
        assert step_slugs == set(SLUGS)

    first_twenty_steps = rows[: 20 * 64]
    prefix_counts = Counter(row["task_metadata"]["environment_slug"] for row in first_twenty_steps)
    assert set(prefix_counts) == set(SLUGS)
    assert max(prefix_counts.values()) - min(prefix_counts.values()) <= 1

    horizon_15_rows = load_rows(MANIFEST_ROOT / "requested_envs_combined_train_17920_h15_t1024.jsonl")
    assert len(horizon_15_rows) == len(rows)
    assert all(row["horizon_cap"] == 15 for row in horizon_15_rows)
    assert [row["task_id"] for row in horizon_15_rows] == [row["task_id"] for row in rows]


def test_requested_uniform_horizon_20_manifests() -> None:
    combined_rows = load_rows(MANIFEST_ROOT / "requested_envs_combined_train_17920_h20_t1024.jsonl")
    assert len(combined_rows) == len(SLUGS) * 20 * 64
    assert all(row["horizon_cap"] == 20 for row in combined_rows)
    assert Counter(row["task_metadata"]["environment_slug"] for row in combined_rows) == {
        slug: 20 * 64 for slug in SLUGS
    }

    for slug in SLUGS:
        train_rows = load_rows(MANIFEST_ROOT / f"{slug}_easy_train_1280_h20_t1024.jsonl")
        smoke_rows = load_rows(MANIFEST_ROOT / f"{slug}_easy_smoke_16_h20_t1024.jsonl")
        assert len(train_rows) == 20 * 64
        assert smoke_rows == train_rows[:16]
        assert all(row["horizon_cap"] == 20 for row in train_rows)


def test_default_shaping_weight_matches_resources_server() -> None:
    """The generator duplicates this constant instead of importing app.py.

    (create_fourteen_env_data.py stays free of the resources server's
    FastAPI/aiohttp dependencies so it can run standalone.) A drift here
    means every generated manifest silently reward-shapes with a weight the
    server wasn't reviewed against.
    """
    assert create_fourteen_env_data.DEFAULT_SHAPING_WEIGHT == visgym_app.DEFAULT_SHAPING_WEIGHT


def test_every_reward_shaping_weight_is_within_the_resources_server_ceiling() -> None:
    """weight > MAX_SHAPING_WEIGHT is silently clamped at runtime (see
    VisGymResourcesServer._clamped_shaping_weight), which would make every
    manifest row generated with an out-of-range weight quietly train at a
    different value than the one committed to the row. Catch it at
    generation time instead.
    """
    for spec in create_fourteen_env_data.ENV_SPECS.values():
        if spec.reward_shaping is None:
            continue
        weight = spec.reward_shaping.get("weight")
        assert weight is not None
        assert 0.0 < weight <= visgym_app.MAX_SHAPING_WEIGHT


def test_requested_asset_fixtures_exist() -> None:
    assert len(list((ASSET_ROOT / "images").glob("*.png"))) == 32
    color_images = list((ASSET_ROOT / "colorization").glob("*.png"))
    assert len(color_images) == 32
    with Image.open(color_images[0]) as image:
        assert image.size == (128, 128)
    assert len(list((ASSET_ROOT / "counting").glob("count_*.png"))) == 32
    annotation_path = ASSET_ROOT / "counting" / "lvis_v1_train.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 32
    assert len(payload["annotations"]) >= 64
    assert {(image["width"], image["height"]) for image in payload["images"]} == {(224, 168)}
    refcoco_dir = ASSET_ROOT / "refcoco" / "refcoco+"
    assert len(list((refcoco_dir / "images/mscoco/images/train2014").glob("*.jpg"))) == 32
    refcoco_instances = json.loads((refcoco_dir / "instances.json").read_text(encoding="utf-8"))
    assert len(refcoco_instances["images"]) == 32
    assert len(refcoco_instances["annotations"]) == 32
    assert (refcoco_dir / "refs(unc).p").is_file()


def test_requested_agent_config_uses_combined_smoke_manifest() -> None:
    config_path = VISGYM_ROOT / "configs" / "visgym_requested_direct_action_agent.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    agent = config["visgym_agent"]["responses_api_agents"]["visgym_agent"]
    assert agent["done_if_no_boxed_answer"] is True
    assert agent["max_steps"] >= 35
    dataset_path = VISGYM_ROOT / agent["datasets"][0]["jsonl_fpath"]
    assert len(load_rows(dataset_path)) == len(SLUGS) * 16


def test_requested_robotics_wheels_carry_their_assets() -> None:
    """Validate the optional robotics wheels once they have been built.

    Both are forks that VisGym ships as source, so they are built locally by
    scripts/build_vendor_wheels.sh rather than committed. Only the fetch_* and
    refcoco_plus tasks need them; skip instead of failing a clean checkout.
    """
    robotics_wheel = VISGYM_ROOT / "vendor_wheels" / "gymnasium_robotics-1.4.1-py3-none-any.whl"
    lvis_wheel = VISGYM_ROOT / "vendor_wheels" / "lvis-0.5.3-py3-none-any.whl"
    if not robotics_wheel.is_file() or not lvis_wheel.is_file():
        pytest.skip(
            "Optional VisGym robotics wheels are not built; run "
            "resources_servers/visgym/scripts/build_vendor_wheels.sh "
            "<VisGym checkout> to enable the fetch_* and refcoco_plus tasks."
        )
    assert robotics_wheel.stat().st_size > 500_000
    with zipfile.ZipFile(robotics_wheel) as archive:
        members = set(archive.namelist())
    assert "gymnasium_robotics/envs/assets/textures/block.png" in members
    assert "gymnasium_robotics/envs/assets/textures/block_hidden.png" in members
