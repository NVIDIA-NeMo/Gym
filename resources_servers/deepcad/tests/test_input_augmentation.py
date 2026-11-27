# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import copy
import io
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from resources_servers.deepcad.dataset.augment_inputs import (
    FOREGROUND_COLORS,
    augment_inputs,
    colorize_image,
    sample_annotations,
    sample_foreground_color,
    view_prompt,
)
from resources_servers.deepcad.dataset.prepare_gym import sample_view_mode


def example(mode: str) -> dict:
    # Identical tiles test that labels follow image positions, not image values.
    source = Image.new("RGB", (524, 524) if mode == "diagonal" else (256, 256), "white")
    source.paste((128, 128, 128), (50, 50, 150, 150))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    content = [{"type": "input_text", "text": "Overall bounding-box dimensions in mm (X,Y,Z): 20, 30, 4."}]
    for _ in range(1 if mode == "diagonal" else 3):
        content.extend(
            [
                {"type": "input_text", "text": "Old view:"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
                },
            ]
        )
    return {
        "responses_create_params": {
            "input": [{"role": "system", "content": "old instructions"}, {"role": "user", "content": content}]
        },
        "verifier_metadata": {"task_id": "part", "reference_step": "private.step", "difficulty": 8, "view_mode": mode},
    }


@pytest.mark.parametrize("mode", ["axis", "diagonal"])
def test_matching_prompt_and_identical_grpo_inputs(mode: str) -> None:
    source = example(mode)
    result = augment_inputs(copy.deepcopy(source))
    # Preparation retries, multiple responses and reloaded rows get identical bytes.
    assert result == augment_inputs(copy.deepcopy(source))
    assert result == augment_inputs(copy.deepcopy(result))
    prompt, user = result["responses_create_params"]["input"]
    assert "Output ONLY a single ```python fenced block" in prompt["content"]
    assert "import cadquery as cq" in prompt["content"] and "no prose" in prompt["content"]
    assert "private.step" not in str(result["responses_create_params"])
    assert "20, 30, 4" in user["content"][0]["text"]
    expected = (
        ["Four diagonal views (2x2 composite):"] if mode == "diagonal" else ["Front view:", "Side view:", "Top view:"]
    )
    assert [item["text"] for item in user["content"][1:] if item["type"] == "input_text"] == expected
    for item in user["content"]:
        if item["type"] != "input_image":
            continue
        with Image.open(io.BytesIO(base64.b64decode(item["image_url"].split(",")[1]))) as image:
            assert image.getpixel((0, 0)) == (255, 255, 255)
            assert image.getpixel((75, 75)) == FOREGROUND_COLORS[result["verifier_metadata"]["foreground_color"]]


def test_prompts_use_correct_cameras_and_units() -> None:
    axis = view_prompt("axis")
    diagonal = view_prompt("diagonal")
    assert "3 separate dimensioned front, side, and top" in axis
    assert "Front: camera on -Y, looking along +Y; X right, Z up" in axis
    assert "Side: camera on +X, looking along -X; Y left, Z up" in axis
    assert "Top: camera on +Z, looking along -Z; X right, Y up" in axis
    assert "Use millimetres in your code" in axis and "normalized" not in axis
    assert "2x2 composite of 4 diagonal" in diagonal
    assert "bbox centered at [0.5, 0.5, 0.5]" in diagonal
    assert "Bottom-left:  [ 1, -1,  1]" in diagonal
    assert "Bottom-right: [-1,  1, -1]" in diagonal
    assert "Front:" not in diagonal and "Top-left:" not in axis


@pytest.mark.parametrize("color", FOREGROUND_COLORS)
def test_white_background_and_shading_survive_color_augmentation(color: str) -> None:
    gray = np.arange(256, dtype=np.uint8)[None, :]
    image = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2))
    tinted = np.asarray(colorize_image(image, color=color))
    assert np.array_equal(tinted[0, 0], [0, 0, 0])
    assert np.array_equal(tinted[0, -1], [255, 255, 255])
    assert np.array_equal(tinted[0, 128], FOREGROUND_COLORS[color])
    assert np.all(np.diff(tinted.astype(int), axis=1) >= 0)
    assert len(np.unique(tinted.reshape(-1, 3), axis=0)) > 200
    transparent = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert colorize_image(transparent, color=color).getpixel((0, 0)) == (255, 255, 255)


def test_colors_independent_of_views_and_task_order() -> None:
    ids = [f"part-{index}" for index in range(6000)]
    colors = {identifier: sample_foreground_color(identifier) for identifier in ids}
    assert colors == {identifier: sample_foreground_color(identifier) for identifier in reversed(ids)}
    assert colors != {identifier: sample_foreground_color(identifier, seed=43) for identifier in ids}
    for mode in ("axis", "diagonal"):
        counts = Counter(colors[identifier] for identifier in ids if sample_view_mode(identifier) == mode)
        assert set(counts) == set(FOREGROUND_COLORS)
        assert all(0.12 < count / sum(counts.values()) < 0.21 for count in counts.values())


def test_saved_inputs_require_original_to_change_color_seed() -> None:
    result = augment_inputs(example("axis"))
    new_seed = next(
        seed for seed in range(100) if sample_foreground_color("part", seed=seed) != sample_foreground_color("part")
    )
    with pytest.raises(ValueError, match="original uncolored dataset"):
        augment_inputs(result, seed=new_seed)


@pytest.mark.parametrize("mode", ["axis", "diagonal"])
def test_annotation_free_prompts_hide_dimensions(mode: str) -> None:
    row = example(mode)
    row["verifier_metadata"]["engineering_annotations"] = False
    result = augment_inputs(row)
    prompt = str(result["responses_create_params"])
    assert "20, 30, 4" not in prompt
    assert "Overall bounding-box dimensions" not in prompt
    assert "R denotes" not in prompt and "depth denotes" not in prompt
    assert "dimensioned" not in prompt
    assert result == augment_inputs(copy.deepcopy(result))


def test_annotation_probability_and_reproducibility() -> None:
    ids = [f"part-{index}" for index in range(6000)]
    selected = {identifier: sample_annotations(identifier) for identifier in ids}
    assert selected == {identifier: sample_annotations(identifier) for identifier in reversed(ids)}
    assert 0.72 < sum(not value for value in selected.values()) / len(ids) < 0.78
    assert all(sample_annotations(identifier, no_annotations_probability=0) for identifier in ids[:100])
    assert not any(sample_annotations(identifier, no_annotations_probability=1) for identifier in ids[:100])
    assert selected != {identifier: sample_annotations(identifier, seed=43) for identifier in ids}
    for mode in ("axis", "diagonal"):
        values = [selected[identifier] for identifier in ids if sample_view_mode(identifier) == mode]
        assert 0.70 < sum(not value for value in values) / len(values) < 0.80


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_annotation_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        sample_annotations("part", no_annotations_probability=probability)
