# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageFilter, ImageOps

from responses_api_agents.agentic_mme_agent.tools import (
    IMAGE_TOOLS,
    MAX_MODEL_IMAGE_PIXELS,
    ImageWorkspace,
    function_schema,
    image_data_url,
    model_image_url,
)


@pytest.fixture
def workspace() -> ImageWorkspace:
    state = ImageWorkspace(max_pixels=10000, max_total_pixels=100000)
    pixels = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(8, 12, 3)
    state.add(Image.fromarray(pixels))
    return state


@pytest.mark.parametrize("name", IMAGE_TOOLS)
def test_all_image_tools(workspace, name) -> None:
    args = {"image_index": 0}
    if name == "crop":
        args["bbox_2d"] = [0, 0, 500, 500]
    if name == "rotate":
        args["angle"] = 90
    if name == "resize":
        args["scale"] = 2
    result = workspace.apply(name, args)
    assert result["new_image_index"] == 1
    assert result["source_image_index"] == 0
    with Image.open(BytesIO(base64.b64decode(result["image_url"].split(",")[1]))) as observed:
        assert observed.size == tuple(result["size"])
        assert np.array_equal(np.asarray(observed), np.asarray(workspace.get(1)))
    assert len(workspace.images) == 2
    assert function_schema(name, *IMAGE_TOOLS[name])["name"] == name


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("flip", {"direction": "horizontal"}, lambda im: ImageOps.mirror(im)),
        ("flip", {"direction": "vertical"}, lambda im: ImageOps.flip(im)),
        ("flip", {"direction": "both"}, lambda im: ImageOps.flip(ImageOps.mirror(im))),
        ("invert", {}, lambda im: ImageOps.invert(im)),
        ("grayscale", {}, lambda im: im.convert("L").convert("RGB")),
        ("equalize", {}, lambda im: ImageOps.equalize(im)),
        ("sharpen", {}, lambda im: im.filter(ImageFilter.SHARPEN)),
        ("enhance", {"brightness": 0}, lambda im: Image.new("RGB", im.size)),
        ("crop", {"bbox_2d": [500, 500, 0, 0]}, lambda im: im.crop((0, 0, 6, 4))),
        ("rotate", {"angle": 90}, lambda im: im.transpose(Image.Transpose.ROTATE_90)),
        ("resize", {"width": 6, "height": 4}, lambda im: im.resize((6, 4), Image.Resampling.LANCZOS)),
        (
            "crop",
            {"bbox_2d": [0, 0, 500, 500], "zoom_scale": 2},
            lambda im: im.crop((0, 0, 6, 4)).resize((12, 8), Image.Resampling.LANCZOS),
        ),
    ],
)
def test_pixel_semantics(workspace, name, args, expected) -> None:
    original = workspace.get(0).copy()
    workspace.apply(name, {"image_index": 0, **args})
    assert np.array_equal(np.asarray(workspace.get(1)), np.asarray(expected(original)))
    assert np.array_equal(np.asarray(workspace.get(0)), np.asarray(original))


@pytest.mark.parametrize("mode", ["binary", "binary_inv", "trunc", "tozero"])
def test_threshold(workspace, mode) -> None:
    workspace.add(Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)))
    workspace.apply("threshold", {"image_index": 1, "value": 128, "mode": mode})
    expected = {"binary": [0, 0, 255], "binary_inv": [255, 255, 0], "trunc": [0, 128, 128], "tozero": [0, 0, 255]}[
        mode
    ]
    assert np.asarray(workspace.get(2))[0, :, 0].tolist() == expected


@pytest.mark.parametrize("method", ["canny", "sobel", "simple"])
def test_edges(workspace, method) -> None:
    workspace.apply("edge_detect", {"image_index": 0, "method": method})
    assert workspace.get(1).mode == "RGB"
    assert workspace.get(1).size == workspace.get(0).size
    assert not np.array_equal(np.asarray(workspace.get(0)), np.asarray(workspace.get(1)))


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("crop", {"bbox_2d": [0, 0, 0, 0]}),
        ("crop", {"bbox_2d": [-1, 0, 500, 500]}),
        ("crop", {"bbox_2d": [0, 0, 500]}),
        ("resize", {"width": 100000, "height": 100000}),
        ("resize", {}),
        ("rotate", {"angle": float("nan")}),
        ("flip", {"direction": "diagonal"}),
        ("grayscale", {"image_index": -1}),
        ("grayscale", {"image_index": 99}),
        ("grayscale", {"image_index": True}),
        ("grayscale", {"path": "/etc/passwd"}),
        ("unknown", {}),
    ],
)
def test_invalid_tools_do_not_change_state(workspace, name, args) -> None:
    with pytest.raises(ValueError):
        workspace.apply(name, {"image_index": 0, **args})
    assert len(workspace.images) == 1


def test_load_and_isolation(workspace) -> None:
    other = ImageWorkspace()
    other.load(image_data_url(workspace.get(0)))
    workspace.apply("invert", {"image_index": 0})
    assert len(other.images) == 1
    assert np.array_equal(np.asarray(other.get(0)), np.asarray(workspace.get(0)))


@pytest.mark.parametrize(
    "url", ["/etc/passwd", "file:///tmp/a.png", "https://example.com/a.png", "data:image/png;base64,broken"]
)
def test_input_access_restricted(url) -> None:
    with pytest.raises((ValueError, OSError)):
        ImageWorkspace().load(url)


def test_image_limit_refuses_the_next_image(workspace) -> None:
    workspace.max_images = len(workspace.images) + 1
    workspace.apply("invert", {"image_index": 0})
    with pytest.raises(ValueError, match="image limit reached: a model request holds at most"):
        workspace.apply("invert", {"image_index": 0})
    assert len(workspace.images) == workspace.max_images


def test_pixel_budget(workspace) -> None:
    workspace.max_total_pixels = workspace.total_pixels
    with pytest.raises(ValueError, match="budget"):
        workspace.apply("invert", {"image_index": 0})


def decode(url: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_model_image_is_capped_and_workspace_stays_lossless() -> None:
    workspace = ImageWorkspace()
    rng = np.random.default_rng(0)
    workspace.add(Image.fromarray(rng.integers(0, 256, (1500, 1500, 3), dtype=np.uint8)))
    result = workspace.apply("crop", {"image_index": 0, "bbox_2d": [0, 0, 1000, 1000], "zoom_scale": 2})
    sent = decode(result["image_url"])
    assert result["size"] == [3000, 3000] and workspace.get(1).size == (3000, 3000)
    assert sent.width * sent.height <= MAX_MODEL_IMAGE_PIXELS and abs(sent.width - sent.height) <= 1
    assert sent.width >= 2047


@pytest.mark.parametrize(
    ("image", "mime"),
    [
        (Image.fromarray(np.random.default_rng(1).integers(0, 256, (400, 400, 3), dtype=np.uint8)), "jpeg"),
        (Image.new("RGB", (400, 400), "white"), "png"),
    ],
    ids=["photo_like", "flat"],
)
def test_model_image_uses_the_smaller_encoding(image, mime) -> None:
    url = model_image_url(image)
    assert url.startswith(f"data:image/{mime};base64,")
    assert decode(url).size == image.size
    assert model_image_url(image) == url  # deterministic: resending the same image keeps the prefix
