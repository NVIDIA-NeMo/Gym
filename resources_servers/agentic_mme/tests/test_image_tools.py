# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageFilter, ImageOps

from resources_servers.agentic_mme import image_tools
from resources_servers.agentic_mme.image_tools import (
    IMAGE_TOOLS,
    MAX_MODEL_IMAGE_PIXELS,
    ImageWorkspace,
    InputImageError,
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


@pytest.mark.parametrize(
    "url",
    ["/etc/passwd", "data:image/png;base64,broken", "data:image/png;base64," + base64.b64encode(b"x" * 64).decode()],
    ids=["not_a_data_url", "bad_base64", "not_an_image"],
)
def test_refused_inputs_are_input_image_errors(url) -> None:
    with pytest.raises(InputImageError):
        ImageWorkspace().load(url)


@pytest.mark.parametrize("max_pixels", [16_000_000, 64_000_000])
def test_encoded_size_limit_follows_the_pixel_limit(max_pixels) -> None:
    # An RGB PNG at the pixel limit fits: a 50 Mpx photo inlined as PNG (over 48 MB of image data, past the
    # former fixed 64,000,000-character cap) loads when max_pixels allows 50 Mpx.
    assert ImageWorkspace(max_pixels=max_pixels).max_input_bytes >= 3 * max_pixels
    workspace = ImageWorkspace(max_pixels=1)
    encoded = "A" * ((workspace.max_input_bytes + 8) * 4 // 3)
    with pytest.raises(InputImageError, match="MB encoded"):
        workspace.load("data:image/png;base64," + encoded)  # refused before decoding


def test_encoded_size_cap_keeps_the_former_allowance() -> None:
    # At a small pixel limit the cap is never below the former fixed 48,000,000 bytes.
    assert ImageWorkspace(max_pixels=1).max_input_bytes == image_tools._MIN_INPUT_BYTES == 48_000_000
    assert ImageWorkspace(max_pixels=64_000_000).max_input_bytes > 48_000_000


def test_dense_jpeg_over_four_bytes_per_pixel_loads(monkeypatch) -> None:
    # A noise JPEG at quality 100 without chroma subsampling takes more than 4 bytes per pixel, so the per-pixel
    # bound alone would refuse it; the allowance floor accepts it. Scaled down: no 1 MiB overhead, and a floor
    # just above this JPEG (the real-size case is 3300x3300, 44.8 MB, under the 48 MB floor).
    stream = BytesIO()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)).save(
        stream, format="JPEG", quality=100, subsampling=0
    )
    data = stream.getvalue()
    assert len(data) > 4 * 300 * 300
    url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    monkeypatch.setattr(image_tools, "_ENCODING_OVERHEAD_BYTES", 0)
    monkeypatch.setattr(image_tools, "_MIN_INPUT_BYTES", 0)
    with pytest.raises(InputImageError, match="MB encoded"):
        ImageWorkspace(max_pixels=300 * 300).load(url)
    monkeypatch.setattr(image_tools, "_MIN_INPUT_BYTES", len(data) + 1)
    workspace = ImageWorkspace(max_pixels=300 * 300)
    workspace.load(url)
    assert workspace.get(0).size == (300, 300)


def test_source_over_the_input_ceiling_is_refused_before_decoding(monkeypatch) -> None:
    # With downscaling on, the input ceiling is Pillow's decompression-bomb limit; a source over it is refused
    # from its header, before the raster is decoded or downscaled.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
    workspace = ImageWorkspace(max_pixels=10_000, downscale_inputs=True)
    assert workspace.max_input_pixels == 1_000_000
    decoded = []
    monkeypatch.setattr(image_tools, "fit_to_pixels", lambda *args: decoded.append(args))
    with pytest.raises(InputImageError, match="input image 1500x1000: image dimensions exceed the 1,000,000-pixel"):
        workspace.load(image_data_url(Image.new("RGB", (1500, 1000), "red")))
    assert decoded == [] and workspace.images == []
    # With Pillow's own limit disabled, the ceiling is Pillow's default rather than none.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    assert workspace.max_input_pixels == image_tools._PILLOW_DEFAULT_MAX_IMAGE_PIXELS


@pytest.mark.parametrize("size", [(10_000, 1), (1, 10_000)], ids=["wide", "tall"])
def test_elongated_input_is_downscaled_within_the_budget(size) -> None:
    workspace = ImageWorkspace(max_pixels=100, downscale_inputs=True)
    workspace.load(image_data_url(Image.new("RGB", size, "red")))
    width, height = workspace.get(0).size
    assert width * height <= 100 and min(width, height) >= 1


def test_oversized_input_is_refused_with_its_size() -> None:
    with pytest.raises(InputImageError, match="input image 40x10: image dimensions exceed"):
        ImageWorkspace(max_pixels=100).load(image_data_url(Image.new("RGB", (40, 10), "red")))


@pytest.mark.parametrize("encoding", ["PNG", "JPEG"])
def test_oversized_input_is_downscaled_when_enabled(encoding) -> None:
    stream = BytesIO()
    Image.new("RGB", (400, 100), "red").save(stream, format=encoding)
    mime = encoding.lower()
    workspace = ImageWorkspace(max_pixels=10_000, downscale_inputs=True)
    workspace.load(f"data:image/{mime};base64," + base64.b64encode(stream.getvalue()).decode())
    assert workspace.get(0).size == (200, 50)
    assert workspace.downscaled_inputs == [((400, 100), (200, 50))]
    assert workspace.total_pixels == 10_000
    # Inputs within the limit are kept as they are.
    workspace.max_total_pixels = 100_000
    workspace.load(image_data_url(Image.new("RGB", (20, 10), "blue")))
    assert workspace.get(1).size == (20, 10) and len(workspace.downscaled_inputs) == 1


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
