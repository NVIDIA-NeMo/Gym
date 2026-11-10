# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest

from nemo_gym.web.task_images import resolve_local_task_image_path, resolve_task_image_url


@pytest.mark.parametrize(
    "reference",
    [
        "data:image/png;base64,AAAA",
        "http://example.test/image.png",
        "https://example.test/image.png",
    ],
)
def test_task_image_url_passes_supported_urls_through(reference: str) -> None:
    assert resolve_task_image_url(f"  {reference}  ", image_root=None, max_bytes=1) == reference


def test_task_image_url_encodes_local_image(tmp_path) -> None:
    payload = b"fake-png-payload"
    image = tmp_path / "nested" / "sample.PNG"
    image.parent.mkdir()
    image.write_bytes(payload)

    resolved = resolve_task_image_url("nested/sample.PNG", image_root=tmp_path, max_bytes=len(payload))

    assert resolved == f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"
    path, mime_type = resolve_local_task_image_path(
        str(image.resolve()),
        image_root=tmp_path,
        max_bytes=len(payload),
    )
    assert path == image.resolve()
    assert mime_type == "image/png"


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("", "must not be empty"),
        ("ftp://example.test/image.png", "unsupported task image URL scheme"),
    ],
)
def test_task_image_url_rejects_invalid_references(reference: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_task_image_url(reference, image_root=None, max_bytes=10)


def test_local_task_image_requires_root_and_positive_limit(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"x")

    with pytest.raises(ValueError, match="task_image_root is required"):
        resolve_local_task_image_path(str(image), image_root=None, max_bytes=1)
    with pytest.raises(ValueError, match="must be positive"):
        resolve_local_task_image_path(str(image), image_root=tmp_path, max_bytes=0)


def test_local_task_image_rejects_escape_directory_size_and_type(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="outside task_image_root"):
        resolve_local_task_image_path(str(outside), image_root=root, max_bytes=100)
    with pytest.raises(ValueError, match="regular file"):
        resolve_local_task_image_path(str(root), image_root=root, max_bytes=100)

    large = root / "large.jpg"
    large.write_bytes(b"1234")
    with pytest.raises(ValueError, match=r"byte limit \(4 > 3\)"):
        resolve_local_task_image_path("large.jpg", image_root=root, max_bytes=3)

    text = root / "not-image.txt"
    text.write_text("plain text")
    with pytest.raises(ValueError, match="unsupported task image type"):
        resolve_local_task_image_path("not-image.txt", image_root=root, max_bytes=100)


def test_local_task_image_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    outside = tmp_path / "outside.webp"
    outside.write_bytes(b"outside")
    (root / "escape.webp").symlink_to(outside)

    with pytest.raises(ValueError, match="outside task_image_root"):
        resolve_local_task_image_path("escape.webp", image_root=root, max_bytes=100)
