# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact, replayable WebVoyager judge evidence stored in rollout responses."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


WEBVOYAGER_JUDGE_EVIDENCE_SCHEMA_VERSION = 1


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return value
    return None


def response_image_urls(response: Any) -> list[str]:
    """Return input-image URLs already retained in a response trajectory."""

    response_mapping = _mapping(response)
    if response_mapping is None:
        return []
    images: list[str] = []
    output = response_mapping.get("output")
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        return images
    for raw_item in output:
        item = _mapping(raw_item)
        if item is None:
            continue
        content = item.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for raw_block in content:
            block = _mapping(raw_block)
            if block is None or block.get("type") != "input_image":
                continue
            image_url = block.get("image_url")
            if isinstance(image_url, Mapping):
                image_url = image_url.get("url")
            if isinstance(image_url, str) and image_url:
                images.append(image_url)
    return images


def compact_webvoyager_judge_evidence(
    *,
    response: Any,
    final_answer: str,
    screenshots: Sequence[str],
    page_urls: Sequence[str],
) -> dict[str, Any]:
    """Reference trajectory images by index and inline only missing edge images."""

    response_images = response_image_urls(response)
    available: dict[str, deque[int]] = defaultdict(deque)
    for index, image_url in enumerate(response_images):
        available[image_url].append(index)

    sequence: list[dict[str, Any]] = []
    for screenshot in screenshots:
        positions = available.get(screenshot)
        if positions:
            sequence.append({"response_image_index": positions.popleft()})
        else:
            sequence.append({"inline_data_url": screenshot})
    return {
        "schema_version": WEBVOYAGER_JUDGE_EVIDENCE_SCHEMA_VERSION,
        "final_answer": final_answer,
        "page_urls": list(page_urls),
        "screenshot_sequence": sequence,
    }


def expand_webvoyager_judge_screenshots(evidence: Mapping[str, Any], response: Any) -> list[str]:
    """Expand compact screenshot references from a persisted rollout response."""

    # Backward compatibility for rows produced by the first standard-judge
    # implementation before compact trajectory references were introduced.
    legacy = evidence.get("screenshots")
    if isinstance(legacy, list) and all(isinstance(item, str) for item in legacy):
        return list(legacy)

    version = evidence.get("schema_version")
    if version != WEBVOYAGER_JUDGE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported WebVoyager judge evidence schema version: {version!r}")
    sequence = evidence.get("screenshot_sequence")
    if not isinstance(sequence, list):
        raise ValueError("WebVoyager judge evidence is missing screenshot_sequence")
    response_images = response_image_urls(response)
    screenshots: list[str] = []
    for item in sequence:
        if not isinstance(item, Mapping):
            raise ValueError("WebVoyager screenshot_sequence entries must be objects")
        if "response_image_index" in item:
            index = item["response_image_index"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(response_images):
                raise ValueError(f"invalid WebVoyager response image index: {index!r}")
            screenshots.append(response_images[index])
            continue
        inline = item.get("inline_data_url")
        if not isinstance(inline, str) or not inline:
            raise ValueError("WebVoyager screenshot_sequence entry has no image reference")
        screenshots.append(inline)
    return screenshots
