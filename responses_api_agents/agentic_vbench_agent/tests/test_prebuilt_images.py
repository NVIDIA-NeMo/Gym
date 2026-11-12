# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from responses_api_agents.agentic_vbench_agent.core import (
    prebuilt_image_name,
    resolve_prebuilt_image,
)


def test_prebuilt_image_name_matches_prebuild_script_tag() -> None:
    assert (
        prebuilt_image_name("agentic-vbench-assembly-task1") == "localhost/avb-prebuild-agentic-vbench-assembly-task1"
    )
    assert prebuilt_image_name("Exp-Declip-Task01") == "localhost/avb-prebuild-exp-declip-task01"


def test_resolve_uses_prebuilt_image_when_present() -> None:
    seen = []

    async def exists(image):
        seen.append(image)
        return True

    assert (
        asyncio.run(resolve_prebuilt_image("exp-declip-task01", environ={}, image_exists=exists))
        == "localhost/avb-prebuild-exp-declip-task01"
    )
    assert seen == ["localhost/avb-prebuild-exp-declip-task01"]


def test_resolve_falls_back_to_build_when_image_missing() -> None:
    async def exists(image):
        return False

    assert asyncio.run(resolve_prebuilt_image("exp-declip-task01", environ={}, image_exists=exists)) is None


def test_resolve_can_be_disabled() -> None:
    async def exists(image):
        raise AssertionError("must not look up images when disabled")

    assert (
        asyncio.run(
            resolve_prebuilt_image(
                "exp-declip-task01", environ={"AGENTIC_VBENCH_PREBUILT_IMAGES": "0"}, image_exists=exists
            )
        )
        is None
    )
