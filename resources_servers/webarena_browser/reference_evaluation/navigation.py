# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-site navigation boundary used by reference WebArena evaluators."""

from __future__ import annotations

from typing import Any


async def goto(page: Any, url: str, **kwargs: Any) -> Any:
    """Navigate an async Playwright page.

    The internal source routes this call through its public-web Cloudflare
    helper. WebArena uses self-hosted sites and does not need
    CAPTCHA/proxy intervention during evaluation, so their Gym evaluator keeps
    that concern out of the scoring package.
    """

    return await page.goto(url, **kwargs)


def resolve_after_navigation_sync(_page: Any) -> bool:
    """No-op counterpart of the public-web obstruction resolver."""

    return False
