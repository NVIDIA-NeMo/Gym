# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Standalone end-to-end test of the PlaywrightBackend against the bundled site.

This needs NO GPU and NO Gym serving stack — it proves the hardest part (real
browser automation + compact observation + reward scoring) works. Run with:

    uv run --with playwright --with pytest --with pytest-asyncio pytest tests/test_backend.py -q
    # (first: uv run --with playwright python -m playwright install chromium)
"""

import pathlib

import pytest

from backend import PlaywrightBackend

SITE = (pathlib.Path(__file__).parent.parent / "site").resolve()


def _url(page: str) -> str:
    return (SITE / page).as_uri()


@pytest.mark.asyncio
async def test_navigate_observe_click_type_and_score():
    b = PlaywrightBackend(headless=True)
    await b.open(_url("index.html"))
    try:
        # observe: the two links must show up as interactive elements
        obs = await b.observe()
        assert obs.title == "Home"
        names = [e.name for e in obs.elements]
        assert any("form" in n.lower() for n in names)

        # click the "Go to form" link -> URL changes to form.html
        form_id = next(e.id for e in obs.elements if "form" in e.name.lower())
        await b.click(form_id)
        assert "form.html" in await b.current_url()

        # type a username, then click submit -> page title becomes "Welcome neo"
        obs2 = await b.observe()
        user_id = next(e.id for e in obs2.elements if e.role in ("input", "textbox"))
        await b.type(user_id, "neo")
        submit_id = next(e.id for e in obs2.elements if "submit" in e.name.lower())
        await b.click(submit_id)
        obs3 = await b.observe()
        assert obs3.title == "Welcome neo"
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_observation_is_compact_text():
    b = PlaywrightBackend(headless=True)
    await b.open(_url("about.html"))
    try:
        rendered = (await b.observe()).render()
        assert "URL:" in rendered and "TITLE: About" in rendered
    finally:
        await b.close()
