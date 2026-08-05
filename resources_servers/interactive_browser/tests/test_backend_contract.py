# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""One contract, run against every shipped backend.

Needs NO GPU and NO Gym serving stack — it proves the hardest part (real browser
automation + compact observation + the state reward scoring reads) works, and
that a locally-launched browser and one reached over CDP behave identically.
Run with:

    uv run --with playwright --with pytest --with pytest-asyncio python -m pytest tests -q
    # (first: uv run --with playwright python -m playwright install chromium)
"""

import pathlib

import pytest
from browser import create_backend


SITE = (pathlib.Path(__file__).parent.parent / "site").resolve()


def _url(page: str) -> str:
    return (SITE / page).as_uri()


@pytest.fixture(params=["local_playwright", "remote_cdp"])
def backend_config(request):
    """Backend config for each shipped backend, as it would appear in YAML."""
    if request.param == "local_playwright":
        return {"local_playwright": {"headless": True}}
    # Remote path, minus the third-party service: a real CDP endpoint served by
    # a browser this test started.
    cdp_url = request.getfixturevalue("cdp_endpoint")
    return {"remote_cdp": {"session_provider": {"static_cdp": {"cdp_url": cdp_url}}}}


@pytest.mark.asyncio
async def test_navigate_observe_click_type_and_score(backend_config):
    b = create_backend(backend_config)
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
        # what `dom_contains` scoring reads
        assert "welcome neo" in (obs3.title + " " + await b.text()).lower()
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_observe_stops_at_the_element_budget(backend_config):
    b = create_backend(backend_config)
    await b.open(_url("index.html"))
    try:
        obs = await b.observe(max_elements=1)
        assert len(obs.elements) == 1
        assert obs.truncated is True
        # The policy must be able to see that the list is incomplete.
        assert "truncated at 1 elements" in obs.render(max_elements=1)
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_observation_is_compact_text(backend_config):
    b = create_backend(backend_config)
    await b.open(_url("about.html"))
    try:
        rendered = (await b.observe()).render()
        assert "URL:" in rendered and "TITLE: About" in rendered
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_isolated_state_between_episodes(backend_config):
    """Two rollouts on one backend config must not share page state."""
    first = create_backend(backend_config)
    second = create_backend(backend_config)
    await first.open(_url("form.html"))
    await second.open(_url("about.html"))
    try:
        assert "form.html" in await first.current_url()
        assert "about.html" in await second.current_url()
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_driving_a_closed_backend_raises(backend_config):
    b = create_backend(backend_config)
    await b.open(_url("index.html"))
    await b.close()
    with pytest.raises(RuntimeError):
        await b.observe()
