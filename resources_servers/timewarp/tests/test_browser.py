# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The browser harness against a real Chromium and a local two-origin site (see conftest.py)."""

import re
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.timewarp.app import SiteUrls, TimewarpResourcesServer, TimewarpResourcesServerConfig
from resources_servers.timewarp.browser import (
    BrowserSession,
    SessionLimits,
    SharedBrowser,
    ToolInputError,
    normalize_ref,
    origin_of,
    split_into_parts,
)


LIMITS = SessionLimits(max_observation_chars=4000, action_timeout_ms=5000, navigation_timeout_ms=10000)


def ref_for(observation: str, label: str) -> str:
    """The ref of the first snapshot line naming ``label``."""
    match = re.search(rf'"{re.escape(label)}"[^\n]*\[ref=([a-z0-9]+)\]', observation)
    assert match, f"{label!r} not in observation:\n{observation}"
    return match.group(1)


@pytest.fixture
async def session(site):
    browser = SharedBrowser(headless=True)
    session = await BrowserSession.open(
        await browser.get(),
        start_url=f"{site['allowed']}/",
        allowed_origins=frozenset({site["allowed"]}),
        limits=LIMITS,
    )
    yield session
    await session.close()
    await browser.close()


async def test_click_follows_a_link_by_its_ref(session, site):
    home = await session.observe()
    assert home.startswith(f"URL: {site['allowed']}/\nTitle: Home\n")

    article = await session.click(ref_for(home, "Biology"))

    assert f"URL: {site['allowed']}/article" in article
    assert 'heading "Biology"' in article


async def test_fill_then_enter_submits_the_search_form(session):
    home = await session.observe()
    search = ref_for(home, "Search articles")
    await session.fill(search, "kiwi")

    results = await session.press(search, "Enter")

    assert "/search?q=kiwi" in results
    assert 'heading "Results for kiwi"' in results


async def test_select_option_by_label(session):
    home = await session.observe()
    era = ref_for(home, "Era")
    await session.select_option(era, "2025")
    assert await session._page.locator(f"aria-ref={era}").input_value() == "2025"


async def test_refs_from_a_previous_page_are_rejected(session):
    home = await session.observe()
    stale = ref_for(home, "Biology")
    await session.click(stale)
    with pytest.raises(ToolInputError, match="refs change when the page changes"):
        await session.click(stale)


async def test_goto_outside_the_tasks_sites_is_refused(session, site):
    with pytest.raises(ToolInputError, match="only this task's websites are reachable"):
        await session.goto(f"{site['other']}/")


async def test_clicking_a_link_off_the_tasks_sites_is_blocked(session, site):
    home = await session.observe()
    observation = await session.click(ref_for(home, "Elsewhere"))
    assert f"URL: {site['allowed']}/" in observation
    assert f"Blocked navigation to {site['other']}/" in observation
    # The note is reported once.
    assert "Blocked navigation" not in await session.observe()


async def test_a_link_to_a_new_tab_opens_in_the_current_one(session, site):
    home = await session.observe()
    observation = await session.click(ref_for(home, "Biology in a new tab"))
    assert f"URL: {site['allowed']}/article" in observation
    assert "Tabs:" not in observation
    assert f"URL: {site['allowed']}/\n" in await session.go_back()


async def test_window_open_loads_in_the_current_tab(session, site):
    await session._page.evaluate("url => window.open(url, '_blank')", f"{site['allowed']}/article")
    await session._page.wait_for_url(f"{site['allowed']}/article")
    assert "Tabs:" not in await session.observe()


async def test_long_pages_are_read_in_parts(session, site):
    await session.goto(f"{site['allowed']}/article")
    first = await session.observe()
    total = int(re.search(r"part 1 of (\d+)", first).group(1))
    assert total > 1
    last = await session.observe(total)
    assert "Paragraph 299 of the Biology article" in last and "read further" not in last
    with pytest.raises(ToolInputError, match=f"between 1 and {total}"):
        await session.observe(total + 1)


async def test_back_and_forward(session, site):
    await session.goto(f"{site['allowed']}/article")
    assert f"URL: {site['allowed']}/\n" in await session.go_back()
    assert f"URL: {site['allowed']}/article" in await session.go_forward()
    with pytest.raises(ToolInputError, match="no next page"):
        await session.go_forward()


def test_rollout_through_the_server_routes(site):
    """Seed, browse with tool calls and verify through the FastAPI app, as simple_agent does."""
    server = TimewarpResourcesServer(
        config=TimewarpResourcesServerConfig(
            name="timewarp",
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            site_urls={2: SiteUrls(wiki=site["allowed"], news=site["allowed"], webshop=site["allowed"])},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    with TestClient(server.setup_webserver()) as client:
        client.post("/seed_session", json={"ui_version": 2, "start_site": "wiki"}).raise_for_status()
        home = client.post("/observe", json={}).text
        search = ref_for(home, "Search articles")
        client.post("/fill", json={"ref": search, "text": "kiwi"})
        assert "Results for kiwi" in client.post("/press", json={"ref": search, "key": "Enter"}).text
        assert "Biology" in client.post("/open_site", json={"site": "news"}).text
        assert client.post("/goto", json={"url": f"{site['other']}/"}).text.startswith("Error: cannot open")

        verify = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
                "response": {
                    "id": "r",
                    "created_at": 0,
                    "model": "m",
                    "object": "response",
                    "output": [
                        {
                            "id": "m1",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "Kiwi.", "annotations": []}],
                        }
                    ],
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
                "verifier_metadata": {"eval_types": ["string_match"], "reference_answers": {"must_include": ["kiwi"]}},
                "intent": "q",
                "ui_version": 2,
                "sites": ["wiki"],
            },
        )
        assert verify.json()["reward"] == 1.0
        assert server._episodes == {}


async def test_session_reports_a_dead_browser(site):
    browser = SharedBrowser(headless=True)
    session = await BrowserSession.open(
        await browser.get(),
        start_url=f"{site['allowed']}/",
        allowed_origins=frozenset({site["allowed"]}),
        limits=LIMITS,
    )
    assert session.is_alive()
    await browser.close()
    assert not session.is_alive()


class TestHelpers:
    @pytest.mark.parametrize("raw", ["e12", "[ref=e12]", "ref=e12", " e12 "])
    def test_ref_spellings(self, raw):
        assert normalize_ref(raw) == "e12"

    def test_non_ref_is_rejected(self):
        with pytest.raises(ToolInputError):
            normalize_ref("css=#search")

    def test_origin(self):
        assert origin_of("http://LOCALHOST:5301/abc?x=1") == "http://localhost:5301"
        assert origin_of("about:blank") == ""

    def test_parts_never_exceed_the_budget(self):
        snapshot = "\n".join(["short line"] * 50 + ["x" * 95] + ["tail"])
        parts = split_into_parts(snapshot, 40)
        assert all(len(part) <= 40 for part in parts)
        assert "".join(part.replace("\n", "") for part in parts) == snapshot.replace("\n", "")
