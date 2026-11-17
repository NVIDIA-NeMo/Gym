# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Chromium checks for focused input; run in the provisioned worker runtime."""

import asyncio
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import pytest_asyncio

from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT


pytest.importorskip("playwright", reason="Real browser checks require the SaaS worker runtime")
pytest.importorskip("mcp", reason="Real browser checks require the SaaS worker runtime")
sys.path.insert(0, str(BENCHMARK_ROOT))
from saas_bench.opencode import browser as browser_module


HTML = b"""<body>
<div role="textbox" aria-label="Document" contenteditable="true"></div>
<canvas id="canvas" width="200" height="100" tabindex="0"></canvas>
<textarea id="helper" style="position:fixed;left:-200px;width:100px"></textarea>
<input aria-label="Normal"><iframe src="/child"></iframe>
<script>
const edit=document.querySelector('[contenteditable]');
edit.textContent=localStorage.saved||'';
edit.oninput=()=>localStorage.saved=edit.textContent;
window.keydowns=0;document.addEventListener('keydown',()=>window.keydowns++);
document.querySelector('#canvas').onclick=()=>document.querySelector('#helper').focus();
</script></body>"""


@pytest_asyncio.fixture
async def browser(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'<textarea aria-label="Child"></textarea>' if self.path == "/child" else HTML)

        def log_message(self, *args):
            pass

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)]
    for server in servers:
        Thread(target=server.serve_forever, daemon=True).start()
    urls = [f"http://127.0.0.1:{s.server_port}" for s in servers]
    tools = browser_module.BrowserTools({"allowed_origins": urls, "input_files": [], "artifacts": str(tmp_path)})
    monkeypatch.setattr(browser_module, "BrowserTools", lambda config: tools)
    server = browser_module.create_server({})
    try:
        await tools.start()
        await tools.page.goto(urls[0])
        yield tools, server, urls
    finally:
        await tools.close()
        for http in servers:
            http.shutdown()
            http.server_close()


def function(server, name):
    return server._tool_manager.get_tool(name).fn


def observation(result):
    return json.loads(result[0])["observation_id"]


@pytest.mark.asyncio
async def test_focused_document_insert_replace_and_persist(browser):
    tools, server, urls = browser
    click = function(server, "interact")
    insert = function(server, "insert_text")
    press = function(server, "press_key")
    result = await click(action="click", role="textbox", name="Document", expected_url=urls[0])
    result = await insert(
        text="Caf\u00e9 \u4f60\u597d\nsecond line", observation_id=observation(result), expected_url=urls[0]
    )
    assert "Caf\u00e9 \u4f60\u597d" in await tools.page.get_by_role("textbox", name="Document").inner_text()
    result = await press(key="Control+A", observation_id=observation(result), expected_url=urls[0])
    result = await insert(text="Persisted replacement", observation_id=observation(result), expected_url=urls[0])
    assert json.loads(result[0])["action_status"] == "succeeded"
    await tools.page.reload()
    assert await tools.page.get_by_role("textbox", name="Document").inner_text() == "Persisted replacement"


@pytest.mark.asyncio
async def test_canvas_focus_hidden_textarea_and_key_event_modes(browser):
    tools, server, urls = browser
    result = await function(server, "interact")(action="click", selector="#canvas", expected_url=urls[0])
    result = await function(server, "insert_text")(
        text="echo hello", observation_id=observation(result), expected_url=urls[0]
    )
    assert await tools.page.locator("#helper").input_value() == "echo hello"
    assert await tools.page.evaluate("window.keydowns") == 0
    result = await function(server, "insert_text")(
        text="abc", mode="key_events", observation_id=observation(result), expected_url=urls[0]
    )
    assert await tools.page.locator("#helper").input_value() == "echo helloabc"
    assert await tools.page.evaluate("window.keydowns") == 3


@pytest.mark.asyncio
async def test_focused_child_frame_and_application_binding(browser):
    tools, server, urls = browser
    result = await function(server, "interact")(
        action="click", selector="textarea", frame_path=["iframe"], expected_url=urls[0]
    )
    result = await function(server, "insert_text")(
        text="Inside frame", observation_id=observation(result), expected_url=urls[0]
    )
    assert await tools.page.frame_locator("iframe").locator("textarea").input_value() == "Inside frame"
    with pytest.raises(ValueError, match="outside assigned"):
        await function(server, "insert_text")(
            text="Wrong app", observation_id=observation(result), expected_url="http://127.0.0.1:1"
        )
    second = await tools.context.new_page()
    await second.goto(urls[1])
    await second.get_by_role("textbox", name="Normal").fill("untouched")
    with pytest.raises(ValueError, match="fresh screenshot"):
        await function(server, "insert_text")(
            text="Wrong tab", observation_id=f"{tools.sequence:05d}", expected_url=urls[1]
        )
    assert await second.get_by_role("textbox", name="Normal").input_value() == "untouched"
    schema = next(t for t in await server.list_tools() if t.name == "insert_text").inputSchema
    assert "expected_url" in schema["required"] and "observation_id" in schema["required"]


@pytest.mark.asyncio
async def test_stale_observation_and_invalid_chord_do_not_insert_text(browser):
    tools, server, urls = browser
    result = await function(server, "interact")(action="click", role="textbox", name="Normal", expected_url=urls[0])
    stale = observation(result)
    await tools.observe()
    with pytest.raises(ValueError, match="latest observation_id"):
        await function(server, "insert_text")(text="wrong", observation_id=stale, expected_url=urls[0])
    with pytest.raises(ValueError, match="insert_text"):
        await function(server, "press_key")(
            key="Control+not-a-key-name", observation_id=f"{tools.sequence:05d}", expected_url=urls[0]
        )
    result = await function(server, "press_key")(key="x", observation_id=f"{tools.sequence:05d}", expected_url=urls[0])
    assert await tools.page.get_by_role("textbox", name="Normal").input_value() == "x"
    assert json.loads(result[0])["screenshot_available"]


@pytest.mark.asyncio
async def test_renderer_crash_records_trusted_fault_and_stops_actions(browser):
    tools, server, urls = browser
    await tools.observe()
    client = await tools.context.new_cdp_session(tools.page)
    # This is actual Chromium renderer termination, not a mocked error string.
    with pytest.raises((browser_module.PlaywrightError, asyncio.TimeoutError)):
        await asyncio.wait_for(client.send("Page.crash"), timeout=3)
    for _ in range(50):
        if tools.browser_fault:
            break
        await asyncio.sleep(0.02)
    receipt = json.loads((tools.artifacts / "browser_fault.json").read_text())
    assert receipt["kind"] == "renderer_crash"
    assert receipt["observation_id"] == f"{tools.sequence:05d}"
    assert "limits" in receipt and "shared_memory" in receipt
    with pytest.raises(RuntimeError, match="fresh runtime retry"):
        await function(server, "navigate")(url=urls[1])
    assert len(tools.context.pages) == 1  # No replay or replacement tab.


@pytest.mark.asyncio
async def test_browser_disconnect_is_fault_but_clean_shutdown_is_not(browser):
    tools, server, urls = browser
    await tools.browser.close()  # Unexpected backend disappearance, not tools.close().
    assert json.loads((tools.artifacts / "browser_fault.json").read_text())["kind"] == "browser_disconnected"
    await tools.close()
    assert tools.browser_fault["kind"] == "browser_disconnected"  # First cause retained.


@pytest.mark.asyncio
async def test_normal_tab_close_and_selector_errors_are_not_browser_faults(browser):
    tools, server, urls = browser
    await function(server, "navigate")(url=urls[1])
    await function(server, "close_tab")()
    with pytest.raises(ValueError):
        await function(server, "interact")(action="click", selector='button "Target crashed"', expected_url=urls[0])
    assert tools.browser_fault is None
    await tools.close()
    assert not (tools.artifacts / "browser_fault.json").exists()


@pytest.mark.asyncio
async def test_target_creation_failure_is_classified_only_at_browser_api(browser, monkeypatch):
    tools, server, urls = browser

    async def fail_creation():
        raise browser_module.PlaywrightError("BrowserContext.new_page: Target crashed")

    monkeypatch.setattr(tools.context, "new_page", fail_creation)
    with pytest.raises(browser_module.PlaywrightError, match="Target crashed"):
        await function(server, "navigate")(url=urls[1])
    receipt = json.loads((tools.artifacts / "browser_fault.json").read_text())
    assert receipt["kind"] == "target_creation_crash"
    assert len(tools.context.pages) == 1
