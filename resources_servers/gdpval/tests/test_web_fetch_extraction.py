# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fetch that returns a fragment, or nothing, must not look like a successful read.

trafilatura defaults to precision over recall and drops link text, so a heavily templated
page can extract to a few hundred characters. Observed on a real run: 313 characters
returned for a 141 KB page, with no error - the model reasoned on the fragment and only
recovered because it happened to notice and re-fetch with curl. A total extraction failure
was worse: an empty ``<body>`` with no error, indistinguishable from a blank page.
"""

import asyncio
import threading

import httpx

from resources_servers.gdpval import tavily_search
from resources_servers.gdpval.tavily_search import fetch_web_page


_LINK_HEAVY = """<html><body>
<nav><a href="/a">Home</a></nav>
<main>
  <h1>Service tiers</h1>
  <ul>
    <li><a href="/bronze">Bronze tier covers same-day courier dispatch</a></li>
    <li><a href="/silver">Silver tier adds pharmacy coordination windows</a></li>
    <li><a href="/gold">Gold tier guarantees a two-hour infusion slot</a></li>
  </ul>
  <table>
    <tr><th>Tier</th><th>Price</th></tr>
    <tr><td>Bronze</td><td>1200</td></tr>
    <tr><td>Gold</td><td>4800</td></tr>
  </table>
</main></body></html>"""

_NO_TEXT = "<html><head><title>t</title></head><body><script>var a=1;</script></body></html>"


def _fetch(html: str, status: int = 200, url: str = "https://example.test/p") -> str:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=html, headers={"content-type": "text/html; charset=utf-8"})

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_web_page(url, client)

    return asyncio.run(run())


def test_a_sliver_of_a_large_page_is_flagged_as_a_fragment():
    """The real failure: 313 characters returned for a 141 KB page, consumed as the whole page."""
    html = (
        "<html><body><main><p>One short sentence of real content.</p></main>"
        "<script>" + "var padding = 1;" * 4000 + "</script></body></html>"
    )
    assert len(html) > 20_000

    text = _fetch(html)

    assert "likely a fragment" in text, "the model was handed a sliver with no indication of it"
    assert "One short sentence of real content." in text, "the extracted text itself must survive"


def test_a_short_page_is_not_flagged():
    """A genuinely short page is not a fragment; crying wolf would teach the model to ignore it."""
    text = _fetch(_LINK_HEAVY)

    assert "likely a fragment" not in text
    assert "Service tiers" in text


def test_unextractable_page_is_an_error_not_an_empty_body():
    result = _fetch(_NO_TEXT)

    assert "<error>" in result
    assert "<body></body>" not in result, "an empty body reads to the model as a blank page"
    assert "no readable text" in result


def test_unextractable_page_reports_what_was_fetched():
    """The model needs to know bytes arrived, so it retries rather than concluding 'empty'."""
    result = _fetch(_NO_TEXT)

    assert str(len(_NO_TEXT)) in result


def test_normal_page_still_returns_a_body():
    result = _fetch(_LINK_HEAVY)

    assert "<error>" not in result
    assert "<body>" in result
    assert "Service tiers" in result


def test_an_http_error_shows_the_model_the_status_and_url():
    result = _fetch("not found", status=404)

    assert "<error>Client error &#x27;404 Not Found&#x27; for url &#x27;https://example.test/p&#x27;" in result


def test_an_invalid_url_is_reported_to_the_model_instead_of_raised():
    result = _fetch("unused", url="http://[::1")

    assert result.startswith("<web_fetch><url>http://[::1</url><error>")


def test_extraction_runs_off_the_event_loop(monkeypatch):
    threads = []
    monkeypatch.setattr(
        tavily_search, "_extract_page_text", lambda html: threads.append(threading.current_thread()) or "text"
    )

    _fetch(_LINK_HEAVY)

    assert threads and threads[0] is not threading.main_thread()


async def _read_streamed(chunks: list[bytes], encoding: str = "utf-8") -> tuple[str, int]:
    """Run ``_read_page`` over a streamed body; return its text and how many chunks were pulled."""
    pulled = 0

    async def body():
        nonlocal pulled
        for chunk in chunks:
            pulled += 1
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers={"content-type": f"text/html; charset={encoding}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await tavily_search._read_page("https://example.test/p", client), pulled


def test_a_page_over_the_size_cap_stops_being_read_and_is_cut(monkeypatch):
    monkeypatch.setattr(tavily_search, "MAX_BODY_BYTES", 1000)

    text, pulled = asyncio.run(_read_streamed([b"x" * 400] * 10))

    assert text == "x" * 1000
    assert pulled < 10, "the reader keeps pulling chunks after the cap"


def test_the_cut_through_a_multibyte_character_decodes_with_a_replacement(monkeypatch):
    monkeypatch.setattr(tavily_search, "MAX_BODY_BYTES", 1000)
    chunks = [b"a" + "é".encode() * 250, "é".encode() * 250]

    text, _ = asyncio.run(_read_streamed(chunks))

    assert text == "a" + "é" * 499 + "\ufffd"


def test_only_the_part_of_a_large_page_before_the_cap_reaches_the_model(monkeypatch):
    monkeypatch.setattr(tavily_search, "MAX_BODY_BYTES", 1000)
    head = "<html><body><main><p>The early paragraph is read.</p>"
    filler = "<p>" + "padding words " * 30 + "</p>"
    tail = "<p>The late paragraph is never read.</p></main></body></html>"
    page = (head + filler * 5 + tail).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=page, headers={"content-type": "text/html; charset=utf-8"})

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_web_page("https://example.test/p", client)

    result = asyncio.run(run())

    assert "The early paragraph is read." in result
    assert "The late paragraph is never read." not in result
