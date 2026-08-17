# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fetch that returns a fragment, or nothing, must not look like a successful read.

trafilatura defaults to precision over recall and drops link text, so a heavily templated
page can extract to a few hundred characters. Observed on a real run: 313 characters
returned for a 141 KB page, with no error - the model reasoned on the fragment and only
recovered because it happened to notice and re-fetch with curl. A total extraction failure
was worse: an empty ``<body>`` with ``success=True``, indistinguishable from a blank page.
"""

import asyncio

from responses_api_agents.stirrup_agent.tavily_search import _fetch_executor, _FetchParams


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


class _Resp:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, text: str):
        self._text = text

    async def get(self, url, headers=None):  # noqa: ARG002 - signature mirrors httpx
        return _Resp(self._text)


def _fetch(html: str):
    return asyncio.run(_fetch_executor(_FetchParams(url="https://example.test/p"), client=_Client(html)))


def test_a_sliver_of_a_large_page_is_flagged_as_a_fragment():
    """The real failure: 313 characters returned for a 141 KB page, consumed as the whole page."""
    html = (
        "<html><body><main><p>One short sentence of real content.</p></main>"
        "<script>" + "var padding = 1;" * 4000 + "</script></body></html>"
    )
    assert len(html) > 20_000

    text = _fetch(html).content

    assert "likely a fragment" in text, "the model was handed a sliver with no indication of it"
    assert "One short sentence of real content." in text, "the extracted text itself must survive"


def test_a_short_page_is_not_flagged():
    """A genuinely short page is not a fragment; crying wolf would teach the model to ignore it."""
    text = _fetch(_LINK_HEAVY).content

    assert "likely a fragment" not in text
    assert "Service tiers" in text


def test_unextractable_page_is_an_error_not_an_empty_body():
    result = _fetch(_NO_TEXT)

    assert result.success is False
    assert "<body></body>" not in result.content, "an empty body reads to the model as a blank page"
    assert "no readable text" in result.content


def test_unextractable_page_reports_what_was_fetched():
    """The model needs to know bytes arrived, so it retries rather than concluding 'empty'."""
    result = _fetch(_NO_TEXT)

    assert str(len(_NO_TEXT)) in result.content


def test_normal_page_still_returns_a_body():
    result = _fetch(_LINK_HEAVY)

    assert result.success is True
    assert "<body>" in result.content
    assert "Service tiers" in result.content
