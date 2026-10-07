# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tavily web search and page fetch behind the ``web_search`` and ``fetch_web_page`` tools.

Both return the text the model sees and never raise. ``TavilySearch`` accepts a single key, a list
of keys, or a comma-separated string with optional surrounding ``[...]`` brackets (the format EFB
injects). With several keys it rotates round-robin per call and retries key-specific failures
(401, 403, 429) and 5xx with the next key, up to ``max_sweeps × len(keys)`` attempts per call.
"""

from __future__ import annotations

from html import escape
from typing import List, Optional, Union

import httpx


MAX_LENGTH = 40_000
TIMEOUT = 60 * 3

# HTTP statuses that indicate the *current key* is the problem (auth or
# quota). On these, we rotate to the next key and retry.
KEY_ROTATION_RETRY_STATUSES = frozenset({401, 403, 429})


def _should_rotate_on_status(status: int) -> bool:
    """Whether to rotate to the next key after seeing this HTTP status.

    Rotates on key-specific 4xx (auth/quota) AND any 5xx. The 5xx rationale
    is conservative: even though all keys hit the same upstream
    (api.tavily.com), Tavily fronts multiple backends and occasional 5xx
    are observed for individual requests; a free retry with the next key
    costs nothing and often succeeds. Non-rotating failures (404, 4xx that
    aren't auth/quota, malformed-query) bail after one attempt — rotating
    wouldn't help and would mask the real issue.
    """
    return status in KEY_ROTATION_RETRY_STATUSES or 500 <= status < 600


def web_client() -> httpx.AsyncClient:
    """Return a client for one tool call."""
    return httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)


def truncate_msg(msg: str, max_length: int) -> str:
    """Truncate long messages by removing middle portion, keeping start and end with ellipsis indicator."""
    msg_len = len(msg)
    if msg_len <= max_length:
        return msg
    else:
        return (
            msg[: max_length // 2]
            + f"\n... This content has been truncated from an original {msg_len} characters to stay below {max_length} characters ...\n"
            + msg[-max_length // 2 :]
        )


class TavilySearch:
    """Run Tavily searches with key rotation."""

    def __init__(self, *, api_keys: Union[str, List[str]], max_sweeps: int = 1) -> None:
        if max_sweeps < 1:
            raise ValueError(f"max_sweeps must be >= 1, got {max_sweeps}")
        self._api_keys: List[str] = self._parse_keys(api_keys)
        if not self._api_keys:
            raise ValueError("at least one Tavily API key is required")
        self._key_idx: int = 0
        self._max_sweeps = max_sweeps

    @staticmethod
    def _parse_keys(value: Union[str, List[str]]) -> List[str]:
        """Normalise ``value`` into a deduplicated, non-empty list of keys.

        Accepts:
        - ``List[str]`` directly,
        - ``"k1,k2,k3"`` (comma-separated string),
        - ``"[k1,k2,k3]"`` (EFB env-var format with surrounding brackets),
        - ``""`` or ``None`` → empty list.
        """
        if isinstance(value, list):
            keys = value
        else:
            s = (value or "").strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            keys = s.split(",") if s else []
        out: List[str] = []
        seen: set[str] = set()
        for k in keys:
            ks = k.strip()
            if ks and ks not in seen:
                seen.add(ks)
                out.append(ks)
        return out

    def _next_key(self) -> str:
        """Return the next key in round-robin order."""
        k = self._api_keys[self._key_idx % len(self._api_keys)]
        self._key_idx += 1
        return k

    async def search(self, query: str, client: httpx.AsyncClient) -> str:
        last_status: Optional[int] = None
        n_keys = len(self._api_keys)
        n_attempts = self._max_sweeps * n_keys
        for _ in range(n_attempts):
            api_key = self._next_key()
            try:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "max_results": 5, "include_answer": True},
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                # Network-level failure — not key-specific. Bail with this error.
                return f"<error>{escape(str(exc))}</error>"

            if _should_rotate_on_status(resp.status_code):
                last_status = resp.status_code
                continue  # rotate to next key

            try:
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                return f"<error>{escape(str(exc))}</error>"

            data = resp.json()
            parts: list[str] = []
            if data.get("answer"):
                parts.append(f"<answer>{escape(data['answer'])}</answer>")

            results = data.get("results", [])
            results_xml = "\n".join(
                f"<result>\n<title>{escape(r.get('title', ''))}</title>"
                f"\n<url>{escape(r.get('url', ''))}</url>"
                f"\n<content>{escape(r.get('content', ''))}</content>\n</result>"
                for r in results
            )
            parts.append(f"<results>\n{results_xml}\n</results>")

            return truncate_msg("\n".join(parts), MAX_LENGTH)

        # All attempts exhausted on retryable errors (auth, quota, or 5xx).
        return (
            f"<error>Tavily exhausted {n_attempts} attempt(s) "
            f"({n_keys} key(s) × {self._max_sweeps} sweep(s)) on retryable errors "
            f"(last status={last_status}). Refresh keys or check upstream.</error>"
        )


# A markdown extraction this much smaller than the page it came from is usually a fragment,
# not a short page. Observed on a real run: 313 characters returned for a 141 KB page (0.2%),
# which the model consumed as if it were the whole page.
_SUSPICIOUS_EXTRACTION_RATIO = 0.01
_MIN_HTML_FOR_RATIO_CHECK = 20_000


def _extract_page_text(html: str) -> str:
    """Extract readable text from *html* as markdown, falling back to a bare pass.

    The fallback matters because ``extract`` returning ``None`` is otherwise indistinguishable
    from a genuinely empty page.
    """
    import trafilatura

    extracted = trafilatura.extract(html, output_format="markdown")
    if extracted and extracted.strip():
        return extracted.strip()
    return (trafilatura.extract(html, include_comments=True) or "").strip()


async def fetch_web_page(url: str, client: httpx.AsyncClient) -> str:
    try:
        resp = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        resp.raise_for_status()

        body_md = _extract_page_text(resp.text)
        if not body_md:
            # An empty <body> with no error reads to the model as "this page is blank",
            # and it will happily reason on from nothing. Say what happened instead.
            return (
                f"<web_fetch><url>{url}</url>"
                f"<error>fetched {len(resp.text)} bytes but no readable text could be extracted; "
                f"the page is likely script-rendered. Try fetching a different URL, or retrieve it "
                f"inside code_exec if you need the raw markup.</error></web_fetch>"
            )
        if len(resp.text) >= _MIN_HTML_FOR_RATIO_CHECK and (
            len(body_md) < _SUSPICIOUS_EXTRACTION_RATIO * len(resp.text)
        ):
            # Extraction succeeded but returned a sliver. Without saying so, the model treats
            # the sliver as the whole page; with it, it can re-fetch the raw markup itself.
            body_md = (
                f"[extracted only {len(body_md)} characters of readable text from a "
                f"{len(resp.text)}-byte page; this is likely a fragment, so re-fetch the raw "
                f"markup inside code_exec if you need the rest]\n\n{body_md}"
            )
        return f"<web_fetch><url>{url}</url><body>{truncate_msg(body_md, MAX_LENGTH)}</body></web_fetch>"
    except httpx.HTTPError as exc:
        return f"<web_fetch><url>{url}</url><error>{escape(str(exc))}</error></web_fetch>"
