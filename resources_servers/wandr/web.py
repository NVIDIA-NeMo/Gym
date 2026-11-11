# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-web page fetching shared by the agent's fetch tool and the verifier's direct and Exa page fetches."""

import asyncio
import ipaddress
import multiprocessing
import socket
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from io import BytesIO
from typing import ClassVar, Optional
from urllib.parse import urljoin, urlsplit

import trafilatura
from aiohttp import ClientError, ClientResponse, ClientTimeout
from pypdf import PdfReader
from yarl import URL

from nemo_gym.client import request


FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_PDF_PAGES = 300
MAX_REDIRECTS = 10
MAX_ADDRESSES = 3
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
READ_CHUNK_BYTES = 1 << 16
# trafilatura/lxml corrupt the heap when several threads parse at once (SIGABRT), so extraction runs in
# worker processes: a crash then costs one page, not the server or the verifier.
EXTRACT_WORKERS = 4

EXA_BASE_URL = "https://api.exa.ai"
# Exa answers these when the key is missing, invalid or out of credit: every page would fail, so it is fatal.
EXA_FATAL_STATUSES = frozenset({401, 402, 403})

# Sends one GET without following redirects: (url, timeout_s, headers, server_hostname) -> response.
GetFunction = Callable[[str, float, dict[str, str], Optional[str]], Awaitable[ClientResponse]]
# Sends one JSON POST: (url, headers, payload, timeout_s) -> (HTTP status, decoded JSON body or None).
PostFunction = Callable[[str, dict[str, str], dict, float], Awaitable[tuple[int, object]]]


class UnsafeURLError(ValueError):
    """A URL is malformed or can reach a non-public address."""


@dataclass
class PageText:
    title: str = ""
    text: str = ""
    description: str = ""
    author: str = ""
    published_date: str = ""


@dataclass
class FetchedPage:
    url: str
    final_url: str
    content_type: str
    body: bytes
    truncated: bool


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (
        address.is_link_local
        or address.is_loopback
        or address.is_multicast
        or address.is_private
        or address.is_reserved
        or address.is_unspecified
    )


def check_http_url(url: object) -> tuple[str, int]:
    """Return (hostname, port) of an http(s) URL that WANDR's verifier accepts, else raise UnsafeURLError.

    The rules are those of the pinned verifier's ``parse_http_url``, so a URL that passes here is not
    the one row that makes the verifier reject a whole submission.
    """
    if not isinstance(url, str) or not url:
        raise UnsafeURLError("url must be a non-empty string")
    if "\\" in url or any(char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159 for char in url):
        raise UnsafeURLError("url contains whitespace, control characters or backslashes")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise UnsafeURLError(f"url is malformed: {error}") from error
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError("url scheme must be http or https")
    if not parsed.hostname:
        raise UnsafeURLError("url has no hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("url must not contain credentials")
    if "%" in parsed.hostname:
        raise UnsafeURLError("url must not contain an IPv6 zone identifier")
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None and not _is_public(literal):
        raise UnsafeURLError("url points to a non-public address")
    return parsed.hostname, port or (443 if scheme == "https" else 80)


async def resolve(hostname: str, port: int, *, timeout: float) -> list[str]:
    """The addresses a hostname resolves to, in resolver order; an IP literal resolves to itself."""
    try:
        answers = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                hostname, port, type=socket.SOCK_STREAM, flags=socket.AI_ADDRCONFIG
            ),
            timeout=timeout,
        )
    except (OSError, asyncio.TimeoutError) as error:
        raise UnsafeURLError("url hostname could not be resolved") from error
    return list(dict.fromkeys(str(answer[4][0]).split("%", 1)[0] for answer in answers if answer[4]))


async def public_addresses(url: str, *, timeout: float) -> list[str]:
    """The URL host's addresses in resolver order; raise UnsafeURLError if any of them is not public."""
    check_http_url(url)
    parsed = URL(url)
    # The name aiohttp itself would resolve: UTS46-encoded, without a trailing dot.
    name = parsed.raw_host.rstrip(".")
    if name.lower() == "localhost" or name.lower().endswith(".localhost"):
        raise UnsafeURLError("url points to a non-public address")
    addresses = await resolve(name, parsed.port, timeout=timeout)
    if not addresses:
        raise UnsafeURLError("url hostname returned no addresses")
    if not all(_is_public(ipaddress.ip_address(address)) for address in addresses):
        raise UnsafeURLError("url points to a non-public address")
    return addresses


def _pinned_request(url: str, address: str) -> tuple[str, dict[str, str], Optional[str]]:
    """The URL with its host replaced by a checked address, plus the Host header and TLS name aiohttp would use."""
    parsed = URL(url)
    pinned = parsed.with_host(address)
    server_hostname = parsed.raw_host.rstrip(".") if parsed.scheme == "https" else None
    return str(pinned), {"Host": parsed.host_port_subcomponent}, server_hostname


async def gym_get(url: str, timeout: float, headers: dict[str, str], server_hostname: Optional[str]) -> ClientResponse:
    """One GET through Gym's shared aiohttp client, without following redirects."""
    return await request(
        "GET",
        url=url,
        headers={**FETCH_HEADERS, **headers},
        timeout=ClientTimeout(total=timeout),
        allow_redirects=False,
        server_hostname=server_hostname,
        _max_num_tries=2,
    )


async def gym_post_json(url: str, headers: dict[str, str], payload: dict, timeout: float) -> tuple[int, object]:
    """One JSON POST through Gym's shared aiohttp client; the body is decoded only for HTTP 200."""
    response = await request(
        "POST", url=url, headers=headers, json=payload, timeout=ClientTimeout(total=timeout), _max_num_tries=3
    )
    try:
        if response.status != 200:
            return response.status, None
        return response.status, await response.json(content_type=None)
    finally:
        response.release()


async def read_body(response: ClientResponse, max_bytes: int) -> tuple[bytes, bool]:
    """Read the body until EOF or max_bytes; returns (body, truncated)."""
    chunks: list[bytes] = []
    size = 0
    while size < max_bytes:
        chunk = await response.content.read(min(READ_CHUNK_BYTES, max_bytes - size))
        if not chunk:
            return b"".join(chunks), False
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks), not response.content.at_eof()


async def _connect(get: GetFunction, url: str, *, timeout: float) -> ClientResponse:
    """GET the URL from the first of its checked addresses that accepts a connection."""
    error: Optional[OSError] = None
    for address in (await public_addresses(url, timeout=timeout))[:MAX_ADDRESSES]:
        pinned_url, headers, server_hostname = _pinned_request(url, address)
        try:
            return await get(pinned_url, timeout, headers, server_hostname)
        except OSError as unreachable:  # includes aiohttp's connector errors and timeouts
            error = unreachable
    raise error


async def fetch_public(get: GetFunction, url: str, *, timeout: float, max_bytes: int) -> FetchedPage:
    """GET a public page; raises UnsafeURLError or ValueError.

    Every hop, redirects included, connects to an address checked to be public, so a DNS answer that
    changes between the check and the connection cannot reach a private network.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        response = await _connect(get, current, timeout=timeout)
        try:
            if response.status in REDIRECT_STATUSES and response.headers.get("Location"):
                current = urljoin(current, response.headers["Location"])
            elif response.status >= 400:
                raise ValueError(f"HTTP {response.status}")
            else:
                body, truncated = await read_body(response, max_bytes)
                return FetchedPage(url, current, response.headers.get("Content-Type", ""), body, truncated)
        finally:
            response.release()
    raise ValueError(f"more than {MAX_REDIRECTS} redirects")


def _charset(content_type: str) -> str:
    for part in content_type.split(";"):
        name, _, value = part.strip().partition("=")
        if name.lower() == "charset" and value:
            return value.strip("\"'")
    return "utf-8"


def extract_page(body: bytes, content_type: str, url: str) -> PageText:
    """Return the main text of an HTML, PDF or plain-text response body; raises ValueError if unreadable."""
    media_type = content_type.split(";")[0].strip().lower()
    if media_type == "application/pdf" or body[:5] == b"%PDF-":
        try:
            reader = PdfReader(BytesIO(body))
            text = "\n".join(page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES])
            title = str((reader.metadata or {}).get("/Title") or "")
        except Exception as error:  # pypdf raises many unrelated types on broken or encrypted files
            raise ValueError(f"unreadable PDF: {type(error).__name__}: {error}") from error
        return PageText(title=title, text=text.strip())

    try:
        markup = body.decode(_charset(content_type), errors="replace")
    except LookupError:
        markup = body.decode("utf-8", errors="replace")
    if "html" not in media_type and not markup.lstrip()[:1] == "<":
        return PageText(text=markup.strip())

    text = trafilatura.extract(markup, url=url, include_tables=True, include_comments=False, favor_recall=True)
    metadata = trafilatura.extract_metadata(markup, default_url=url)
    return PageText(
        title=(metadata.title if metadata else None) or "",
        text=(text or "").strip(),
        description=(metadata.description if metadata else None) or "",
        author=(metadata.author if metadata else None) or "",
        published_date=(metadata.date if metadata else None) or "",
    )


_extract_pool: Optional[ProcessPoolExecutor] = None


def _pool() -> ProcessPoolExecutor:
    global _extract_pool
    if _extract_pool is None:
        _extract_pool = ProcessPoolExecutor(
            max_workers=EXTRACT_WORKERS, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=500
        )
    return _extract_pool


async def extract_page_isolated(
    body: bytes, content_type: str, url: str, *, extract: Callable[[bytes, str, str], PageText] = extract_page
) -> PageText:
    """extract_page in a worker process; a crashed worker becomes ValueError and the pool is rebuilt."""
    global _extract_pool
    pool = _pool()
    try:
        return await asyncio.get_running_loop().run_in_executor(pool, extract, body, content_type, url)
    except BrokenProcessPool as error:
        if _extract_pool is pool:
            _extract_pool = None
            pool.shutdown(wait=False, cancel_futures=True)
        raise ValueError(f"page text extraction crashed: {error}") from error


@dataclass
class VerifierPage:
    """The page fields WANDR's verifier reads from its fetch client (``components._parse_page``)."""

    url: str
    title: str = ""
    description: str = ""
    author: str = ""
    hostname: Optional[str] = None
    published_date: str = ""
    is_paywall: bool = False
    content: Optional[str] = None
    error: Optional[str] = None


@dataclass
class VerifierPages:
    pages: list[VerifierPage]


class DirectFetchClient:
    """WANDR's fetch-client contract (``client.content.fetch(urls=...)`` returning ``.pages``) over direct GETs."""

    def __init__(self, get: GetFunction, *, timeout: float, max_bytes: int) -> None:
        self._get = get
        self._timeout = timeout
        self._max_bytes = max_bytes
        self.content = self

    async def close(self) -> None:
        """The upstream pool calls this at shutdown; the shared Gym client needs no closing."""

    async def fetch(self, urls: list[str]) -> VerifierPages:
        return VerifierPages(pages=list(await asyncio.gather(*(self._page(url) for url in urls))))

    async def _page(self, url: str) -> VerifierPage:
        hostname = None
        try:
            hostname, _ = check_http_url(url)
            fetched = await fetch_public(self._get, url, timeout=self._timeout, max_bytes=self._max_bytes)
            page = await extract_page_isolated(fetched.body, fetched.content_type, fetched.final_url)
        except (ClientError, OSError, asyncio.TimeoutError, ValueError) as error:
            return VerifierPage(url=url, hostname=hostname, error=f"{type(error).__name__}: {error}")
        return VerifierPage(
            url=url,
            title=page.title,
            description=page.description,
            author=page.author,
            hostname=hostname,
            published_date=page.published_date,
            content=page.text,
            error=None if page.text else "empty page text",
        )


class ExaFetchClient:
    """WANDR's fetch-client contract over Exa's ``/contents``: page text, title, author and published date."""

    # The upstream fetch retries and then swallows exceptions, so an authentication failure is recorded here
    # and run_verifier.py fails the verification instead of scoring every record against a missing page.
    fatal_errors: ClassVar[list[str]] = []

    def __init__(
        self,
        post: PostFunction,
        *,
        api_key: str,
        timeout: float,
        base_url: str = EXA_BASE_URL,
        max_age_hours: Optional[int] = None,
    ) -> None:
        self._post = post
        self._api_key = api_key
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._max_age_hours = max_age_hours
        self.content = self

    async def close(self) -> None:
        """The upstream pool calls this at shutdown; the shared Gym client needs no closing."""

    async def fetch(self, urls: list[str]) -> VerifierPages:
        pages: dict[str, VerifierPage] = {}
        hostnames: dict[str, str] = {}
        for url in urls:
            try:
                hostnames[url], _ = check_http_url(url)
            except UnsafeURLError as error:
                pages[url] = VerifierPage(url=url, error=f"UnsafeURLError: {error}")
        wanted = list(dict.fromkeys(url for url in urls if url not in pages))
        if wanted:
            payload: dict = {"urls": wanted, "text": True}
            if self._max_age_hours is not None:
                payload["maxAgeHours"] = self._max_age_hours
            status, data = await self._post(
                f"{self._base_url}/contents", {"x-api-key": self._api_key}, payload, self._timeout
            )
            if status in EXA_FATAL_STATUSES:
                ExaFetchClient.fatal_errors.append(f"Exa /contents answered HTTP {status}")
            if status != 200 or not isinstance(data, dict):
                # Provider error bodies can echo credentials; report the status only.
                raise ValueError(f"Exa /contents failed with HTTP {status}")
            results: dict[str, dict] = {}
            for result in data.get("results") or []:
                if isinstance(result, dict):
                    for key in (result.get("id"), result.get("url")):
                        if isinstance(key, str):
                            results.setdefault(key, result)
            statuses = {item.get("id"): item for item in data.get("statuses") or [] if isinstance(item, dict)}
            for url in wanted:
                pages[url] = self._page(url, hostnames[url], results.get(url), statuses.get(url))
        return VerifierPages(pages=[pages[url] for url in urls])

    @staticmethod
    def _page(url: str, hostname: str, result: Optional[dict], status: Optional[dict]) -> VerifierPage:
        if result is None or (status or {}).get("status") == "error":
            error = (status or {}).get("error") or {}
            reason = " ".join(str(part) for part in (error.get("tag"), error.get("httpStatusCode")) if part)
            return VerifierPage(url=url, hostname=hostname, error=f"Exa: {reason or 'no result'}")
        text = str(result.get("text") or "")
        return VerifierPage(
            url=url,
            title=str(result.get("title") or ""),
            author=str(result.get("author") or ""),
            hostname=hostname,
            published_date=str(result.get("publishedDate") or ""),
            content=text,
            error=None if text else "empty page text",
        )
