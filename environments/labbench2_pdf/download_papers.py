#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download publicly reachable LabBench2 source papers from their DOI metadata.

The downloaded PDFs are a local, gitignored cache. The resolver does not use
credentials, subscription cookies, search-engine scraping, or paywall bypasses.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.labbench2_pdf.data_utils import (
    DEFAULT_PAPERS_DIR,
    SOURCE_POLICIES,
    SUPPORTED_BENCHMARKS,
    build_coverage_audit,
    doi_filename_candidates,
    load_questions,
    source_to_doi,
)


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
UNPAYWALL_API_URL = "https://api.unpaywall.org/v2"
CROSSREF_API_URL = "https://api.crossref.org/works"
PMC_S3_BUCKET = "pmc-oa-opendata"
PMC_S3_HTTPS = "https://pmc-oa-opendata.s3.amazonaws.com"
DEFAULT_USER_AGENT = "NeMoGym-LabBench2-PaperDownloader/1.0"
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    body: bytes
    url: str
    content_type: str


@dataclass(frozen=True, slots=True)
class DownloadCandidate:
    url: str
    resolver: str
    license: str | None = None
    referer: str | None = None
    expected_md5: str | None = None
    pmcid: str | None = None
    version: str | int | None = None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    doi: str
    status: str
    pdf: str | None
    resolver: str | None
    source_url: str | None
    license: str | None
    pmcid: str | None
    version: str | int | None
    sha256: str | None
    bytes: int
    attempts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaperDownloadOptions:
    questions_dir: Path
    papers_dir: Path = DEFAULT_PAPERS_DIR
    benchmarks: tuple[str, ...] = SUPPORTED_BENCHMARKS
    source_policy: str = "all"
    jobs: int = 6
    timeout: float = 45.0
    retries: int = 3
    max_pdf_mb: int = 100
    min_pdf_bytes: int = 1000
    user_agent: str = DEFAULT_USER_AGENT
    contact_email: str | None = None
    openalex_api_key: str | None = None
    openalex_content: bool = False
    unpaywall_email: str | None = None
    crossref: bool = True
    retry_failed: bool = False
    overwrite: bool = False


class RequestError(RuntimeError):
    """A concise HTTP error safe to include in the generated manifest."""


class HTTPClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float,
        retries: int,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries

    def get(
        self,
        url: str,
        *,
        accept: str = "application/pdf,text/html;q=0.9,application/xhtml+xml;q=0.8,*/*;q=0.1",
        referer: str | None = None,
        max_bytes: int,
    ) -> HTTPResponse:
        headers = {"Accept": accept, "User-Agent": self.user_agent}
        if referer:
            headers["Referer"] = referer
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > max_bytes:
                        raise ValueError(f"remote object exceeds {max_bytes} bytes")
                    body = response.read(max_bytes + 1)
                    if len(body) > max_bytes:
                        raise ValueError(f"remote object exceeds {max_bytes} bytes")
                    content_type = response.headers.get_content_type()
                    return HTTPResponse(body=body, url=response.geturl(), content_type=content_type)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in TRANSIENT_HTTP_CODES:
                    break
            except (OSError, urllib.error.URLError, ValueError) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        assert last_error is not None
        if isinstance(last_error, urllib.error.HTTPError):
            raise RequestError(f"HTTP {last_error.code}") from last_error
        raise RequestError(f"{type(last_error).__name__}: {last_error}") from last_error


class _PDFLinkParser(HTMLParser):
    META_NAMES = {
        "citation_pdf_url",
        "dc.identifier.pdf",
        "eprints.document_url",
        "wkhealth_pdf_url",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.metadata_urls: list[str] = []
        self.alternate_urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value for key, value in attrs if value is not None}
        if tag.casefold() == "meta":
            name = (values.get("name") or values.get("property") or "").casefold()
            content = values.get("content")
            if name in self.META_NAMES and content:
                self.metadata_urls.append(content)
        elif tag.casefold() == "link":
            media_type = (values.get("type") or "").split(";", 1)[0].strip().casefold()
            relations = {value.casefold() for value in (values.get("rel") or "").split()}
            href = values.get("href")
            if href and media_type == "application/pdf" and relations.intersection({"alternate", "download"}):
                self.alternate_urls.append(href)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_doi(source: str) -> str:
    return source_to_doi(source).casefold()


def _preferred_pdf_name(doi: str) -> str:
    return doi_filename_candidates(doi)[-1]


def _manifest_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _with_api_key(url: str, api_key: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("api_key", api_key))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def _url_with_query(base: str, values: dict[str, str | int]) -> str:
    return f"{base}?{urllib.parse.urlencode(values)}"


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _request_json(client: HTTPClient, url: str) -> dict[str, Any]:
    response = client.get(url, accept="application/json", max_bytes=32 * 1024 * 1024)
    value = json.loads(response.body)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {_manifest_url(url)}")
    return value


def _looks_like_pdf(payload: bytes, min_pdf_bytes: int) -> bool:
    return len(payload) >= min_pdf_bytes and b"%PDF-" in payload[:1024]


def _extract_pdf_urls(payload: bytes, base_url: str) -> list[str]:
    parser = _PDFLinkParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    urls = [urllib.parse.urljoin(base_url, value) for value in parser.metadata_urls + parser.alternate_urls]
    return list(dict.fromkeys(url for url in urls if urllib.parse.urlsplit(url).scheme in {"http", "https"}))


def _s3_pdf_url(value: str) -> tuple[str, str | None]:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "s3" or parsed.netloc != PMC_S3_BUCKET:
        raise ValueError(f"unexpected PMC object URL: {_manifest_url(value)}")
    expected_md5 = urllib.parse.parse_qs(parsed.query).get("md5", [None])[0]
    path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/")
    return f"{PMC_S3_HTTPS}{path}", expected_md5


def _europe_pmc_records(client: HTTPClient, dois: Sequence[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for batch in _chunks(dois, 30):
        query = " OR ".join(f'DOI:"{doi}"' for doi in batch)
        url = _url_with_query(
            EUROPE_PMC_SEARCH_URL,
            {"query": query, "format": "json", "resultType": "core", "pageSize": 1000},
        )
        response = _request_json(client, url)
        for record in response.get("resultList", {}).get("result", []):
            doi = _canonical_doi(str(record.get("doi") or ""))
            if doi in batch:
                records[doi] = record
    return records


def _pmc_candidates(client: HTTPClient, pmcid: str) -> list[DownloadCandidate]:
    listing_url = _url_with_query(
        f"{PMC_S3_HTTPS}/",
        {"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
    )
    response = client.get(listing_url, accept="application/xml", max_bytes=2 * 1024 * 1024)
    root = ET.fromstring(response.body)
    prefixes = [
        node.text.rstrip("/")
        for node in root.findall(".//{*}CommonPrefixes/{*}Prefix")
        if node.text and node.text.startswith(f"{pmcid}.")
    ]
    metadata: list[dict[str, Any]] = []
    for prefix in prefixes:
        try:
            metadata.append(_request_json(client, f"{PMC_S3_HTTPS}/{prefix}/{prefix}.json"))
        except (RequestError, ValueError, json.JSONDecodeError):
            continue
    metadata.sort(
        key=lambda item: (
            bool(item.get("is_manuscript")),
            -int(item.get("version") or 0),
        )
    )
    candidates: list[DownloadCandidate] = []
    for item in metadata:
        pdf_url = item.get("pdf_url")
        if not pdf_url or item.get("is_retracted"):
            continue
        try:
            url, expected_md5 = _s3_pdf_url(str(pdf_url))
        except ValueError:
            continue
        candidates.append(
            DownloadCandidate(
                url=url,
                resolver="pmc_cloud",
                license=str(item.get("license_code") or "") or None,
                expected_md5=expected_md5,
                pmcid=pmcid,
                version=item.get("version"),
            )
        )
    return candidates


def resolve_europe_pmc(
    client: HTTPClient,
    dois: Sequence[str],
    *,
    jobs: int,
) -> dict[str, list[DownloadCandidate]]:
    candidates = {doi: [] for doi in dois}
    records = _europe_pmc_records(client, dois)
    pmcid_to_doi = {str(record["pmcid"]): doi for doi, record in records.items() if record.get("pmcid")}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_pmc_candidates, client, pmcid): pmcid for pmcid in pmcid_to_doi}
        for future in concurrent.futures.as_completed(futures):
            pmcid = futures[future]
            try:
                candidates[pmcid_to_doi[pmcid]].extend(future.result())
            except (RequestError, ET.ParseError, ValueError):
                pass
    for doi, record in records.items():
        urls = record.get("fullTextUrlList", {}).get("fullTextUrl", [])
        for item in urls:
            if item.get("availabilityCode") in {"F", "OA"} and item.get("documentStyle") == "pdf":
                candidates[doi].append(
                    DownloadCandidate(
                        url=str(item["url"]),
                        resolver="europe_pmc",
                        pmcid=str(record.get("pmcid") or "") or None,
                    )
                )
    return candidates


def resolve_openalex(
    client: HTTPClient,
    dois: Sequence[str],
    *,
    api_key: str | None,
    use_content_api: bool = False,
) -> dict[str, list[DownloadCandidate]]:
    candidates = {doi: [] for doi in dois}
    for batch in _chunks(dois, 50):
        values: dict[str, str | int] = {
            "filter": "doi:" + "|".join(f"https://doi.org/{doi}" for doi in batch),
            "per-page": 100,
        }
        if api_key:
            values["api_key"] = api_key
        response = _request_json(client, _url_with_query(OPENALEX_WORKS_URL, values))
        for work in response.get("results", []):
            doi = _canonical_doi(str(work.get("doi") or ""))
            if doi not in candidates:
                continue
            locations = list(work.get("locations") or [])
            best = work.get("best_oa_location")
            if isinstance(best, dict):
                locations.insert(0, best)
            best_license = str(best.get("license") or "") or None if isinstance(best, dict) else None
            content_pdf = (work.get("content_urls") or {}).get("pdf")
            is_open_access = bool((work.get("open_access") or {}).get("is_oa"))
            for location in locations:
                pdf_url = location.get("pdf_url")
                if location.get("is_oa") and pdf_url:
                    candidates[doi].append(
                        DownloadCandidate(
                            url=str(pdf_url),
                            resolver="openalex",
                            license=str(location.get("license") or "") or None,
                            referer=str(location.get("landing_page_url") or "") or None,
                        )
                    )
                landing_page_url = location.get("landing_page_url")
                if location.get("is_oa") and landing_page_url:
                    candidates[doi].append(
                        DownloadCandidate(
                            url=str(landing_page_url),
                            resolver="openalex_landing",
                            license=str(location.get("license") or "") or None,
                        )
                    )
            if use_content_api and api_key and content_pdf and is_open_access:
                candidates[doi].append(
                    DownloadCandidate(
                        url=_with_api_key(str(content_pdf), api_key),
                        resolver="openalex_content",
                        license=best_license,
                    )
                )
    return candidates


def _unpaywall_candidates_for_doi(
    client: HTTPClient,
    doi: str,
    email: str,
) -> list[DownloadCandidate]:
    url = _url_with_query(
        f"{UNPAYWALL_API_URL}/{urllib.parse.quote(doi, safe='')}",
        {"email": email},
    )
    response = _request_json(client, url)
    candidates: list[DownloadCandidate] = []
    locations = list(response.get("oa_locations") or [])
    best = response.get("best_oa_location")
    if isinstance(best, dict):
        locations.insert(0, best)
    for location in locations:
        pdf_url = location.get("url_for_pdf")
        if pdf_url:
            candidates.append(
                DownloadCandidate(
                    url=str(pdf_url),
                    resolver="unpaywall",
                    license=str(location.get("license") or "") or None,
                    referer=str(location.get("url_for_landing_page") or "") or None,
                )
            )
    return candidates


def resolve_unpaywall(
    client: HTTPClient,
    dois: Sequence[str],
    *,
    email: str,
    jobs: int,
) -> dict[str, list[DownloadCandidate]]:
    candidates = {doi: [] for doi in dois}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_unpaywall_candidates_for_doi, client, doi, email): doi for doi in dois}
        for future in concurrent.futures.as_completed(futures):
            doi = futures[future]
            try:
                candidates[doi].extend(future.result())
            except (RequestError, ValueError, json.JSONDecodeError):
                pass
    return candidates


def _crossref_candidates_for_doi(client: HTTPClient, doi: str) -> list[DownloadCandidate]:
    url = f"{CROSSREF_API_URL}/{urllib.parse.quote(doi, safe='')}"
    response = _request_json(client, url)
    candidates: list[DownloadCandidate] = []
    for link in response.get("message", {}).get("link", []):
        link_url = link.get("URL")
        content_type = str(link.get("content-type") or "").split(";", 1)[0].casefold()
        if link_url and content_type == "application/pdf":
            candidates.append(DownloadCandidate(url=str(link_url), resolver="crossref"))
    return candidates


def resolve_crossref(
    client: HTTPClient,
    dois: Sequence[str],
    *,
    jobs: int,
) -> dict[str, list[DownloadCandidate]]:
    candidates = {doi: [] for doi in dois}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_crossref_candidates_for_doi, client, doi): doi for doi in dois}
        for future in concurrent.futures.as_completed(futures):
            doi = futures[future]
            try:
                candidates[doi].extend(future.result())
            except (RequestError, ValueError, json.JSONDecodeError):
                pass
    return candidates


def _provider_candidates(doi: str) -> list[DownloadCandidate]:
    quoted_doi = urllib.parse.quote(doi, safe="/():;._-")
    suffix = doi.split("/", 1)[1] if "/" in doi else doi
    candidates: list[DownloadCandidate] = []
    if doi.startswith("10.1101/"):
        candidates.extend(
            [
                DownloadCandidate(
                    url=f"https://www.biorxiv.org/content/{quoted_doi}.full.pdf",
                    resolver="biorxiv",
                ),
                DownloadCandidate(
                    url=f"https://www.medrxiv.org/content/{quoted_doi}.full.pdf",
                    resolver="medrxiv",
                ),
            ]
        )
    if doi.startswith("10.1038/"):
        candidates.append(DownloadCandidate(url=f"https://www.nature.com/articles/{suffix}.pdf", resolver="nature"))
    if doi.startswith("10.1128/"):
        candidates.append(
            DownloadCandidate(
                url=f"https://journals.asm.org/doi/pdf/{quoted_doi}?download=true",
                resolver="asm",
            )
        )
    if doi.startswith("10.1073/"):
        candidates.append(DownloadCandidate(url=f"https://www.pnas.org/doi/pdf/{quoted_doi}", resolver="pnas"))
    if doi.startswith("10.1126/"):
        candidates.append(DownloadCandidate(url=f"https://www.science.org/doi/pdf/{quoted_doi}", resolver="science"))
    if doi.startswith("10.1056/"):
        candidates.append(DownloadCandidate(url=f"https://www.nejm.org/doi/pdf/{quoted_doi}", resolver="nejm"))
    if doi.startswith("10.1021/"):
        candidates.append(DownloadCandidate(url=f"https://pubs.acs.org/doi/pdf/{quoted_doi}", resolver="acs"))
    if doi.startswith(("10.1002/", "10.1111/")):
        candidates.append(
            DownloadCandidate(
                url=f"https://onlinelibrary.wiley.com/doi/pdfdirect/{quoted_doi}",
                resolver="wiley",
            )
        )
    if doi.startswith("10.1080/"):
        candidates.append(
            DownloadCandidate(
                url=f"https://www.tandfonline.com/doi/pdf/{quoted_doi}?download=1",
                resolver="taylor_francis",
            )
        )
    if doi.startswith("10.1152/"):
        candidates.append(
            DownloadCandidate(
                url=f"https://journals.physiology.org/doi/pdf/{quoted_doi}",
                resolver="aps",
            )
        )
    if doi.startswith("10.7554/elife."):
        article_id = suffix.removeprefix("elife.").split(".", 1)[0]
        candidates.extend(
            [
                DownloadCandidate(
                    url=f"https://elifesciences.org/articles/{article_id}.pdf",
                    resolver="elife",
                ),
                DownloadCandidate(
                    url=f"https://elifesciences.org/reviewed-preprints/{article_id}/pdf",
                    resolver="elife",
                ),
            ]
        )
    candidates.append(DownloadCandidate(url=f"https://doi.org/{quoted_doi}", resolver="doi_landing"))
    return candidates


def _deduplicate_candidates(candidates: Iterable[DownloadCandidate]) -> list[DownloadCandidate]:
    unique: list[DownloadCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        parsed = urllib.parse.urlsplit(candidate.url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            continue
        if candidate.url in seen:
            continue
        seen.add(candidate.url)
        unique.append(candidate)
    return unique


def _existing_pdf(papers_dir: Path, doi: str, min_pdf_bytes: int) -> Path | None:
    for name in doi_filename_candidates(doi):
        path = papers_dir / name
        if path.is_file() and path.stat().st_size >= min_pdf_bytes:
            with path.open("rb") as stream:
                if _looks_like_pdf(stream.read(1024), min_pdf_bytes=0):
                    return path
    return None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_pdf_atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def download_one(
    doi: str,
    candidates: Sequence[DownloadCandidate],
    *,
    papers_dir: Path,
    client: HTTPClient,
    max_pdf_bytes: int,
    min_pdf_bytes: int,
    overwrite: bool,
) -> DownloadResult:
    existing = None if overwrite else _existing_pdf(papers_dir, doi, min_pdf_bytes)
    if existing is not None:
        return DownloadResult(
            doi=doi,
            status="cached",
            pdf=existing.name,
            resolver=None,
            source_url=None,
            license=None,
            pmcid=None,
            version=None,
            sha256=_sha256_path(existing),
            bytes=existing.stat().st_size,
            attempts=(),
        )

    attempts: list[str] = []
    for candidate in candidates:
        pending = [(candidate.url, candidate.referer, candidate.expected_md5)]
        seen_urls: set[str] = set()
        while pending and len(seen_urls) < 5:
            url, referer, expected_md5 = pending.pop(0)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            display_url = _manifest_url(url)
            try:
                response = client.get(
                    url,
                    referer=referer,
                    max_bytes=max_pdf_bytes,
                )
            except RequestError as exc:
                attempts.append(f"{candidate.resolver}: {display_url}: {exc}")
                continue
            if _looks_like_pdf(response.body, min_pdf_bytes):
                if expected_md5:
                    actual_md5 = hashlib.md5(response.body, usedforsecurity=False).hexdigest()
                    if actual_md5 != expected_md5:
                        attempts.append(f"{candidate.resolver}: {display_url}: MD5 mismatch")
                        continue
                destination = papers_dir / _preferred_pdf_name(doi)
                _write_pdf_atomic(destination, response.body)
                return DownloadResult(
                    doi=doi,
                    status="downloaded",
                    pdf=destination.name,
                    resolver=candidate.resolver,
                    source_url=_manifest_url(response.url),
                    license=candidate.license,
                    pmcid=candidate.pmcid,
                    version=candidate.version,
                    sha256=_sha256(response.body),
                    bytes=len(response.body),
                    attempts=tuple(attempts),
                )
            pdf_urls = [url for url in _extract_pdf_urls(response.body, response.url) if url not in seen_urls]
            if not pdf_urls:
                attempts.append(f"{candidate.resolver}: {display_url}: not a PDF ({response.content_type})")
            pending.extend((pdf_url, response.url, None) for pdf_url in pdf_urls)
    return DownloadResult(
        doi=doi,
        status="failed",
        pdf=None,
        resolver=None,
        source_url=None,
        license=None,
        pmcid=None,
        version=None,
        sha256=None,
        bytes=0,
        attempts=tuple(attempts),
    )


def _extend_candidates(
    destination: dict[str, list[DownloadCandidate]],
    source: dict[str, list[DownloadCandidate]],
) -> None:
    for doi, values in source.items():
        destination[doi].extend(values)


def _coverage_summary(
    rows_by_benchmark: dict[str, list[dict[str, Any]]],
    papers_dir: Path,
    *,
    source_policy: str,
) -> dict[str, Any]:
    audit, _ = build_coverage_audit(rows_by_benchmark, papers_dir, source_policy=source_policy)
    return {
        "row_count": audit["row_count"],
        "answerable_count": audit["answerable_count"],
        "missing_count": audit["missing_count"],
        "partial_count": audit["partial_count"],
        "source_policy": source_policy,
        "by_benchmark": audit["by_benchmark"],
        "missing_rows": [item for item in audit["items"] if item["missing"]],
    }


def _previous_records(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records = manifest.get("records", []) if isinstance(manifest, dict) else []
    return {
        str(record["doi"]): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("doi"), str)
    }


def _restore_cached_provenance(
    result: DownloadResult,
    previous: dict[str, dict[str, Any]],
) -> DownloadResult:
    record = previous.get(result.doi)
    if result.status != "cached" or not record or record.get("sha256") != result.sha256:
        return result
    return replace(
        result,
        resolver=record.get("resolver"),
        source_url=record.get("source_url"),
        license=record.get("license"),
        pmcid=record.get("pmcid"),
        version=record.get("version"),
        attempts=tuple(str(value) for value in record.get("attempts") or []),
    )


def _failed_result_from_record(record: dict[str, Any]) -> DownloadResult | None:
    """Rehydrate a prior failure so ordinary preparation can remain offline."""
    if record.get("status") != "failed" or not isinstance(record.get("doi"), str):
        return None
    try:
        return DownloadResult(
            doi=record["doi"],
            status="failed",
            pdf=None,
            resolver=record.get("resolver"),
            source_url=record.get("source_url"),
            license=record.get("license"),
            pmcid=record.get("pmcid"),
            version=record.get("version"),
            sha256=None,
            bytes=int(record.get("bytes") or 0),
            attempts=tuple(str(value) for value in record.get("attempts") or []),
        )
    except (TypeError, ValueError):
        return None


def download_corpus(options: PaperDownloadOptions) -> dict[str, Any]:
    questions_dir = options.questions_dir.expanduser().resolve()
    papers_dir = options.papers_dir.expanduser().resolve()
    rows_by_benchmark = load_questions(questions_dir, options.benchmarks)
    target_dois = sorted(
        {_canonical_doi(source) for rows in rows_by_benchmark.values() for row in rows for source in row["sources"]}
    )
    papers_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = papers_dir / "download_manifest.json"
    previous = _previous_records(manifest_path)
    user_agent = options.user_agent
    if options.contact_email:
        user_agent = f"{user_agent} (mailto:{options.contact_email})"
    client = HTTPClient(user_agent=user_agent, timeout=options.timeout, retries=options.retries)
    candidates = {doi: [] for doi in target_dois}
    resolver_errors: dict[str, str] = {}
    existing_dois = {doi for doi in target_dois if _existing_pdf(papers_dir, doi, options.min_pdf_bytes) is not None}
    reused_failures: dict[str, DownloadResult] = {}
    if not options.overwrite and not options.retry_failed:
        for doi in target_dois:
            if doi in existing_dois:
                continue
            prior_result = _failed_result_from_record(previous.get(doi, {}))
            if prior_result is not None:
                reused_failures[doi] = prior_result
    pending_dois = [
        doi for doi in target_dois if doi not in reused_failures and (options.overwrite or doi not in existing_dois)
    ]

    print(
        f"Found {len(existing_dois)} cached PDFs; resolving {len(pending_dois)} DOIs...",
        flush=True,
    )
    if reused_failures:
        print(
            f"Reusing {len(reused_failures)} recorded failures; pass --retry-failed to try them again.",
            flush=True,
        )
    if pending_dois:
        print("Resolving through Europe PMC...", flush=True)
        try:
            _extend_candidates(candidates, resolve_europe_pmc(client, pending_dois, jobs=options.jobs))
        except (RequestError, ValueError, json.JSONDecodeError) as exc:
            resolver_errors["europe_pmc"] = f"{type(exc).__name__}: {exc}"
        print("Resolving OpenAlex locations...", flush=True)
        try:
            _extend_candidates(
                candidates,
                resolve_openalex(
                    client,
                    pending_dois,
                    api_key=options.openalex_api_key,
                    use_content_api=options.openalex_content,
                ),
            )
        except (RequestError, ValueError, json.JSONDecodeError) as exc:
            resolver_errors["openalex"] = f"{type(exc).__name__}: {exc}"
        if options.unpaywall_email:
            print("Resolving Unpaywall locations...", flush=True)
            try:
                _extend_candidates(
                    candidates,
                    resolve_unpaywall(
                        client,
                        pending_dois,
                        email=options.unpaywall_email,
                        jobs=options.jobs,
                    ),
                )
            except (RequestError, ValueError, json.JSONDecodeError) as exc:
                resolver_errors["unpaywall"] = f"{type(exc).__name__}: {exc}"
        if options.crossref:
            print("Resolving Crossref publisher links...", flush=True)
            try:
                _extend_candidates(candidates, resolve_crossref(client, pending_dois, jobs=options.jobs))
            except (RequestError, ValueError, json.JSONDecodeError) as exc:
                resolver_errors["crossref"] = f"{type(exc).__name__}: {exc}"
    for doi in pending_dois:
        candidates[doi].extend(_provider_candidates(doi))
        candidates[doi] = _deduplicate_candidates(candidates[doi])

    results = list(reused_failures.values())
    active_dois = [doi for doi in target_dois if doi not in reused_failures]
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.jobs) as pool:
        futures = {
            pool.submit(
                download_one,
                doi,
                candidates[doi],
                papers_dir=papers_dir,
                client=client,
                max_pdf_bytes=options.max_pdf_mb * 1024 * 1024,
                min_pdf_bytes=options.min_pdf_bytes,
                overwrite=options.overwrite,
            ): doi
            for doi in active_dois
        }
        completed = len(reused_failures)
        for future in concurrent.futures.as_completed(futures):
            result = _restore_cached_provenance(future.result(), previous)
            results.append(result)
            completed += 1
            print(
                f"[{completed:03d}/{len(target_dois):03d}] {result.status:10} {result.doi}"
                + (f" via {result.resolver}" if result.resolver else ""),
                flush=True,
            )

    results.sort(key=lambda item: item.doi)
    coverage = _coverage_summary(rows_by_benchmark, papers_dir, source_policy=options.source_policy)
    manifest = {
        "schema_version": "labbench2_pdf_download.v1",
        "created_at": _utc_now(),
        "questions_dir": str(questions_dir),
        "papers_dir": str(papers_dir),
        "benchmarks": list(options.benchmarks),
        "source_policy": options.source_policy,
        "retry_failed": options.retry_failed,
        "doi_count": len(target_dois),
        "downloaded_count": sum(result.status == "downloaded" for result in results),
        "cached_count": sum(result.status == "cached" for result in results),
        "failed_count": sum(result.status == "failed" for result in results),
        "reused_failed_count": len(reused_failures),
        "resolver_errors": resolver_errors,
        "resolver_counts": {
            resolver: sum(result.resolver == resolver for result in results)
            for resolver in sorted({result.resolver for result in results if result.resolver})
        },
        "coverage": coverage,
        "records": [asdict(result) for result in results],
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=SUPPORTED_BENCHMARKS,
        default=list(SUPPORTED_BENCHMARKS),
    )
    parser.add_argument(
        "--source-policy",
        choices=SOURCE_POLICIES,
        default="all",
        help="report a question as answerable only when all sources resolve (default), or any source resolves",
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-pdf-mb", type=int, default=100)
    parser.add_argument("--min-pdf-bytes", type=int, default=1000)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--contact-email")
    parser.add_argument(
        "--openalex-api-key",
        default=os.environ.get("OPENALEX_API_KEY"),
        help="OpenAlex key (defaults to OPENALEX_API_KEY)",
    )
    parser.add_argument(
        "--openalex-content",
        action="store_true",
        help="use OpenAlex's $0.01/PDF content API; requires --openalex-api-key",
    )
    parser.add_argument(
        "--unpaywall-email",
        default=os.environ.get("UNPAYWALL_EMAIL"),
        help="enable Unpaywall using this email (defaults to UNPAYWALL_EMAIL)",
    )
    parser.add_argument("--crossref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry DOI failures recorded by an earlier run",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> PaperDownloadOptions:
    return PaperDownloadOptions(
        questions_dir=args.questions_dir,
        papers_dir=args.papers_dir,
        benchmarks=tuple(args.benchmarks),
        source_policy=args.source_policy,
        jobs=args.jobs,
        timeout=args.timeout,
        retries=args.retries,
        max_pdf_mb=args.max_pdf_mb,
        min_pdf_bytes=args.min_pdf_bytes,
        user_agent=args.user_agent,
        contact_email=args.contact_email,
        openalex_api_key=args.openalex_api_key,
        openalex_content=args.openalex_content,
        unpaywall_email=args.unpaywall_email,
        crossref=args.crossref,
        retry_failed=args.retry_failed,
        overwrite=args.overwrite,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if args.timeout <= 0 or args.retries < 1:
        raise SystemExit("--timeout and --retries must be positive")
    if args.max_pdf_mb < 1 or args.min_pdf_bytes < 1:
        raise SystemExit("--max-pdf-mb and --min-pdf-bytes must be positive")
    if args.openalex_content and not args.openalex_api_key:
        raise SystemExit("--openalex-content requires --openalex-api-key")
    try:
        manifest = download_corpus(options_from_args(args))
    except (FileNotFoundError, RequestError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    coverage = manifest["coverage"]
    print(
        f"Downloaded or reused {manifest['doi_count'] - manifest['failed_count']}/{manifest['doi_count']} "
        f"papers. {coverage['answerable_count']}/{coverage['row_count']} questions are answerable; "
        f"{coverage['missing_count']} would be skipped."
    )
    print(f"Manifest: {Path(manifest['papers_dir']) / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
