# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from environments.labbench2_pdf import download_papers as download_papers_module
from environments.labbench2_pdf.download_papers import (
    DownloadCandidate,
    DownloadResult,
    HTTPResponse,
    PaperDownloadOptions,
    _extract_pdf_urls,
    _restore_cached_provenance,
    _s3_pdf_url,
    download_corpus,
    download_one,
    resolve_europe_pmc,
    resolve_openalex,
)


class FakeHTTPClient:
    def __init__(self, responses: dict[str, HTTPResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str | None]] = []

    def get(
        self,
        url: str,
        *,
        accept: str = "",
        referer: str | None = None,
        max_bytes: int,
    ) -> HTTPResponse:
        del accept, max_bytes
        self.calls.append((url, referer))
        return self.responses[url]


class StaticJSONClient:
    def __init__(self, value: dict) -> None:
        self.value = value
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        accept: str = "",
        referer: str | None = None,
        max_bytes: int,
    ) -> HTTPResponse:
        del accept, referer, max_bytes
        self.urls.append(url)
        return HTTPResponse(body=json.dumps(self.value).encode(), url=url, content_type="application/json")


def _pdf_payload(marker: bytes = b"paper") -> bytes:
    return b"%PDF-1.7\n" + marker + (b"x" * 128)


def test_extract_pdf_urls_uses_declared_pdf_metadata_only() -> None:
    html = b"""
    <html><head>
      <meta name="citation_pdf_url" content="/article/main.pdf">
      <link rel="alternate" type="application/pdf; charset=binary" href="supplement.pdf">
    </head><body>
      <a href="unrelated.pdf">not declared metadata</a>
    </body></html>
    """

    assert _extract_pdf_urls(html, "https://example.org/article/index.html") == [
        "https://example.org/article/main.pdf",
        "https://example.org/article/supplement.pdf",
    ]


def test_s3_pdf_url_accepts_only_the_public_pmc_bucket_and_preserves_md5() -> None:
    url, expected_md5 = _s3_pdf_url(
        "s3://pmc-oa-opendata/oa_package/aa/bb/PMC1234567.tar.gz/PMC1234567.pdf?md5=012345"
    )

    assert url == ("https://pmc-oa-opendata.s3.amazonaws.com/oa_package/aa/bb/PMC1234567.tar.gz/PMC1234567.pdf")
    assert expected_md5 == "012345"


def test_europe_pmc_includes_free_author_manuscript_pdf() -> None:
    doi = "10.1000/free-manuscript"
    pdf_url = "https://europepmc.org/articles/PMC123?pdf=render"
    client = StaticJSONClient(
        {
            "resultList": {
                "result": [
                    {
                        "doi": doi,
                        "fullTextUrlList": {
                            "fullTextUrl": [
                                {
                                    "availabilityCode": "F",
                                    "documentStyle": "pdf",
                                    "url": pdf_url,
                                }
                            ]
                        },
                    }
                ]
            }
        }
    )

    candidates = resolve_europe_pmc(client, [doi], jobs=1)  # type: ignore[arg-type]

    assert candidates[doi] == [DownloadCandidate(url=pdf_url, resolver="europe_pmc")]


def test_openalex_tries_free_locations_before_opt_in_content_api() -> None:
    doi = "10.1000/open"
    client = StaticJSONClient(
        {
            "results": [
                {
                    "doi": f"https://doi.org/{doi}",
                    "best_oa_location": {
                        "is_oa": True,
                        "pdf_url": "https://repository.example/paper.pdf",
                        "landing_page_url": "https://repository.example/item",
                        "license": "cc-by",
                    },
                    "locations": [],
                    "open_access": {"is_oa": True},
                    "content_urls": {"pdf": "https://content.openalex.org/works/W123.pdf"},
                }
            ]
        }
    )

    candidates = resolve_openalex(
        client,  # type: ignore[arg-type]
        [doi],
        api_key="secret",
        use_content_api=True,
    )[doi]

    assert [candidate.resolver for candidate in candidates] == [
        "openalex",
        "openalex_landing",
        "openalex_content",
    ]
    assert candidates[-1].url == "https://content.openalex.org/works/W123.pdf?api_key=secret"


def test_download_follows_publisher_declared_pdf_and_writes_doi_name(tmp_path: Path) -> None:
    landing_url = "https://publisher.example/article"
    pdf_url = "https://publisher.example/article.pdf?token=ephemeral"
    payload = _pdf_payload()
    client = FakeHTTPClient(
        {
            landing_url: HTTPResponse(
                body=b'<meta name="citation_pdf_url" content="/article.pdf?token=ephemeral">',
                url=landing_url,
                content_type="text/html",
            ),
            pdf_url: HTTPResponse(body=payload, url=pdf_url, content_type="application/pdf"),
        }
    )

    result = download_one(
        "10.1000/example",
        [DownloadCandidate(url=landing_url, resolver="doi_landing")],
        papers_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
        max_pdf_bytes=1024,
        min_pdf_bytes=100,
        overwrite=False,
    )

    assert result.status == "downloaded"
    assert result.pdf == "10.1000_example.pdf"
    assert result.source_url == "https://publisher.example/article.pdf"
    assert (tmp_path / result.pdf).read_bytes() == payload
    assert client.calls == [(landing_url, None), (pdf_url, landing_url)]


def test_bad_pmc_checksum_falls_through_to_next_public_location(tmp_path: Path) -> None:
    bad_url = "https://pmc.example/wrong.pdf"
    good_url = "https://repository.example/paper.pdf"
    bad_payload = _pdf_payload(b"wrong")
    good_payload = _pdf_payload(b"right")
    client = FakeHTTPClient(
        {
            bad_url: HTTPResponse(body=bad_payload, url=bad_url, content_type="application/pdf"),
            good_url: HTTPResponse(body=good_payload, url=good_url, content_type="application/pdf"),
        }
    )

    result = download_one(
        "10.1000/checksum",
        [
            DownloadCandidate(
                url=bad_url,
                resolver="pmc_cloud",
                expected_md5=hashlib.md5(b"different", usedforsecurity=False).hexdigest(),
            ),
            DownloadCandidate(url=good_url, resolver="openalex", license="cc-by"),
        ],
        papers_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
        max_pdf_bytes=1024,
        min_pdf_bytes=100,
        overwrite=False,
    )

    assert result.status == "downloaded"
    assert result.resolver == "openalex"
    assert result.license == "cc-by"
    assert result.attempts == (f"pmc_cloud: {bad_url}: MD5 mismatch",)
    assert (tmp_path / "10.1000_checksum.pdf").read_bytes() == good_payload


def test_valid_existing_pdf_is_reused_without_network(tmp_path: Path) -> None:
    payload = _pdf_payload(b"cached")
    cached = tmp_path / "10.1000_cached.pdf"
    cached.write_bytes(payload)
    client = FakeHTTPClient({})

    result = download_one(
        "10.1000/cached",
        [DownloadCandidate(url="https://unused.example/paper.pdf", resolver="openalex")],
        papers_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
        max_pdf_bytes=1024,
        min_pdf_bytes=100,
        overwrite=False,
    )

    assert result.status == "cached"
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert client.calls == []


def test_cached_pdf_retains_verified_acquisition_provenance() -> None:
    cached = DownloadResult(
        doi="10.1000/cached",
        status="cached",
        pdf="10.1000_cached.pdf",
        resolver=None,
        source_url=None,
        license=None,
        pmcid=None,
        version=None,
        sha256="abc123",
        bytes=123,
        attempts=(),
    )
    previous = {
        cached.doi: {
            "doi": cached.doi,
            "sha256": cached.sha256,
            "resolver": "pmc_cloud",
            "source_url": "https://pmc.example/paper.pdf",
            "license": "cc-by",
            "pmcid": "PMC123",
            "version": 2,
            "attempts": ["first location failed"],
        }
    }

    result = _restore_cached_provenance(cached, previous)

    assert result.resolver == "pmc_cloud"
    assert result.source_url == "https://pmc.example/paper.pdf"
    assert result.license == "cc-by"
    assert result.pmcid == "PMC123"
    assert result.version == 2
    assert result.attempts == ("first location failed",)


def test_recorded_failures_are_reused_without_network_by_default(monkeypatch, tmp_path: Path) -> None:
    questions_dir = tmp_path / "questions"
    papers_dir = tmp_path / "papers"
    questions_dir.mkdir()
    papers_dir.mkdir()
    question = {
        "id": "item-id",
        "tag": "litqa3",
        "version": "2.0",
        "question": "Question?",
        "ideal": "answer",
        "sources": ["https://doi.org/10.1000/missing"],
    }
    (questions_dir / "litqa3.jsonl").write_text(json.dumps(question) + "\n", encoding="utf-8")
    record = {
        "doi": "10.1000/missing",
        "status": "failed",
        "pdf": None,
        "resolver": None,
        "source_url": None,
        "license": None,
        "pmcid": None,
        "version": None,
        "sha256": None,
        "bytes": 0,
        "attempts": ["openalex: unavailable"],
    }
    (papers_dir / "download_manifest.json").write_text(
        json.dumps({"records": [record]}),
        encoding="utf-8",
    )

    def unexpected_resolver(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a recorded failure should not trigger a resolver")

    monkeypatch.setattr(download_papers_module, "resolve_europe_pmc", unexpected_resolver)
    monkeypatch.setattr(download_papers_module, "resolve_openalex", unexpected_resolver)
    monkeypatch.setattr(download_papers_module, "resolve_crossref", unexpected_resolver)

    manifest = download_corpus(
        PaperDownloadOptions(
            questions_dir=questions_dir,
            papers_dir=papers_dir,
            benchmarks=("litqa3",),
            jobs=1,
            min_pdf_bytes=1,
        )
    )

    assert manifest["failed_count"] == 1
    assert manifest["reused_failed_count"] == 1
    assert manifest["records"][0]["attempts"] == ("openalex: unavailable",)
