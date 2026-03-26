#!/usr/bin/env python3
"""Pre-fetch SEC filing metadata for all companies in a ticker config.

Populates the same filings_metadata/{CIK}.json cache that app.py uses,
including pagination through filings.files for complete filing history.

Idempotent: skips companies whose cache file already exists.

Usage:
    python prefetch_sec_metadata.py \
        --cache_dir /path/to/gym_cache/finance_sec_search \
        --ticker_config /path/to/sp500.yaml

    # Or with explicit ticker list:
    python prefetch_sec_metadata.py \
        --cache_dir /path/to/gym_cache/finance_sec_search \
        --tickers AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
USER_AGENT = "Gym-SEC-Prefetch/1.0 (research@nvidia.com)"
MAX_REQUESTS_PER_SECOND = 10


class RateLimiter:
    """Sliding-window rate limiter matching app.py's fixed version."""

    def __init__(self, max_requests: int = MAX_REQUESTS_PER_SECOND, window: float = 1.0):
        self.max_requests = max_requests
        self.window = window
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.requests and (now - self.requests[0]) >= self.window:
                    self.requests.popleft()
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return
                sleep_time = self.window - (now - self.requests[0])
            await asyncio.sleep(max(sleep_time, 0.01))


def parse_filings_columns(columns: Dict[str, Any], cik: str, ticker: str) -> Dict[str, Dict[str, Any]]:
    """Parse SEC columnar filing data -- same logic as app.py._parse_filings_columns."""
    acc_numbers = columns.get("accessionNumber", [])
    forms = columns.get("form", [])
    dates = columns.get("filingDate", [])
    report_dates = columns.get("reportDate", [])
    primary_docs = columns.get("primaryDocument", [])

    filings: Dict[str, Dict[str, Any]] = {}
    for acc, form, fdate, rdate, pdoc in zip(acc_numbers, forms, dates, report_dates, primary_docs):
        acc_nodash = acc.replace("-", "")
        filings[acc_nodash] = {
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "filing_date": fdate,
            "report_date": rdate,
            "accession_number": acc,
            "primary_document": pdoc,
            "filing_url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_nodash}/{pdoc}",
        }
    return filings


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    rate_limiter: RateLimiter,
    max_retries: int = 3,
) -> Optional[str]:
    timeout = aiohttp.ClientTimeout(total=30)
    for attempt in range(max_retries):
        await rate_limiter.acquire()
        try:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.text()
                if resp.status in (403, 429, 503):
                    logger.warning("SEC %d on attempt %d for %s", resp.status, attempt + 1, url)
                    await asyncio.sleep(2**attempt)
                    continue
                logger.warning("SEC %d (non-retryable) for %s", resp.status, url)
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("Fetch error attempt %d for %s", attempt + 1, url, exc_info=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
    return None


async def fetch_company_filings(
    session: aiohttp.ClientSession,
    cik: str,
    ticker: str,
    rate_limiter: RateLimiter,
) -> Dict[str, Dict[str, Any]]:
    """Fetch full filing history for one company (recent + supplementary files)."""
    data = await fetch_with_retry(session, SEC_SUBMISSIONS_URL.format(cik=cik), rate_limiter)
    if not data:
        return {}

    try:
        filings_data = json.loads(data).get("filings", {})
        recent = filings_data.get("recent", {})
        filings = parse_filings_columns(recent, cik, ticker)

        for file_ref in filings_data.get("files", []):
            filename = file_ref.get("name", "")
            if not filename:
                continue
            extra_data = await fetch_with_retry(
                session, f"https://data.sec.gov/submissions/{filename}", rate_limiter
            )
            if extra_data:
                try:
                    extra = json.loads(extra_data)
                    filings.update(parse_filings_columns(extra, cik, ticker))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse %s for CIK %s", filename, cik)

        return filings
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to parse submissions for CIK %s (%s)", cik, ticker, exc_info=True)
        return {}


def load_ticker_list(ticker_config: str) -> list[str]:
    """Load tickers from a YAML config file (expects a 'tickers' key with a list)."""
    import yaml

    with open(ticker_config, "r") as f:
        cfg = yaml.safe_load(f)

    if isinstance(cfg, list):
        return cfg
    if isinstance(cfg, dict):
        for key in ("tickers", "companies", "symbols"):
            if key in cfg and isinstance(cfg[key], list):
                return cfg[key]
    raise ValueError(f"Cannot find ticker list in {ticker_config}. Expected a list or dict with 'tickers' key.")


async def resolve_tickers(
    session: aiohttp.ClientSession,
    tickers: list[str],
    rate_limiter: RateLimiter,
) -> Dict[str, Dict[str, str]]:
    """Resolve tickers to CIKs using SEC company_tickers.json."""
    data = await fetch_with_retry(session, SEC_TICKERS_URL, rate_limiter)
    if not data:
        raise RuntimeError("Failed to fetch SEC company tickers")

    raw = json.loads(data)
    lookup: Dict[str, Dict[str, str]] = {}
    for item in raw.values():
        t = item["ticker"].upper()
        lookup[t] = {"cik": str(item["cik_str"]).zfill(10), "name": item["title"]}

    resolved = {}
    for t in tickers:
        t_upper = t.strip().upper()
        if t_upper in lookup:
            resolved[t_upper] = lookup[t_upper]
        else:
            logger.warning("Ticker %s not found in SEC registry, skipping", t_upper)
    return resolved


async def prefetch(cache_dir: str, tickers: list[str], force: bool = False) -> None:
    metadata_dir = Path(cache_dir) / "filings_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter()
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT}, connector=connector
    ) as session:
        companies = await resolve_tickers(session, tickers, rate_limiter)
        logger.info("Resolved %d / %d tickers", len(companies), len(tickers))

        skipped = 0
        fetched = 0
        failed = 0
        total = len(companies)

        for idx, (ticker, info) in enumerate(companies.items(), 1):
            cik = info["cik"]
            cache_path = metadata_dir / f"{cik}.json"
            if cache_path.exists() and not force:
                skipped += 1
                continue

            filings = await fetch_company_filings(session, cik, ticker, rate_limiter)
            if filings:
                with open(cache_path, "w") as f:
                    json.dump(filings, f)
                fetched += 1
                if fetched % 50 == 0 or idx == total:
                    logger.info("Progress: %d/%d fetched (%d skipped, %d failed)", fetched, total, skipped, failed)
            else:
                failed += 1
                logger.warning("No filings fetched for %s (CIK %s) [%d/%d]", ticker, cik, idx, total)

        logger.info(
            "Prefetch complete: %d fetched, %d skipped (cached), %d failed",
            fetched, skipped, failed,
        )


def main():
    parser = argparse.ArgumentParser(description="Pre-fetch SEC filing metadata cache")
    parser.add_argument("--cache_dir", required=True, help="Cache directory (same as app.py cache_dir)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache exists (use after adding pagination)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker_config", help="YAML file with ticker list")
    group.add_argument("--tickers", nargs="+", help="Explicit list of tickers")
    args = parser.parse_args()

    if args.ticker_config:
        tickers = load_ticker_list(args.ticker_config)
    else:
        tickers = args.tickers

    logger.info("Prefetching metadata for %d tickers into %s", len(tickers), args.cache_dir)
    asyncio.run(prefetch(args.cache_dir, tickers, force=args.force))


if __name__ == "__main__":
    main()
