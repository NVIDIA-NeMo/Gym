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
"""Prefetch Tiingo price history into the on-disk cache, one ticker at a time.

Populates the same per-(endpoint, ticker) master files that ``CachedPriceHistory``
reads at runtime, so a later eval run (with ``use_cache: true``) serves every
already-prefetched price query from disk instead of calling Tiingo.

Design notes:
  - Fetches the widest useful window once per ticker: ``--start`` .. ``MAX_END_DATE``
    (the pinned upstream ``finance_agent.tools.MAX_END_DATE``). Because history is
    immutable within that window, one fetch per ticker suffices.
  - Sequential + throttled (``--sleep``) to respect limited API keys.
  - Idempotent/resumable: a ticker whose master already covers the window is
    skipped unless ``--force`` is given.

Usage:
    python scripts/prefetch_prices.py \
        --cache-dir /shared/cache/finance_agent_v2 \
        --tickers AAPL MSFT NVDA \
        --asset-class equity
    # or from a file (one "TICKER[,asset_class]" per line):
    python scripts/prefetch_prices.py --cache-dir ... --tickers-file tickers.txt
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make the resource server package importable whether run from its dir or elsewhere.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent.parent))

from finance_agent.tools import MAX_END_DATE  # noqa: E402

try:
    from resources_servers.finance_agent_v2.cache import ToolCache
    from resources_servers.finance_agent_v2.cached_tools import CachedPriceHistory
except ImportError:  # pragma: no cover - flat execution fallback
    sys.path.insert(0, str(_PKG_ROOT))
    from cache import ToolCache  # type: ignore
    from cached_tools import CachedPriceHistory  # type: ignore


def _parse_ticker_line(line: str, default_asset_class: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "," in line:
        ticker, _, ac = line.partition(",")
        return ticker.strip(), (ac.strip() or default_asset_class)
    return line, default_asset_class


def _load_tickers(args: argparse.Namespace) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for t in args.tickers or []:
        pairs.append((t, args.asset_class))
    if args.tickers_file:
        for raw in Path(args.tickers_file).read_text().splitlines():
            parsed = _parse_ticker_line(raw, args.asset_class)
            if parsed:
                pairs.append(parsed)
    # De-dup preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


async def _prefetch_one(
    tool: CachedPriceHistory, ticker: str, asset_class: str, start: str, end: str, force: bool
) -> str:
    endpoint = tool._route_endpoint(asset_class)  # normalizes equity/etf -> equity
    recs_path, meta_path = tool._master_paths(endpoint, ticker)
    if not force:
        meta = tool._cache.read_json(meta_path)
        if isinstance(meta, dict) and meta.get("cov_start", "9999") <= start and meta.get("cov_end", "0000") >= end:
            return "skip (covered)"
    # _fetch performs the union-fetch + merge + write via the same path used at runtime.
    records = await tool._fetch(endpoint, ticker, start, end)
    return f"ok ({len(records)} rows in window)"


async def _main_async(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("TIINGO_API_KEY") or os.getenv("PRICING_DATA_API_KEY")
    if not api_key:
        print("ERROR: no Tiingo API key (pass --api-key or set TIINGO_API_KEY).", file=sys.stderr)
        return 2

    cache = ToolCache(args.cache_dir, use_cache=True)
    if not cache.enabled:
        print("ERROR: cache is disabled; --cache-dir is required.", file=sys.stderr)
        return 2

    tool = CachedPriceHistory(api_key, cache)
    end = min(args.end, MAX_END_DATE)
    pairs = _load_tickers(args)
    if not pairs:
        print("ERROR: no tickers given (use --tickers or --tickers-file).", file=sys.stderr)
        return 2

    print(f"Prefetching {len(pairs)} ticker(s) into {cache.root} over {args.start}..{end}")
    failures = 0
    for i, (ticker, asset_class) in enumerate(pairs, 1):
        try:
            status = await _prefetch_one(tool, ticker, asset_class, args.start, end, args.force)
            print(f"[{i}/{len(pairs)}] {ticker} ({asset_class}): {status}")
        except Exception as e:  # noqa: BLE001 - report and continue to next ticker
            failures += 1
            print(f"[{i}/{len(pairs)}] {ticker} ({asset_class}): FAILED - {type(e).__name__}: {e}", file=sys.stderr)
        if args.sleep > 0 and i < len(pairs):
            await asyncio.sleep(args.sleep)

    print(f"Done. {len(pairs) - failures} ok, {failures} failed.")
    return 1 if failures else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch Tiingo price history into the finance_agent_v2 cache.")
    p.add_argument("--cache-dir", required=True, help="Cache root (same as the server's cache_dir).")
    p.add_argument("--tickers", nargs="*", help="Ticker symbols to prefetch.")
    p.add_argument("--tickers-file", help="File with one 'TICKER[,asset_class]' per line.")
    p.add_argument(
        "--asset-class", default="equity", choices=["equity", "etf", "crypto", "fx"],
        help="Default asset class for tickers without an explicit one (default: equity).",
    )
    p.add_argument("--start", default="1990-01-01", help="Window start date YYYY-MM-DD (default: 1990-01-01).")
    p.add_argument("--end", default=MAX_END_DATE, help=f"Window end date YYYY-MM-DD (clamped to {MAX_END_DATE}).")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between tickers (default: 1.0).")
    p.add_argument("--force", action="store_true", help="Refetch even if the master already covers the window.")
    p.add_argument("--api-key", help="Tiingo API key (else TIINGO_API_KEY / PRICING_DATA_API_KEY env).")
    args = p.parse_args()
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
