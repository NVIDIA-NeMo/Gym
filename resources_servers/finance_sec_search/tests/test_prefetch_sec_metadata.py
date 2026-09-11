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

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.finance_sec_search.scripts import prefetch_sec_metadata as prefetch_mod


LIVE_TICKERS = {
    "0": {"cik_str": "320193", "ticker": "AAPL", "title": "APPLE INC."},
}

SNAP_OVERLAY = {
    "35": {"cik_str": "1564408", "ticker": "SNAP", "title": "Snap Inc"},
}

SNAP_CIK = "0001564408"

SNAP_SUBMISSIONS = json.dumps(
    {
        "filings": {
            "recent": {
                "accessionNumber": ["0001564408-24-000001"],
                "form": ["10-K"],
                "filingDate": ["2024-01-01"],
                "reportDate": ["2023-12-31"],
                "primaryDocument": ["snap.htm"],
            },
            "files": [],
        }
    }
)

AAPL_SUBMISSIONS = json.dumps(
    {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000001"],
                "form": ["10-K"],
                "filingDate": ["2024-01-01"],
                "reportDate": ["2023-12-31"],
                "primaryDocument": ["aapl.htm"],
            },
            "files": [],
        }
    }
)


def _tickers_in_cache(cache_dir: Path) -> list[str]:
    raw = json.loads((cache_dir / "tickers.json").read_text())
    return [item["ticker"].upper() for item in raw.values()]


def _mock_fetch(_session, url, *_args, **_kwargs):
    if url == prefetch_mod.SEC_TICKERS_URL:
        return json.dumps(LIVE_TICKERS)
    if f"CIK{SNAP_CIK}.json" in url:
        return SNAP_SUBMISSIONS
    if "CIK0000320193.json" in url:
        return AAPL_SUBMISSIONS
    return None


@pytest.mark.asyncio
async def test_overlay_persist_includes_supplementary_ticker(tmp_path):
    overlay = tmp_path / "supplementary_tickers.json"
    overlay.write_text(json.dumps(SNAP_OVERLAY))
    cache_dir = tmp_path / "cache"

    with patch.object(prefetch_mod, "fetch_with_retry", new=AsyncMock(side_effect=_mock_fetch)):
        await prefetch_mod.prefetch(
            str(cache_dir),
            ["AAPL"],
            supplementary_tickers=str(overlay),
        )

    persisted = _tickers_in_cache(cache_dir)
    assert sorted(persisted) == ["AAPL", "SNAP"]


def test_overlay_replaces_matching_live_row():
    """A ticker in both sources leaves one row, carrying the overlay's values."""
    live = {
        "0": {"cik_str": "320193", "ticker": "AAPL", "title": "APPLE INC."},
        "1": {"cik_str": "21344", "ticker": "KO", "title": "STALE NAME"},
    }
    overlay = {"0": {"cik_str": "21344", "ticker": "ko", "title": "COCA COLA CO"}}

    merged = prefetch_mod.overlay_tickers_registry(live, overlay)

    assert sorted(item["ticker"].upper() for item in merged.values()) == ["AAPL", "KO"]
    lookup = prefetch_mod.registry_lookup(merged)
    assert lookup["KO"]["name"] == "COCA COLA CO"
    assert lookup["AAPL"]["cik"] == "0000320193"


def test_overlay_keys_do_not_collide_with_sec_numbering():
    live = {"0": {"cik_str": "320193", "ticker": "AAPL", "title": "APPLE INC."}}
    overlay = {"0": {"cik_str": "1564408", "ticker": "SNAP", "title": "Snap Inc"}}

    merged = prefetch_mod.overlay_tickers_registry(live, overlay)

    assert len(merged) == 2
    assert merged["0"]["ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_prefetch_writes_metadata_for_overlay_cik(tmp_path):
    overlay = tmp_path / "supplementary_tickers.json"
    overlay.write_text(json.dumps(SNAP_OVERLAY))
    cache_dir = tmp_path / "cache"

    with patch.object(prefetch_mod, "fetch_with_retry", new=AsyncMock(side_effect=_mock_fetch)):
        await prefetch_mod.prefetch(
            str(cache_dir),
            ["AAPL"],
            supplementary_tickers=str(overlay),
        )

    meta_path = cache_dir / "filings_metadata" / f"{SNAP_CIK}.json"
    assert meta_path.is_file()
    filings = json.loads(meta_path.read_text())
    assert any(item.get("ticker") == "SNAP" for item in filings.values())


@pytest.mark.asyncio
async def test_unknown_ticker_without_overlay_is_skipped(tmp_path, caplog):
    cache_dir = tmp_path / "cache"

    with (
        caplog.at_level("WARNING"),
        patch.object(prefetch_mod, "fetch_with_retry", new=AsyncMock(side_effect=_mock_fetch)),
    ):
        await prefetch_mod.prefetch(str(cache_dir), ["AAPL", "NOTEXIST"])

    assert (cache_dir / "filings_metadata" / "0000320193.json").is_file()
    assert not (cache_dir / "filings_metadata" / "0000000000.json").exists()
    assert "Ticker NOTEXIST not found in SEC registry, skipping" in caplog.text
    assert "SNAP" not in _tickers_in_cache(cache_dir)


@pytest.mark.asyncio
async def test_omitting_supplementary_flag_does_not_require_json(tmp_path):
    cache_dir = tmp_path / "cache"
    missing = tmp_path / "does-not-exist.json"
    assert not missing.exists()

    with patch.object(prefetch_mod, "fetch_with_retry", new=AsyncMock(side_effect=_mock_fetch)):
        await prefetch_mod.prefetch(str(cache_dir), ["AAPL"])

    assert (cache_dir / "tickers.json").is_file()
    assert "AAPL" in _tickers_in_cache(cache_dir)


@pytest.mark.asyncio
async def test_existing_metadata_is_skipped(tmp_path):
    overlay = tmp_path / "supplementary_tickers.json"
    overlay.write_text(json.dumps(SNAP_OVERLAY))
    cache_dir = tmp_path / "cache"
    meta_dir = cache_dir / "filings_metadata"
    meta_dir.mkdir(parents=True)
    snap_path = meta_dir / f"{SNAP_CIK}.json"
    snap_path.write_text(json.dumps({"cached": True}))
    aapl_path = meta_dir / "0000320193.json"
    aapl_path.write_text(json.dumps({"cached": True}))

    fetch = AsyncMock(side_effect=_mock_fetch)
    with patch.object(prefetch_mod, "fetch_with_retry", new=fetch):
        await prefetch_mod.prefetch(
            str(cache_dir),
            ["AAPL"],
            supplementary_tickers=str(overlay),
        )

    assert json.loads(snap_path.read_text()) == {"cached": True}
    assert json.loads(aapl_path.read_text()) == {"cached": True}
    called_urls = [call.args[1] for call in fetch.call_args_list]
    assert prefetch_mod.SEC_TICKERS_URL in called_urls
    assert not any("submissions/CIK" in url for url in called_urls)
