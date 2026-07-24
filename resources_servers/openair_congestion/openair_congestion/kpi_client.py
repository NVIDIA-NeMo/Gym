# SPDX-License-Identifier: Apache-2.0
"""Lightweight Prometheus ``/metrics`` scraper for the openair-rl-gym kpi-exporter.

The kpi-exporter (``services/kpi-exporter/``) ships five gauge families:

- ``openair_prb_util{cell}``         — DL PRB utilisation 0..1
- ``openair_sinr_db{cell,ue}``       — DL SINR in dB
- ``openair_throughput_mbps{cell,ue}`` — delivered DL throughput in Mbps
- ``openair_active_ue_count{cell}``  — active connected UEs
- ``openair_bler{cell,ue}``          — block error rate 0..1

This module is intentionally tiny: a ``KpiSnapshot`` dataclass plus a
synchronous ``fetch(url)`` function, parsing the Prometheus text exposition
format inline. We deliberately avoid pulling in ``prometheus_client`` for
parsing — that library is built for *exposing* metrics, not consuming them,
and adds a non-trivial start-up cost.

Used by :mod:`openair_congestion.env` to build observations on every ``/step``.
"""

from __future__ import annotations

import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict


DEFAULT_URL = "http://localhost:9101/metrics"
DEFAULT_TIMEOUT_S = 3.0

# Pattern matches a single Prometheus sample line like:
#   openair_prb_util{cell="0"} 0.7009
#   openair_sinr_db{cell="1",ue="3"} -0.39
#   openair_exporter_scrapes 332735.0
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|NaN|\+Inf|-Inf)\s*$"
)
_LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')


@dataclass
class KpiSnapshot:
    """Parsed ``/metrics`` snapshot, indexed for O(1) lookups by env builder."""

    prb_util: Dict[int, float] = field(default_factory=dict)
    sinr_db: Dict[tuple[int, int], float] = field(default_factory=dict)
    throughput_mbps: Dict[tuple[int, int], float] = field(default_factory=dict)
    active_ue_count: Dict[int, float] = field(default_factory=dict)
    bler: Dict[tuple[int, int], float] = field(default_factory=dict)
    source_mode: str = "unknown"
    snapshot_fresh: bool | None = None
    snapshot_age_s: float | None = None
    snapshot_revision: int | None = None
    snapshot_id: str | None = None
    scrape_count: float = 0.0
    raw_lines: int = 0

    def cell_ids(self) -> list[int]:
        return sorted(set(self.prb_util.keys()) | set(self.active_ue_count.keys()))

    def ues_in_cell(self, cell_id: int) -> list[int]:
        return sorted({u for c, u in self.sinr_db.keys() if c == cell_id})

    def ue_throughput(self, cell_id: int, ue_id: int, default: float = 0.0) -> float:
        return self.throughput_mbps.get((cell_id, ue_id), default)

    def ue_sinr(self, cell_id: int, ue_id: int, default: float = -10.0) -> float:
        return self.sinr_db.get((cell_id, ue_id), default)

    def ue_bler(self, cell_id: int, ue_id: int, default: float = 0.0) -> float:
        return self.bler.get((cell_id, ue_id), default)


class KpiScrapeError(RuntimeError):
    """Raised when the kpi-exporter is unreachable or returns malformed text."""


def _int_label(labels: dict[str, str], key: str) -> int | None:
    try:
        return int(labels[key])
    except (KeyError, TypeError, ValueError):
        return None


def parse(text: str) -> KpiSnapshot:
    """Parse a Prometheus exposition payload into a :class:`KpiSnapshot`."""
    snap = KpiSnapshot()
    for raw in text.splitlines():
        snap.raw_lines += 1
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if m is None:
            continue
        name = m.group("name")
        labels_str = m.group("labels") or ""
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = {lm.group("key"): lm.group("value") for lm in _LABEL_RE.finditer(labels_str)}
        if name == "openair_prb_util" and "cell" in labels:
            cell_id = _int_label(labels, "cell")
            if cell_id is not None:
                snap.prb_util[cell_id] = value
        elif name == "openair_sinr_db" and "cell" in labels and "ue" in labels:
            cell_id = _int_label(labels, "cell")
            ue_id = _int_label(labels, "ue")
            if cell_id is not None and ue_id is not None:
                snap.sinr_db[(cell_id, ue_id)] = value
        elif name == "openair_throughput_mbps" and "cell" in labels and "ue" in labels:
            cell_id = _int_label(labels, "cell")
            ue_id = _int_label(labels, "ue")
            if cell_id is not None and ue_id is not None:
                snap.throughput_mbps[(cell_id, ue_id)] = value
        elif name == "openair_active_ue_count" and "cell" in labels:
            cell_id = _int_label(labels, "cell")
            if cell_id is not None:
                snap.active_ue_count[cell_id] = value
        elif name == "openair_bler" and "cell" in labels and "ue" in labels:
            cell_id = _int_label(labels, "cell")
            ue_id = _int_label(labels, "ue")
            if cell_id is not None and ue_id is not None:
                snap.bler[(cell_id, ue_id)] = value
        elif name == "openair_exporter_scrapes":
            snap.scrape_count = value
        elif name == "openair_exporter_source" and "mode" in labels:
            snap.source_mode = labels["mode"]
        elif name == "openair_runner_snapshot_fresh":
            snap.snapshot_fresh = value >= 0.5
        elif name == "openair_runner_snapshot_age_seconds":
            snap.snapshot_age_s = value
        elif name == "openair_runner_snapshot_revision":
            snap.snapshot_revision = int(value)
        elif name == "openair_runner_snapshot_identity" and "snapshot_id" in labels:
            if value >= 0.5:
                snap.snapshot_id = labels["snapshot_id"]
    return snap


def fetch(url: str = DEFAULT_URL, timeout_s: float = DEFAULT_TIMEOUT_S) -> KpiSnapshot:
    """Scrape the kpi-exporter once. Raises :class:`KpiScrapeError` on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise KpiScrapeError(f"kpi-exporter {url} returned HTTP {resp.status}")
            payload = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise KpiScrapeError(f"kpi-exporter {url} unreachable: {e}") from e
    except (TimeoutError, OSError) as e:  # network errors
        raise KpiScrapeError(f"kpi-exporter {url} I/O error: {e}") from e
    return parse(payload)


__all__ = [
    "DEFAULT_URL",
    "DEFAULT_TIMEOUT_S",
    "KpiSnapshot",
    "KpiScrapeError",
    "parse",
    "fetch",
]
