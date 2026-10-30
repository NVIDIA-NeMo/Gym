# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flat-file cache for Exa MCP responses, shared by every shard and every sandbox.

LAYOUT -- deliberately the legacy one (the previous stack's
responses_api_agents/swe_agents/search_cache.py), not a new design:

    <root>/search/<ab>/<cd>/<sha256(norm_query)>.json  -> {query, norm, provider, payload}
    <root>/browse/<ab>/<cd>/<sha256(norm_url)>.json    -> {url,   norm, provider, payload}

One store with two sections, content-addressed, two-level hex sharding. Keys are GLOBAL rather
than partitioned per question: measured over a completed run, cross-question reuse is
essentially nil at both an exact match and a high Jaccard threshold, so partitioning would buy
nothing while fencing off reuse if the question set ever changes.

WHY FUZZY MATCHING IS NOT OPTIONAL. Measured on the same data: exact-key reuse across two runs
of an identical question set is negligible, while near-duplicate reuse is an order of magnitude
larger, and most questions issue several near-duplicate queries. An exact-only cache would look
broken.

WHAT IS STORED: the text the transport returned, BEFORE the contamination guard and the TDM
domain filter run. Those are applied by the plugin on the way out, on hits and misses alike. Cache
post-filter text and a later run with a different exclude-list silently inherits the old
filtering; legacy made the same choice for the same reason.

PROVIDER TAGGING: entries carry the provider that produced them, so an Exa entry is never served
to a Tavily caller or vice versa. We only run Exa today, but a mixed store that forgets which is
which is silently wrong rather than loudly broken.

CONCURRENCY: writes are tmp-file + os.replace, which is atomic within a directory on Lustre, so a
reader never sees a half-written entry and concurrent writers of the same key are last-wins on
identical content. No locks, no SQLite, nothing that takes a cross-node lease -- the failure mode
of a lock on this filesystem is worse than the duplicate work it would prevent.
"""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_query(text: str) -> str:
    """Lowercase, strip one layer of surrounding quotes, collapse whitespace.

    Byte-compatible with the legacy normaliser (the previous stack's
    swe_agents/search_cache.py) ON PURPOSE. The key is sha256(norm), so a normaliser that differs
    by even a trailing character addresses a different file and every entry already on disk
    becomes invisible. Do not "improve" this without a new root.
    """
    q = (text or "").strip()
    if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
        q = q[1:-1]
    return " ".join(q.split()).strip().lower()


def normalize_url(url: str) -> str:
    """Drop the #fragment and one trailing slash (legacy search_cache.py:122-128)."""
    u = (url or "").strip().split("#", 1)[0]
    return u[:-1] if u.endswith("/") else u


def normalize(text: str, kind: str = "search") -> str:
    return normalize_query(text) if kind == "search" else normalize_url(text)


def render_tavily(results: list) -> str:
    """Render Tavily's structured results into the Title:/URL:/... text Exa returns.

    Needed because the legacy store is ~90% Tavily by volume and those entries hold a list of
    dicts, while our plugin parses the rendered Exa SERP. Field order matches the fork's renderer
    so the model sees the shape it expects.

    This is a TRANSLATION, not a recording: the content came from Tavily, not Exa. Fine for cost
    (the model still gets real search results) but it breaks strict comparability with an
    Exa-only baseline, which is why it sits behind its own flag.
    """
    blocks = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        entry = "Title: " + str(r.get("title") or "") + "\nURL: " + str(r.get("url") or "")
        entry += "\nPublished: " + str(r.get("published_date") or "N/A")
        body = r.get("raw_content") or r.get("content") or ""
        if body:
            entry += "\nText: " + str(body)
        blocks.append(entry)
    return "\n\n".join(blocks)


def to_sse(text: str) -> str:
    """Wrap rendered SERP text in the one-frame SSE envelope the plugin parses.

    parseMcpBody reads `data: ` frames and takes result.content[0].text, so a legacy entry -- which
    stores only the rendered text -- has to be re-enveloped before it can be served as a cache hit.
    """
    import json as _json

    return "event: message\ndata: " + _json.dumps({"result": {"content": [{"type": "text", "text": text}]}}) + "\n\n"


def _tokens(norm: str) -> frozenset:
    return frozenset(_TOKEN_RE.findall(norm))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ExaCache:
    """Read/write the flat-file store, with an in-process index for near-duplicate lookup.

    The index is built once by walking the store and then kept current by every put(). It holds
    only (norm, key) pairs -- roughly 60 bytes per entry -- so a million entries is ~60MB, which is
    affordable in the resources server. Payloads are never held in memory.
    """

    def __init__(
        self,
        root: str,
        fuzzy_threshold: float = 0.8,
        provider: str = "exa",
        legacy_root: str = "",
        use_tavily: bool = True,
    ) -> None:
        self.root = Path(root)
        # A READ-ONLY second store, searched by exact key only. The legacy cache holds ~2M
        # entries; walking it to build a fuzzy index would hang server startup while rollouts
        # wait on seed_session, and an exact lookup is a direct path from the hash, so it costs
        # one stat() regardless of store size. We never write here.
        self.legacy_root = Path(legacy_root) if legacy_root else None
        # Legacy entries are ~90% Tavily by volume. Those hold structured results rather than
        # rendered Exa text, so serving them means TRANSLATING (render_tavily). The model still
        # gets real search results, but they did not come from Exa -- acceptable for cost,
        # not for a strict comparison against an Exa-only baseline.
        self.use_tavily = use_tavily
        self.threshold = fuzzy_threshold
        self.provider = provider
        self._lock = threading.Lock()
        # kind -> list[(norm, tokens, key)]
        self._index: Dict[str, list] = {"search": [], "browse": []}
        self.stats = collections_counter()
        self._loaded = False

    # ---- paths -------------------------------------------------------------------------------
    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / key[:2] / key[2:4] / f"{key}.json"

    @staticmethod
    def _key(norm: str) -> str:
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    # ---- index -------------------------------------------------------------------------------
    def load(self) -> None:
        """Walk the store once to build the near-duplicate index.

        Done lazily on first use rather than at construction: a cold root costs nothing, and a warm
        one should not delay server startup while rollouts wait on seed_session.
        """
        with self._lock:
            if self._loaded:
                return
            started = time.time()
            n = 0
            for kind in ("search", "browse"):
                base = self.root / kind
                if not base.is_dir():
                    continue
                for path in base.glob("*/*/*.json"):
                    try:
                        rec = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue  # a torn or deleted entry is a miss, never an error
                    norm = rec.get("norm")
                    if not norm:
                        continue
                    self._index[kind].append((norm, _tokens(norm), path.stem))
                    n += 1
            self._loaded = True
            self.stats["index_entries"] = n
            self.stats["index_load_ms"] = int((time.time() - started) * 1000)

    # ---- read/write --------------------------------------------------------------------------
    def get(self, kind: str, text: str) -> Tuple[Optional[Any], str]:
        """Return (payload, how) where how is 'exact' | 'fuzzy' | 'miss'."""
        self.load()
        norm = normalize(text, kind)
        key = self._key(norm)
        rec = self._read(kind, key)
        if rec is not None:
            self.stats["hit_exact"] += 1
            return rec.get("payload"), "exact"

        legacy = self._read_legacy(kind, key)
        if legacy is not None:
            self.stats["hit_legacy"] += 1
            return legacy, "legacy"

        toks = _tokens(norm)
        best, best_score = None, 0.0
        for cand_norm, cand_toks, cand_key in self._index.get(kind, ()):
            score = _jaccard(toks, cand_toks)
            if score > best_score:
                best, best_score = cand_key, score
        if best is not None and best_score >= self.threshold:
            rec = self._read(kind, best)
            if rec is not None:
                self.stats["hit_fuzzy"] += 1
                return rec.get("payload"), "fuzzy"
        self.stats["miss"] += 1
        return None, "miss"

    def put(self, kind: str, text: str, payload: Any) -> None:
        self.load()
        norm = normalize(text, kind)
        key = self._key(norm)
        path = self._path(kind, key)
        rec = {
            "query" if kind == "search" else "url": text,
            "norm": norm,
            "provider": self.provider,
            "payload": payload,
            "ts": int(time.time()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # tmp + replace: atomic within the directory, so a concurrent reader never sees a
            # partial file and two writers of the same key are last-wins on identical content.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(rec, fh)
            os.replace(tmp, path)
        except OSError as e:
            # A cache that cannot write must not break a rollout. Lustre returns EIO on roughly 1%
            # of creates here, which is exactly the case this swallows.
            self.stats["put_error"] += 1
            print(f"[exa-cache] put failed ({type(e).__name__}): {e}", flush=True)
            return
        with self._lock:
            self._index.setdefault(kind, []).append((norm, _tokens(norm), key))
        self.stats["put"] += 1

    def _read_legacy(self, kind: str, key: str) -> Optional[str]:
        """Exact-key read from the legacy store, re-enveloped into what the plugin parses.

        Legacy records store `results`, not `payload`: a rendered SERP string for Exa entries, a
        list of structured results for Tavily ones. Both have to be wrapped in the one-frame SSE
        envelope before the plugin's parseMcpBody can read them.
        """
        if self.legacy_root is None:
            return None
        path = self.legacy_root / kind / key[:2] / key[2:4] / f"{key}.json"
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        results = rec.get("results")
        provider = rec.get("provider") or "tavily"  # untagged legacy entries predate the field
        if isinstance(results, str):
            text = results  # exa: already the rendered SERP
        elif isinstance(results, list):
            if not self.use_tavily and provider != "exa":
                self.stats["legacy_tavily_skipped"] += 1
                return None
            text = render_tavily(results)  # tavily: structured -> rendered
            self.stats["legacy_tavily_served"] += 1
        else:
            return None
        if not text.strip():
            return None
        return to_sse(text)

    def _read(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(kind, key)
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        # Never serve an entry produced by a different provider: the rendered text differs.
        if rec.get("provider") not in (None, self.provider):
            return None
        return rec


def collections_counter():
    from collections import Counter

    return Counter()
