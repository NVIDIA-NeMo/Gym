# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Offline builder for the search-norms fuzzy index (see search_cache.py).

Scans <cache_root>/search/**/*.json, extracts each entry's normalized query,
and writes <cache_root>/index/search_norms.txt (one norm per line, atomic
replace). Server processes load this base file plus their own append shards.

Payloads store "norm" before the (large) "results" field, so a bounded head
read + regex avoids parsing 400k full JSON bodies; falls back to a full parse.

Usage:
    python build_cache_index.py <cache_root> [--workers N]
"""

import argparse
import json
import os
import re
import sys
from multiprocessing import Pool

from resources_servers.browsecomp_advanced_harness.search_cache import INDEX_BASE, INDEX_DIR
from resources_servers.browsecomp_advanced_harness.sqlite_fuzzy_index import (
    DEFAULT_TOKENIZER,
    build_master_db,
    master_db_path,
)


_HEAD_BYTES = 8192
_NORM_RE = re.compile(r'"norm":\s*"((?:[^"\\]|\\.)*)"')


def _extract_norm(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(_HEAD_BYTES)
        m = _NORM_RE.search(head)
        if m:
            return json.loads(f'"{m.group(1)}"')
        with open(path, "r", encoding="utf-8") as f:  # norm beyond head (huge query) — rare
            return json.load(f).get("norm")
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _scan_subdir(subdir):
    norms = []
    for dirpath, _, fnames in os.walk(subdir):
        for fname in fnames:
            if fname.endswith(".json"):
                norm = _extract_norm(os.path.join(dirpath, fname))
                if norm:
                    norms.append(norm)
    return norms


def build_index(cache_root, workers=1, tokenizer=DEFAULT_TOKENIZER):
    """Scan the search tree and (re)write the base index. Returns #norms.

    Writes BOTH artifacts, in this order:
      index/search_norms.txt  the plain-text base index (the durable record)
      index/master.db         the FTS5 index the servers actually query

    master.db lands last so its mtime is the merge marker: at startup a server
    loads only the .txt shards newer than the master, which are exactly the
    norms cached since this build. See SearchCache._unmerged_norms.
    """
    search_dir = os.path.join(cache_root, "search")
    subdirs = sorted(
        os.path.join(search_dir, d) for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d))
    )
    if workers > 1:
        with Pool(workers) as pool:
            chunks = pool.map(_scan_subdir, subdirs)
    else:
        chunks = [_scan_subdir(d) for d in subdirs]
    norms = sorted({n for chunk in chunks for n in chunk})

    index_dir = os.path.join(cache_root, INDEX_DIR)
    os.makedirs(index_dir, exist_ok=True)
    out = os.path.join(index_dir, INDEX_BASE)
    tmp = f"{out}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(norms) + ("\n" if norms else ""))
    os.replace(tmp, out)

    build_master_db(master_db_path(cache_root), norms, tokenizer=tokenizer)
    return len(norms)


def build_master_from_text_index(cache_root, tokenizer=DEFAULT_TOKENIZER):
    """Rebuild ONLY master.db, from the existing .txt index. Returns #norms.

    The full build_index rescans every entry under search/ (430k+ JSON files,
    minutes of Lustre metadata traffic). When the text index is already current
    — it is the durable record, and servers append to it as they cache — the
    master can be rebuilt from it in seconds instead.

    Use this to add a master to a cache root that predates it, or to re-merge
    accumulated shards. Use build_index when the text index itself is stale.
    """
    index_dir = os.path.join(cache_root, INDEX_DIR)
    seen, norms = set(), []
    for fname in sorted(os.listdir(index_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(index_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                norm = line.rstrip("\n")
                if norm and norm not in seen:
                    seen.add(norm)
                    norms.append(norm)
    return build_master_db(master_db_path(cache_root), norms, tokenizer=tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, choices=["trigram", "unicode61"])
    parser.add_argument(
        "--from-text-index",
        action="store_true",
        help="rebuild only master.db from the existing .txt index (seconds, no JSON rescan)",
    )
    args = parser.parse_args()
    if args.from_text_index:
        n = build_master_from_text_index(args.cache_root, tokenizer=args.tokenizer)
        print(f"wrote {n} norms -> {master_db_path(args.cache_root)} (fts5 {args.tokenizer}, from text index)")
        return 0
    n = build_index(args.cache_root, workers=args.workers, tokenizer=args.tokenizer)
    print(f"wrote {n} norms -> {os.path.join(args.cache_root, INDEX_DIR, INDEX_BASE)}")
    print(f"wrote {n} norms -> {master_db_path(args.cache_root)} (fts5 {args.tokenizer})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
