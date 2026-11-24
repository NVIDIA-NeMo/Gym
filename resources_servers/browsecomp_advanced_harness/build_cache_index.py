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

With a pack (pack/MANIFEST.json), packed search entries whose flat file is gone
are indexed too, so deleting flat files after packing keeps their queries
visible to fuzzy matching. Each shard <ab> is one task: its flat files are
scanned first, and a packed entry with a flat file is skipped. For the rest only
a prefix of the blob is read and inflated (the same head read as a flat file);
an entry whose norm lies past that prefix is read in full.

Usage:
    python build_cache_index.py <cache_root> [--workers N]
"""

import argparse
import json
import os
import re
import sys
import zlib
from collections import Counter
from multiprocessing import Pool
from os import PathLike

from resources_servers.browsecomp_advanced_harness.search_cache import INDEX_BASE, INDEX_DIR
from resources_servers.browsecomp_advanced_harness.search_cache_pack import (
    INDEX_HEAD_SIZE,
    LEN_MASK,
    REC_SIZE,
    RECORD,
    STORED,
    load_manifest,
    open_pack,
    pack_root_of,
    read_index_head,
    shard_paths,
)
from resources_servers.browsecomp_advanced_harness.sqlite_fuzzy_index import (
    DEFAULT_TOKENIZER,
    build_master_db,
    master_db_path,
)


_HEAD_BYTES = 8192
# Compressed bytes read for a packed entry's head; at the pack's ~3x ratio they
# inflate to more than _HEAD_BYTES.
_PACK_PREFIX_BYTES = 4096
_NORM_RE = re.compile(r'"norm":\s*"((?:[^"\\]|\\.)*)"')


def _norm_in_head(head):
    m = _NORM_RE.search(head)
    return json.loads(f'"{m.group(1)}"') if m else None


def _extract_norm(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(_HEAD_BYTES)
        norm = _norm_in_head(head)
        if norm is not None:
            return norm
        with open(path, "r", encoding="utf-8") as f:  # norm beyond head (huge query) — rare
            return json.load(f).get("norm")
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _packed_norm_from_prefix(fd, offset, length, stored):
    """The norm from the first bytes of a packed blob, or None when it is not in them."""
    try:
        if stored:
            head = os.pread(fd, min(length, _HEAD_BYTES), offset)
        else:
            head = zlib.decompressobj().decompress(os.pread(fd, min(length, _PACK_PREFIX_BYTES), offset), _HEAD_BYTES)
        return _norm_in_head(head.decode("utf-8", errors="replace"))
    except (OSError, zlib.error, ValueError):
        return None


def _packed_norm_in_full(fd, offset, length, raw_len, stored):
    """The norm from the whole blob, or None if it is unreadable."""
    try:
        blob = os.pread(fd, length, offset)
        raw = blob if stored else zlib.decompress(blob, 15, raw_len)
        norm = json.loads(raw.decode("utf-8")).get("norm")
        return norm if isinstance(norm, str) else None
    except (OSError, zlib.error, ValueError, AttributeError):
        return None


def _packed_shard_norms(pack_root, ab, flat_names, counts):
    """Norms of the packed search entries of shard <ab> that have no flat file in flat_names."""
    idx_path, dat_path = shard_paths(pack_root, "search", ab)
    with open(idx_path, "rb") as f:
        data = f.read()
    count, _ = read_index_head(data[:INDEX_HEAD_SIZE], len(data), idx_path)
    norms = []
    fd = os.open(dat_path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        for i in range(count):
            digest, offset, length, raw_len = RECORD.unpack_from(data, INDEX_HEAD_SIZE + i * REC_SIZE)
            if digest.hex() in flat_names:
                counts["skipped_flat"] += 1
                continue
            stored, length = bool(length & STORED), length & LEN_MASK
            norm = _packed_norm_from_prefix(fd, offset, length, stored)
            if norm is not None:
                counts["prefix"] += 1
            else:
                norm = _packed_norm_in_full(fd, offset, length, raw_len, stored)
                counts["full" if norm is not None else "unreadable"] += 1
            if norm:
                norms.append(norm)
    finally:
        os.close(fd)
    return norms


def _scan_shard(task):
    """(norms, counts) of one shard: the flat files under search/<ab>/, then the packed entries without one."""
    cache_root, ab, pack_root = task
    norms, flat_names, counts = [], set(), Counter()
    for dirpath, _, fnames in os.walk(os.path.join(cache_root, "search", ab)):
        for fname in fnames:
            if fname.endswith(".json"):
                flat_names.add(fname[: -len(".json")])
                norm = _extract_norm(os.path.join(dirpath, fname))
                if norm:
                    norms.append(norm)
    if pack_root is not None:
        norms.extend(_packed_shard_norms(pack_root, ab, flat_names, counts))
    return norms, counts


def build_index(cache_root: str | PathLike[str], *, workers: int = 1, tokenizer: str = DEFAULT_TOKENIZER) -> int:
    """Scan the search tree and (re)write the base index. Returns #norms.

    Writes BOTH artifacts, in this order:
      index/search_norms.txt  the plain-text base index (the durable record)
      index/master.db         the FTS5 index the servers actually query

    master.db lands last so its mtime is the merge marker: at startup a server
    loads only the .txt shards newer than the master, which are exactly the
    norms cached since this build. See SearchCache._unmerged_norms.
    """
    search_dir = os.path.join(cache_root, "search")
    subdirs = set()
    if os.path.isdir(search_dir):  # with a pack, the flat tree can be gone
        subdirs = {d for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d))}
    # open_pack refuses a pack whose shard files do not match MANIFEST.json; only shards it lists are read.
    reader = open_pack(cache_root)
    packed = set()
    if reader is not None:
        reader.close()
        packed = {key.split("/", 1)[1] for key in load_manifest(cache_root)["shards"] if key.startswith("search/")}
    pack_root = pack_root_of(cache_root)
    tasks = [(str(cache_root), ab, pack_root if ab in packed else None) for ab in sorted(subdirs | packed)]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(_scan_shard, tasks)
    else:
        results = [_scan_shard(t) for t in tasks]
    counts = sum((c for _, c in results), Counter())
    if packed:
        print(
            f"packed search entries: {counts['prefix']} read from a prefix, {counts['full']} read in full, "
            f"{counts['skipped_flat']} skipped (flat file present), {counts['unreadable']} unreadable",
            file=sys.stderr,
        )
    norms = sorted({n for chunk, _ in results for n in chunk})

    index_dir = os.path.join(cache_root, INDEX_DIR)
    os.makedirs(index_dir, exist_ok=True)
    out = os.path.join(index_dir, INDEX_BASE)
    tmp = f"{out}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(norms) + ("\n" if norms else ""))
    os.replace(tmp, out)

    build_master_db(master_db_path(cache_root), norms, tokenizer=tokenizer)
    return len(norms)


def build_master_from_text_index(cache_root: str | PathLike[str], *, tokenizer: str = DEFAULT_TOKENIZER) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--from-text-index",
        action="store_true",
        help="rebuild only master.db from the existing .txt index (seconds, no JSON rescan)",
    )
    args = parser.parse_args()
    if args.from_text_index:
        # The server queries the index with DEFAULT_TOKENIZER, so the index is built with it too.
        n = build_master_from_text_index(args.cache_root)
        print(f"wrote {n} norms -> {master_db_path(args.cache_root)} (fts5 {DEFAULT_TOKENIZER}, from text index)")
        return 0
    n = build_index(args.cache_root, workers=args.workers)
    print(f"wrote {n} norms -> {os.path.join(args.cache_root, INDEX_DIR, INDEX_BASE)}")
    print(f"wrote {n} norms -> {master_db_path(args.cache_root)} (fts5 {DEFAULT_TOKENIZER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
