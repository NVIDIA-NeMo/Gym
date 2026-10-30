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
"""Offline builder and verifier for the cache pack (format: search_cache_pack.py).

Packs the flat entries of each shard <kind>/<ab> -- the files
<root>/<kind>/<ab>/<cd>/<sha256>.json -- into <root>/pack/<kind>/<ab>.{idx,dat},
verifies them, and last writes <root>/pack/MANIFEST.json, which is what turns
the pack on for SearchCache. The flat tree is only listed and read; `unpack`
is the one command that writes there, and it only creates missing files.

Usage (Gym root on PYTHONPATH):
    python -m resources_servers.browsecomp_advanced_harness.build_cache_pack <command> <cache_root> [options]

    build       pack the selected shards (resumable: built shards are skipped)
    verify      check built shards; records a verified flag per shard
    finalize    write MANIFEST.json once all 512 shards are built and verified
    status      built / verified counts and totals
    unpack      rollback: recreate missing flat files of shards from the pack
    candidates  list flat files verified byte-identical to their packed copy

Shard selection (build, verify, status, unpack, candidates):
    --shards SPEC [SPEC ...]   SPEC = all | <kind>:all | <kind>:<ab>[-<ab>][,...]
                               e.g. search:00-3f  browse:all  search:7f,a3
    --part I/N                 keep every N-th selected shard from the I-th on,
                               to split one selection over N nodes

build, per shard: list <kind>/<ab>/*/; every ^[0-9a-f]{64}\\.json$ file whose
name matches its directory is read and must decode as UTF-8 and json.loads
(how SearchCache serves it). Files that do not are skipped and recorded: they
stay flat-only, a miss exactly as today. *.tmp files and other names (e.g. the
0-byte rsync partials .<name>.json.XXXXXX) are counted and listed. The shard is
written to node-local tmp, copied to pack/<kind>/<ab>.{dat,idx}.tmp, fsynced
and renamed into place; its manifest (manifests/<kind>_<ab>.json) goes last and
records, per packed entry, relpath, size, mtime_ns and sha256 of the raw bytes,
so a later flat-file deletion can be gated on "unchanged since packed".

verify, per shard, through the production PackReader: index structure (sorted,
fan-out consistent, blob extents tile the .dat, stored flag iff >= 8 MiB);
every manifest entry's pack bytes against the manifest sha256, and against the
current flat bytes when the flat file's size and mtime still match the
manifest. Flat files changed, gone or new since the build are counted, not
failed. A sample is also compared at the SearchCache level: the entry served
from the flat file equals the entry served from the pack. The outcome and a
verified flag are written into the shard manifest; finalize requires them.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import re
import shutil
import socket
import sys
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager

from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache, normalize_query, normalize_url
from resources_servers.browsecomp_advanced_harness.search_cache_pack import (
    FORMAT,
    INDEX_HEAD_SIZE,
    KINDS,
    LEN_MASK,
    MANIFEST,
    REC_SIZE,
    RECORD,
    STORE_RAW_MIN,
    STORED,
    VERSION,
    ZLIB_LEVEL,
    PackReader,
    all_shards,
    encode_blob,
    pack_root_of,
    read_index_head,
    shard_key,
    shard_manifest_path,
    shard_paths,
    write_index,
)


NAME_RE = re.compile(r"^[0-9a-f]{64}\.json$")
SHARD_FORMAT = "bcpack-shard"
ENTRY_FIELDS = ["relpath", "size", "mtime_ns", "sha256"]
DEFAULT_SAMPLE = 128
_CHUNK = 64
_MAX_LISTED_FAILURES = 1000
# Per-entry outcomes that are not failures and are listed in the verify record.
_FLAT_LISTS = ("flat_changed", "flat_missing", "flat_unreadable")


# ---- shard selection ------------------------------------------------------
def parse_shards(specs):
    """[(kind, ab)] for SPECs (see module doc), in all_shards() order."""
    selected = set()
    for spec in specs:
        if spec == "all":
            selected.update(all_shards())
            continue
        kind, _, sel = spec.partition(":")
        if kind not in KINDS or not sel:
            raise ValueError(f"bad shard spec {spec!r}: want all, <kind>:all or <kind>:<ab>[-<ab>][,...]")
        for item in sel.split(","):
            if item == "all":
                lo, hi = 0, 255
            else:
                ends = item.split("-")
                if len(ends) > 2 or not all(re.fullmatch(r"[0-9a-f]{2}", e) for e in ends):
                    raise ValueError(f"bad shard range {item!r} in {spec!r}: want two lowercase hex digits")
                lo, hi = int(ends[0], 16), int(ends[-1], 16)
            selected.update((kind, f"{b:02x}") for b in range(lo, hi + 1))
    return [s for s in all_shards() if s in selected]


def take_part(shards, part):
    """Every N-th shard from the I-th on, for part "I/N"."""
    if not part:
        return shards
    i, n = (int(x) for x in part.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad --part {part!r}: want I/N with 0 <= I < N")
    return shards[i::n]


# ---- small file helpers ---------------------------------------------------
def _relpath(kind, h):
    return f"{kind}/{h[:2]}/{h[2:4]}/{h}.json"


def _size(path):
    try:
        return os.stat(path).st_size
    except FileNotFoundError:
        return None


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _write_json_atomic(path, obj):
    """Write obj as JSON via temp + fsync + rename. A top-level "entries" list
    goes last, one entry per line, so the summary stays readable with head."""
    head = {k: v for k, v in obj.items() if k != "entries"}
    text = json.dumps(head, indent=1, ensure_ascii=False)
    if "entries" in obj:
        rows = ",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in obj["entries"])
        text = text[: -len("\n}")] + ',\n "entries": [\n' + rows + ("\n" if rows else "") + " ]\n}"
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(src, dst):
    """Copy the node-local src to <dst>.tmp, fsync, and rename it over dst."""
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, dst)
    _fsync_dir(os.path.dirname(dst))


@contextmanager
def _pool(workers):
    if workers <= 1:
        yield None
        return
    # fork: workers start with the modules already imported.
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        yield pool


def _imap(pool, fn, items, chunksize):
    if pool is None:
        return map(fn, items)
    return pool.imap_unordered(fn, items, chunksize=chunksize)


def load_shard_manifest(root, kind, ab):
    return _load_json(shard_manifest_path(pack_root_of(root), kind, ab))


def _brief(m):
    """Shard manifest without its per-entry and per-file lists, for logs."""
    out = {k: v for k, v in m.items() if k not in ("entries", "skipped", "tmp_files", "other_files", "verify")}
    v = m.get("verify")
    if v:
        out["verify"] = {k: x for k, x in v.items() if not isinstance(x, list)}
    return out


# ---- build ----------------------------------------------------------------
def list_flat_shard(root, kind, ab):
    """Classify the files under <root>/<kind>/<ab>/.

    Returns (names, tmp_files, other_files, misplaced): names are the 64-hex
    keys of well-placed entry files; the rest are relpaths. A misplaced entry
    file (name not under its own <ab>/<cd>) is unreachable for SearchCache, so
    packing it would turn today's miss into a hit."""
    base = os.path.join(root, kind, ab)
    names, tmp_files, other_files, misplaced = [], [], [], []
    try:
        subdirs = sorted(os.listdir(base))
    except FileNotFoundError:
        return names, tmp_files, other_files, misplaced
    for cd in subdirs:
        rel_dir = f"{kind}/{ab}/{cd}"
        d = os.path.join(base, cd)
        if not os.path.isdir(d):
            other_files.append(rel_dir)
            continue
        with os.scandir(d) as it:
            for e in it:
                rel = f"{rel_dir}/{e.name}"
                if NAME_RE.match(e.name):
                    if e.name[:2] == ab and e.name[2:4] == cd:
                        names.append(e.name[:64])
                    else:
                        misplaced.append(rel)
                elif e.name.endswith(".tmp"):
                    tmp_files.append(rel)
                else:
                    other_files.append(rel)
    for lst in (names, tmp_files, other_files, misplaced):
        lst.sort()
    return names, tmp_files, other_files, misplaced


def _unparseable(raw):
    """None if raw parses the way SearchCache serves it, else (reason, detail)."""
    try:
        json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as e:
        return "utf8_error", repr(e)[:300]
    except ValueError as e:
        return "json_error", repr(e)[:300]
    except RecursionError as e:
        return "json_recursion", repr(e)[:300]
    return None


def _load_entry(item):
    """Pool worker: (hex, "ok", (size, mtime_ns, raw_len, sha256, blob, length field)) or (hex, reason, detail)."""
    h, path = item
    try:
        with open(path, "rb") as f:
            st = os.fstat(f.fileno())
            raw = f.read()
    except FileNotFoundError:
        return h, "vanished", None
    except OSError as e:
        return h, "oserror", repr(e)[:300]
    if len(raw) != st.st_size:
        return h, "size_changed_while_reading", f"read {len(raw)} of {st.st_size} bytes"
    bad = _unparseable(raw)
    if bad is not None:
        return h, bad[0], bad[1]
    try:
        blob, length = encode_blob(raw)
    except ValueError as e:
        return h, "too_large", str(e)
    return h, "ok", (st.st_size, st.st_mtime_ns, len(raw), hashlib.sha256(raw).hexdigest(), blob, length)


def is_built(root, kind, ab):
    """True if the shard's manifest says built and its files are at the recorded sizes."""
    m = load_shard_manifest(root, kind, ab)
    if not m or not m.get("built"):
        return False
    idx_path, dat_path = shard_paths(pack_root_of(root), kind, ab)
    return _size(idx_path) == m["idx_bytes"] and _size(dat_path) == m["dat_bytes"]


def build_shard(root, kind, ab, tmp_dir, pool=None):
    """Pack one shard and write its manifest. Returns the manifest."""
    pack_root = pack_root_of(root)
    t_start = time.time()
    names, tmp_files, other_files, misplaced = list_flat_shard(root, kind, ab)
    t_listed = time.time()

    os.makedirs(tmp_dir, exist_ok=True)
    local_dat = os.path.join(tmp_dir, f"{kind}_{ab}.dat")
    local_idx = os.path.join(tmp_dir, f"{kind}_{ab}.idx")
    records, entries = [], []
    skipped = [{"relpath": rel, "reason": "misplaced", "detail": None} for rel in misplaced]
    offset = raw_bytes = stored_count = stored_bytes = 0
    items = [(h, os.path.join(root, _relpath(kind, h))) for h in names]
    with open(local_dat, "wb") as dat:
        for h, status, payload in _imap(pool, _load_entry, items, chunksize=4):
            rel = _relpath(kind, h)
            if status != "ok":
                skipped.append({"relpath": rel, "reason": status, "detail": payload})
                continue
            size, mtime_ns, raw_len, sha, blob, length = payload
            dat.write(blob)
            records.append((bytes.fromhex(h), offset, length, raw_len))
            entries.append([rel, size, mtime_ns, sha])
            offset += len(blob)
            raw_bytes += raw_len
            if length & STORED:
                stored_count += 1
                stored_bytes += raw_len
        dat.flush()
        os.fsync(dat.fileno())
    records.sort()
    entries.sort()
    skipped.sort(key=lambda s: s["relpath"])
    with open(local_idx, "wb") as f:
        write_index(f, records)
        f.flush()
        os.fsync(f.fileno())
    t_built = time.time()

    idx_path, dat_path = shard_paths(pack_root, kind, ab)
    mpath = shard_manifest_path(pack_root, kind, ab)
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    if os.path.exists(mpath):
        # A rebuild: retire the old record before its files change, so no
        # crash can leave a verified manifest describing the new files.
        os.replace(mpath, mpath + ".replaced")
    _publish(local_dat, dat_path)
    _publish(local_idx, idx_path)
    os.remove(local_dat)
    os.remove(local_idx)
    dat_bytes, idx_bytes = _size(dat_path), _size(idx_path)
    if dat_bytes != offset:
        raise RuntimeError(f"{dat_path} is {dat_bytes} bytes after publish, wrote {offset}")
    t_done = time.time()

    manifest = {
        "format": SHARD_FORMAT,
        "version": VERSION,
        "kind": kind,
        "ab": ab,
        "build_id": uuid.uuid4().hex,
        "built": True,
        "verified": False,
        "zlib_level": ZLIB_LEVEL,
        "store_raw_min_bytes": STORE_RAW_MIN,
        "flat_files_listed": len(names) + len(misplaced),
        "entries_count": len(entries),
        "raw_bytes": raw_bytes,
        "dat_bytes": dat_bytes,
        "idx_bytes": idx_bytes,
        "stored_raw_count": stored_count,
        "stored_raw_bytes": stored_bytes,
        "skipped_count": len(skipped),
        "skipped_reasons": dict(Counter(s["reason"] for s in skipped)),
        "tmp_count": len(tmp_files),
        "other_count": len(other_files),
        "build_host": socket.gethostname(),
        "build_started_unix": t_start,
        "build_finished_unix": t_done,
        "list_s": round(t_listed - t_start, 3),
        "pack_s": round(t_built - t_listed, 3),
        "publish_s": round(t_done - t_built, 3),
        "verify": None,
        "skipped": skipped,
        "tmp_files": tmp_files,
        "other_files": other_files,
        "entry_fields": ENTRY_FIELDS,
        "entries": entries,
    }
    _write_json_atomic(mpath, manifest)
    return manifest


# ---- verify ---------------------------------------------------------------
def check_index(idx_path, dat_size, ab):
    """(records, problems) for one .idx against its .dat size; problems == [] means sound."""
    with open(idx_path, "rb") as f:
        data = f.read()
    try:
        count, fan = read_index_head(data[:INDEX_HEAD_SIZE], len(data), idx_path)
    except ValueError as e:
        return None, [str(e)]
    records = [RECORD.unpack_from(data, INDEX_HEAD_SIZE + i * REC_SIZE) for i in range(count)]
    problems = []
    first = int(ab, 16)
    seen_fan = [0] * 257
    for i, (digest, _, length, raw_len) in enumerate(records):
        if i and digest <= records[i - 1][0]:
            problems.append(f"record {i} is not strictly after record {i - 1}")
        if digest[0] != first:
            problems.append(f"record {i} ({digest.hex()}) does not belong to shard {ab}")
        seen_fan[digest[1] + 1] += 1
        stored = bool(length & STORED)
        if stored != (raw_len >= STORE_RAW_MIN):
            problems.append(f"record {i}: stored={stored} for a {raw_len}-byte entry")
        if stored and (length & LEN_MASK) != raw_len:
            problems.append(f"record {i}: stored blob of {length & LEN_MASK} bytes for a {raw_len}-byte entry")
        if len(problems) > 50:
            break
    for b in range(256):
        seen_fan[b + 1] += seen_fan[b]
    if tuple(seen_fan) != tuple(fan):
        problems.append("fan-out table does not match the records")
    pos = 0
    for offset, length in sorted((r[1], r[2] & LEN_MASK) for r in records):
        if offset != pos:
            problems.append(f"blob extents leave a gap or overlap at .dat offset {pos}")
            break
        pos += length
    else:
        if pos != dat_size:
            problems.append(f"blobs cover {pos} bytes, the .dat is {dat_size}")
    return records, problems


def _same(a, b):
    """Equality that also holds for entries containing NaN (NaN != NaN)."""
    return a == b or json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


_VERIFY_CACHES = {}


def _verify_caches(root):
    """Per-process (flat-serving, pack-serving) SearchCache pair for the sample check."""
    pair = _VERIFY_CACHES.get(root)
    if pair is None:
        flat_cache = SearchCache(root, read_only=True)
        # A root that does not exist has no flat files and no manifest, so every
        # _read on it falls through to whichever reader is attached as _pack.
        pack_cache = SearchCache(os.path.join(pack_root_of(root), f".no-such-root-{uuid.uuid4().hex}"), read_only=True)
        pair = _VERIFY_CACHES[root] = (flat_cache, pack_cache)
    return pair


def _sample_check(root, kind, digest, flat_raw, reader):
    """None if SearchCache serves the same entry from the flat file and from the pack, else a failure name."""
    obj = json.loads(flat_raw.decode("utf-8"))
    norm = obj.get("norm") if isinstance(obj, dict) else None
    if not isinstance(norm, str) or hashlib.sha256(norm.encode("utf-8")).digest() != digest:
        # Reachable only by its file name, which the byte check already covers.
        return "sample_norm_mismatch"
    flat_cache, pack_cache = _verify_caches(root)
    pack_cache._pack = reader
    before = pack_cache.stats()
    if not _same(flat_cache._read(kind, norm), pack_cache._read(kind, norm, count_pack_hit=True)):
        return "sample_read_mismatch"
    served = 1
    if kind == "search" and normalize_query(norm) == norm:
        if not _same(flat_cache.get_search(norm), pack_cache.get_search(norm)):
            return "sample_get_mismatch"
        served += 1
    elif kind == "browse" and normalize_url(norm) == norm:
        if not _same(flat_cache.get_extract(norm), pack_cache.get_extract(norm)):
            return "sample_get_mismatch"
        served += 1
    after = pack_cache.stats()
    if (
        after[f"{kind}_pack_hits"] - before[f"{kind}_pack_hits"] != served
        or after["pack_errors"] != before["pack_errors"]
    ):
        return "sample_not_served_from_pack"
    return None


def _verify_chunk(task):
    """Pool worker: check a run of manifest entries. Returns (counts, lists, failures)."""
    root, kind, chunk = task
    reader = PackReader(pack_root_of(root))
    counts, lists, failures = Counter(), {k: [] for k in _FLAT_LISTS}, []

    def fail(rel, check, detail=None):
        counts[check] += 1
        failures.append({"relpath": rel, "check": check, "detail": detail})

    try:
        for (rel, size, mtime_ns, sha), in_sample in chunk:
            digest = bytes.fromhex(os.path.basename(rel)[:64])
            errors_before = reader.errors
            pack_raw = reader.get_bytes(kind, digest)
            if pack_raw is None:
                fail(rel, "pack_read_error" if reader.errors > errors_before else "pack_missing")
                continue
            if len(pack_raw) != size or hashlib.sha256(pack_raw).hexdigest() != sha:
                fail(rel, "pack_sha_mismatch", f"{len(pack_raw)} bytes")
                continue
            try:
                with open(os.path.join(root, rel), "rb") as f:
                    st = os.fstat(f.fileno())
                    unchanged = st.st_size == size and st.st_mtime_ns == mtime_ns
                    flat_raw = f.read() if unchanged else None
            except FileNotFoundError:
                counts["flat_missing"] += 1
                lists["flat_missing"].append(rel)
                continue
            except OSError:
                counts["flat_unreadable"] += 1
                lists["flat_unreadable"].append(rel)
                continue
            if flat_raw is None:
                counts["flat_changed"] += 1
                lists["flat_changed"].append(rel)
                continue
            if flat_raw != pack_raw:
                fail(rel, "flat_bytes_mismatch", "flat size and mtime match the manifest, bytes do not")
                continue
            counts["identical"] += 1
            if in_sample:
                counts["sample_checked"] += 1
                bad = _sample_check(root, kind, digest, flat_raw, reader)
                if bad == "sample_norm_mismatch":
                    counts[bad] += 1
                elif bad is not None:
                    fail(rel, bad)
    finally:
        reader.close()
    return counts, lists, failures


def verify_shard(root, kind, ab, pool=None, sample=DEFAULT_SAMPLE):
    """Verify one built shard and record the outcome in its manifest. Returns the verify record."""
    pack_root = pack_root_of(root)
    mpath = shard_manifest_path(pack_root, kind, ab)
    m = _load_json(mpath)
    if not m or not m.get("built"):
        return {"kind": kind, "ab": ab, "verified": False, "failures": [{"check": "not_built"}], "failure_count": 1}
    t0 = time.time()
    failures = []
    idx_path, dat_path = shard_paths(pack_root, kind, ab)
    idx_size, dat_size = _size(idx_path), _size(dat_path)
    if (idx_size, dat_size) != (m["idx_bytes"], m["dat_bytes"]):
        failures.append(
            {
                "check": "file_sizes",
                "detail": f"idx {idx_size} / dat {dat_size}, manifest {m['idx_bytes']} / {m['dat_bytes']}",
            }
        )
    records = None
    if idx_size is not None and dat_size is not None:
        records, problems = check_index(idx_path, dat_size, ab)
        failures += [{"check": "index", "detail": p} for p in problems]
    entries = m["entries"]
    if records is not None:
        in_index = {r[0] for r in records}
        in_manifest = {bytes.fromhex(os.path.basename(e[0])[:64]) for e in entries}
        if in_index != in_manifest:
            failures.append(
                {
                    "check": "index_vs_manifest",
                    "detail": f"{len(in_index - in_manifest)} indexed keys not in the manifest, "
                    f"{len(in_manifest - in_index)} manifest keys not indexed",
                }
            )

    rng = random.Random(f"{kind}/{ab}/{m['build_id']}")
    in_sample = set(rng.sample(range(len(entries)), min(sample, len(entries))))
    flagged = [(e, i in in_sample) for i, e in enumerate(entries)]
    tasks = [(root, kind, flagged[i : i + _CHUNK]) for i in range(0, len(flagged), _CHUNK)]
    counts = Counter()
    lists = {k: [] for k in _FLAT_LISTS}
    entry_failures = []
    for c, ls, fs in _imap(pool, _verify_chunk, tasks, chunksize=1):
        counts.update(c)
        for k, v in ls.items():
            lists[k].extend(v)
        entry_failures.extend(fs)

    names, _, _, _ = list_flat_shard(root, kind, ab)
    packed = {os.path.basename(e[0])[:64] for e in entries}
    skipped = {os.path.basename(s["relpath"])[:64] for s in m["skipped"] if s["reason"] != "misplaced"}
    new_flats = [_relpath(kind, h) for h in names if h not in packed and h not in skipped]
    skipped_still_flat = sum(1 for h in names if h in skipped)

    failure_count = len(failures) + len(entry_failures)
    record = {
        "build_id": m["build_id"],
        "verified": failure_count == 0,
        "verified_at_unix": time.time(),
        "host": socket.gethostname(),
        "verify_s": None,
        "idx_bytes": idx_size,
        "dat_bytes": dat_size,
        "entries_checked": len(entries),
        "identical_count": counts["identical"],
        "flat_changed_count": counts["flat_changed"],
        "flat_missing_count": counts["flat_missing"],
        "flat_unreadable_count": counts["flat_unreadable"],
        "new_flat_count": len(new_flats),
        "skipped_still_flat": skipped_still_flat,
        "sample_checked": counts["sample_checked"],
        "sample_norm_mismatch": counts["sample_norm_mismatch"],
        "failure_count": failure_count,
        "failure_checks": dict(Counter(f["check"] for f in failures + entry_failures)),
        "failures": (failures + sorted(entry_failures, key=lambda f: f["relpath"]))[:_MAX_LISTED_FAILURES],
        "flat_changed": sorted(lists["flat_changed"]),
        "flat_missing": sorted(lists["flat_missing"]),
        "flat_unreadable": sorted(lists["flat_unreadable"]),
        "new_flats": new_flats,
    }
    record["verify_s"] = round(time.time() - t0, 3)
    m["verify"] = record
    m["verified"] = record["verified"]
    _write_json_atomic(mpath, m)
    return {"kind": kind, "ab": ab, **record}


# ---- finalize / status / candidates / unpack -------------------------------
def finalize(root):
    """Write MANIFEST.json if every shard is built and verified at its current files.

    Returns (manifest or None, problems)."""
    pack_root = pack_root_of(root)
    out = os.path.join(pack_root, MANIFEST)
    if os.path.exists(out):
        return None, [f"{out} already exists"]
    problems, shards = [], {}
    totals = Counter()
    for kind, ab in all_shards():
        key = shard_key(kind, ab)
        m = load_shard_manifest(root, kind, ab)
        if m is None:
            problems.append(f"{key}: no shard manifest")
            continue
        v = m.get("verify") or {}
        if not m.get("built"):
            problems.append(f"{key}: not built")
            continue
        if not (m.get("verified") and v.get("verified")):
            why = f"{v['failure_count']} failures" if v else "verify never ran"
            problems.append(f"{key}: not verified ({why})")
            continue
        if v.get("build_id") != m.get("build_id"):
            problems.append(f"{key}: verify record is for another build")
            continue
        sizes = tuple(_size(p) for p in shard_paths(pack_root, kind, ab))
        if sizes != (m["idx_bytes"], m["dat_bytes"]) or sizes != (v["idx_bytes"], v["dat_bytes"]):
            problems.append(f"{key}: files are {sizes}, built {(m['idx_bytes'], m['dat_bytes'])}")
            continue
        shards[key] = {
            "idx_bytes": m["idx_bytes"],
            "dat_bytes": m["dat_bytes"],
            "entries": m["entries_count"],
            "raw_bytes": m["raw_bytes"],
            "stored_raw_count": m["stored_raw_count"],
            "build_id": m["build_id"],
        }
        for k in ("entries_count", "raw_bytes", "dat_bytes", "idx_bytes", "stored_raw_count", "skipped_count"):
            totals[f"{kind}_{k}"] += m[k]
            totals[k] += m[k]
    if problems:
        return None, problems
    now = time.time()
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "generation": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now)) + "-" + uuid.uuid4().hex[:8],
        "created_unix": now,
        "host": socket.gethostname(),
        "zlib_level": ZLIB_LEVEL,
        "store_raw_min_bytes": STORE_RAW_MIN,
        "totals": dict(totals),
        "shards": shards,
    }
    _write_json_atomic(out, manifest)
    return manifest, []


def status(root, shards):
    """Per-kind totals over the selected shards' manifests."""
    rows = {}
    for kind, ab in shards:
        r = rows.setdefault(kind, Counter())
        r["shards"] += 1
        m = load_shard_manifest(root, kind, ab)
        if not m or not m.get("built"):
            continue
        r["built"] += 1
        for k in ("entries_count", "raw_bytes", "dat_bytes", "idx_bytes", "stored_raw_count"):
            r[k] += m[k]
        for k in ("skipped_count", "tmp_count", "other_count"):
            r[k] += m[k]
        v = m.get("verify") or {}
        if m.get("verified") and v.get("build_id") == m["build_id"]:
            r["verified"] += 1
        elif v:
            r["verify_failed"] += 1
        for k in (
            "identical_count",
            "flat_changed_count",
            "flat_missing_count",
            "flat_unreadable_count",
            "new_flat_count",
        ):
            r[k] += v.get(k, 0) or 0
    return {kind: dict(r) for kind, r in rows.items()}


def candidates(root, shards, out):
    """Write the flat files verified byte-identical to their packed copy.

    One line per file: relpath, size, mtime_ns, sha256 (tab-separated). Only
    from verified shards; a deletion must still re-check each file against
    these values right before removing it. Returns (count, unverified shards)."""
    n, unverified = 0, []
    for kind, ab in shards:
        m = load_shard_manifest(root, kind, ab)
        v = (m or {}).get("verify") or {}
        if not (m and m.get("verified") and v.get("build_id") == m.get("build_id")):
            unverified.append(shard_key(kind, ab))
            continue
        excluded = set(v["flat_changed"]) | set(v["flat_missing"]) | set(v["flat_unreadable"])
        for rel, size, mtime_ns, sha in m["entries"]:
            if rel not in excluded:
                out.write(f"{rel}\t{size}\t{mtime_ns}\t{sha}\n")
                n += 1
    return n, unverified


def unpack_shard(root, kind, ab, dry_run=False):
    """Recreate the shard's missing flat files from the pack (rollback). Returns counts.

    Never replaces an existing flat file: each file is written to a temp name,
    stamped with the manifest mtime and published with os.link, which fails
    instead of overwriting when the name appeared meanwhile."""
    m = load_shard_manifest(root, kind, ab)
    if not m or not m.get("built"):
        raise RuntimeError(f"{shard_key(kind, ab)} is not built")
    reader = PackReader(pack_root_of(root))
    counts = Counter()
    try:
        for rel, size, mtime_ns, sha in m["entries"]:
            path = os.path.join(root, rel)
            if os.path.lexists(path):
                counts["present"] += 1
                continue
            raw = reader.get_bytes(kind, bytes.fromhex(os.path.basename(rel)[:64]))
            if raw is None or len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
                counts["pack_error"] += 1
                print(f"[unpack] {rel}: pack copy missing or does not match the manifest; left absent", flush=True)
                continue
            if dry_run:
                counts["would_restore"] += 1
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.unpack.tmp"
            with open(tmp, "wb") as f:
                f.write(raw)
            os.utime(tmp, ns=(mtime_ns, mtime_ns))
            try:
                os.link(tmp, path)
                counts["restored"] += 1
            except FileExistsError:
                counts["present"] += 1
            finally:
                os.unlink(tmp)
    finally:
        reader.close()
    return dict(counts)


# ---- CLI ------------------------------------------------------------------
def _selected(args):
    return take_part(parse_shards(args.shards), args.part)


def _cmd_build(args):
    if os.path.exists(os.path.join(pack_root_of(args.cache_root), MANIFEST)):
        print("refusing to build: the pack is finalized (MANIFEST.json exists) and live readers check its sizes")
        return 2
    shards = _selected(args)
    tmp_dir = args.tmp_dir or os.path.join(tempfile.gettempdir(), f"bcpack-{socket.gethostname()}-{os.getpid()}")
    built = skipped = 0
    with _pool(args.workers) as pool:
        for kind, ab in shards:
            if not args.force and is_built(args.cache_root, kind, ab):
                skipped += 1
                print(json.dumps({"kind": kind, "ab": ab, "status": "already_built"}), flush=True)
                continue
            m = build_shard(args.cache_root, kind, ab, tmp_dir, pool=pool)
            built += 1
            print(json.dumps(_brief(m)), flush=True)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    print(f"build: {built} shard(s) built, {skipped} already built, of {len(shards)} selected", flush=True)
    return 0


def _cmd_verify(args):
    shards = _selected(args)
    failed = []
    with _pool(args.workers) as pool:
        for kind, ab in shards:
            r = verify_shard(args.cache_root, kind, ab, pool=pool, sample=args.sample)
            if not r["verified"]:
                failed.append(shard_key(kind, ab))
            print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}), flush=True)
    print(f"verify: {len(shards) - len(failed)} of {len(shards)} shard(s) verified; failed: {failed}", flush=True)
    return 1 if failed else 0


def _cmd_finalize(args):
    manifest, problems = finalize(args.cache_root)
    if problems:
        print(f"finalize: NOT writing {MANIFEST}: {len(problems)} problem(s)")
        for p in problems[:50]:
            print(f"  {p}")
        return 1
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=1))
    return 0


def _cmd_status(args):
    print(json.dumps(status(args.cache_root, _selected(args)), indent=1))
    return 0


def _cmd_unpack(args):
    total = Counter()
    for kind, ab in _selected(args):
        c = unpack_shard(args.cache_root, kind, ab, dry_run=args.dry_run)
        total.update(c)
        print(json.dumps({"kind": kind, "ab": ab, **c}), flush=True)
    print(json.dumps({"unpack_total": dict(total), "dry_run": args.dry_run}), flush=True)
    return 1 if total["pack_error"] else 0


def _cmd_candidates(args):
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        n, unverified = candidates(args.cache_root, _selected(args), out)
    finally:
        if args.out:
            out.close()
    print(f"candidates: {n} flat file(s); unverified shards skipped: {unverified}", file=sys.stderr)
    return 1 if unverified else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, shards=True):
        p = sub.add_parser(name)
        p.add_argument("cache_root")
        if shards:
            p.add_argument("--shards", nargs="+", default=["all"])
            p.add_argument("--part", default=None, help="I/N: every N-th selected shard from the I-th on")
        p.set_defaults(fn=fn)
        return p

    p = add("build", _cmd_build)
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--tmp-dir", default=None, help="node-local staging dir (default: $TMPDIR/bcpack-<host>-<pid>)")
    p.add_argument("--force", action="store_true", help="rebuild shards that are already built")
    p = add("verify", _cmd_verify)
    p.add_argument("--workers", type=int, default=28)
    p.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE, help="entries per shard for the SearchCache-level check"
    )
    add("finalize", _cmd_finalize, shards=False)
    add("status", _cmd_status)
    p = add("unpack", _cmd_unpack)
    p.add_argument("--dry-run", action="store_true")
    p = add("candidates", _cmd_candidates)
    p.add_argument("--out", default=None, help="output file (default: stdout)")

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
