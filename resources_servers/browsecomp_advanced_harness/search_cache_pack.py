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
"""Read-only packed store behind the flat-file search/browse cache (search_cache.py).

Why: a long-lived cache root holds millions of one-entry files. That many small
files is expensive on lustre (metadata load, file-count quota, copies). The
pack holds the same entries in 512 shard file pairs. The flat files stay the
write path and are read FIRST; the pack only answers when the flat file does
not exist. A pack can therefore never shadow a newer flat entry, and removing
the pack (or never building one) leaves the cache exactly as it was.

Layout (under <root>/pack/, built by build_cache_pack.py):
    <kind>/<ab>.idx               index of shard <kind>/<ab>
    <kind>/<ab>.dat               entry blobs of shard <kind>/<ab>
    manifests/<kind>_<ab>.json    per-shard build + verify record
    MANIFEST.json                 written last, once every shard is built and
                                  verified; its presence is what turns the
                                  pack on (see open_pack)

<kind> is "search" or "browse"; <ab> is the first byte of the entry's sha256
key in lowercase hex, the same <ab> as the flat path <kind>/<ab>/<cd>/<hex>.json.
So a shard holds exactly the flat files of one <kind>/<ab>/ directory.

.idx, version 1, all integers little-endian:
    header   8s   magic b"BCPACKIX"
             u32  version (1)
             u32  record size (48)
             u64  record count N
    fan-out  257 x u32: fan[b] = number of records whose digest[1] < b,
             so fan[0] = 0 and fan[256] = N
    records  N x 48 bytes, strictly ascending by digest:
             32s  sha256 digest of the normalized key (the flat file name)
             u64  offset of the blob in .dat
             u32  blob length; top bit (STORED) set = blob is stored raw
             u32  raw length (size of the original flat file)

.dat is the blobs back to back, in no particular order, nothing in between. A
blob is the entry's ORIGINAL flat-file bytes: zlib-compressed (level 6, zlib
container) when the file is smaller than STORE_RAW_MIN (8 MiB), stored as-is
otherwise. Inflating a giant entry costs ~3 ms/MB on the get path, and the
handful of entries above 8 MiB are a small share of the bytes.

Lookup: shard by digest[0]; the fan-out bounds the records sharing digest[1]
(~180 in a search shard); one os.pread of that block, binary search in memory,
one os.pread of the blob from .dat, inflate unless stored. Everything is
pread on long-lived fds: no shared file position, no mmap. An mmap'd index
page-faults with the GIL held, which stalled every other thread in the server
under load; pread releases the GIL for the I/O.

Any read or corruption error is a miss: counted in PackReader.errors (the
cache's pack_errors stat) and logged loudly for the first few, never raised
into a rollout.
"""

import json
import os
import struct
import threading
import zlib


PACK_DIR = "pack"
MANIFEST = "MANIFEST.json"
SHARD_MANIFEST_DIR = "manifests"
KINDS = ("search", "browse")
FORMAT = "bcpack"
VERSION = 1

MAGIC = b"BCPACKIX"
HEADER = struct.Struct("<8sIIQ")
FANOUT = struct.Struct("<257I")
RECORD = struct.Struct("<32sQII")
INDEX_HEAD_SIZE = HEADER.size + FANOUT.size
REC_SIZE = RECORD.size

STORED = 0x80000000
LEN_MASK = 0x7FFFFFFF
STORE_RAW_MIN = 8 << 20
ZLIB_LEVEL = 6

_LOGGED_ERRORS = 20


class PackManifestError(RuntimeError):
    """MANIFEST.json is present but the pack it describes is not there intact."""


def all_shards():
    """Every (kind, ab) shard, in build order: all search shards, then all browse."""
    return [(kind, f"{b:02x}") for kind in KINDS for b in range(256)]


def shard_key(kind, ab):
    return f"{kind}/{ab}"


def pack_root_of(root):
    return os.path.join(root, PACK_DIR)


def shard_paths(pack_root, kind, ab):
    """(<pack_root>/<kind>/<ab>.idx, <pack_root>/<kind>/<ab>.dat)."""
    base = os.path.join(pack_root, kind, ab)
    return base + ".idx", base + ".dat"


def shard_manifest_path(pack_root, kind, ab):
    return os.path.join(pack_root, SHARD_MANIFEST_DIR, f"{kind}_{ab}.json")


def encode_blob(raw):
    """(blob, length field) for one entry's raw bytes, as the .dat stores it."""
    if len(raw) > LEN_MASK:
        raise ValueError(f"entry of {len(raw)} bytes exceeds the {LEN_MASK}-byte record limit")
    if len(raw) >= STORE_RAW_MIN:
        return raw, len(raw) | STORED
    blob = zlib.compress(raw, ZLIB_LEVEL)
    return blob, len(blob)


def write_index(f, records):
    """Write a v1 .idx to the binary file f.

    records: (digest, offset, length field, raw length) tuples, strictly
    ascending by digest, all sharing digest[0]."""
    fan = [0] * 257
    prev = None
    for digest, _, _, _ in records:
        if prev is not None and digest <= prev:
            raise ValueError("index records must be strictly ascending by digest")
        if prev is not None and digest[0] != prev[0]:
            raise ValueError("index records must share the first digest byte")
        prev = digest
        fan[digest[1] + 1] += 1
    for b in range(256):
        fan[b + 1] += fan[b]
    f.write(HEADER.pack(MAGIC, VERSION, REC_SIZE, len(records)))
    f.write(FANOUT.pack(*fan))
    for rec in records:
        f.write(RECORD.pack(*rec))


def read_index_head(head, idx_size, idx_path):
    """(count, fan) from the first INDEX_HEAD_SIZE bytes of an .idx; ValueError if malformed."""
    if len(head) < INDEX_HEAD_SIZE:
        raise ValueError(f"truncated pack index header in {idx_path}")
    magic, version, rec_size, count = HEADER.unpack_from(head, 0)
    if magic != MAGIC or version != VERSION or rec_size != REC_SIZE:
        raise ValueError(f"bad pack index header in {idx_path}: {magic!r} v{version} rec{rec_size}")
    if idx_size != INDEX_HEAD_SIZE + count * REC_SIZE:
        raise ValueError(f"pack index {idx_path} is {idx_size} bytes, header says {count} records")
    fan = FANOUT.unpack_from(head, HEADER.size)
    if fan[0] != 0 or fan[256] != count or any(fan[b] > fan[b + 1] for b in range(256)):
        raise ValueError(f"inconsistent fan-out table in {idx_path}")
    return count, fan


def pread_exact(fd, n, offset):
    """os.pread that returns exactly n bytes or raises OSError (EOF before n)."""
    buf = os.pread(fd, n, offset)
    if len(buf) == n:
        return buf
    parts, got = [buf], len(buf)
    while got < n:
        chunk = os.pread(fd, n - got, offset + got)
        if not chunk:
            raise OSError(f"short read: {got} of {n} bytes at offset {offset}")
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


class _Shard:
    __slots__ = ("idx_fd", "dat_fd", "fan", "dat_size")

    def __init__(self, idx_path, dat_path):
        self.idx_fd = os.open(idx_path, os.O_RDONLY | os.O_CLOEXEC)
        self.dat_fd = -1
        try:
            idx_size = os.fstat(self.idx_fd).st_size
            head = os.pread(self.idx_fd, INDEX_HEAD_SIZE, 0)
            _, self.fan = read_index_head(head, idx_size, idx_path)
            self.dat_fd = os.open(dat_path, os.O_RDONLY | os.O_CLOEXEC)
            self.dat_size = os.fstat(self.dat_fd).st_size
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in (self.idx_fd, self.dat_fd):
            if fd >= 0:
                os.close(fd)
        self.idx_fd = self.dat_fd = -1


class PackReader:
    """Thread-safe reader over <pack_root>/<kind>/<ab>.{idx,dat}.

    A shard opens lazily on first use, under its own lock, and keeps its two
    fds for the life of the reader; there is no reader-wide lock on the get
    path. shards: the "kind/ab" keys the manifest lists (None = all 512); an
    unlisted shard is a clean miss, never an error.
    """

    def __init__(self, pack_root, shards=None, generation=None):
        self.pack_root = pack_root
        self.generation = generation
        listed = set(shards) if shards is not None else None
        self._locks = {
            (kind, ab): threading.Lock()
            for kind, ab in all_shards()
            if listed is None or shard_key(kind, ab) in listed
        }
        self._shards = {}
        self._err_lock = threading.Lock()
        self.errors = 0

    def _shard(self, kind, ab):
        key = (kind, ab)
        shard = self._shards.get(key)
        if shard is not None:
            return shard
        lock = self._locks.get(key)
        if lock is None:
            return None
        with lock:
            shard = self._shards.get(key)
            if shard is None:
                # An open that fails is not cached: the next get retries, so a
                # transient lustre error does not turn a shard off for good.
                shard = _Shard(*shard_paths(self.pack_root, kind, ab))
                self._shards[key] = shard
        return shard

    def _locate(self, kind, digest):
        """(shard, offset, blob length, raw length, stored) or None; raises on I/O or format errors."""
        shard = self._shard(kind, f"{digest[0]:02x}")
        if shard is None:
            return None
        lo, hi = shard.fan[digest[1]], shard.fan[digest[1] + 1]
        if lo >= hi:
            return None
        block = pread_exact(shard.idx_fd, (hi - lo) * REC_SIZE, INDEX_HEAD_SIZE + lo * REC_SIZE)
        a, z = 0, hi - lo
        while a < z:
            mid = (a + z) >> 1
            pos = mid * REC_SIZE
            key = block[pos : pos + 32]
            if key < digest:
                a = mid + 1
            elif key > digest:
                z = mid
            else:
                _, offset, length, raw_len = RECORD.unpack_from(block, pos)
                return shard, offset, length & LEN_MASK, raw_len, bool(length & STORED)
        return None

    def lookup(self, kind, digest):
        """(offset, blob length, raw length, stored) for digest, or None if absent; raises on errors."""
        loc = self._locate(kind, digest)
        return None if loc is None else loc[1:]

    def get_bytes(self, kind, digest):
        """The entry's original flat-file bytes, or None (absent, or any error -> counted miss)."""
        try:
            loc = self._locate(kind, digest)
            if loc is None:
                return None
            shard, offset, length, raw_len, stored = loc
            if offset + length > shard.dat_size or (stored and length != raw_len):
                raise ValueError(f"index record out of range: offset={offset} len={length} raw={raw_len}")
            buf = pread_exact(shard.dat_fd, length, offset)
            if stored:
                return buf
            # bufsize=raw_len: one exactly-sized output buffer instead of
            # repeated growth; the compressed buffer is dropped before return.
            raw = zlib.decompress(buf, 15, raw_len)
            del buf
            if len(raw) != raw_len:
                raise ValueError(f"inflated to {len(raw)} bytes, index says {raw_len}")
            return raw
        except Exception as e:  # a cache read must never fail the rollout
            self.note_error("read", kind, digest, e)
            return None

    def load(self, kind, digest):
        """The parsed entry (json.loads of its bytes), or None like get_bytes.

        A packed entry that does not parse is corruption (the builder only packs
        entries that parse), so it counts as an error, not just a miss."""
        raw = self.get_bytes(kind, digest)
        if raw is None:
            return None
        try:
            text = raw.decode("utf-8")
            del raw  # free the bytes before the parse allocates the objects
            return json.loads(text)
        except ValueError as e:
            self.note_error("parse", kind, digest, e)
            return None

    def note_error(self, what, kind, digest, exc):
        with self._err_lock:
            self.errors += 1
            n = self.errors
        if n <= _LOGGED_ERRORS:
            print(
                f"[cache][pack][ERROR] {what} failed for {kind}/{digest.hex()} in {self.pack_root}: {exc!r}"
                f" -- served as a miss (pack error #{n})",
                flush=True,
            )
            if n == _LOGGED_ERRORS:
                print("[cache][pack][ERROR] further pack errors are counted in pack_errors, not logged", flush=True)

    def close(self):
        for key in list(self._shards):
            self._shards.pop(key).close()


def load_manifest(root):
    """The parsed <root>/pack/MANIFEST.json, or None if absent; PackManifestError if unreadable."""
    path = os.path.join(pack_root_of(root), MANIFEST)
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise PackManifestError(f"search cache pack manifest {path} is unreadable: {e!r}") from e
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise PackManifestError(
            f"search cache pack manifest {path} is {manifest.get('format')!r} v{manifest.get('version')!r}, "
            f"this reader understands {FORMAT!r} v{VERSION}"
        )
    return manifest


def open_pack(root):
    """PackReader for <root>/pack, or None when the pack is not turned on.

    The pack is on iff <root>/pack/MANIFEST.json exists. When it does, every
    shard it lists must have its .idx and .dat at the recorded sizes, else
    PackManifestError: a missing shard would otherwise be silent misses, each
    one a paid provider call.
    """
    manifest = load_manifest(root)
    if manifest is None:
        return None
    pack_root = pack_root_of(root)
    shards = manifest.get("shards") or {}
    problems = []
    for key, info in sorted(shards.items()):
        kind, ab = key.split("/")
        for path, want in zip(shard_paths(pack_root, kind, ab), (info["idx_bytes"], info["dat_bytes"])):
            try:
                size = os.stat(path).st_size
            except OSError as e:
                problems.append(f"{os.path.relpath(path, pack_root)}: {e.strerror or e!r}")
                continue
            if size != want:
                problems.append(f"{os.path.relpath(path, pack_root)}: {size} bytes, manifest says {want}")
    if problems:
        shown = "; ".join(problems[:10]) + (f"; ... {len(problems) - 10} more" if len(problems) > 10 else "")
        raise PackManifestError(
            f"search cache pack {pack_root} (generation {manifest.get('generation')}) does not match its "
            f"MANIFEST.json: {len(problems)} problem(s): {shown}. Restore the shard files, or move "
            f"MANIFEST.json aside to serve from the flat files alone."
        )
    return PackReader(pack_root, shards=shards.keys(), generation=manifest.get("generation"))
