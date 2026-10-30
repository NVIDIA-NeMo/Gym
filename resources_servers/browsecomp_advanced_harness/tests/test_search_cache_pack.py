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
"""Tests for the read-only cache pack (search_cache_pack.py, build_cache_pack.py).

Two layers:
1. SearchCache with a pack -- flat file first, pack only when the flat file is
   absent; small/compressed and large/stored entries round-trip; a corrupt
   blob is a counted miss; no pack (or an unfinalized one) changes nothing; a
   manifest whose shards are missing fails at construction; read-only mode,
   the prefer-exa rule and fuzzy all see packed entries; many threads at once.
2. The offline tool -- what build packs and what it skips and lists, resume,
   verify outcomes (changed/new flats counted, corruption failed), finalize
   gating, index structure checks, unpack (rollback) and deletion candidates.
"""

import hashlib
import io
import json
import os
import random
import shutil
import threading

import pytest

from resources_servers.browsecomp_advanced_harness import build_cache_pack as bcp
from resources_servers.browsecomp_advanced_harness import search_cache_pack as scp
from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache, normalize_query, normalize_url


def _digest(norm):
    return hashlib.sha256(norm.encode("utf-8")).digest()


def _flat(root, kind, norm):
    h = _digest(norm).hex()
    return os.path.join(root, kind, h[:2], h[2:4], h + ".json")


def _make_pack(root, stage):
    """Build, verify and finalize a pack of everything under root, in-process."""
    for kind, ab in scp.all_shards():
        bcp.build_shard(root, kind, ab, stage)
    for kind, ab in scp.all_shards():
        r = bcp.verify_shard(root, kind, ab)
        assert r["verified"], r["failures"]
    manifest, problems = bcp.finalize(root)
    assert problems == []
    return manifest


def _drop_flats(root):
    for kind in scp.KINDS:
        shutil.rmtree(os.path.join(root, kind), ignore_errors=True)


def _tree(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        out += [os.path.join(dirpath, n) for n in dirnames + filenames]
    return sorted(out)


SMALL = [{"title": "T", "url": "https://x.com/a", "content": "snippet", "raw_content": "RAW ONE"}]
COMPRESSIBLE = [
    {"title": f"T{i}", "url": f"https://x.com/{i}", "raw_content": "lorem ipsum " * 2000} for i in range(20)
]
LARGE = [{"title": "huge", "url": "https://x.com/huge", "raw_content": "z" * (scp.STORE_RAW_MIN + (1 << 20))}]


@pytest.fixture
def stage(tmp_path):
    return str(tmp_path / "stage")


@pytest.fixture
def packed_root(tmp_path, stage):
    """A cache root with search + browse entries, packed and finalized; flat files still present."""
    root = str(tmp_path / "cache")
    w = SearchCache(root)
    w.put_search("small query", SMALL, provider="tavily")
    w.put_search("compressible query", COMPRESSIBLE, provider="exa")
    w.put_search("large query", LARGE, provider="exa")
    w.put_extract("https://x.com/page", "PAGE BODY", provider="tavily")
    _make_pack(root, stage)
    return root


class TestPackServing:
    def test_roundtrip_small_compressed_and_large_stored(self, packed_root) -> None:
        originals = {
            q: open(_flat(packed_root, "search", normalize_query(q)), "rb").read()
            for q in ("small query", "compressible query", "large query")
        }
        _drop_flats(packed_root)
        cache = SearchCache(packed_root)
        assert cache.get_search("small query") == ("tavily", SMALL)
        assert cache.get_search("compressible query") == ("exa", COMPRESSIBLE)
        assert cache.get_search("large query") == ("exa", LARGE)
        assert cache.get_extract("https://x.com/page/") == ("tavily", "PAGE BODY")
        # byte-exact, and stored iff >= 8 MiB
        for q, raw in originals.items():
            d = _digest(normalize_query(q))
            assert cache._pack.get_bytes("search", d) == raw
            _, length, raw_len, stored = cache._pack.lookup("search", d)
            assert raw_len == len(raw)
            assert stored == (len(raw) >= scp.STORE_RAW_MIN)
            if not stored:
                assert length < raw_len  # really compressed
        assert cache._pack.lookup("search", _digest(normalize_query("large query")))[3] is True
        s = cache.stats()
        assert s["search_pack_hits"] == 3 and s["browse_pack_hits"] == 1 and s["pack_errors"] == 0

    def test_flat_first_when_both_exist(self, packed_root) -> None:
        # A newer flat entry for a packed key (written after the pack was built)
        SearchCache(packed_root).put_search("small query", [{"r": "newer"}], provider="exa")
        cache = SearchCache(packed_root)
        assert cache.get_search("small query") == ("exa", [{"r": "newer"}])
        assert cache.stats()["search_pack_hits"] == 0

    def test_pack_hit_only_when_flat_absent(self, packed_root) -> None:
        os.remove(_flat(packed_root, "search", "small query"))
        cache = SearchCache(packed_root)
        assert cache.get_search("small query") == ("tavily", SMALL)  # pack
        assert cache.get_search("compressible query") == ("exa", COMPRESSIBLE)  # flat
        s = cache.stats()
        assert s["search_hits"] == 2 and s["search_pack_hits"] == 1

    def test_corrupt_flat_is_not_masked_by_pack(self, packed_root) -> None:
        with open(_flat(packed_root, "search", "small query"), "w") as f:
            f.write("{not json")
        cache = SearchCache(packed_root)
        assert cache.get_search("small query") is None
        assert cache.stats()["search_pack_hits"] == 0

    def test_never_packed_key_is_plain_miss(self, packed_root) -> None:
        cache = SearchCache(packed_root)
        assert cache.get_search("never cached") is None
        assert cache.get_extract("https://never.example") is None
        s = cache.stats()
        assert s["search_misses"] == 1 and s["browse_misses"] == 1 and s["pack_errors"] == 0


class TestPackCorruption:
    def _corrupt(self, root, norm, at, data):
        cache = SearchCache(root)
        offset, length, _, stored = cache._pack.lookup("search", _digest(norm))
        dat = scp.shard_paths(scp.pack_root_of(root), "search", _digest(norm)[:1].hex())[1]
        with open(dat, "r+b") as f:
            f.seek(offset + (at if at >= 0 else length + at))
            f.write(data)
        return stored

    def test_corrupt_compressed_blob_is_counted_miss(self, packed_root, capsys) -> None:
        assert self._corrupt(packed_root, "compressible query", 100, b"\x00\xff\x00\xff") is False
        _drop_flats(packed_root)
        cache = SearchCache(packed_root)
        assert cache.get_search("compressible query") is None
        assert cache.get_search("small query") == ("tavily", SMALL)  # other entries unaffected
        s = cache.stats()
        assert s["pack_errors"] == 1 and s["search_misses"] == 1 and s["search_pack_hits"] == 1
        assert "[cache][pack][ERROR]" in capsys.readouterr().out

    def test_corrupt_stored_blob_is_counted_miss(self, packed_root) -> None:
        assert self._corrupt(packed_root, "large query", 0, b"\x00") is True
        _drop_flats(packed_root)
        cache = SearchCache(packed_root)
        assert cache.get_search("large query") is None
        assert cache.stats()["pack_errors"] == 1

    def test_truncated_dat_is_counted_miss(self, packed_root) -> None:
        cache = SearchCache(packed_root)
        _drop_flats(packed_root)
        d = _digest("small query")
        dat = scp.shard_paths(scp.pack_root_of(packed_root), "search", d[:1].hex())[1]
        cache._pack._shard("search", d[:1].hex())  # open before the file shrinks
        os.truncate(dat, 0)
        assert cache.get_search("small query") is None
        assert cache.stats()["pack_errors"] == 1


class TestPackEnabling:
    def test_no_pack_dir_is_todays_behaviour(self, tmp_path) -> None:
        root = str(tmp_path / "cache")
        cache = SearchCache(root)
        assert cache._pack is None and cache.pack_generation is None
        cache.put_search("q", SMALL, provider="tavily")
        assert cache.get_search("q") == ("tavily", SMALL)
        assert "pack" not in cache.summary_str()
        assert not os.path.exists(os.path.join(root, "pack"))

    def test_unfinalized_pack_is_ignored(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        SearchCache(root).put_search("q", SMALL, provider="tavily")
        for kind, ab in scp.all_shards():
            bcp.build_shard(root, kind, ab, stage)
        _drop_flats(root)
        cache = SearchCache(root)
        assert cache._pack is None
        assert cache.get_search("q") is None

    def test_generation_is_reported(self, packed_root) -> None:
        manifest = scp.load_manifest(packed_root)
        assert SearchCache(packed_root).pack_generation == manifest["generation"]

    def test_manifest_with_missing_shard_fails_init(self, packed_root) -> None:
        idx, dat = scp.shard_paths(scp.pack_root_of(packed_root), "browse", "7f")
        os.rename(dat, dat + ".moved")
        with pytest.raises(scp.PackManifestError, match=r"browse/7f\.dat"):
            SearchCache(packed_root)

    def test_manifest_with_wrong_size_fails_init(self, packed_root) -> None:
        idx, _ = scp.shard_paths(scp.pack_root_of(packed_root), "search", "00")
        with open(idx, "ab") as f:
            f.write(b"x")
        with pytest.raises(scp.PackManifestError, match=r"search/00\.idx"):
            SearchCache(packed_root, read_only=True)

    def test_unreadable_manifest_fails_init(self, packed_root) -> None:
        with open(os.path.join(scp.pack_root_of(packed_root), scp.MANIFEST), "w") as f:
            f.write("{")
        with pytest.raises(scp.PackManifestError):
            SearchCache(packed_root)

    def test_disabled_cache_ignores_pack(self) -> None:
        cache = SearchCache(None)
        assert cache._pack is None and cache.stats()["pack_errors"] == 0


class TestPackWithCacheFeatures:
    def test_read_only_serves_pack_and_writes_nothing(self, packed_root) -> None:
        _drop_flats(packed_root)
        before = _tree(packed_root)
        cache = SearchCache(packed_root, read_only=True)
        assert cache.get_search("small query") == ("tavily", SMALL)
        assert cache.get_extract("https://x.com/page") == ("tavily", "PAGE BODY")
        cache.put_search("brand new", SMALL, provider="exa")
        cache.put_extract("https://x.com/new", "BODY", provider="exa")
        assert _tree(packed_root) == before

    def test_tavily_put_blocked_by_packed_exa_entry(self, packed_root) -> None:
        _drop_flats(packed_root)
        cache = SearchCache(packed_root)
        cache.put_search("compressible query", [{"src": "tavily"}], provider="tavily")
        assert not os.path.exists(_flat(packed_root, "search", "compressible query"))
        assert cache.get_search("compressible query") == ("exa", COMPRESSIBLE)
        # the prefer-exa check read the pack but is not a served hit
        assert cache.stats()["search_pack_hits"] == 1
        # an exa refresh is still written, and the flat copy now wins
        cache.put_search("compressible query", [{"src": "exa2"}], provider="exa")
        assert cache.get_search("compressible query") == ("exa", [{"src": "exa2"}])

    def test_tavily_put_over_packed_tavily_entry_is_written(self, packed_root) -> None:
        _drop_flats(packed_root)
        cache = SearchCache(packed_root)
        cache.put_search("small query", [{"src": "tavily2"}], provider="tavily")
        assert cache.get_search("small query") == ("tavily", [{"src": "tavily2"}])

    def test_fuzzy_hit_served_from_pack(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        SearchCache(root, fuzzy_threshold=0.8).put_search("countries elected 68th session CSW 2024", SMALL, "tavily")
        _make_pack(root, stage)
        _drop_flats(root)  # index/ (the fuzzy norms) stays
        cache = SearchCache(root, fuzzy_threshold=0.8)
        assert cache.get_search("68th session CSW 2024 countries elected") == ("tavily", SMALL)
        s = cache.stats()
        assert s["search_fuzzy_hits"] == 1 and s["search_pack_hits"] == 1 and s["search_hits"] == 1

    def test_concurrent_reads(self, tmp_path, stage, monkeypatch) -> None:
        root = str(tmp_path / "cache")
        w = SearchCache(root)
        expected = {}
        for i in range(300):
            q = f"query number {i}"
            results = [{"i": i, "raw_content": "x" * random.Random(i).randint(10, 50_000)}]
            w.put_search(q, results, provider="exa" if i % 2 else "tavily")
            expected[q] = ("exa" if i % 2 else "tavily", results)
        _make_pack(root, stage)
        _drop_flats(root)

        opened = []

        class CountingShard(scp._Shard):
            def __init__(self, idx_path, dat_path):
                opened.append(idx_path)
                super().__init__(idx_path, dat_path)

        monkeypatch.setattr(scp, "_Shard", CountingShard)
        cache = SearchCache(root)
        queries = list(expected) + [f"missing {i}" for i in range(50)]
        barrier = threading.Barrier(32)
        wrong = []

        def worker(seed):
            rng = random.Random(seed)
            barrier.wait()  # every thread hits the lazy shard opens at once
            for _ in range(300):
                q = rng.choice(queries)
                if cache.get_search(q) != expected.get(q):
                    wrong.append(q)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert wrong == []
        s = cache.stats()
        assert s["pack_errors"] == 0
        assert s["search_hits"] + s["search_misses"] == 32 * 300
        assert s["search_pack_hits"] == s["search_hits"]
        assert len(opened) == len(set(opened))  # each shard opened once

    def test_stats_keys(self, packed_root, tmp_path) -> None:
        plain = SearchCache(str(tmp_path / "plain"))
        old_keys = {
            "search_hits",
            "search_misses",
            "search_fuzzy_hits",
            "browse_hits",
            "browse_misses",
            "search_get_s",
            "search_get_max_s",
            "search_get_slow",
            "browse_get_s",
            "browse_get_max_s",
            "browse_get_slow",
        }
        new_keys = {"search_pack_hits", "browse_pack_hits", "pack_errors"}
        assert set(plain.stats()) == old_keys | new_keys
        assert all(plain.stats()[k] == 0 for k in new_keys)
        assert new_keys <= set(plain.stats_report())

        _drop_flats(packed_root)
        cache = SearchCache(packed_root)
        cache.get_search("small query")
        cache.get_extract("https://x.com/page")
        r = cache.stats_report()
        assert (r["search_pack_hits"], r["browse_pack_hits"], r["pack_errors"]) == (1, 1, 0)
        assert "pack hits search 1, browse 1, errors 0" in cache.summary_str()


class TestBuild:
    def test_skips_unparseable_and_lists_tmp_and_partials(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        SearchCache(root).put_search("good", SMALL, provider="tavily")

        def place(norm, data, name=None):
            path = _flat(root, "search", norm)
            if name is not None:
                path = os.path.join(os.path.dirname(path), name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return os.path.relpath(path, root)

        bad_json = place("bad json", b"{not json")
        bad_utf8 = place("bad utf8", b'{"norm": "\xff"}')
        h = _digest("partial").hex()
        tmp_file = place("partial", b"{}", name=f"{h}.json.123.456.tmp")
        partial = place("partial", b"", name=f".{h}.json.Ab12Cd")
        misplaced = place("misplaced", b"{}", name=_digest("elsewhere").hex() + ".json")

        for kind, ab in scp.all_shards():
            bcp.build_shard(root, kind, ab, stage)
        manifests = [bcp.load_shard_manifest(root, k, ab) for k, ab in scp.all_shards()]
        skipped = {s["relpath"]: s["reason"] for m in manifests for s in m["skipped"]}
        assert skipped == {bad_json: "json_error", bad_utf8: "utf8_error", misplaced: "misplaced"}
        assert [f for m in manifests for f in m["tmp_files"]] == [tmp_file]
        assert [f for m in manifests for f in m["other_files"]] == [partial]
        assert sum(m["entries_count"] for m in manifests) == 1
        good = bcp.load_shard_manifest(root, "search", _digest("good")[:1].hex())
        rel, size, mtime_ns, sha = good["entries"][0]
        st = os.stat(os.path.join(root, rel))
        raw = open(os.path.join(root, rel), "rb").read()
        assert (size, mtime_ns, sha) == (st.st_size, st.st_mtime_ns, hashlib.sha256(raw).hexdigest())
        assert good["entry_fields"] == ["relpath", "size", "mtime_ns", "sha256"]

        # skipped files stay flat-only: today's miss, never served from the pack
        for kind, ab in scp.all_shards():
            assert bcp.verify_shard(root, kind, ab)["verified"]
        assert bcp.finalize(root)[1] == []
        cache = SearchCache(root)
        assert cache._pack.lookup("search", _digest("bad json")) is None
        assert cache.get_search("bad json") is None
        assert cache.get_search("good") == ("tavily", SMALL)

    def test_cli_build_is_resumable_and_uses_workers(self, tmp_path, stage, capsys) -> None:
        root = str(tmp_path / "cache")
        w = SearchCache(root)
        for i in range(40):
            w.put_search(f"q{i}", SMALL, provider="tavily")
        assert (
            bcp.main(["build", root, "--shards", "search:all", "--part", "0/2", "--workers", "2", "--tmp-dir", stage])
            == 0
        )
        first = {
            ab: bcp.load_shard_manifest(root, "search", ab)["build_id"]
            for _, ab in bcp.parse_shards(["search:all"])[0::2]
        }
        assert bcp.main(["build", root, "--workers", "2", "--tmp-dir", stage]) == 0
        out = capsys.readouterr().out
        assert out.count('"already_built"') == 128
        for ab, build_id in first.items():  # not rebuilt
            assert bcp.load_shard_manifest(root, "search", ab)["build_id"] == build_id
        assert bcp.main(["verify", root, "--workers", "2"]) == 0
        assert bcp.main(["finalize", root]) == 0
        assert bcp.main(["build", root, "--shards", "search:00"]) == 2  # finalized: refuses
        _drop_flats(root)
        cache = SearchCache(root)
        assert all(cache.get_search(f"q{i}") == ("tavily", SMALL) for i in range(40))

    def test_force_rebuild_retires_verified_manifest(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        SearchCache(root).put_search("q", SMALL, provider="tavily")
        ab = _digest("q")[:1].hex()
        bcp.build_shard(root, "search", ab, stage)
        assert bcp.verify_shard(root, "search", ab)["verified"]
        old = bcp.load_shard_manifest(root, "search", ab)
        bcp.build_shard(root, "search", ab, stage)
        new = bcp.load_shard_manifest(root, "search", ab)
        assert new["build_id"] != old["build_id"] and new["verified"] is False
        assert os.path.exists(scp.shard_manifest_path(scp.pack_root_of(root), "search", ab) + ".replaced")

    def test_parse_shards(self) -> None:
        assert bcp.parse_shards(["search:00-03"]) == [
            ("search", "00"),
            ("search", "01"),
            ("search", "02"),
            ("search", "03"),
        ]
        assert len(bcp.parse_shards(["browse:all"])) == 256
        assert len(bcp.parse_shards(["all"])) == 512
        assert bcp.parse_shards(["browse:7f,a3", "search:ff"]) == [
            ("search", "ff"),
            ("browse", "7f"),
            ("browse", "a3"),
        ]
        parts = [bcp.take_part(bcp.parse_shards(["all"]), f"{i}/4") for i in range(4)]
        assert sorted(s for p in parts for s in p) == sorted(bcp.parse_shards(["all"]))
        for bad in (["search:0g"], ["web:00"], ["search:"], ["search:000"]):
            with pytest.raises(ValueError):
                bcp.parse_shards(bad)


class TestVerifyAndFinalize:
    def test_changed_new_and_missing_flats_are_counted_not_failed(self, packed_root) -> None:
        w = SearchCache(packed_root)
        w.put_search("small query", [{"r": "changed"}], provider="exa")
        w.put_search("a new query", SMALL, provider="tavily")
        os.remove(_flat(packed_root, "search", "compressible query"))
        totals = {}
        for kind, ab in scp.all_shards():
            r = bcp.verify_shard(packed_root, kind, ab)
            assert r["verified"], r["failures"]
            for k in ("identical_count", "flat_changed_count", "flat_missing_count", "new_flat_count"):
                totals[k] = totals.get(k, 0) + r[k]
        assert totals == {"identical_count": 2, "flat_changed_count": 1, "flat_missing_count": 1, "new_flat_count": 1}
        out = io.StringIO()
        n, unverified = bcp.candidates(packed_root, scp.all_shards(), out)
        assert n == 2 and unverified == []
        listed = {line.split("\t")[0] for line in out.getvalue().splitlines()}
        assert listed == {
            os.path.relpath(_flat(packed_root, "search", "large query"), packed_root),
            os.path.relpath(_flat(packed_root, "browse", normalize_url("https://x.com/page")), packed_root),
        }

    def test_corrupt_dat_fails_verify_and_blocks_finalize(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        SearchCache(root).put_search("q", COMPRESSIBLE, provider="tavily")
        for kind, ab in scp.all_shards():
            bcp.build_shard(root, kind, ab, stage)
        ab = _digest("q")[:1].hex()
        dat = scp.shard_paths(scp.pack_root_of(root), "search", ab)[1]
        with open(dat, "r+b") as f:
            f.seek(50)
            f.write(b"\x00\x00\x00\x00")
        for kind, ab2 in scp.all_shards():
            r = bcp.verify_shard(root, kind, ab2)
            assert r["verified"] == (ab2 != ab or kind != "search")
        assert bcp.load_shard_manifest(root, "search", ab)["verify"]["failure_checks"] == {"pack_read_error": 1}
        manifest, problems = bcp.finalize(root)
        assert manifest is None and problems == [f"search/{ab}: not verified (1 failures)"]
        assert not os.path.exists(os.path.join(scp.pack_root_of(root), scp.MANIFEST))

    def test_flat_bytes_mismatch_with_same_size_and_mtime_fails(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        SearchCache(root).put_search("q", SMALL, provider="tavily")
        path = _flat(root, "search", "q")
        st = os.stat(path)
        bcp.build_shard(root, "search", _digest("q")[:1].hex(), stage)
        raw = open(path, "rb").read()
        with open(path, "wb") as f:
            f.write(raw.replace(b"RAW ONE", b"RAW TWO"))
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        r = bcp.verify_shard(root, "search", _digest("q")[:1].hex())
        assert not r["verified"] and r["failure_checks"] == {"flat_bytes_mismatch": 1}

    def test_check_index_detects_bad_structure(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        w = SearchCache(root)
        qs = [f"q{i}" for i in range(2000)]
        for q in qs:
            w.put_search(q, SMALL, provider="tavily")
        bcp.build_shard(root, "search", "00", stage)
        idx, dat = scp.shard_paths(scp.pack_root_of(root), "search", "00")
        records, problems = bcp.check_index(idx, os.path.getsize(dat), "00")
        assert problems == [] and len(records) >= 2
        assert bcp.check_index(idx, os.path.getsize(dat) + 1, "00")[1]  # extents no longer tile the .dat
        data = bytearray(open(idx, "rb").read())
        a, b = scp.INDEX_HEAD_SIZE, scp.INDEX_HEAD_SIZE + scp.REC_SIZE
        data[a:b], data[b : b + scp.REC_SIZE] = data[b : b + scp.REC_SIZE], data[a:b]  # swap first two records
        with open(idx, "wb") as f:
            f.write(data)
        _, problems = bcp.check_index(idx, os.path.getsize(dat), "00")
        assert any("not strictly after" in p for p in problems)

    def test_finalize_requires_every_shard_verified(self, tmp_path, stage) -> None:
        root = str(tmp_path / "cache")
        SearchCache(root).put_search("q", SMALL, provider="tavily")
        for kind, ab in scp.all_shards():
            bcp.build_shard(root, kind, ab, stage)
        for kind, ab in scp.all_shards():
            if (kind, ab) != ("browse", "ff"):
                assert bcp.verify_shard(root, kind, ab)["verified"]
        os.rename(
            scp.shard_manifest_path(scp.pack_root_of(root), "browse", "00"),
            scp.shard_manifest_path(scp.pack_root_of(root), "browse", "00") + ".moved",
        )
        manifest, problems = bcp.finalize(root)
        assert manifest is None
        assert problems == ["browse/00: no shard manifest", "browse/ff: not verified (verify never ran)"]
        assert SearchCache(root)._pack is None

    def test_finalize_writes_sizes_and_totals(self, packed_root) -> None:
        manifest = scp.load_manifest(packed_root)
        assert len(manifest["shards"]) == 512
        assert manifest["totals"]["entries_count"] == 4
        assert manifest["totals"]["stored_raw_count"] == 1
        idx, dat = scp.shard_paths(scp.pack_root_of(packed_root), "search", "00")
        assert manifest["shards"]["search/00"]["idx_bytes"] == os.path.getsize(idx)
        assert manifest["shards"]["search/00"]["dat_bytes"] == os.path.getsize(dat)
        assert bcp.finalize(packed_root)[0] is None  # never rewritten


class TestUnpack:
    def test_unpack_restores_missing_flats_without_overwriting(self, packed_root) -> None:
        paths = {q: _flat(packed_root, "search", q) for q in ("small query", "compressible query", "large query")}
        originals = {q: (open(p, "rb").read(), os.stat(p).st_mtime_ns) for q, p in paths.items()}
        _drop_flats(packed_root)
        SearchCache(packed_root).put_search("small query", [{"r": "newer"}], provider="exa")
        newer = open(paths["small query"], "rb").read()
        totals = {}
        for kind, ab in scp.all_shards():
            for k, v in bcp.unpack_shard(packed_root, kind, ab).items():
                totals[k] = totals.get(k, 0) + v
        assert totals == {"restored": 3, "present": 1}
        assert open(paths["small query"], "rb").read() == newer  # never overwritten
        for q in ("compressible query", "large query"):
            raw, mtime_ns = originals[q]
            assert open(paths[q], "rb").read() == raw
            assert os.stat(paths[q]).st_mtime_ns == mtime_ns  # matches the manifest again
        assert not [p for p in _tree(packed_root) if p.endswith(".tmp")]

    def test_unpack_dry_run_writes_nothing(self, packed_root) -> None:
        _drop_flats(packed_root)
        before = _tree(packed_root)
        totals = {}
        for kind, ab in scp.all_shards():
            for k, v in bcp.unpack_shard(packed_root, kind, ab, dry_run=True).items():
                totals[k] = totals.get(k, 0) + v
        assert totals == {"would_restore": 4}
        assert _tree(packed_root) == before


class TestIndexFormat:
    def test_write_index_layout(self) -> None:
        recs = sorted((bytes([7, b]) + os.urandom(30), i * 10, 10, 20) for i, b in enumerate([0, 0, 5, 255]))
        buf = io.BytesIO()
        scp.write_index(buf, recs)
        data = buf.getvalue()
        assert data[:8] == b"BCPACKIX"
        count, fan = scp.read_index_head(data[: scp.INDEX_HEAD_SIZE], len(data), "mem")
        assert count == 4 and fan[0] == 0 and fan[1] == 2 and fan[5] == 2 and fan[6] == 3 and fan[255] == 3
        assert fan[256] == 4
        assert len(data) == scp.INDEX_HEAD_SIZE + 4 * 48

    def test_write_index_rejects_unsorted(self) -> None:
        d1, d2 = bytes([1, 2]) + bytes(30), bytes([1, 1]) + bytes(30)
        with pytest.raises(ValueError):
            scp.write_index(io.BytesIO(), [(d1, 0, 1, 1), (d2, 1, 1, 1)])

    def test_encode_blob_threshold(self) -> None:
        small = json.dumps({"x": "a" * 1000}).encode()
        blob, length = scp.encode_blob(small)
        assert not length & scp.STORED and length == len(blob) < len(small)
        big = b'"' + b"a" * scp.STORE_RAW_MIN + b'"'
        blob, length = scp.encode_blob(big)
        assert blob is big and length == len(big) | scp.STORED
