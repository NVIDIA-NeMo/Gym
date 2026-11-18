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

import errno
import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from resources_servers.opencode_search import exa_cache
from resources_servers.opencode_search.exa_cache import ExaCache, normalize_query, to_sse
from resources_servers.opencode_search.tests.support import within


STORED = "capital city of the country australia"
# Jaccard 5/6 against STORED: a fuzzy hit at the default threshold of 0.8, never an exact one.
NEAR = "capital city of country australia"
# Jaccard 4/5 against NEAR: also a fuzzy hit for it, but a worse one than STORED.
FARTHER = "capital city of australia"
LATER = "tallest mountain on the south american continent"
LATER_NEAR = "tallest mountain on south american continent"  # Jaccard 6/7 against LATER


def _key(text: str) -> str:
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()


def _entry(root: Path, text: str) -> Path:
    key = _key(text)
    return root / "search" / key[:2] / key[2:4] / f"{key}.json"


def _payload(text: str) -> str:
    return f"payload for {text}"


def _seed(root: Path, *texts: str) -> None:
    """Store one search entry per text, through a writer that builds no index."""
    writer = ExaCache(str(root), exact_only=True)
    for text in texts:
        assert writer.put("search", text, _payload(text)) is True


def _index_file(store: Path, *texts: str, **fields) -> dict:
    """The content of store's index file when it lists the search entries for texts; fields replace keys."""
    content = {
        "version": 1,
        "root": str(store.resolve()),
        "entries": {"search": {_key(text): normalize_query(text) for text in texts}, "browse": {}},
    }
    return {**content, **fields}


@contextmanager
def _walk_blocked(cache: ExaCache):
    """Hold the store walk until the block exits, then wait for the loader thread to finish."""
    release = threading.Event()
    walk = cache._walk

    def blocked_walk() -> tuple:
        release.wait(30)
        return walk()

    cache._walk = blocked_walk
    try:
        yield
    finally:
        release.set()
        if cache._loader is not None:
            cache._loader.join(5)
            assert not cache._loader.is_alive()


def _snapshot(base: Path) -> list:
    return sorted((str(path.relative_to(base)), path.stat().st_mtime_ns) for path in base.rglob("*"))


def test_exact_and_legacy_root_hits_do_not_wait_for_the_index(tmp_path):
    root, legacy_root = tmp_path / "store", tmp_path / "legacy"
    _seed(root, STORED)
    legacy_text = "population of the largest city in canada"
    legacy_entry = _entry(legacy_root, legacy_text)
    legacy_entry.parent.mkdir(parents=True)
    legacy_entry.write_text(json.dumps({"results": "Title: rendered text", "provider": "exa"}))
    cache = ExaCache(str(root), legacy_root=str(legacy_root))

    with _walk_blocked(cache):
        cache.load_in_background()
        assert within(5, cache.get, "search", STORED) == (_payload(STORED), "exact")
        assert within(5, cache.get, "search", legacy_text) == (to_sse("Title: rendered text"), "legacy")
        assert cache._loader.is_alive()


def test_fuzzy_matching_is_skipped_until_the_index_is_ready(tmp_path):
    _seed(tmp_path / "store", STORED)
    cache = ExaCache(str(tmp_path / "store"))

    with _walk_blocked(cache):
        cache.load_in_background()
        assert within(5, cache.get, "search", NEAR) == (None, "miss")
        assert cache.stats["fuzzy_skipped"] == 1

    assert cache._ready.is_set()
    assert cache.get("search", NEAR) == (_payload(STORED), "fuzzy")
    assert cache.stats["fuzzy_skipped"] == 1


def test_put_does_not_wait_for_the_index_and_is_indexed_once(tmp_path):
    cache = ExaCache(str(tmp_path / "store"))

    with _walk_blocked(cache):
        cache.load_in_background()
        # Twice, as when concurrent misses for one query are all stored.
        assert within(5, cache.put, "search", STORED, "payload") is True
        assert within(5, cache.put, "search", STORED, "payload") is True
        assert within(5, cache.get, "search", STORED) == ("payload", "exact")
        assert cache._loader.is_alive()

    assert cache.get("search", NEAR) == ("payload", "fuzzy")
    # Neither the second put() nor the walk, which listed the entry, indexed it again.
    assert len(cache._index["search"]) == 1
    assert cache.stats["index_from_store"] == 0


def test_index_file_round_trip_reads_no_entry_files(tmp_path, monkeypatch):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    texts = [STORED, LATER, "who wrote the novel middlemarch"]
    _seed(root, *texts)
    first = ExaCache(str(root), index_path=str(index_file))
    umask = os.umask(0o022)
    try:
        first.load()
    finally:
        os.umask(umask)
    assert json.loads(index_file.read_text()) == _index_file(root, *texts)
    # It holds every stored query, so like the entries it is readable by its owner only.
    assert stat.S_IMODE(index_file.stat().st_mode) == 0o600
    saved = index_file.stat()

    entry_reads = []
    read_text, read_bytes = Path.read_text, Path.read_bytes

    def counting(read):
        def wrapper(path, *args, **kwargs):
            if root / "search" in path.parents:
                entry_reads.append(path)
            return read(path, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(Path, "read_text", counting(read_text))
    monkeypatch.setattr(Path, "read_bytes", counting(read_bytes))
    second = ExaCache(str(root), index_path=str(index_file))
    second.load()

    assert (second.stats["index_from_file"], second.stats["index_from_store"]) == (3, 0)
    assert entry_reads == []
    assert second.get("search", NEAR) == (_payload(STORED), "fuzzy")
    # Nothing new was found, so the index file was not rewritten.
    assert (index_file.stat().st_ino, index_file.stat().st_mtime_ns) == (saved.st_ino, saved.st_mtime_ns)


def test_index_file_makes_fuzzy_matching_ready_before_the_walk(tmp_path):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    ExaCache(str(root), index_path=str(index_file)).load()
    cache = ExaCache(str(root), index_path=str(index_file))

    with _walk_blocked(cache):
        cache.load_in_background()
        assert cache._ready.wait(5)
        assert within(5, cache.get, "search", NEAR) == (_payload(STORED), "fuzzy")
        assert cache._loader.is_alive()
        assert "index_load_ms" not in cache.stats


def test_walk_indexes_entries_written_after_the_index_file_and_resaves_it(tmp_path):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    ExaCache(str(root), index_path=str(index_file)).load()
    _seed(root, LATER)

    cache = ExaCache(str(root), index_path=str(index_file))
    cache.load()

    assert (cache.stats["index_from_file"], cache.stats["index_from_store"]) == (1, 1)
    assert cache.get("search", LATER_NEAR) == (_payload(LATER), "fuzzy")
    assert json.loads(index_file.read_text()) == _index_file(root, STORED, LATER)


def test_walk_drops_index_file_keys_whose_entries_are_gone(tmp_path):
    """Only the best fuzzy match is read, so a gone entry left in the index would hide FARTHER on every start."""
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED, FARTHER)
    ExaCache(str(root), index_path=str(index_file)).load()
    _entry(root, STORED).unlink()

    cache = ExaCache(str(root), index_path=str(index_file))
    cache.load()

    assert (cache.stats["index_from_file"], cache.stats["index_dropped"]) == (2, 1)
    assert cache.get("search", NEAR) == (_payload(FARTHER), "fuzzy")
    assert json.loads(index_file.read_text()) == _index_file(root, FARTHER)


def test_entry_stored_again_after_the_walk_passed_it_stays_indexed(tmp_path):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    ExaCache(str(root), index_path=str(index_file)).load()
    _entry(root, STORED).unlink()
    cache = ExaCache(str(root), index_path=str(index_file))
    walk = cache._walk

    def walk_then_put() -> tuple:
        listed = walk()  # STORED's entry is gone, so the walk does not list it
        assert cache.put("search", STORED, "stored again") is True
        return listed

    cache._walk = walk_then_put
    cache.load()

    assert cache.stats["index_dropped"] == 0
    assert cache.get("search", NEAR) == ("stored again", "fuzzy")


@pytest.mark.parametrize(
    "content",
    [
        lambda root: b"not json",
        lambda root: json.dumps(_index_file(root, LATER, version=999)).encode(),
        lambda root: json.dumps(_index_file(root, LATER, root=str(root.parent / "another-store"))).encode(),
        lambda root: json.dumps(_index_file(root, entries=[])).encode(),
    ],
    ids=["not-json", "other-version", "another-root", "entries-not-a-mapping"],
)
def test_unusable_index_file_is_rebuilt_from_the_store(tmp_path, content):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    index_file.write_bytes(content(root))
    cache = ExaCache(str(root), index_path=str(index_file))

    cache.load()

    # The files that parse list LATER, which is not in the store: a file that was used would have indexed it.
    assert (cache.stats["index_from_file"], cache.stats["index_from_store"]) == (0, 1)
    assert cache.get("search", NEAR) == (_payload(STORED), "fuzzy")
    assert json.loads(index_file.read_text()) == _index_file(root, STORED)


def test_index_file_is_used_through_a_symlink_to_its_root(tmp_path):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    ExaCache(str(root), index_path=str(index_file)).load()
    link = tmp_path / "link"
    link.symlink_to(root)

    cache = ExaCache(str(link), index_path=str(index_file))
    cache.load()

    assert (cache.stats["index_from_file"], cache.stats["index_from_store"]) == (1, 0)


def test_index_file_keys_that_are_not_entry_names_are_not_indexed(tmp_path):
    """A key is joined into an entry path, so one that is not a sha256 name must never be read."""
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, LATER)
    content = _index_file(root, LATER)
    content["entries"]["search"]["../../elsewhere"] = normalize_query(STORED)
    index_file.write_text(json.dumps(content))
    cache = ExaCache(str(root), index_path=str(index_file))

    cache.load()

    assert cache.stats["index_from_file"] == 1
    assert [key for _, _, key in cache._index["search"]] == [_key(LATER)]
    assert cache.get("search", NEAR) == (None, "miss")


def test_walk_skips_entries_it_cannot_index(tmp_path):
    root = tmp_path / "store"
    _seed(root, STORED)
    unusable = [b'{"norm": "capital city', b"[]", b'{"norm": 5}', b'{"norm": ""}', b"\xff\xfe not utf-8"]
    for i, content in enumerate(unusable):
        path = _entry(root, f"unusable entry {i}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    cache = ExaCache(str(root))

    cache.load()

    assert cache.stats["index_from_store"] == 1
    assert [key for _, _, key in cache._index["search"]] == [_key(STORED)]
    assert cache.get("search", NEAR) == (_payload(STORED), "fuzzy")


def test_failed_index_file_save_keeps_the_previous_file(tmp_path, monkeypatch, capsys):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    ExaCache(str(root), index_path=str(index_file)).load()
    previous = index_file.read_bytes()
    _seed(root, LATER)

    def dump_then_fail(obj, fh, **kwargs) -> None:
        fh.write(json.dumps(obj)[:20])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(exa_cache.json, "dump", dump_then_fail)
    cache = ExaCache(str(root), index_path=str(index_file))
    cache.load()

    assert "[exa-cache] index file not saved (OSError)" in capsys.readouterr().out
    assert index_file.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ["index.json", "store"]
    assert cache._loaded
    assert cache.get("search", LATER_NEAR) == (_payload(LATER), "fuzzy")


def test_exact_only_builds_no_index(tmp_path):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    cache = ExaCache(str(root), exact_only=True, index_path=str(index_file))

    cache.load_in_background()
    assert cache._loader is None
    cache.load()

    assert within(5, cache.get, "search", STORED) == (_payload(STORED), "exact")
    assert within(5, cache.get, "search", NEAR) == (None, "miss")
    assert cache.stats["fuzzy_skipped"] == 0
    assert cache.put("search", LATER, "later payload") is True
    assert within(5, cache.get, "search", LATER) == ("later payload", "exact")
    assert cache._index == {"search": [], "browse": []}
    assert not index_file.exists()


def test_read_only_writes_nothing(tmp_path):
    root, index_file = tmp_path / "store", tmp_path / "index.json"
    _seed(root, STORED)
    before = _snapshot(tmp_path)
    cache = ExaCache(str(root), read_only=True, index_path=str(index_file))

    assert cache.put("search", LATER, "later payload") is False
    assert cache.stats["put_skipped_read_only"] == 1
    cache.load()

    # The walk indexed an entry the index file lacked, which makes a writable cache save the file.
    assert cache.stats["index_from_store"] == 1
    assert cache.get("search", NEAR) == (_payload(STORED), "fuzzy")
    assert cache.get("search", STORED) == (_payload(STORED), "exact")
    assert cache.get("search", LATER) == (None, "miss")
    assert not index_file.exists()
    assert _snapshot(tmp_path) == before


def test_load_in_background_starts_one_thread_and_load_is_idempotent(tmp_path):
    _seed(tmp_path / "store", STORED)
    cache = ExaCache(str(tmp_path / "store"))
    walks = []
    walk = cache._walk

    def counting_walk() -> tuple:
        walks.append(1)
        return walk()

    cache._walk = counting_walk

    with _walk_blocked(cache):
        cache.load_in_background()
        loader = cache._loader
        cache.load_in_background()
        assert cache._loader is loader
        # A daemon, so stopping the server does not wait for the rest of the walk.
        assert loader.daemon

    cache.load()
    assert len(walks) == 1
    assert cache.stats["index_entries"] == 1


def test_failed_index_build_leaves_exact_hits_working(tmp_path, capsys):
    _seed(tmp_path / "store", STORED)
    cache = ExaCache(str(tmp_path / "store"))

    def failing_walk() -> tuple:
        raise RuntimeError("store unreadable")

    cache._walk = failing_walk
    cache.load_in_background()
    cache._loader.join(5)

    assert "fuzzy index build failed (RuntimeError: store unreadable); serving exact hits only" in (
        capsys.readouterr().out
    )
    assert not cache._ready.is_set()
    # The index never becomes ready, so a lookup that waited for it would never return.
    assert within(5, cache.get, "search", STORED) == (_payload(STORED), "exact")
    assert within(5, cache.get, "search", NEAR) == (None, "miss")
    assert cache.stats["fuzzy_skipped"] == 1


def test_defaults_fuzzy_match_and_write_back_without_an_index_file(tmp_path):
    cache = ExaCache(str(tmp_path / "store"))
    assert (cache.threshold, cache.exact_only, cache.read_only, cache.index_path) == (0.8, False, False, None)

    assert cache.put("search", STORED, "payload") is True
    cache.load()

    assert cache.get("search", NEAR) == ("payload", "fuzzy")
    assert [path.name for path in tmp_path.rglob("*") if path.is_file()] == [f"{_key(STORED)}.json"]
