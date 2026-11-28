# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for benchmarks/asr_leaderboard/prepare.py: a fake load_dataset and a fake soundfile."""

import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from benchmarks.asr_leaderboard import prepare as prepare_module


# The (config, split) pairs hf-audio/open-asr-leaderboard serves. TED-LIUM is not one of them: upstream deleted
# tedlium/ on 2026-05-27 ("Remove Tedlium dataset (#9)"), so asking for it fails with NonMatchingSplitsSizesError.
UPSTREAM_SUBSETS = {
    ("librispeech", "test.clean"),
    ("librispeech", "test.other"),
    ("voxpopuli", "test"),
    ("gigaspeech", "test"),
    ("spgispeech", "test"),
    ("earnings22", "test"),
    ("ami", "test"),
}


class _FakeAudioDecoder:
    """What datasets>=4 yields for an Audio column: "array" and "sampling_rate" can be indexed (each "array" read
    decodes the clip again), but it is no mapping, so `"array" in decoder` raises like datasets' wrapper does."""

    def __init__(self, num_samples: int, sampling_rate: int = 16000) -> None:
        self.num_samples = num_samples
        self.sampling_rate = sampling_rate
        self.decodes = 0

    def __getitem__(self, key: str) -> Any:
        if key == "array":
            self.decodes += 1
            return [0.0] * self.num_samples
        if key == "sampling_rate":
            return self.sampling_rate
        raise TypeError("'torchcodec.decoders.AudioDecoder' object is not subscriptable")


def _entries(config: str, split: str, n: int = 2) -> list[dict[str, Any]]:
    return [
        {"id": f"{config}-{split}-{i}", "text": f"utterance {i}", "audio": _FakeAudioDecoder(1600)} for i in range(n)
    ]


@pytest.fixture
def fake_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[tuple, dict[str, Any]]]:
    """Point the output at tmp_path, record every load_dataset call, and stub soundfile.write."""
    calls: list[tuple[tuple, dict[str, Any]]] = []

    def load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((args, kwargs))
        _, config = args
        return _entries(config, kwargs["split"])

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda path, array, sampling_rate: Path(path).write_bytes(b"flac")
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path / "data")
    return calls


def test_prepare_loads_only_subsets_upstream_serves_at_a_pinned_revision(
    fake_sources: list[tuple[tuple, dict[str, Any]]], tmp_path: Path
) -> None:
    prepare_module.prepare(work_dir=tmp_path / "audio")

    requested = {(args[1], kwargs["split"]) for args, kwargs in fake_sources}
    assert requested == UPSTREAM_SUBSETS
    for args, kwargs in fake_sources:
        # The canonical repo name, not the old one HF answers with a 307 redirect.
        assert args[0] == "hf-audio/open-asr-leaderboard"
        # A commit SHA, not a branch: upstream main can drop a subset again.
        assert re.fullmatch(r"[0-9a-f]{40}", kwargs.get("revision") or "")


def test_prepare_writes_every_subset_into_the_benchmark_jsonl(
    fake_sources: list[tuple[tuple, dict[str, Any]]], tmp_path: Path
) -> None:
    out_path = prepare_module.prepare(work_dir=tmp_path / "audio")

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").split("\n") if line]
    assert out_path == tmp_path / "data" / "asr_leaderboard_benchmark.jsonl"
    assert len(rows) == 2 * len(prepare_module.DATASET_CONFIGS)
    assert {row["subset_for_metrics"] for row in rows} == set(prepare_module.DATASET_CONFIGS)
    assert "tedlium" not in prepare_module.DATASET_CONFIGS
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["asr_leaderboard_benchmark.jsonl"]


@pytest.mark.parametrize("previous", [None, '{"previous": "complete run"}\n'])
def test_a_failed_subset_leaves_no_partial_benchmark_jsonl(
    fake_sources: list[tuple[tuple, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    previous: str | None,
) -> None:
    out_path = tmp_path / "data" / "asr_leaderboard_benchmark.jsonl"
    if previous is not None:
        out_path.parent.mkdir(parents=True)
        out_path.write_text(previous, encoding="utf-8")

    def load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if args[1] == "voxpopuli":
            raise RuntimeError("NonMatchingSplitsSizesError: recorded num_examples=0")
        return _entries(args[1], kwargs["split"])

    monkeypatch.setattr("datasets.load_dataset", load_dataset)

    with pytest.raises(RuntimeError, match="NonMatchingSplitsSizesError"):
        prepare_module.prepare(work_dir=tmp_path / "audio")

    # librispeech rows were produced before voxpopuli failed; none of them may reach the benchmark file, which
    # `gym eval prepare +use_cached_prepared_benchmarks=true` and collate would otherwise accept as prepared.
    if previous is None:
        assert not out_path.exists()
    else:
        assert out_path.read_text(encoding="utf-8") == previous
    assert sorted(p.name for p in out_path.parent.iterdir()) == ([] if previous is None else [out_path.name])


@pytest.mark.parametrize("shape", ["decoder", "dict"])
def test_prepare_decodes_each_clip_once_and_skips_too_short_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: str
) -> None:
    """datasets>=4 rows carry an audio decoder, datasets<4 rows a decoded dict; both must produce the same rows."""
    clips = {"spk/long": _FakeAudioDecoder(1600), "spk/short": _FakeAudioDecoder(800)}  # 0.1 s and 0.05 s
    entries = [
        {
            "id": raw_id,
            "text": "hello",
            "audio": clip if shape == "decoder" else {"array": clip["array"], "sampling_rate": clip["sampling_rate"]},
        }
        for raw_id, clip in clips.items()
    ]
    for clip in clips.values():
        clip.decodes = 0
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: entries)
    written: dict[str, tuple[int, int]] = {}
    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda path, array, sampling_rate: written.update({Path(path).name: (len(array), sampling_rate)})
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path / "data")

    out_path = prepare_module.prepare(work_dir=tmp_path / "audio", datasets=("ami",))

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").split("\n") if line]
    assert [row["sample_id"] for row in rows] == ["spk/long"]
    assert (
        rows[0]["responses_create_params"]["metadata"]["audio_path"]
        == "/dataset/asr-leaderboard/data/ami/spk_long.flac"
    )
    assert written == {"spk_long.flac": (1600, 16000)}
    if shape == "decoder":
        assert [clip.decodes for clip in clips.values()] == [1, 1]
