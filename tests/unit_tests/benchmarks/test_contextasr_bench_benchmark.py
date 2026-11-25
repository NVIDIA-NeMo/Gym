# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for ContextASR-Bench: declaration, discovery, row building, archive reading, prepare()."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks import audio_manifest
from benchmarks.contextasr_bench import prepare as ctx
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "contextasr_bench"


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "MIT"


def test_profiles_cover_every_set_and_mode() -> None:
    profiles = audio_manifest.load_profiles(ctx.PROFILES)["profiles"]
    expected = [f"{source['profile']}.{mode}" for source in ctx.SOURCES.values() for mode in ctx.MODES]
    assert list(profiles) == expected and len(expected) == 12
    for name, contract in profiles.items():
        assert contract["prompt_override"] == {"system_prompt": ""}
        assert contract["skills"]["METRICS_TYPE"] == "contextasr"
    # The English speech profiles keep the greedy default; the others need room for longer transcripts.
    controls = {name: audio_manifest.request_controls(c)["max_output_tokens"] for name, c in profiles.items()}
    assert controls["contextasr-bench.fine"] == 256
    assert controls["contextasr-bench-mandarin.fine"] == 1024
    assert controls["contextasr-bench-dialogue.coarse"] == controls["contextasr-bench-dialogue-mandarin.fine"] == 4096
    assert sum(source["rows"] for source in ctx.SOURCES.values()) == 41329
    for source in ctx.SOURCES.values():
        assert len(source["jsonl_sha256"]) == 64 and int(source["jsonl_sha256"], 16) >= 0
    assert ctx.HF_REVISION == "7906fd98d1d093bf244df80e726e07fc8afed6a6"  # pragma: allowlist secret


def test_narrowing_arguments_default_to_all_and_accept_any_case() -> None:
    assert ctx.select_subsets(None) == ctx.SUBSETS and ctx.select_languages(None) == ctx.LANGUAGES
    assert ctx.select_modes(None) == ctx.MODES
    assert ctx.select_subsets("Speech") == ("speech",) and ctx.select_languages(["mandarin"]) == ("Mandarin",)
    assert ctx.select_modes("fine, contextless") == ("contextless", "fine")
    for bad in ("nope", "speech,nope", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            ctx.select_subsets(bad)


SPEECH = {
    "uniq_id": "A-1_EN",
    "language": "English",
    "duration": 1.5,
    "audio": "audio/ContextASR-Speech/English/A-1_EN.wav",
    "text": "Hello Anna.",
    "domain_label": "Greetings",
    "entity_list": ["Anna", "Hello Anna"],
}
DIALOGUE = {
    "uniq_id": "0-1",
    "language": "Mandarin",
    "duration": 12.0,
    "audio": "audio/ContextASR-Dialogue/Mandarin/0-1.wav",
    "text": "你好 小王",
    "dialogue": [{"role": "a", "text": "你好"}, {"role": "b", "text": "小王"}],
    "movie_name": "僵尸至尊",
    "entity_list": ["小王"],
}


def test_english_speech_prompts() -> None:
    task = "Transcribe the English audio into text, ensuring all punctuation marks are included."
    prompts = {
        mode: ctx.format_source_row(SPEECH, mode, "speech", "English", "/r")["messages"][0]["content"]
        for mode in ctx.MODES
    }
    assert prompts["contextless"] == task
    assert prompts["coarse"] == f"This audio belongs to the Greetings field. {task}"
    assert prompts["fine"] == (
        "This audio belongs to the Greetings field and may contain the following words or phrases: "
        f"Anna, Hello Anna. {task}"
    )
    source = ctx.format_source_row(SPEECH, "fine", "speech", "English", "/r")
    assert source["messages"][0]["audio"] == {"path": "/r/audio/ContextASR-Speech/English/A-1_EN.wav", "duration": 1.5}
    assert source["audio_filepath"] == "/r/audio/ContextASR-Speech/English/A-1_EN.wav"
    assert source["expected_answer"] == "Hello Anna." and source["subset_for_metrics"] == source["domain_label"]
    assert (
        source["source_provenance"]["expected_rows"] == 15326
        and source["source_provenance"]["dataset"] == ctx.HF_REPO_ID
    )
    assert set(source) == {
        "messages", "expected_answer", "entity_list", "domain_label", "subset_for_metrics", "uniq_id",
        "duration", "audio_filepath", "source_provenance",
    }  # fmt: skip
    with pytest.raises(ValueError, match="Unknown mode"):
        ctx.format_source_row(SPEECH, "other", "speech", "English", "/r")


def test_dialogue_rows_name_the_movie_and_language() -> None:
    source = ctx.format_source_row(DIALOGUE, "coarse", "dialogue", "Mandarin", "/r")
    assert source["messages"][0]["content"] == (
        "This audio is a dialogue about the movie 僵尸至尊. "
        "Transcribe the Mandarin audio into text, ensuring all punctuation marks are included."
    )
    assert source["movie_name"] == "僵尸至尊" and "subset_for_metrics" not in source and "domain_label" not in source
    fine = ctx.format_source_row(DIALOGUE, "fine", "dialogue", "Mandarin", "/r")["messages"][0]["content"]
    assert "and may contain the following words or phrases: 小王. Transcribe the Mandarin" in fine


def _wav_bytes(seconds: float = 0.1) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
    return buffer.getvalue()


def _write_tar(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w") as archive:
        for name in names:
            payload = _wav_bytes()
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


class _FakeFileSystem:
    """Serves local tar files under the Hugging Face path layout the extraction code uses."""

    def __init__(self, tars: dict[str, Path]) -> None:
        self.tars = tars
        self.opened: list[str] = []

    def open(self, path: str, mode: str):
        assert mode == "rb"
        name = path.split("@", 1)[1].split("/", 1)[1]
        self.opened.append(name)
        return open(self.tars[name], "rb")


def _row(subset: str, language: str, name: str) -> dict:
    return {"audio": f"audio/ContextASR-{subset.capitalize()}/{language}/{name}.wav"}


def test_extract_audio_reads_only_the_archive_holding_the_clips(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(ctx.SOURCES, ("speech", "English"), ctx.SOURCES["speech", "English"] | {"tars": 3})
    paths = ctx.tar_paths("speech", "English")
    assert paths[0] == "audio/ContextASR-Speech/English/ContextASR-Speech_English_1.tar" and len(paths) == 3
    tars = {}
    for path, names in zip(paths, (["A-1.wav", "B-1.wav"], ["C-1.wav", "D-1.wav"], ["E-1.wav", "F-1.wav"])):
        tars[path] = tmp_path / Path(path).name
        _write_tar(tars[path], names)
    filesystem = _FakeFileSystem(tars)
    monkeypatch.setattr(ctx, "HfFileSystem", lambda: filesystem)

    data_dir = tmp_path / "data"
    rows = [_row("speech", "English", "D-1"), _row("speech", "English", "A-1")]
    ctx.extract_audio("speech", "English", rows, data_dir)
    assert sorted(p.name for p in (data_dir / "audio/ContextASR-Speech/English").iterdir()) == ["A-1.wav", "D-1.wav"]
    assert set(filesystem.opened) == set(paths[:2]) | {paths[2]}, "first clip names are read from every archive"
    assert filesystem.opened.count(paths[2]) == 1, "the third archive holds no wanted clip"
    ctx.check_audio(rows, data_dir)

    filesystem.opened.clear()
    ctx.extract_audio("speech", "English", rows, data_dir)
    assert filesystem.opened == [], "existing clips must not be fetched again"

    # A clip that the sorted archives do not announce is found by the fallback scan; a missing one is reported.
    ctx.extract_audio("speech", "English", [_row("speech", "English", "B-1")], data_dir)
    assert (data_dir / "audio/ContextASR-Speech/English/B-1.wav").is_file()
    with pytest.raises(FileNotFoundError, match="missing from the archives"):
        ctx.extract_audio("speech", "English", [_row("speech", "English", "Z-9")], data_dir)


def test_extract_audio_skips_archives_that_hold_no_wanted_clip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(ctx.SOURCES, ("speech", "English"), ctx.SOURCES["speech", "English"] | {"tars": 3})
    paths = ctx.tar_paths("speech", "English")
    tars = {}
    for path, names in zip(paths, (["A-1.wav"], ["C-1.wav"], ["E-1.wav", "F-1.wav"])):
        tars[path] = tmp_path / Path(path).name
        _write_tar(tars[path], names)
    filesystem = _FakeFileSystem(tars)
    monkeypatch.setattr(ctx, "HfFileSystem", lambda: filesystem)
    ctx.extract_audio("speech", "English", [_row("speech", "English", "F-1")], tmp_path / "data")
    assert filesystem.opened.count(paths[0]) == filesystem.opened.count(paths[1]) == 1, "only the name peek"
    assert filesystem.opened.count(paths[2]) == 2


def test_check_audio_rejects_missing_or_empty_clips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing or unreadable"):
        ctx.check_audio([_row("speech", "English", "nope")], tmp_path)
    import soundfile as sf

    empty = tmp_path / "audio/ContextASR-Speech/English/empty.wav"
    empty.parent.mkdir(parents=True)
    sf.write(str(empty), np.zeros(0, dtype=np.float32), 16000, format="WAV")
    with pytest.raises(ValueError, match="empty"):
        ctx.check_audio([_row("speech", "English", "empty")], tmp_path)


def _tiny_rows(subset: str, language: str) -> list[dict]:
    base = SPEECH if subset == "speech" else DIALOGUE
    directory = f"audio/ContextASR-{subset.capitalize()}/{language}"
    return [base | {"uniq_id": f"U-{i}", "language": language, "audio": f"{directory}/U-{i}.wav"} for i in (1, 2)]


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> Path:
    """Run prepare() against two-clip versions of all four test sets, served from local files."""
    files, tars = {}, {}
    for subset, language in ctx.SOURCES:
        rows = _tiny_rows(subset, language)
        path = tmp_path / ctx.metadata_name(subset, language)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        files[path.name] = path
        source = ctx.SOURCES[subset, language]
        monkeypatch.setitem(
            ctx.SOURCES,
            (subset, language),
            source | {"jsonl_sha256": audio_manifest.sha256_file(path), "rows": len(rows), "tars": 1},
        )
        (tar_path,) = ctx.tar_paths(subset, language)
        tars[tar_path] = tmp_path / Path(tar_path).name
        _write_tar(tars[tar_path], sorted(Path(row["audio"]).name for row in rows))
    monkeypatch.setattr(ctx, "hf_hub_download", lambda repo, filename, **_: str(files[filename]))
    monkeypatch.setattr(ctx, "HfFileSystem", lambda: _FakeFileSystem(tars))
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ctx, "DATA_DIR", data_dir)
    monkeypatch.setattr(ctx, "OUTPUT_FPATH", data_dir / "contextasr_bench_benchmark.jsonl")
    return data_dir


def test_prepare_defaults_to_every_set_and_mode(offline_prepare: Path) -> None:
    output = ctx.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 4 * 3 * 2
    profiles = list(dict.fromkeys(row["benchmark_profile"] for row in rows))
    assert profiles == [f"{source['profile']}.{mode}" for source in ctx.SOURCES.values() for mode in ctx.MODES]
    assert all(
        row["sample_id"] == f"{row['benchmark_profile']}:{i}"
        for name in profiles
        for i, row in enumerate((r for r in rows if r["benchmark_profile"] == name), 1)
    )
    assert ctx.is_prepared_data_current(output) and not ctx.is_prepared_data_current(offline_prepare / "other.jsonl")
    mandarin = next(row for row in rows if row["benchmark_profile"] == "contextasr-bench-mandarin.contextless")
    assert mandarin["responses_create_params"]["input"][0]["content"].startswith("Transcribe the Mandarin audio")


def test_prepare_one_set_end_to_end(offline_prepare: Path) -> None:
    output = ctx.prepare(subsets="speech", languages="English")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["benchmark_profile"] for row in rows[::2]] == [f"contextasr-bench.{mode}" for mode in ctx.MODES]
    assert [row["sample_id"] for row in rows[:2]] == [
        "contextasr-bench.contextless:1",
        "contextasr-bench.contextless:2",
    ]
    first = rows[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (offline_prepare / "audio/ContextASR-Speech/English/U-1.wav").resolve()
    assert audio_path.is_file()
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert first["responses_create_params"]["input"][0]["content"].startswith("Transcribe the English audio")
    assert ctx.is_prepared_data_current(output) is False, "narrowed data must be prepared again by default"


def test_prepare_audio_root_and_row_cap(offline_prepare: Path) -> None:
    output = ctx.prepare(
        subsets="dialogue", languages="Mandarin", modes="fine", audio_root="/data/c", max_samples_per_profile=1
    )
    (line,) = output.read_text().splitlines()
    row = json.loads(line)
    assert row["benchmark_profile"] == "contextasr-bench-dialogue-mandarin.fine"
    assert (
        row["responses_create_params"]["metadata"]["audio_path"]
        == "/data/c/audio/ContextASR-Dialogue/Mandarin/U-1.wav"
    )
    assert (offline_prepare / "audio/ContextASR-Dialogue/Mandarin/U-1.wav").is_file()
    assert row["responses_create_params"]["max_output_tokens"] == 4096


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_prepare_rejects_a_bad_row_cap(bad, offline_prepare: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ctx.prepare(max_samples_per_profile=bad)


def test_prepare_rejects_profiles_that_disagree_with_sources(offline_prepare: Path, monkeypatch) -> None:
    monkeypatch.setitem(ctx.SOURCES, ("speech", "English"), ctx.SOURCES["speech", "English"] | {"profile": "other"})
    with pytest.raises(ValueError, match="disagree"):
        ctx.prepare()


def test_prepare_rejects_a_drifted_source(offline_prepare: Path, monkeypatch) -> None:
    key = ("speech", "English")
    monkeypatch.setitem(ctx.SOURCES, key, ctx.SOURCES[key] | {"rows": 3})
    with pytest.raises(ValueError, match="row-count drift"):
        ctx.load_metadata(*key)
    monkeypatch.setitem(ctx.SOURCES, key, ctx.SOURCES[key] | {"jsonl_sha256": "0" * 64})
    with pytest.raises(ValueError, match="hash drift"):
        ctx.load_metadata(*key)


def test_load_metadata_rejects_duplicate_ids_and_audio_paths(offline_prepare: Path, monkeypatch) -> None:
    key = ("speech", "English")
    rows = _tiny_rows(*key)
    for field in ("uniq_id", "audio"):
        duplicated = [rows[0], rows[1] | {field: rows[0][field]}]
        path = offline_prepare.parent / "dup.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in duplicated), encoding="utf-8")
        monkeypatch.setattr(ctx, "hf_hub_download", lambda *args, **kwargs: str(path))
        monkeypatch.setitem(ctx.SOURCES, key, ctx.SOURCES[key] | {"jsonl_sha256": audio_manifest.sha256_file(path)})
        with pytest.raises(ValueError, match="duplicate ids or audio paths"):
            ctx.load_metadata(*key)
