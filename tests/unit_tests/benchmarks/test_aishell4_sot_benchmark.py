# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the AISHELL-4 SOT benchmark and the Mandarin meeting helpers other meeting benchmarks share."""

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import audio_manifest
from benchmarks import mandarin_meeting as meeting
from benchmarks.aishell4_sot import prepare as aishell4
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "aishell4_sot"


def textgrid(*tiers: tuple[str, list[tuple[float, float, str]]]) -> str:
    """A long-format TextGrid with one interval tier per speaker."""
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "item []:"]
    for number, (speaker, intervals) in enumerate(tiers, 1):
        lines += [f"    item [{number}]:", '        class = "IntervalTier"', f'        name = "{speaker}"']
        lines.append(f"        intervals: size = {len(intervals)}")
        for position, (start, end, text) in enumerate(intervals, 1):
            lines += [
                f"        intervals [{position}]:",
                f"            xmin = {start}",
                f"            xmax = {end}",
                f'            text = "{text}"',
            ]
    return "\n".join(lines) + "\n"


MEETING = textgrid(
    ("tier-b", [(0.0, 1.0, ""), (2.0, 3.0, "你好")]),
    ("tier-a", [(0.0, 1.5, "大家好"), (3.0, 4.0, "再见")]),
)


def write_multichannel_wav(path: Path, *, channels: int = 3, rate: int = 16000, seconds: float = 0.25) -> None:
    import soundfile as sf

    frames = int(rate * seconds)
    samples = np.stack([np.full(frames, 0.1 * (channel + 1), dtype=np.float32) for channel in range(channels)], axis=1)
    sf.write(path, samples, rate, subtype="PCM_16")


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental


def test_profile_contract_values() -> None:
    contract = audio_manifest.load_profiles(aishell4.PROFILES)["profiles"]["aishell4-sot"]
    controls = audio_manifest.request_controls(contract)
    assert controls["max_output_tokens"] == 16384 and controls["temperature"] == 0.0
    assert "++eval_config.normalization=aishell4-sot" in contract["skills"]["EVAL_ARGS"]


def test_session_contract() -> None:
    assert len(aishell4.SESSION_IDS) == len(set(aishell4.SESSION_IDS)) == 20
    assert list(aishell4.SESSION_IDS) == sorted(aishell4.SESSION_IDS)


def test_parse_textgrid_keeps_non_empty_intervals_and_rejects_malformed_ones() -> None:
    intervals = meeting.parse_long_textgrid(MEETING, "m.TextGrid")
    assert [(i.speaker, i.start, i.end, i.text) for i in intervals] == [
        ("tier-b", 2.0, 3.0, "你好"),
        ("tier-a", 0.0, 1.5, "大家好"),
        ("tier-a", 3.0, 4.0, "再见"),
    ]
    with pytest.raises(ValueError, match="No non-empty"):
        meeting.parse_long_textgrid(textgrid(("a", [(0.0, 1.0, "")])), "empty")
    with pytest.raises(ValueError, match="Non-positive"):
        meeting.parse_long_textgrid(textgrid(("a", [(2.0, 1.0, "x")])), "backwards")
    with pytest.raises(ValueError, match="before tier name"):
        meeting.parse_long_textgrid('intervals [1]:\n xmin = 0\n xmax = 1\n text = "x"\n', "orphan")
    with pytest.raises(ValueError, match="quoted Praat string"):
        meeting.parse_long_textgrid(textgrid(("a", [(0.0, 1.0, "x")])).replace('"x"', "x"), "bare")


def test_reference_orders_turns_chronologically_and_numbers_speakers_by_first_appearance() -> None:
    reference, speakers, changes = meeting.intervals_to_sot(meeting.parse_long_textgrid(MEETING, "m"))
    assert reference == "<spk:0> 大家好 <spk:1> 你好 <spk:0> 再见"
    assert (speakers, changes) == (2, 2)


def test_write_first_channel_wav_keeps_only_channel_zero(tmp_path: Path) -> None:
    source, destination = tmp_path / "multi.wav", tmp_path / "out" / "mono.wav"
    write_multichannel_wav(source)
    meeting.write_first_channel_wav(source, destination)
    with wave.open(str(destination)) as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 16000)
        frames = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
    assert frames.size == 4000 and np.abs(frames / 32768 - 0.1).max() < 1e-3
    assert not list(destination.parent.glob("*.partial"))
    assert meeting.wav_duration(destination) == pytest.approx(0.25)


class _FakeHfFileSystem:
    """Serves the FLAC headers the loader reads with range requests."""

    def __init__(self, flacs: dict[str, Path]) -> None:
        self.flacs = flacs
        self.opened: list[str] = []

    def open(self, path: str, mode: str):
        assert f"{aishell4.HF_REPO_ID}@{aishell4.HF_REVISION}/" in path
        self.opened.append(path)
        return open(self.flacs[Path(path).stem], mode)


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """Run prepare() against two synthetic meetings, writing under tmp_path; records the files downloaded."""
    import soundfile as sf

    sessions = ("L_R003S01C02", "M_R003S01C01")
    flacs, grids = {}, {}
    for session in sessions:
        flacs[session] = tmp_path / "hub" / f"{session}.flac"
        flacs[session].parent.mkdir(exist_ok=True)
        sf.write(flacs[session], np.zeros((4000, 3), dtype=np.float32), 16000)
        grids[session] = tmp_path / "hub" / f"{session}.TextGrid"
        grids[session].write_text(MEETING, encoding="utf-8")
    downloads: list[str] = []

    def fake_download(repo_id, filename, *, repo_type, revision, local_dir=None):
        assert (repo_id, repo_type, revision) == (aishell4.HF_REPO_ID, "dataset", aishell4.HF_REVISION)
        downloads.append(filename)
        stem = Path(filename).stem
        return str(flacs[stem] if filename.endswith(".flac") else grids[stem])

    monkeypatch.setattr(aishell4, "hf_hub_download", fake_download)
    monkeypatch.setattr(aishell4, "HfFileSystem", lambda: _FakeHfFileSystem(flacs))
    monkeypatch.setattr(aishell4, "SESSION_IDS", sessions)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(aishell4, "DATA_DIR", data_dir)
    monkeypatch.setattr(aishell4, "OUTPUT_FPATH", data_dir / "aishell4_sot_benchmark.jsonl")
    return SimpleNamespace(data_dir=data_dir, downloads=downloads)


def test_prepare_writes_one_combined_jsonl_with_existing_audio(offline_prepare: SimpleNamespace) -> None:
    data_dir = offline_prepare.data_dir
    output = aishell4.prepare()
    assert output == data_dir / "aishell4_sot_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["aishell4-sot:1", "aishell4-sot:2"]
    assert {row["benchmark_profile"] for row in rows} == {"aishell4-sot"}

    first = rows[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (data_dir / "audio" / "L_R003S01C02.wav").resolve() and audio_path.is_file()
    assert first["responses_create_params"]["input"][0]["content"].startswith("Transcribe this audio.")
    assert first["responses_create_params"]["max_output_tokens"] == 16384
    assert first["expected_answer"] == "<spk:0> 大家好 <spk:1> 你好 <spk:0> 再见" == first["text"]
    assert first["id"] == "L_R003S01C02" and first["mandarin_corpus"] == "aishell4-sot"
    assert first["duration"] == pytest.approx(0.25) and first["num_speakers"] == 2 and first["num_changes"] == 2
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    extra = first["extra_fields"]
    assert extra["source_repo"] == "AISHELL/AISHELL-4" and extra["source_revision"] == aishell4.HF_REVISION
    assert extra["source_audio"] == "test/wav/L_R003S01C02.flac"
    assert (extra["source_audio_channels"], extra["source_audio_frames"], extra["source_audio_sample_rate"]) == (
        3,
        4000,
        16000,
    )
    assert extra["reference_intervals"] == 3

    assert aishell4.is_prepared_data_current(output)
    assert not aishell4.is_prepared_data_current(data_dir / "other.jsonl")
    assert not list((data_dir / "audio").glob(".download-*")), "downloads are removed after conversion"


def test_prepare_is_idempotent_and_honors_audio_root_and_row_cap(offline_prepare: SimpleNamespace) -> None:
    aishell4.prepare()
    flac_downloads = [name for name in offline_prepare.downloads if name.endswith(".flac")]
    assert len(flac_downloads) == 2
    output = aishell4.prepare(audio_root="/data/aishell4_sot", max_samples_per_profile=1)
    assert [name for name in offline_prepare.downloads if name.endswith(".flac")] == flac_downloads, "no re-download"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["responses_create_params"]["metadata"]["audio_path"] == "/data/aishell4_sot/audio/L_R003S01C02.wav"
    assert not aishell4.is_prepared_data_current(output), "capped or remapped data must be prepared again by default"


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "2"])
def test_prepare_rejects_an_invalid_row_cap_before_downloading(offline_prepare: SimpleNamespace, cap) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        aishell4.prepare(max_samples_per_profile=cap)
    assert offline_prepare.downloads == []
