# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row building shared by the Mandarin meeting benchmarks (AISHELL-4 and AliMeeting).

Both corpora publish long-format Praat TextGrid annotations (one tier per speaker). The reference of a session is its
speaker turns in chronological order, serialized as ``<spk:N> text`` with speaker numbers in order of first
appearance; the evaluator applies the corpus text normalization and character tokenization before scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SPEAKER_PROMPT = "Transcribe the mixed audio. Put a distinct <spk:N> tag before every speaker turn."
_ITEM_RE = re.compile(r"^\s*item\s*\[\d+\]:\s*$")
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[\d+\]:\s*$")
_ASSIGN_RE = re.compile(r"^\s*(name|xmin|xmax|text)\s*=\s*(.*?)\s*$")


@dataclass(frozen=True)
class MeetingInterval:
    """One non-empty TextGrid interval: the speaker (tier name), its start and end in seconds, and the text."""

    speaker: str
    start: float
    end: float
    text: str


def _praat_string(value: str) -> str:
    value = value.strip()
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        raise ValueError(f"Expected a quoted Praat string, got {value!r}")
    return value[1:-1].replace('""', '"')


def parse_long_textgrid(text: str, source: str) -> list[MeetingInterval]:
    """Parse a long-format TextGrid into its non-empty intervals; ``source`` names the file in error messages."""
    lines = text.splitlines()
    intervals: list[MeetingInterval] = []
    speaker: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if _ITEM_RE.match(line):
            speaker = None
            index += 1
            continue
        match = _ASSIGN_RE.match(line)
        if match and match.group(1) == "name":
            speaker = _praat_string(match.group(2))
            index += 1
            continue
        if not _INTERVAL_RE.match(line):
            index += 1
            continue
        if speaker is None:
            raise ValueError(f"Interval encountered before tier name in {source}:{index + 1}")
        values: dict[str, str] = {}
        index += 1
        while index < len(lines) and len(values) < 3:
            field = _ASSIGN_RE.match(lines[index])
            if field and field.group(1) in {"xmin", "xmax", "text"}:
                values[field.group(1)] = field.group(2)
            index += 1
        if set(values) != {"xmin", "xmax", "text"}:
            raise ValueError(f"Incomplete interval in {source} near line {index + 1}")
        interval_text = _praat_string(values["text"])
        if interval_text:
            start, end = float(values["xmin"]), float(values["xmax"])
            if end <= start:
                raise ValueError(f"Non-positive interval in {source}: {start}..{end}")
            intervals.append(MeetingInterval(speaker=speaker, start=start, end=end, text=interval_text))
    if not intervals:
        raise ValueError(f"No non-empty speech intervals found in {source}")
    return intervals


def intervals_to_sot(intervals: list[MeetingInterval]) -> tuple[str, int, int]:
    """Serialize turns chronologically: returns the reference, the number of speakers and of speaker changes."""
    speaker_ids: dict[str, int] = {}
    chunks: list[str] = []
    previous: int | None = None
    changes = 0
    for interval in sorted(intervals, key=lambda item: (item.start, item.end, item.speaker)):
        speaker_id = speaker_ids.setdefault(interval.speaker, len(speaker_ids))
        changes += previous is not None and speaker_id != previous
        previous = speaker_id
        chunks.append(f"<spk:{speaker_id}> {interval.text.strip()}")
    return " ".join(chunks), len(speaker_ids), changes


def wav_duration(path: Path) -> float:
    """Duration of the audio file at ``path`` in seconds."""
    import soundfile as sf

    return float(sf.info(path).duration)


def write_first_channel_wav(source: Path, destination: Path) -> None:
    """Stream channel zero of ``source`` into a PCM16 mono WAV (a meeting never fits in RAM).

    The single channel is the released single-distant-microphone condition. Written atomically.
    """
    import soundfile as sf

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with (
        sf.SoundFile(source) as input_audio,
        sf.SoundFile(
            partial, mode="w", samplerate=input_audio.samplerate, channels=1, format="WAV", subtype="PCM_16"
        ) as output_audio,
    ):
        while len(block := input_audio.read(1_048_576, dtype="float32", always_2d=True)):
            output_audio.write(block[:, :1])
    partial.replace(destination)


def format_source_row(
    *,
    corpus: str,
    session_id: str,
    audio_path: str,
    duration: float,
    textgrid: str,
    source_metadata: dict,
    source_audio: str,
    source_textgrid: str,
    source_info: dict,
) -> dict:
    """Build the source row of one session from its TextGrid text and the converted mono audio.

    ``source_audio``, ``source_textgrid`` (paths of the originals within the source) and ``source_info`` (channels,
    frames, sample_rate of the multichannel audio) are recorded as provenance.
    """
    intervals = parse_long_textgrid(textgrid, source_textgrid)
    reference, num_speakers, num_changes = intervals_to_sot(intervals)
    return {
        "id": session_id,
        "sample_id": session_id,
        "audio_filepath": audio_path,
        "duration": duration,
        "expected_answer": reference,
        "text": reference,
        "messages": [{"role": "user", "content": SPEAKER_PROMPT, "audio": {"path": audio_path, "duration": duration}}],
        "source_lang": "zh",
        "target_lang": "zh",
        "task_type": "Mandarin-MS-ASR",
        "category": "nonjudge",
        "mandarin_corpus": corpus,
        "num_speakers": num_speakers,
        "num_changes": num_changes,
        "subset_for_metrics": "test",
        "extra_fields": {
            **source_metadata,
            "source_audio": source_audio,
            "source_audio_stem": Path(source_audio).stem,
            "source_audio_channels": source_info["channels"],
            "source_audio_frames": source_info["frames"],
            "source_audio_sample_rate": source_info["sample_rate"],
            "source_textgrid": source_textgrid,
            "input_channel": 0,
            "reference_intervals": len(intervals),
            "normalization": f"released-{corpus}",
            "metric": "full-session-cpCER",
        },
    }
