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
"""Multi-judge panel support shared by the GDPVal rubric and comparison scorers.

A *panel* is a set of LLM judges (each a distinct upstream model + reasoning
settings, e.g. GPT-5.5 medium, Gemini 3.1 Pro Preview high, Claude Opus 4.8
high). For every individual scoring/comparison we *sample* one panel member —
the per-trial grader is drawn here. Sampling is seeded (via :func:`make_rng`)
so a rerun of the same task lands on the same judges and the result is
reproducible.

Tasks whose deliverables/references carry audio or video files are instead
*routed* per-modality to the judges that read that modality (audio and video are
tracked separately via :func:`dir_media_modalities` and each member's
``handles_audio`` / ``handles_video`` flags), since most judges can't read those
modalities natively — and some (e.g. MiniMax-M3) read video but not audio.

This module is intentionally connection-agnostic: a :class:`ResolvedJudge`
carries only the upstream coordinates (base URL / model / api key / create
overrides). The rubric scorers build an ``AsyncOpenAI`` client from these; the
comparison scorer wraps them in its own client-bearing ``Judge`` for the
threaded sync path.
"""

from __future__ import annotations

import hashlib
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, TypeVar, Union


@dataclass
class ResolvedJudge:
    """A single panel member resolved to concrete upstream coordinates.

    ``create_overrides`` holds provider-specific generation/reasoning knobs
    (e.g. ``reasoning_effort``, ``extra_body``, ``temperature``) that are merged
    into the ``chat.completions.create`` kwargs via :func:`merge_create_kwargs`.
    A ``None`` value there *removes* the default key (so a reasoning model can
    drop ``temperature``).
    """

    name: str
    base_url: str
    model: str
    api_key: str = "sk-dummy"
    create_overrides: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    # Per-modality media capability, tracked SEPARATELY because a judge may read
    # one but not the other:
    #   - Gemini (e.g. Gemini 3.1 Pro Preview): audio AND video.
    #   - self-hosted MiniMax-M3: video ONLY (its config has no audio tower).
    #   - GPT / Claude: neither.
    # Tasks are routed to capable members per modality (see the resources
    # server's routing).
    handles_audio: bool = False
    handles_video: bool = False


class _HasWeight(Protocol):
    name: str
    model: str
    weight: float


class _HasAV(Protocol):
    name: str
    handles_audio: bool
    handles_video: bool


_J = TypeVar("_J", bound=_HasWeight)
_A = TypeVar("_A", bound=_HasAV)


# Media extensions the frontier judges cannot read as plain text. Audio and video
# are kept as SEPARATE sets because judge capability is per-modality (e.g. MiniMax-M3
# reads video but not audio) — a task is routed per the modalities it actually
# contains. Kept lowercase, dot-prefixed.
_AUDIO_EXTS = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".wma",
        ".aiff",
        ".aif",
    }
)
_VIDEO_EXTS = frozenset(
    {
        ".mp4",
        ".m4v",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".wmv",
        ".flv",
        ".mpeg",
        ".mpg",
        ".3gp",
    }
)
_AUDIO_VIDEO_EXTS = _AUDIO_EXTS | _VIDEO_EXTS


def make_rng(seed: Optional[int], *parts: Any) -> random.Random:
    """Return a ``random.Random`` seeded deterministically from *seed* + *parts*.

    Used to make per-comparison judge sampling reproducible: callers seed with
    a stable identity (e.g. the task id and reference repeat) so the same task
    always draws the same judges across reruns. When *seed* is ``None`` the
    parts alone determine the stream (still reproducible per task); pass no
    parts for a fully fresh stream. Parts may be any value — they are stringified
    before hashing, so ints/enums work as identity components too.
    """
    payload = "|".join([repr(seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def sample_judge(judges: Sequence[_J], rng: random.Random) -> _J:
    """Sample one judge from the panel using each member's ``weight``.

    Duck-typed on ``.weight`` / ``.name`` so it works for both
    :class:`ResolvedJudge` (rubric path) and the comparison scorer's
    client-bearing ``Judge``. Non-positive total weight falls back to a uniform
    choice.
    """
    if not judges:
        raise ValueError("sample_judge requires a non-empty judge panel")
    if len(judges) == 1:
        return judges[0]
    weights = [j.weight if (j.weight and j.weight > 0) else 0.0 for j in judges]
    if sum(weights) <= 0:
        return rng.choice(list(judges))
    return rng.choices(list(judges), weights=weights, k=1)[0]


def merge_create_kwargs(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-judge ``create_overrides`` onto *base* create kwargs.

    A ``None`` override value deletes the key from the result (lets a reasoning
    judge drop ``temperature``); any other value replaces the default.
    """
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def panel_summary(judges: Sequence[_HasWeight]) -> List[Dict[str, Any]]:
    """A small JSON-friendly description of the panel for verify responses."""
    return [{"name": j.name, "model": j.model, "weight": j.weight} for j in judges]


def is_audio_file(name: str) -> bool:
    return Path(name).suffix.lower() in _AUDIO_EXTS


def is_video_file(name: str) -> bool:
    return Path(name).suffix.lower() in _VIDEO_EXTS


def is_audio_video_file(name: str) -> bool:
    return Path(name).suffix.lower() in _AUDIO_VIDEO_EXTS


def _has_av_ext(name: str) -> bool:  # backwards-compat alias
    return is_audio_video_file(name)


def dir_media_modalities(dir_path: Optional[Union[str, Path]]) -> Set[str]:
    """The set of media modalities present under *dir_path* (recursively).

    Returns a subset of ``{"audio", "video"}``. Audio and video are reported
    separately so callers can route per-modality (a judge may read video but not
    audio). Zip archives are peeked into — a deliverable is often shipped as a
    single ``.zip`` whose members include the media. A missing/None path or an
    unreadable archive contributes nothing rather than raising.
    """
    modalities: Set[str] = set()
    if not dir_path:
        return modalities
    root = Path(dir_path)
    if not root.is_dir():
        # Allow a single-file path (used by tests / single-deliverable checks).
        if root.is_file():
            if is_audio_file(root.name):
                modalities.add("audio")
            if is_video_file(root.name):
                modalities.add("video")
        return modalities

    def _classify(name: str) -> None:
        if is_audio_file(name):
            modalities.add("audio")
        if is_video_file(name):
            modalities.add("video")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        _classify(path.name)
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist():
                        _classify(member)
            except (zipfile.BadZipFile, OSError):
                continue
        if modalities == {"audio", "video"}:
            break
    return modalities


def dir_contains_audio_video(dir_path: Optional[Union[str, Path]]) -> bool:
    """True when *dir_path* holds any audio/video file (recursively).

    Thin wrapper over :func:`dir_media_modalities` for callers that only need a
    boolean. A single-file path is also accepted.
    """
    if dir_path is None:
        return False
    root = Path(dir_path)
    if root.is_file():
        return is_audio_video_file(root.name)
    return bool(dir_media_modalities(dir_path))


def select_av_judges(judges: Sequence[_A]) -> List[_A]:
    """The AV-capable subset of *judges*, or the full panel as a fallback.

    "AV-capable" means a member reads audio OR video. When no member reads either
    we return every judge so downstream sampling still has a non-empty panel to
    draw from (grading with a text-only judge beats failing the task outright).
    """
    capable = [
        j for j in judges if getattr(j, "handles_audio", False) or getattr(j, "handles_video", False)
    ]
    return capable if capable else list(judges)
