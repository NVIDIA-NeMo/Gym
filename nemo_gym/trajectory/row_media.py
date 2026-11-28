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
"""Rollout rows that store each large ``data:`` URL once, by reference (``row_media: reference``).

A finished row can hold the same screenshot several times: in the agent's observation records, in the
trajectory's invocation conversation and in its tool call outputs, and, with inline model-call payloads, in every
later request. With ``row_media: reference`` the evidence keys of a row (``ROW_MEDIA_KEYS``) keep ``null`` where a
data URL of at least ``BLOB_MIN_CHARS`` characters was, the URL is written once per rollout as a blob named by its
SHA-256, and the row lists every replaced place under ``ng_row_media``, outside the payload:

.. code-block:: json

    {"ng_trajectory": {"tool_calls": [{"output": [{"type": "input_image", "image_url": null}]}]},
     "ng_row_media": {"version": 1, "root": "model_call_capture_dir", "dir": "blobs/3-0",
                      "blobs": [[["ng_trajectory", "tool_calls", 0, "output", 0, "image_url"], "9f2c...e1"]]}}

``root`` says which directory ``dir`` is relative to: the run's ``model_call_capture_dir``, whose ``blobs/<rollout
id>`` the delta capture format also uses, so an image the model was sent is not stored again; or the directory of
the rollouts file when the run has no capture dir. The harness output (``response``) and the task input are never
changed. :func:`restore_row_media` gives back the row exactly as it was before.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    NG_TRAJECTORY_KEY,
    NG_VERIFIER_INVOCATIONS_KEY,
    ObservationModel,
)
from nemo_gym.utils.data_url_blobs import BLOB_MIN_CHARS, SHA256_HEX_PATTERN, extract_data_urls, write_blob


RowMedia = Literal["inline", "reference"]
RowMediaRoot = Literal["model_call_capture_dir", "rollouts_dir"]

NG_ROW_MEDIA_KEY = "ng_row_media"
ROW_MEDIA_VERSION = 1
CAPTURE_ROOT: RowMediaRoot = "model_call_capture_dir"
ROLLOUTS_ROOT: RowMediaRoot = "rollouts_dir"
# The evidence keys Gym attaches to a row. The harness output and the task input stay as the harness returned them:
# verification recovery and training read them.
ROW_MEDIA_KEYS = (NG_AGENT_OBSERVATIONS_KEY, NG_TRAJECTORY_KEY, "ng_model_call_capture", NG_VERIFIER_INVOCATIONS_KEY)
# The capture store's blob directory name; a row's blobs there share the delta capture format's files.
CAPTURE_BLOB_DIR_NAME = "blobs"
_DIR_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


class RowMediaConfig(BaseModel):
    """The run-wide ``row_media`` setting from Gym's global config: ``inline`` (the default) or ``reference``."""

    row_media: RowMedia = "inline"


def row_media_from_config(global_config_dict: Any) -> RowMedia:
    """Whether written rows keep large data URLs inline or reference them."""
    return RowMediaConfig.model_validate(global_config_dict).row_media


class RowMediaUnavailable(ValueError):
    """A row's referenced media cannot be read, or a reader that needs it inline was given references."""


class RowMediaReferences(ObservationModel):
    """The ``ng_row_media`` key of a row: where its data URLs are stored and every place each one was."""

    version: Literal[1] = ROW_MEDIA_VERSION
    root: RowMediaRoot
    dir: str = Field(description="This row's blob directory, relative to ``root``.")
    blobs: list[tuple[list[Union[str, int]], str]] = Field(
        description="``[path, sha256]`` per replaced data URL: the path of keys and list indexes from the row."
    )

    @field_validator("dir")
    @classmethod
    def _relative_dir(cls, value: str) -> str:
        parts = value.split("/")
        if not all(_DIR_PART.fullmatch(part) for part in parts):
            raise ValueError(f"row media dir {value!r} must be a relative path of plain names")
        return value

    @field_validator("blobs")
    @classmethod
    def _valid_blobs(cls, value: list[tuple[list[Union[str, int]], str]]) -> list[tuple[list[Union[str, int]], str]]:
        for path, digest in value:
            if not path or not SHA256_HEX_PATTERN.fullmatch(digest):
                raise ValueError("each row media blob needs a nonempty path and a sha256 digest")
        return value


@dataclass(frozen=True)
class RowMediaLocation:
    """Where one rollout's row media goes: a ``root`` kind, that root's directory, and the relative ``dir``."""

    root: RowMediaRoot
    base: Path
    dir: str

    @property
    def directory(self) -> Path:
        return self.base / self.dir


def capture_media_location(capture_dir: Path, rollout_id: str) -> RowMediaLocation:
    """The capture store's blob directory of ``rollout_id``, which delta capture lines also use."""
    return RowMediaLocation(CAPTURE_ROOT, Path(capture_dir), f"{CAPTURE_BLOB_DIR_NAME}/{_dir_part(rollout_id)}")


def rollouts_media_location(rollouts_fpath: Path, rollout_id: str) -> RowMediaLocation:
    """``<dir>/<stem>_blobs/<rollout_id>`` next to the rollouts file, for a run without a capture dir."""
    rollouts_fpath = Path(rollouts_fpath)
    return RowMediaLocation(
        ROLLOUTS_ROOT, rollouts_fpath.parent, f"{_dir_part(rollouts_fpath.stem + '_blobs')}/{_dir_part(rollout_id)}"
    )


def _dir_part(name: str) -> str:
    if not _DIR_PART.fullmatch(name):
        raise ValueError(f"{name!r} cannot name a row media directory")
    return name


@dataclass(frozen=True)
class RowMediaStats:
    """What :func:`reference_row_media` did to one row."""

    references: int = 0
    unique: int = 0
    referenced_chars: int = 0
    written_blobs: int = 0
    written_bytes: int = 0


def row_media_references(record: Any) -> Optional[RowMediaReferences]:
    """The validated ``ng_row_media`` of a row, or ``None`` when its media is inline."""
    if not isinstance(record, dict) or record.get(NG_ROW_MEDIA_KEY) is None:
        return None
    try:
        return RowMediaReferences.model_validate(record[NG_ROW_MEDIA_KEY])
    except ValueError as error:
        raise RowMediaUnavailable(f"{NG_ROW_MEDIA_KEY} is malformed: {error}") from error


def reference_row_media(
    record: dict[str, Any], location: RowMediaLocation, *, min_chars: int = BLOB_MIN_CHARS
) -> RowMediaStats:
    """Store each data URL of the row's evidence keys once under ``location`` and reference it, in place.

    Each distinct URL is written durably before the row changes, so a row never names a blob that is not there;
    blobs already present (another copy in this row, or the delta capture of the same rollout) are not written
    again. The row keeps ``null`` in each replaced place and lists it under ``ng_row_media``, after its other keys.
    A row that already references some media keeps those references and adds the new ones. Raises ``OSError``
    when a blob cannot be written; the row is then unchanged.
    """
    existing = row_media_references(record)
    if existing is not None and (existing.root, existing.dir) != (location.root, location.dir):
        raise ValueError(f"row already references its media in {existing.root}/{existing.dir}")
    found: list[list[Any]] = []
    digests: dict[int, str] = {}
    replaced: dict[str, Any] = {}
    for key in ROW_MEDIA_KEYS:
        if key in record:
            value = extract_data_urls(record[key], [key], found, min_chars=min_chars, digests=digests)
            if value is not record[key]:
                replaced[key] = value
    if not found:
        return RowMediaStats()
    unique: dict[str, str] = {}
    for _, digest, text in found:
        unique.setdefault(digest, text)
    written = written_bytes = 0
    directory = location.directory
    for digest, text in unique.items():
        if write_blob(directory, digest, text):
            written += 1
            written_bytes += len(text)
    record.update(replaced)
    entries = [list(entry) for entry in existing.blobs] if existing is not None else []
    entries.extend([path, digest] for path, digest, _ in found)
    # Re-added last, so the listing follows the payload it describes.
    record.pop(NG_ROW_MEDIA_KEY, None)
    record[NG_ROW_MEDIA_KEY] = {
        "version": ROW_MEDIA_VERSION,
        "root": location.root,
        "dir": location.dir,
        "blobs": entries,
    }
    return RowMediaStats(
        references=len(found),
        unique=len(unique),
        referenced_chars=sum(len(text) for _, _, text in found),
        written_blobs=written,
        written_bytes=written_bytes,
    )


def restore_row_media(
    record: dict[str, Any],
    *,
    capture_dirs: Sequence[Path] = (),
    rollouts_dir: Optional[Path] = None,
    in_place: bool = False,
) -> dict[str, Any]:
    """The row with every referenced data URL back in its place and no ``ng_row_media``: the row as written.

    ``capture_dirs`` are the run's ``model_call_capture_dir`` candidates and ``rollouts_dir`` the directory of the
    rollouts file (:func:`row_media_dirs` finds both from a rollouts file). Without ``in_place`` only the
    containers on a restored path are copied, so the caller's row is unchanged. Each blob is read once and its text
    shared by every place that names it. A row without ``ng_row_media`` is returned as it is. Raises
    ``RowMediaUnavailable`` naming the blob and the directories searched when one cannot be read.
    """
    references = row_media_references(record)
    if references is None:
        return record
    bases = (
        [Path(directory) for directory in capture_dirs]
        if references.root == CAPTURE_ROOT
        else ([Path(rollouts_dir)] if rollouts_dir is not None else [])
    )
    candidates = [base / references.dir for base in bases]
    texts: dict[str, str] = {}

    def text_for(digest: str) -> str:
        text = texts.get(digest)
        if text is None:
            for directory in candidates:
                try:
                    text = (directory / digest).read_bytes().decode("utf-8")
                    break
                except FileNotFoundError:
                    continue
                except (OSError, UnicodeDecodeError) as error:
                    raise RowMediaUnavailable(f"row media blob {directory / digest} is unreadable: {error}") from error
            if text is None:
                searched = ", ".join(str(directory) for directory in candidates) or "no directory"
                raise RowMediaUnavailable(
                    f"row media blob {digest} is not in {searched}. This row keeps its images in the run's "
                    f"{references.root} (row_media: reference); pass that directory to restore_row_media."
                )
            texts[digest] = text
        return text

    restored = record if in_place else dict(record)
    copied = {id(restored)}
    for path, digest in references.blobs:
        target: Any = restored
        for step in path[:-1]:
            child = _step(target, step, path)
            if not isinstance(child, (dict, list)):
                raise RowMediaUnavailable(f"row media path {path!r} does not resolve at {step!r}")
            if not in_place and id(child) not in copied:
                child = child.copy()
                target[step] = child
                copied.add(id(child))
            target = child
        last = path[-1]
        if _step(target, last, path) is not None:
            raise RowMediaUnavailable(f"row media path {path!r} does not name a placeholder")
        target[last] = text_for(digest)
    restored.pop(NG_ROW_MEDIA_KEY, None)
    return restored


def _step(value: Any, step: Union[str, int], path: list[Any]) -> Any:
    if isinstance(value, dict) and isinstance(step, str) and step in value:
        child = value[step]
    elif isinstance(value, list) and type(step) is int and 0 <= step < len(value):
        child = value[step]
    else:
        raise RowMediaUnavailable(f"row media path {path!r} does not resolve at {step!r}")
    return child


def prune_row_media(projection: dict[str, Any]) -> None:
    """Keep only the ``ng_row_media`` entries whose place is still a placeholder in a projection of a row, in place.

    A projection that leaves out payloads (the exporters' view) drops some referenced places; the listing then names
    only what the projection kept, and goes when nothing is left.
    """
    references = projection.get(NG_ROW_MEDIA_KEY)
    if not isinstance(references, dict) or not isinstance(references.get("blobs"), list):
        return
    kept = []
    for entry in references["blobs"]:
        path = entry[0] if isinstance(entry, list) and entry and isinstance(entry[0], list) else None
        if path and _is_placeholder(projection, path):
            kept.append(entry)
    if kept:
        projection[NG_ROW_MEDIA_KEY] = {**references, "blobs": kept}
    else:
        projection.pop(NG_ROW_MEDIA_KEY)


def _is_placeholder(value: Any, path: list[Any]) -> bool:
    for step in path:
        if isinstance(value, dict) and isinstance(step, str) and step in value:
            value = value[step]
        elif isinstance(value, list) and type(step) is int and 0 <= step < len(value):
            value = value[step]
        else:
            return False
    return value is None


def referenced_paths(record: Any, prefixes: Iterable[str] = ROW_MEDIA_KEYS) -> list[list[Any]]:
    """The places of the row's referenced data URLs under the given top-level keys."""
    references = row_media_references(record)
    if references is None:
        return []
    keys = set(prefixes)
    return [list(path) for path, _ in references.blobs if path[0] in keys]


def require_inline_media(record: Any, *, reader: str, prefixes: Iterable[str] = ROW_MEDIA_KEYS) -> None:
    """Raise ``RowMediaUnavailable`` when ``record`` keeps data URLs that ``reader`` reads by reference only."""
    paths = referenced_paths(record, prefixes)
    if paths:
        raise RowMediaUnavailable(
            f"{reader} needs images inline, but this row stores {len(paths)} by reference (row_media: reference), "
            f"for example at {paths[0]!r}. Restore them first with nemo_gym.trajectory.row_media.restore_row_media."
        )


def row_media_dirs(rollouts_fpath: Path) -> tuple[list[Path], Path]:
    """``(capture_dirs, rollouts_dir)`` for :func:`restore_row_media` on rows of ``rollouts_fpath``.

    The capture dir is read from the run's saved config next to the file (``<stem>_resolved_config.yaml``); a file
    without one, such as a ``run_examples`` shard, gets none.
    """
    from nemo_gym.utils.path_utils import resolved_config_path_for

    rollouts_fpath = Path(rollouts_fpath)
    capture_dirs: list[Path] = []
    config_fpath = resolved_config_path_for(rollouts_fpath)
    if config_fpath.is_file():
        from omegaconf import OmegaConf

        try:
            value = OmegaConf.select(OmegaConf.load(config_fpath), "model_call_capture_dir")
        except Exception:
            value = None
        if isinstance(value, str) and value:
            capture_dirs.append(Path(value))
    return capture_dirs, rollouts_fpath.parent
