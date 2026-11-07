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
"""The line formats of a model-call capture file, and how the delta format is written and read back.

The ``full`` format stores each exchange as one JSON line, request history and all. An agent re-sends its whole
conversation on every call, so a rollout's capture grows with the square of its length.

The opt-in ``delta`` format stores each line's request history (Responses ``input``, Chat or Messages ``messages``)
as the new items only, plus the ``model_call_id`` of an earlier line of the same file whose history shares the
first ``shared_prefix`` items. Other request fields equal to that line's (``tools``, ``instructions``) are named
in ``inherited_keys`` and left ``null``. Base64 ``data:`` URLs of at least ``BLOB_MIN_CHARS`` characters are
stored once per rollout under ``blobs/<rollout_id>/<sha256>`` next to the captures; the line lists the JSON path
of each one it replaced and leaves ``null`` there. All of this sits under one ``capture_format`` key, which a
``full`` line never has, so no value inside an exchange can be mistaken for a reference.

Readers rebuild the full exchange. Consecutive calls share the item objects of their common prefix, and each
blob is read once, so a fully read rollout holds about its unique content rather than every copy of it.
"""

import hashlib
import os
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import orjson

from nemo_gym.utils.data_url_blobs import (
    BLOB_MIN_CHARS,  # noqa: F401  (part of this module's interface)
    SHA256_HEX_PATTERN,
    extract_data_urls,
    write_blob,
)


CaptureFormat = Literal["full", "delta"]

CAPTURE_FORMAT_KEY = "capture_format"
DELTA_FORMAT_VERSION = 2
BLOB_DIR_NAME = "blobs"
# The request fields that hold a conversation's history, by dialect: Responses, then Chat and Messages.
HISTORY_KEYS = ("input", "messages")

# Earlier calls of one rollout a writer compares a new request against. Calls interleave (subagents,
# judges, user simulators), so the base is the best prefix match among these, not just the last call.
_RECENT_CALLS_PER_ROLLOUT = 6
# Rollouts whose recent calls a writer remembers. Past this the least recently written rollout is forgotten,
# and its next call is stored in full.
_RECENT_ROLLOUTS = 1024
_DIGEST_SIZE = 16
# Request fields other than the history are inherited from the base when equal and at least this long.
_INHERIT_MIN_CHARS = 256


class CaptureDecodeError(ValueError):
    """A capture line names a base call, blob or format this reader cannot resolve."""


def history_key(request: Any) -> Optional[str]:
    """The request field holding the conversation history, when it is a list."""
    if not isinstance(request, dict):
        return None
    return next((key for key in HISTORY_KEYS if isinstance(request.get(key), list)), None)


def _digest(value: Any) -> bytes:
    return hashlib.blake2b(orjson.dumps(value, default=str), digest_size=_DIGEST_SIZE).digest()


def _common_prefix_items(left: bytes, right: bytes) -> int:
    """The number of leading item digests two concatenated digest strings share."""
    low, high = 0, min(len(left), len(right)) // _DIGEST_SIZE
    while low < high:
        middle = (low + high + 1) // 2
        end = middle * _DIGEST_SIZE
        if left[:end] == right[:end]:
            low = middle
        else:
            high = middle - 1
    return low


def _inheritable(value: Any) -> bool:
    return (isinstance(value, (list, dict)) and bool(value)) or (
        isinstance(value, str) and len(value) >= _INHERIT_MIN_CHARS
    )


@dataclass(frozen=True, slots=True)
class _WrittenCall:
    """What a writer remembers of a call it wrote: enough to verify the line and match a later prefix."""

    model_call_id: str
    offset: int
    history_key: str
    item_digests: bytes
    value_digests: dict[str, bytes]


@dataclass(slots=True)
class PreparedExchange:
    """Digests of one exchange's request, computed before the file lock is taken."""

    exchange: dict[str, Any]
    history_key: Optional[str] = None
    item_digests: bytes = b""
    value_digests: dict[str, bytes] = field(default_factory=dict)


class DeltaCaptureWriter:
    """Encodes exchanges in the delta format. One per ``CaptureStore`` and process.

    A writer only ever names a base it wrote itself, and checks under the file lock that the base line is still at
    the offset it was written to, so a capture deleted and recreated for a retry, or written by other processes in
    between, never resolves to the wrong line. Without a usable base the line stores the whole history.
    """

    def __init__(self, blob_root: Path) -> None:
        self._blob_root = blob_root
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, deque[_WrittenCall]] = OrderedDict()

    def prepare(self, exchange: dict[str, Any]) -> PreparedExchange:
        """Hash the request's history items and large fields; the costly part, done outside the file lock."""
        prepared = PreparedExchange(exchange=exchange)
        request = exchange.get("request")
        key = history_key(request)
        if key is None:
            return prepared
        prepared.history_key = key
        prepared.item_digests = b"".join(_digest(item) for item in request[key])
        prepared.value_digests = {
            name: _digest(value) for name, value in request.items() if name != key and _inheritable(value)
        }
        return prepared

    def encode(self, rollout_id: str, prepared: PreparedExchange, fd: int) -> bytes:
        """The line to append, holding the file lock on ``fd``; blobs it references are durable on return."""
        base, shared = self._choose_base(rollout_id, prepared, fd)
        exchange = prepared.exchange
        spec: dict[str, Any] = {"version": DELTA_FORMAT_VERSION}
        line: dict[str, Any] = dict(exchange)
        if base is not None:
            key = prepared.history_key
            assert key is not None
            request = dict(exchange["request"])
            request[key] = request[key][shared:]
            inherited = [
                name for name, digest in prepared.value_digests.items() if base.value_digests.get(name) == digest
            ]
            for name in inherited:
                request[name] = None
            line["request"] = request
            spec.update(base_model_call_id=base.model_call_id, history_key=key, shared_prefix=shared)
            if inherited:
                spec["inherited_keys"] = inherited
        blobs: list[list[Any]] = []
        line = extract_data_urls(line, [], blobs)
        if blobs:
            directory = self._blob_root / rollout_id
            for _, digest, text in blobs:
                write_blob(directory, digest, text)
            spec["blobs"] = [[path, digest] for path, digest, _ in blobs]
        line[CAPTURE_FORMAT_KEY] = spec
        return orjson.dumps(line, default=str, option=orjson.OPT_APPEND_NEWLINE)

    def remember(self, rollout_id: str, prepared: PreparedExchange, offset: int) -> None:
        """Record a line that is now durable at ``offset`` as a candidate base for later calls."""
        model_call_id = prepared.exchange.get("model_call_id")
        if prepared.history_key is None or not isinstance(model_call_id, str) or not model_call_id:
            return
        written = _WrittenCall(
            model_call_id=model_call_id,
            offset=offset,
            history_key=prepared.history_key,
            item_digests=prepared.item_digests,
            value_digests=prepared.value_digests,
        )
        with self._lock:
            recent = self._recent.pop(rollout_id, None) or deque(maxlen=_RECENT_CALLS_PER_ROLLOUT)
            recent.append(written)
            self._recent[rollout_id] = recent
            while len(self._recent) > _RECENT_ROLLOUTS:
                self._recent.popitem(last=False)

    def _choose_base(self, rollout_id: str, prepared: PreparedExchange, fd: int) -> tuple[Optional[_WrittenCall], int]:
        if prepared.history_key is None:
            return None, 0
        with self._lock:
            recent = list(self._recent.get(rollout_id, ()))
        candidates = sorted(
            (
                (_common_prefix_items(prepared.item_digests, call.item_digests), position, call)
                for position, call in enumerate(recent)
                if call.history_key == prepared.history_key
            ),
            key=lambda candidate: (candidate[0], candidate[1]),
            reverse=True,
        )
        for shared, _, call in candidates:
            if shared == 0:
                break
            if _line_starts_with_call(fd, call):
                return call, shared
        return None, 0


def _line_starts_with_call(fd: int, call: _WrittenCall) -> bool:
    """Whether the line at ``call.offset`` is still the one this writer wrote there."""
    try:
        head = os.pread(fd, 128, call.offset)
    except OSError:
        return False
    return head.startswith(b"{") and orjson.dumps(call.model_call_id) in head


class CaptureReader:
    """Rebuilds the exchanges of one rollout's capture file, line by line in file order.

    With ``payloads`` the reader returns full exchanges. It keeps each delta line's rebuilt request so later lines
    can extend it, shares the prefix items of a full-format line with the earlier call it continues, and reads each
    blob once. Without ``payloads`` it returns metadata views: the request keeps only its small fields (no history,
    tools or instructions) and ``request_raw`` is dropped, so nothing grows with the conversation.
    """

    def __init__(self, blob_dir: Path, *, payloads: bool = True) -> None:
        self._blob_dir = blob_dir
        self._payloads = payloads
        self._requests: dict[str, dict[str, Any]] = {}
        self._blobs: dict[str, str] = {}
        self._recent_histories: deque[tuple[str, list[Any]]] = deque(maxlen=_RECENT_CALLS_PER_ROLLOUT)

    def decode(self, line: dict[str, Any]) -> dict[str, Any]:
        """The exchange a parsed line stores; raises ``CaptureDecodeError`` when it cannot be rebuilt."""
        if CAPTURE_FORMAT_KEY not in line:
            return self._full_line(line)
        spec = line.pop(CAPTURE_FORMAT_KEY)
        if not isinstance(spec, dict) or spec.get("version") != DELTA_FORMAT_VERSION:
            raise CaptureDecodeError(f"unsupported capture format {spec!r}")
        request = line.get("request")
        base_id = spec.get("base_model_call_id")
        inherited = spec.get("inherited_keys") or []
        key = spec.get("history_key") if base_id is not None else history_key(request)
        if base_id is not None and (not isinstance(request, dict) or key not in HISTORY_KEYS):
            raise CaptureDecodeError("a delta line needs a request with a history field")
        if not isinstance(inherited, list) or not all(isinstance(name, str) for name in inherited):
            raise CaptureDecodeError("inherited_keys must be a list of field names")
        skipped = {key, *inherited} if not self._payloads else set()
        for entry in spec.get("blobs") or []:
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], list):
                raise CaptureDecodeError("malformed blob reference")
            path, digest = entry
            if path[:1] == ["request"] and len(path) > 1 and path[1] in skipped:
                continue
            _set_path(line, path, self._blob(digest))
        if not self._payloads:
            return _metadata_view(line, drop={key, *inherited})
        if base_id is not None:
            base = self._requests.get(base_id) if isinstance(base_id, str) else None
            if base is None:
                raise CaptureDecodeError(f"base call {base_id!r} is not an earlier readable line of this capture")
            shared = spec.get("shared_prefix")
            base_items = base.get(key)
            if not isinstance(base_items, list) or type(shared) is not int or not 0 <= shared <= len(base_items):
                raise CaptureDecodeError("shared_prefix does not fit the base call's history")
            if not isinstance(request.get(key), list):
                raise CaptureDecodeError("a delta line's history must be a list")
            request[key] = base_items[:shared] + request[key]
            for name in inherited:
                if name not in base:
                    raise CaptureDecodeError(f"inherited field {name!r} is not on the base call")
                request[name] = base[name]
        model_call_id = line.get("model_call_id")
        if isinstance(request, dict) and isinstance(model_call_id, str):
            # A copy, so a caller replacing a field of the returned request cannot change what later lines extend.
            self._requests[model_call_id] = dict(request)
        return line

    def _full_line(self, line: dict[str, Any]) -> dict[str, Any]:
        request = line.get("request")
        key = history_key(request)
        if not self._payloads:
            return _metadata_view(line, drop={key})
        if key is not None:
            items = request[key]
            # Reuse the item objects of the earlier call this one continues, so equal items are held once.
            best: list[Any] = []
            best_shared = 0
            for previous_key, previous in reversed(self._recent_histories):
                if previous_key != key:
                    continue
                shared = 0
                for left, right in zip(items, previous):
                    if left is not right and left != right:
                        break
                    shared += 1
                if shared > best_shared:
                    best, best_shared = previous, shared
                if shared == len(previous):
                    break
            if best_shared:
                items[:best_shared] = best[:best_shared]
            self._recent_histories.append((key, items))
        return line

    def _blob(self, digest: Any) -> str:
        if not isinstance(digest, str) or not SHA256_HEX_PATTERN.fullmatch(digest):
            raise CaptureDecodeError(f"malformed blob digest {digest!r}")
        text = self._blobs.get(digest)
        if text is None:
            try:
                text = (self._blob_dir / digest).read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError) as error:
                raise CaptureDecodeError(f"blob {digest} is unreadable: {error}") from error
            self._blobs[digest] = text
        return text


def _metadata_view(line: dict[str, Any], *, drop: set[Optional[str]]) -> dict[str, Any]:
    """The line without its request payload: the request keeps only its small fields, whatever the format."""
    line.pop("request_raw", None)
    request = line.get("request")
    if isinstance(request, dict):
        line["request"] = {
            name: value for name, value in request.items() if name not in drop and not _inheritable(value)
        }
    return line


def _set_path(value: Any, path: list[Any], replacement: str) -> None:
    if not path:
        raise CaptureDecodeError("a blob path cannot be empty")
    target = value
    for step in path[:-1]:
        target = _step(target, step)
    last = path[-1]
    if isinstance(target, dict) and isinstance(last, str) and target.get(last, ...) is None:
        target[last] = replacement
    elif isinstance(target, list) and type(last) is int and 0 <= last < len(target) and target[last] is None:
        target[last] = replacement
    else:
        raise CaptureDecodeError(f"blob path {path!r} does not name a placeholder")


def _step(value: Any, step: Any) -> Any:
    if isinstance(value, dict) and isinstance(step, str) and step in value:
        return value[step]
    if isinstance(value, list) and type(step) is int and 0 <= step < len(value):
        return value[step]
    raise CaptureDecodeError(f"blob path step {step!r} does not resolve")
