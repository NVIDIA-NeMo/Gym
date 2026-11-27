# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the audio benchmarks that the ``audio_eval`` resources server scores.

A benchmark's ``prepare.py`` downloads its public sources, builds one source row per example, and calls
:func:`build_row` for every profile in its ``profiles.json``. :func:`write_jsonl` writes all rows to the single
combined JSONL that Gym's benchmark declaration points at.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from resources_servers.audio_eval.profiles import load_profiles, source_hash  # noqa: F401  (re-exported)


# Sampling controls stored in each row's request. A profile selects one with ``"request": "<preset>"`` or
# ``"request": {"preset": "<preset>", "max_output_tokens": 4096}``; the extra keys override the preset.
_SAMPLING = {
    "frequency_penalty": 0.0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
}
REQUEST_PRESETS: dict[str, dict[str, Any]] = {
    "greedy": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 256,
        "seed": 0,
        "extra_body": {**_SAMPLING, "chat_template_kwargs": {"enable_thinking": False}},
    },
    "reasoning": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_output_tokens": 20480,
        "seed": 0,
        "extra_body": {
            **_SAMPLING,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 16384},
            "thinking_token_budget": 17408,
            "top_k": -1,
        },
    },
}


def request_controls(contract: dict) -> dict:
    """Resolve a profile's ``request`` (preset name, or preset plus overrides) into explicit controls."""
    request = contract["request"]
    request = {"preset": request} if isinstance(request, str) else dict(request)
    controls = deepcopy(REQUEST_PRESETS[request.pop("preset")])
    controls["extra_body"].update(request.pop("extra_body", {}))
    controls.update(request)
    return controls


def _override_prompts(messages: list[dict], override: dict) -> list[dict]:
    """Apply a profile's ``prompt_override``; ``None`` keeps the source prompt, an empty system prompt drops it."""
    user, system = override.get("user_prompt"), override.get("system_prompt")
    if user is not None:
        (message,) = [m for m in messages if m["role"] == "user"]
        if not isinstance(message["content"], str):
            raise ValueError("user_prompt override requires a string user message")
        message["content"] = user
    has_system = bool(messages) and messages[0]["role"] == "system"
    if system == "":
        return messages[1:] if has_system else messages
    if system is not None:
        if has_system:
            messages[0]["content"] = system
        else:
            messages.insert(0, {"role": "system", "content": system})
    return messages


def _split_audio(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Move audio out of the messages: returns text-only messages and the audio paths in message order."""
    paths: list[str] = []
    text_messages = []
    for message in messages:
        message = dict(message)
        audios = message.pop("audios", None) or [a for a in [message.get("audio")] if a]
        message.pop("audio", None)
        paths.extend(str(audio["path"]) for audio in audios)
        text_messages.append(message)
    return text_messages, paths


def build_row(source: dict, *, profile: str, contract: dict, index: int) -> dict:
    """Build one Gym row from a source row.

    ``source`` holds the example's fields plus ``messages``: chat messages whose audio is attached as
    ``{"audio": {"path": ...}}`` or ``{"audios": [{"path": ...}, ...]}``. It is stored as ``ng_source_row`` (the
    scorer's input) after a JSON round trip, and ``ng_source_row_sha256`` is the digest of that stored form, so the
    digest still matches once the row has been written to JSONL and read back: an integer mapping key is already a
    string, and a NaN or an integer beyond 64 bits (which cannot be hashed and compared reliably) fails here, at
    prepare time, instead of failing every verify of the row. The row's own fields are the source fields, the
    profile's prompt override and request controls applied. ``index`` is the 1-based position of the row within its
    profile; it makes ``sample_id`` unique, and ``task_id`` is the same value, the stable identity Gym keeps for
    resume and comparison when a benchmark is re-prepared, narrowed or reordered.
    """
    source = json.loads(json.dumps(source, allow_nan=False))
    messages, audio_paths = _split_audio(deepcopy(source["messages"]))
    if not audio_paths:
        raise ValueError(f"{profile}:{index} has no audio path")
    messages = _override_prompts(messages, contract.get("prompt_override") or {})

    controls = request_controls(contract)
    metadata = (
        {"audio_path": audio_paths[0]}
        if len(audio_paths) == 1
        else {"audio_paths": json.dumps(audio_paths, ensure_ascii=False)}
    )
    metadata["extra_body"] = json.dumps(
        {**controls["extra_body"], "seed": controls["seed"]}, sort_keys=True, separators=(",", ":")
    )
    row = {key: value for key, value in source.items() if key != "messages"}
    row["responses_create_params"] = {
        "input": messages,
        "metadata": metadata,
        "temperature": controls["temperature"],
        "top_p": controls["top_p"],
        "max_output_tokens": controls["max_output_tokens"],
    }
    row["benchmark_profile"] = profile
    row["sample_id"] = f"{profile}:{index}"
    row["task_id"] = row["sample_id"]
    row["ng_source_row"] = deepcopy(source)
    row["ng_source_row_sha256"] = source_hash(source)
    return row


def profile_rows(profiles_path: Path, sources: Mapping[str, Iterable[dict]] | Sequence[dict]) -> Iterator[dict]:
    """Rows for ``{profile name: source rows}``, built with each profile's contract; ``sample_id`` indexes from 1.

    A plain list of source rows is used for every profile of ``profiles_path``. Mapped sources are consumed lazily,
    one profile after the other in mapping order, so a generator may write audio as it is iterated.
    """
    contracts = load_profiles(profiles_path)["profiles"]
    if not isinstance(sources, Mapping):
        sources = {name: sources for name in contracts}
    for name, rows in sources.items():
        for index, source in enumerate(rows, 1):
            yield build_row(source, profile=name, contract=contracts[name], index=index)


def select(
    value: str | Sequence[str] | None,
    valid: Iterable[str],
    label: str,
    aliases: Mapping[str, Sequence[str]] | None = None,
    *,
    ignore_case: bool = False,
    keep_order: bool = False,
) -> tuple[str, ...]:
    """Normalize a narrowing argument (``None`` = all, a comma-separated string or a list) against ``valid``.

    ``aliases`` maps a whole string value to a named selection (such as ``card``). The result follows ``valid``'s
    order, or the request's order (without repeats) when ``keep_order``. ``label`` names the argument in the error
    for an empty or unknown selection.
    """
    valid = tuple(valid)
    if value is None:
        return valid
    if isinstance(value, str):
        value = aliases[value] if aliases and value in aliases else value.split(",")
    canonical = {name.lower() if ignore_case else name: name for name in valid}
    requested = [str(item).strip() for item in value if str(item).strip()]
    requested = [item.lower() if ignore_case else item for item in requested]
    unknown = sorted(set(requested) - set(canonical))
    if not requested or unknown:
        raise ValueError(f"Empty or unknown {label} {unknown}; valid: {', '.join(valid)}")
    if keep_order:
        return tuple(dict.fromkeys(canonical[item] for item in requested))
    return tuple(name for name in valid if (name.lower() if ignore_case else name) in requested)


def audio_prefix(data_dir: Path, audio_root: str | None = None) -> str:
    """Directory rows use to reference audio written under ``data_dir``.

    Defaults to the absolute local path, where the audio exists. Pass ``audio_root`` only when the model server
    sees the same files elsewhere, such as ``/data`` in a container that mounts ``data_dir`` there.
    """
    return (audio_root or str(data_dir.resolve())).rstrip("/")


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file at ``path``, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, sha256: str) -> Path:
    """Download ``url`` to ``destination`` unless a file with the pinned SHA-256 is already there."""
    if destination.is_file() and sha256_file(destination) == sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".tmp")
    with urllib.request.urlopen(url) as response, staged.open("wb") as sink:
        while chunk := response.read(1 << 20):
            sink.write(chunk)
    if (actual := sha256_file(staged)) != sha256:
        staged.unlink()
        raise ValueError(f"SHA-256 mismatch for {url}: expected {sha256}, got {actual}")
    os.replace(staged, destination)
    return destination


def _stamp_path(path: Path) -> Path:
    return path.with_name(path.name + ".stamp")


def _code_digest(code_files: Iterable[Path]) -> str:
    return hashlib.sha256(b"".join(sha256_file(path).encode() for path in code_files)).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict], *, code_files: Iterable[Path], options: dict | None = None) -> Path:
    """Write ``rows`` atomically, plus a stamp of the code that produced them for :func:`is_current`.

    ``options`` records non-default preparation arguments (a row cap, an audio root); data prepared with them is
    never reported as current, so a cached run cannot silently reuse it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    count = 0
    with staged.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if not count:
        staged.unlink()
        raise ValueError(f"No rows to write to {path}")
    os.replace(staged, path)
    _stamp_path(path).write_text(json.dumps({"code": _code_digest(code_files), "options": options or {}}))
    print(f"Wrote {count} rows to {path}")
    return path


def _plain(value: Any) -> Any:
    """``value`` with Hydra ``ListConfig``/``DictConfig`` containers (a command-line override) as plain lists and dicts."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def write_rows(path: Path, rows: Iterable[dict], *, code_files: Iterable[Path], **options: Any) -> Path:
    """:func:`write_jsonl` that stamps only the truthy ``options`` (the arguments a caller set to non-defaults)."""
    return write_jsonl(
        path, rows, code_files=code_files, options={key: _plain(value) for key, value in options.items() if value}
    )


def currency_check(code_files: Iterable[Path]) -> Callable[[Path], bool]:
    """A benchmark's ``is_prepared_data_current(fpath)``: :func:`is_current` against its ``code_files``."""
    code_files = tuple(code_files)
    return lambda fpath: is_current(fpath, code_files=code_files)


def is_current(path: Path, *, code_files: Iterable[Path]) -> bool:
    """Whether ``path`` was prepared with default options by the current code and profiles."""
    try:
        stamp = json.loads(_stamp_path(Path(path)).read_text())
    except (OSError, ValueError):
        return False
    return stamp.get("code") == _code_digest(code_files) and not stamp.get("options")
