# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parameterized tasks: one template folder plus ``tasks.jsonl``, rendered into plain task folders at prepare.

A template folder is a task folder (``task.toml``, ``environment/``, ``tests/`` or ``verifier.py``, ...) with a
``tasks.jsonl`` beside it, one row per task: ``{"task_id": "...", "task_data": {...}}`` (or the fields flat
beside ``task_id``). Files named ``<stem>.template<suffix>`` (``instruction.template.md``,
``solution/solve.template.sh``, ``solution/answer.template.md``) are rendered with the row's fields,
``{{ task_data.question }}`` or ``{{ question }}``, into ``<stem><suffix>``; every other task file is copied,
hidden files under ``environment/``, ``tests/``, ``solution/`` and ``steps/`` included (``environment/.env``,
``environment/.dockerignore``: the task digest and the sandbox uploads see them, so the render must carry them).
Left out: the editor leftovers and byte-code caches the digest skips (:func:`~nemo_gym.data.tasks.harbor.digest.is_ignored`)
at any depth, and at the template root the hidden entries (``.git``, ``.venv``), ``tasks.jsonl``, the dataset-level
files and prepare's records, plus ``README.md``, which describes the template, not a task (a per-task README is
``README.template.md``). A rendered name that collides with another file of its folder (``instruction.md`` beside
``instruction.template.md``) is refused, naming both.
Each rendered folder gets the row's fields as ``task_data.json``, which the loader reads and the dataset's
``task_data.py`` validates (:mod:`nemo_gym.data.tasks.harbor.task_data`). The rendered ``task.toml`` is the template's
with ``[task].name`` rewritten to ``<org>/<task_id>`` (the template's org): Harbor keys a dataset on that name
(``Task.name``, ``harbor add`` merges folders that share one, trial results and the viewer label by it), so every
rendered task needs its own; a template without a ``[task]`` table renders without one (Harbor then names the task
after its folder). Dataset-level files (``dataset.toml``,
``task_data.py``, ``verifier.py``) are copied beside the rendered folders, which then form an ordinary
dataset the loader, the environment server and the publisher read as files.

Rendering is deterministic, so rendered folders digest the same on every prepare of the same template. The
rendered folder is also the dataset root prepare records into, so a re-render replaces the task folders and the
dataset-level files but keeps prepare's own records there (``manifest.toml``, ``compose-images.json``,
``built-images.json`` and their lock files): an image built for the template is reused, not rebuilt and pushed
again, and a base image's recorded configuration is not re-resolved on every run. The render lands in a hidden
sibling of the output first and is then swapped in entry by entry: a task folder whose rendered content did not
change is left in place, a changed one is replaced with a rename, a stale one removed. An identical re-render
therefore touches no task folder at all, so a server still seeding episodes from an earlier prepare of the same
template (it copies each task folder at seed) never meets a folder that is being deleted and rewritten. The swap
runs under an exclusive lock on a hidden sibling of the output (``.<name>.lock``, the record files' ``flock``), since
``gym dataset validate`` and ``gym eval run`` render one template into one folder and may do so at once: the second
install waits for the first and then finds its folders already in place, rather than racing it entry by entry. A
record writer's in-flight temporary at the root (a hidden ``*.tmp``) is kept like the records themselves, so a prepare
recording into the folder while another renders it never loses the file it is about to install.
"""

from __future__ import annotations

import filecmp
import json
import os
import re
import secrets
import shutil
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nemo_gym.data.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.data.tasks.harbor.digest import GITIGNORE, is_ignored
from nemo_gym.data.tasks.harbor.errors import HarborTaskError
from nemo_gym.data.tasks.harbor.hub import validate_task_name
from nemo_gym.data.tasks.harbor.manifest import MANIFEST_FILE
from nemo_gym.data.tasks.harbor.python_verifier import VERIFIER_FILE
from nemo_gym.data.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE, RECORD_LOCK_SUFFIX, record_lock
from nemo_gym.data.tasks.harbor.task import TASK_FILE, is_task_folder
from nemo_gym.data.tasks.harbor.task_data import TASK_DATA_FILE, TASK_DATA_MODULE, is_reserved_key


TASKS_FILE = "tasks.jsonl"
TEMPLATE_MARKER = ".template"
# Files of the template folder that describe the dataset, not a task: copied beside the rendered folders.
DATASET_FILES = (DATASET_FILE, TASK_DATA_MODULE, VERIFIER_FILE)
# What prepare records at the dataset root, here the rendered folder: kept across renders (with their lock files).
RECORD_FILES = (MANIFEST_FILE, IMAGE_CONFIGS_FILE, BUILT_IMAGES_FILE)
# Files of the template folder that are neither a task file nor a dataset file.
SKIPPED_FILES = (TASKS_FILE, "README.md", *RECORD_FILES)
# Names a rendered task folder cannot take: everything else that lives at the rendered dataset root.
RESERVED_TASK_IDS = frozenset(
    (TASKS_FILE, *DATASET_FILES, *RECORD_FILES, *(name + RECORD_LOCK_SUFFIX for name in RECORD_FILES))
)
_PLACEHOLDER = re.compile(r"\{\{\s*(?:task_data\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
# Any other `{{ ... }}` left in a rendered file, or a `{{` that never closes (`{{ a }` directly before a valid
# placeholder): a misspelt placeholder, refused rather than shipped to the agent.
_LEFTOVER = re.compile(r"\{\{.*?\}\}|\{\{[^\n]*", re.DOTALL)
# A task_id is one folder name: 255 bytes is NAME_MAX on Linux and macOS, so a longer one fails at mkdir with a raw
# OSError that names the staging path, not the row.
MAX_TASK_ID_BYTES = 255
# Harbor 0.22.0's ORG_NAME_PATTERN: what a `[task].name` must match (Harbor also refuses `..` in it).
_HARBOR_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$")


def is_harbor_name(name: object) -> bool:
    """Whether ``name`` is an ``org/name`` Harbor accepts for a task or a dataset (its ``ORG_NAME_PATTERN``, no ``..``)."""
    return isinstance(name, str) and bool(_HARBOR_NAME.match(name)) and ".." not in name


# A TOML table header, `[name]` or `[[name]]`, with an optional trailing comment; group 1 is the dotted name.
_TABLE_HEADER = re.compile(r"^\[\[?\s*([^\]]*?)\s*\]\]?\s*(?:#.*)?$")
# `name = "..."` (or '...') on a line of its own, with an optional trailing comment: the line the rewrite replaces.
_NAME_LINE = re.compile(r'^(\s*name\s*=\s*)("[^"\\]*"|\'[^\'\\]*\')(\s*(?:#.*)?)$')


class ParameterizedError(HarborTaskError):
    """``tasks.jsonl`` or a template file cannot be rendered."""


def is_template_folder(path: Path) -> bool:
    """A task folder with ``tasks.jsonl`` beside its ``task.toml``."""
    path = Path(path)
    return is_task_folder(path) and (path / TASKS_FILE).is_file()


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    task_data: dict[str, Any]


def read_rows(template: Path) -> list[TaskRow]:
    """The rows of ``<template>/tasks.jsonl``: a valid, unique ``task_id`` each, the fields a JSON object.

    A ``task_id`` is a single folder name (:func:`validate_task_name`) of at most :data:`MAX_TASK_ID_BYTES` bytes, not
    one of the files the rendered dataset root holds (:data:`RESERVED_TASK_IDS`: ``tasks.jsonl``, the dataset-level
    files, prepare's records and their lock files) and not hidden (a leading ``.``), since task discovery skips hidden
    folders.
    """
    path = Path(template) / TASKS_FILE
    rows: list[TaskRow] = []
    seen: set[str] = set()
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ParameterizedError(f"{path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ParameterizedError(f"{path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ParameterizedError(f"{path}:{number}: a row must be a JSON object, got {type(row).__name__}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ParameterizedError(f"{path}:{number}: a row needs a non-empty string `task_id`")
        try:
            validate_task_name(task_id, error=ParameterizedError)
        except ParameterizedError as exc:
            raise ParameterizedError(f"{path}:{number}: {exc}") from exc
        if (length := len(task_id.encode("utf-8"))) > MAX_TASK_ID_BYTES:
            raise ParameterizedError(
                f"{path}:{number}: task_id {task_id[:40]!r}... is {length} bytes long; a task folder name holds at most "
                f"{MAX_TASK_ID_BYTES}"
            )
        if task_id in RESERVED_TASK_IDS:
            raise ParameterizedError(
                f"{path}:{number}: task_id {task_id!r} is reserved: the rendered dataset root keeps a file of that name"
            )
        if task_id.startswith("."):
            raise ParameterizedError(
                f"{path}:{number}: task_id {task_id!r} cannot start with '.': hidden folders are not task folders"
            )
        if task_id in seen:
            raise ParameterizedError(f"{path}:{number}: duplicate task_id {task_id!r}")
        seen.add(task_id)
        if "task_data" in row:
            fields = row["task_data"]
            extra = sorted(set(row) - {"task_id", "task_data"})
            if extra:
                raise ParameterizedError(
                    f"{path}:{number}: a row with a `task_data` object carries nothing else, got {extra}"
                )
        else:
            fields = {key: value for key, value in row.items() if key != "task_id"}
        if not isinstance(fields, dict):
            raise ParameterizedError(
                f"{path}:{number}: `task_data` must be a JSON object, got {type(fields).__name__}"
            )
        reserved = sorted(key for key in fields if is_reserved_key(key))
        if reserved:
            raise ParameterizedError(f"{path}:{number}: keys {reserved} are reserved")
        rows.append(TaskRow(task_id=task_id, task_data=fields))
    if not rows:
        raise ParameterizedError(f"{path} has no rows")
    return rows


def render_text(text: str, fields: Mapping[str, Any], *, where: str) -> str:
    """Substitute ``{{ task_data.key }}`` and ``{{ key }}`` with the row's values; an unknown key is an error.

    So is any other ``{{ ... }}`` still in the text once the placeholders are filled (``{{ task_data.q.upper }}``,
    ``{{ q | upper }}``, ``{{ task-data.q }}``): only the two spellings above are placeholders, and a misspelt one
    would otherwise reach the agent, or ``solution/answer.md``, as raw text. The check runs over the input text with
    the placeholders removed, never over the substituted values, so a field whose value contains ``{{`` renders. A
    ``{{`` that never closes (``{{ a }`` directly before a valid placeholder, whose own ``}}`` would otherwise be
    taken for its end) is refused the same way.
    """

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in fields:
            raise ParameterizedError(f"{where}: placeholder {match.group(0)!r} names no field of the row")
        value = fields[key]
        return value if isinstance(value, str) else json.dumps(value)

    leftover = _LEFTOVER.search(_PLACEHOLDER.sub("", text))
    if leftover is not None:
        what = "is not a placeholder" if leftover.group(0).endswith("}}") else "opens a placeholder that never closes"
        raise ParameterizedError(
            f"{where}: {leftover.group(0)!r} {what}; write `{{{{ key }}}}` or `{{{{ task_data.key }}}}` "
            "with a field name, or keep a literal `{{ ... }}` in a file without the .template marker"
        )
    return _PLACEHOLDER.sub(replace, text)


def task_name_for(template_name: str, task_id: str) -> str:
    """``<org>/<task_id>``: the Harbor task name of the row rendered from a template whose ``[task].name`` is given."""
    return f"{template_name.rpartition('/')[0]}/{task_id}"


def rendered_task_toml(text: str, task_id: str, *, where: str) -> str:
    """The template's ``task.toml`` for one row: ``[task].name`` rewritten to ``<org>/<task_id>``, every other byte
    kept; the text unchanged when it has no ``[task].name``.

    The rewrite is line-based (the ``name = "..."`` line under the ``[task]`` header) and checked: the result must
    parse to the template's document with only that name changed, so a name spelled some other way (an inline
    ``task = { name = "..." }`` table, a dotted ``task.name`` key, a multi-line string) is a :class:`ParameterizedError`
    asking for the plain form rather than a mangled file. A template name that is not ``org/name``, or a ``task_id``
    outside the characters Harbor admits in a name, is refused with the row named: Harbor's own check would refuse the
    published folder, and the loader would otherwise run a task Harbor cannot.
    """
    try:
        parsed = tomllib.loads(text)
    except ValueError as exc:
        raise ParameterizedError(f"{where}: {exc}") from exc
    task = parsed.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("name"), str):
        return text
    if not is_harbor_name(task["name"]):
        raise ParameterizedError(
            f"{where}: [task].name {task['name']!r} is not `org/name`; the rendered tasks are named `<org>/<task_id>`"
        )
    name = task_name_for(task["name"], task_id)
    if not is_harbor_name(name):
        raise ParameterizedError(
            f"{where}: task_id {task_id!r} cannot be a Harbor task name ({name!r}): the template's [task].name is "
            "rendered as `<org>/<task_id>`, so a task_id holds alphanumerics, hyphens, underscores and dots, does not "
            "start with a dot and has no `..`"
        )
    lines = text.splitlines(keepends=True)
    in_task = False
    rewritten = False
    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        header = _TABLE_HEADER.match(body.strip())
        if header is not None:
            in_task = header.group(1) == "task" and not body.lstrip().startswith("[[")
            continue
        if in_task and not rewritten and (match := _NAME_LINE.match(body)) is not None:
            lines[index] = f'{match.group(1)}"{name}"{match.group(3)}{line[len(body) :]}'
            rewritten = True
    result = "".join(lines)
    expected = {**parsed, "task": {**task, "name": name}}
    try:
        ok = rewritten and tomllib.loads(result) == expected
    except ValueError:
        ok = False
    if not ok:
        raise ParameterizedError(
            f"{where}: [task].name could not be rewritten to {name!r} for the rendered task; write it as a `[task]` "
            'table with `name = "org/name"` on a line of its own'
        )
    return result


def rendered_name(name: str) -> str | None:
    """``instruction.md`` for ``instruction.template.md``; ``None`` for a file that is not a template."""
    if TEMPLATE_MARKER not in name:
        return None
    stem, marker, suffix = name.rpartition(TEMPLATE_MARKER)
    if not stem or (suffix and not suffix.startswith(".")):
        return None
    return stem + suffix


def _render_tree(
    source: Path, target: Path, fields: Mapping[str, Any], *, task_id: str, skip: set[str], at_root: bool = True
) -> None:
    """Render ``source`` into ``target``: templates filled, every other entry copied with its mode.

    What the digest skips (``__pycache__``, ``*.pyc``, editor leftovers) is left out at every depth; hidden entries
    only at the template root (``.git``, ``.venv``, an editor's folder), since below it a hidden file is task content
    (``environment/.env``, ``environment/.dockerignore``) that the digest covers and the sandbox receives. The root
    ``.gitignore`` is the exception: Harbor's packager reads it as the task's ignore rules (``harbor task init`` writes
    one into every task), so every rendered task carries the template's, and publish ships it on. Two entries
    that would land under one name (a plain ``instruction.md`` beside ``instruction.template.md``) are refused, and so
    is a root entry that would land under ``task_data.json`` (:func:`_refuse_task_data_twins`, checked before any row).
    """
    written: dict[str, Path] = {}
    for child in sorted(source.iterdir()):
        hidden_root = at_root and child.name.startswith(".") and child.name != GITIGNORE
        if child.name in skip or is_ignored(child.name) or hidden_root:
            continue
        if child.is_symlink():
            raise ParameterizedError(f"{child}: links are not rendered; copy the file into the template")
        plain = None if child.is_dir() else rendered_name(child.name)
        name = child.name if plain is None else plain
        if name in written:
            raise ParameterizedError(
                f"task {task_id!r}: {written[name]} and {child} would both be written as {name!r} in the rendered "
                "folder; keep one of them"
            )
        written[name] = child
        if child.is_dir():
            (target / name).mkdir()
            _render_tree(child, target / name, fields, task_id=task_id, skip=set(), at_root=False)
            continue
        if plain is None:
            shutil.copyfile(child, target / name)
            shutil.copymode(child, target / name)
            continue
        try:
            text = child.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise ParameterizedError(f"{child}: {exc}") from exc
        (target / name).write_text(render_text(text, fields, where=f"task {task_id!r}, {child.name}"))
        shutil.copymode(child, target / name)


def _refuse_task_data_twins(template: Path) -> None:
    """Refuse a ``task_data.json`` (or a ``task_data.template.json``) at the template root.

    The rendered task's ``task_data.json`` is written from its row once the tree is rendered, so a file the template
    ships under that name would be copied or rendered and then replaced without a word, the author's fields discarded.
    Below the root the name is task content (a ``tests/`` fixture) and is copied like any other file.
    """
    for child in sorted(template.iterdir()):
        if child.name == TASK_DATA_FILE or (child.is_file() and rendered_name(child.name) == TASK_DATA_FILE):
            raise ParameterizedError(
                f"{child}: a template has no {TASK_DATA_FILE} of its own; the rendered task's {TASK_DATA_FILE} is "
                f"written from its {TASKS_FILE} row, so put the fields in each row ({TASK_DATA_MODULE} types them)"
            )


def is_record_file(name: str) -> bool:
    """A record prepare writes at the dataset root, its lock file, or a record writer's in-flight temporary: kept when
    the rendered folder is replaced.

    The temporaries are the hidden ``*.tmp`` names the writers use (``.<record>.<pid>.tmp`` from
    :func:`~nemo_gym.data.tasks.harbor.records.write_json_record`, ``.manifest-<pid>-<thread>-<hex>.tmp`` from the
    manifest's writer); the renderer writes no such name itself. A writer in another process ``fsync``\\ s and then
    ``os.replace``\\ s its temporary over the record, and loses the file to an overlapping render otherwise.
    """
    if name in RECORD_FILES:
        return True
    if name.endswith(RECORD_LOCK_SUFFIX) and name[: -len(RECORD_LOCK_SUFFIX)] in RECORD_FILES:
        return True
    return name.startswith(".") and name.endswith(".tmp")


def _same_tree(a: Path, b: Path) -> bool:
    """Whether two rendered trees hold the same entries, byte for byte and mode for mode (no links in either)."""
    names = sorted(p.name for p in a.iterdir())
    if names != sorted(p.name for p in b.iterdir()):
        return False
    for name in names:
        x, y = a / name, b / name
        if x.is_symlink() or y.is_symlink() or x.is_dir() != y.is_dir():
            return False
        if x.is_dir():
            if not _same_tree(x, y):
                return False
        elif stat.S_IMODE(x.stat().st_mode) != stat.S_IMODE(y.stat().st_mode) or not filecmp.cmp(x, y, shallow=False):
            return False
    return True


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _move_into_place(fresh: Path, target: Path) -> None:
    """Rename ``fresh`` to ``target``, whatever is there now: a file over a file atomically, otherwise the old entry is
    renamed aside first and removed after the new one landed, so the gap is one rename wide, not a tree's rewrite."""
    aside = None
    if (target.is_dir() and not target.is_symlink()) or (fresh.is_dir() and (target.exists() or target.is_symlink())):
        # Not derived from the target's name: a task_id may use the whole 255 bytes a folder name allows.
        aside = target.with_name(f".stale-{secrets.token_hex(4)}")
        os.replace(target, aside)
    os.replace(fresh, target)
    if aside is not None:
        _remove(aside)


def _install_rendered(fresh: Path, output: Path) -> None:
    """Swap the entries of ``fresh`` into ``output``: stale entries go (records stay), unchanged task folders and files
    are left in place, changed ones are replaced by rename."""
    wanted = {child.name for child in fresh.iterdir()}
    for child in list(output.iterdir()):
        if not is_record_file(child.name) and child.name not in wanted:
            _remove(child)
    for child in sorted(fresh.iterdir()):
        target = output / child.name
        if child.is_dir():
            if target.is_dir() and not target.is_symlink() and _same_tree(child, target):
                continue
        elif (
            target.is_file()
            and not target.is_symlink()
            and stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(child.stat().st_mode)
            and filecmp.cmp(child, target, shallow=False)
        ):
            continue
        _move_into_place(child, target)


def render_template_folder(template: Path, output: Path) -> Path:
    """Render every row of ``<template>/tasks.jsonl`` into ``<output>/<task_id>/`` and return ``output``.

    Each rendered ``task.toml`` carries the row's own ``[task].name``, ``<org>/<task_id>``
    (:func:`rendered_task_toml`), when the template has one. Everything in ``output`` is replaced, so a row removed
    from ``tasks.jsonl`` leaves no stale folder behind, except prepare's records (``RECORD_FILES`` and their lock files), which stay: ``output`` is the dataset root
    those records live in, and a built image or a resolved base image must survive the next prepare. Dataset-level
    files are copied beside the task folders. The rows are rendered into a hidden sibling of ``output`` first and
    swapped in afterwards (:func:`_install_rendered`), under an exclusive lock on ``output``'s hidden sibling lock file
    (``.<name>.lock``) so that two commands rendering one template into one folder install one after the other: an
    entry whose content did not change is left in place, so an identical re-render leaves every task folder present
    throughout, and a render that fails leaves ``output`` as it was. Raises :class:`ParameterizedError` (a
    :class:`HarborTaskError`) for a malformed row or template.
    """
    template = Path(template).resolve()
    if not is_template_folder(template):
        raise ParameterizedError(f"{template} is not a template folder ({TASK_FILE} with {TASKS_FILE} beside it)")
    rows = read_rows(template)
    _refuse_task_data_twins(template)
    output = Path(output).resolve()
    if output == template or template in output.parents:
        raise ParameterizedError(f"Rendered tasks cannot be written inside the template folder ({output})")
    if output.exists() and not output.is_dir():
        output.unlink()
    output.mkdir(parents=True, exist_ok=True)
    try:
        task_toml = (template / TASK_FILE).read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ParameterizedError(f"{template / TASK_FILE}: {exc}") from exc
    fresh = output.with_name(f".{output.name}.render-{os.getpid()}-{secrets.token_hex(4)}")
    fresh.mkdir()
    try:
        for name in DATASET_FILES:
            if (template / name).is_file():
                shutil.copyfile(template / name, fresh / name)
        skip = set(DATASET_FILES) | set(SKIPPED_FILES)
        for row in rows:
            folder = fresh / row.task_id
            try:
                folder.mkdir()
            except OSError as exc:
                # The row passed read_rows; what is left is this filesystem's (a shorter name limit, a full disk).
                raise ParameterizedError(
                    f"{template / TASKS_FILE}: task {row.task_id[:40]!r}: its folder cannot be created: {exc.strerror}"
                ) from exc
            _render_tree(template, folder, row.task_data, task_id=row.task_id, skip=skip)
            (folder / TASK_FILE).write_text(
                rendered_task_toml(task_toml, row.task_id, where=f"task {row.task_id!r}, {template / TASK_FILE}")
            )
            (folder / TASK_DATA_FILE).write_text(json.dumps(row.task_data, sort_keys=True) + "\n")
        # `record_lock` locks `<path>.lock`: here `.<name>.lock` beside the output, outside the folder it guards.
        with record_lock(output.with_name(f".{output.name}")):
            _install_rendered(fresh, output)
    finally:
        shutil.rmtree(fresh, ignore_errors=True)
    return output
