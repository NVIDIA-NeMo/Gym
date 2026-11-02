# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym dataset publish --harbor``: a Gym-authored dataset as plain Harbor task folders.

Expands a parameterized template into its task folders, writes the ``tests/test.sh`` shim for every task
graded by a Python verifier (:mod:`nemo_gym.tasks.harbor.python_verifier`) and a ``solution/solve.sh`` for
every ``solution/answer.md``, writes ``dataset.toml`` without its ``[gym]`` table (Harbor's own tables byte for
byte, as the RFC has it: the published file is Harbor's artifact, Gym's license, defaults and overrides stay with the
source), and runs Harbor's own format check on each staged folder when the ``harbor`` package is importable (it is not
a Gym dependency; the report says how to install it otherwise). Gym-only files (``tasks.jsonl``, ``task_data.py``, a
dataset-level ``verifier.py``) stay behind: what is published is what Harbor reads.

Before anything is written, every task's fields pass the dataset's ``TaskData`` (``task_data.py``), the check prepare
runs before a row is written, so Gym and the published dataset agree on which rows exist and no shim carries a value
the model refuses. The folders are published into a staging folder beside ``output``, Harbor's check runs over them
there, and they are moved into ``output`` only once every task published and passed the check: a publish refused by
Gym or by Harbor leaves nothing behind (``output`` stays absent) and a retry into the same folder needs no cleanup.

A published task holds one grader, and the same one Gym runs. A multi-step task (``[[steps]]``) is never graded by
a Python verifier in Gym (the loader grades each step by a ``test.sh``, its own or the task's), so publish copies such a
task as it is and writes no shim for it, even under a dataset-level ``verifier.py``: a shim ``tests/test.sh`` would
become Harbor's fallback grader for every step without its own script, and the published copy would grade differently
from the source. Two published folders that share a ``[task].name`` are refused: Harbor keys a dataset on that name
(``harbor add`` keeps one folder per name), so the published dataset would silently lose a task. A task's own
``tests/verifier.py`` is copied where it is, with the fixtures and helper modules beside it, and the shim's ``test.sh``
runs it there (``/tests/verifier.py``, as Gym's server does), so a module that reads a sibling fixture by
``Path(__file__)`` grades the published task as it graded the source. The published ``tests/`` is then the hub's
helper shape, ``test.sh`` beside ``verifier.py``, which the loader grades by ``test.sh``: the folder loads back into
Gym as a plain ``test.sh`` task that ``gym dataset validate`` plays the way Harbor would. Only a dataset-level
``verifier.py`` travels in the shim, under ``tests/.ng/verifier.py`` beside the runner (a task it grades has no
``tests/`` of its own). A source task that already has ``tests/test.sh`` beside a ``tests/verifier.py`` (the hub's
helper shape) is a ``test.sh`` task to the loader and is copied as it is, with no shim.

A verifier whose ``verify`` declares ``response`` is refused: Harbor's ``test.sh`` sees the container only,
never the agent's response, so such a task stays Gym-only. So is a Python verifier whose task grades in Harbor's
separate verifier mode (``[verifier] environment_mode = "separate"``, or a ``[verifier.environment]`` table, which
implies it): Harbor uploads nothing into a separate verifier container (``skip_tests_upload`` in its trial), so the
published shim would never reach it and whatever ``/tests/test.sh`` the verifier image ships would grade instead.
Gym's own server uploads the shim in both modes, which is why such a task validates under Gym and still cannot be
published. A published ``answer.md`` becomes a ``solve.sh``
that writes the answer to ``answer.md`` in the working directory (:func:`answer_write_command`), which is the only
place Harbor's oracle can put it. Gym's oracle copies the file there before it submits the text as its response (a
command carrying the text would exceed the exec argument limit for a long answer), so a verifier that reads
``workspace / "answer.md"`` grades identically under ``gym dataset validate`` and under Harbor: that is the
publishable answer shape.

What is published is the task folders (of each, what the digest covers, the top-level task files and the
``environment/``, ``tests/``, ``solution/`` and ``steps/`` folders, plus a root ``.gitignore``: together the set
Harbor's own packager reads, the ``.gitignore`` being its ignore rules at ``harbor add`` and ``harbor publish``, so a
task published as it is, with no shim and no ``solve.sh`` written for an ``answer.md``, packages to the source's hub
content hash, unless the source holds a byte-code cache or editor leftover that its own ``.gitignore`` does not
list: publish leaves those out at every depth, where Harbor's packager reads the ``.gitignore`` in place of its
defaults and would package them (the one ``harbor task
init`` writes names ``__pycache__/`` and ``*.pyc`` but no ``*~`` or ``*.swp``); a root ``.env``, ``.git/`` or
``.venv/`` is the author's and stays behind, while a template's root ``.gitignore`` is rendered into every instance,
the one hidden root entry the renderer keeps) and ``dataset.toml`` (``[gym]`` stripped);
everything else at the dataset root (``tasks.jsonl``, ``task_data.py``, a dataset-level ``verifier.py``,
``manifest.toml`` and the image records) is Gym's and stays behind. The published ``dataset.toml`` has to be a
manifest Harbor reads, since ``harbor add`` takes the dataset branch as soon as the file exists: a ``[dataset]`` table
with an ``org/name`` ``name`` is required of what remains (a :class:`PublishError` otherwise, with or without the
``harbor`` package), Harbor's own ``DatasetManifest`` reads the text as part of the format check when the package is
installed, and a ``dataset.toml`` that held Gym's ``[gym]`` table alone is not published at all (the report says so):
the folder is then a plain folder of task folders, which Harbor scans. A task folder holding a symbolic link among
the published paths is refused naming the link: ``copytree`` would follow it and publish the target's bytes, host files
outside the dataset included, where the loader refuses a file that escapes the task folder, the renderer refuses links
and the hub fetcher refuses them.
"""

from __future__ import annotations

import os
import re
import secrets
import shlex
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.digest import GITIGNORE, is_covered
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.parameterized import is_harbor_name, is_template_folder, render_template_folder
from nemo_gym.tasks.harbor.python_verifier import (
    PythonVerifierError,
    VerifierShim,
    python_verifier_for,
    reads_response,
    write_shim,
)
from nemo_gym.tasks.harbor.task import is_task_folder, iter_task_folders
from nemo_gym.tasks.harbor.task_data import TASK_DATA_FILE, TaskDataError, load_task_data_schema, read_task_fields


ANSWER_FILE = "answer.md"
SOLVE_SCRIPT = "solve.sh"
INSTALL_HINT = "install the `harbor` package (for example `uv pip install harbor`) to run Harbor's format check"
# A TOML table header, `[name]` or `[[name]]`, with an optional trailing comment; group 1 is the dotted name.
_TABLE_HEADER = re.compile(r"^\[\[?\s*([^\]]*?)\s*\]\]?\s*(?:#.*)?$")
# A top-level `gym.<key> = ...` dotted key or an inline `gym = {...}` table: the same table spelled without a header.
_GYM_KEY = re.compile(r"^gym\s*[.=]")


class PublishError(HarborTaskError):
    """The dataset cannot be published as Harbor tasks."""


@dataclass
class PublishReport:
    output: Path
    tasks: list[str] = field(default_factory=list)
    wrapped_verifiers: list[str] = field(default_factory=list)
    wrapped_answers: list[str] = field(default_factory=list)
    # Harbor's format check: None when the package is not importable, else task -> problem (an empty dict is clean);
    # the published dataset.toml, read by Harbor's DatasetManifest, is the entry named after the file.
    harbor_check: dict[str, str] | None = None
    harbor_note: str | None = None
    # Why no dataset.toml was published although the source has one.
    dataset_toml_note: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the folders were published: Harbor's check refused none (or did not run)."""
        return not self.harbor_check

    def text(self) -> str:
        if self.harbor_check:
            refused_tasks = [task for task in self.harbor_check if task != DATASET_FILE]
            what = [f"{len(refused_tasks)} of {len(self.tasks)} task(s)"] if refused_tasks else []
            if DATASET_FILE in self.harbor_check:
                what.append(DATASET_FILE)
            lines = [
                f"Nothing published into {self.output}: Harbor's format check refused {' and '.join(what)} "
                f"({self.harbor_note}); fix them and publish again"
            ]
            lines.extend(f"    {task}: {problem}" for task, problem in sorted(self.harbor_check.items()))
            return "\n".join(lines)
        lines = [f"Published {len(self.tasks)} task(s) into {self.output}"]
        if self.dataset_toml_note:
            lines.append(f"  {DATASET_FILE} not published: {self.dataset_toml_note}")
        if self.wrapped_verifiers:
            lines.append(f"  Python verifiers wrapped in tests/test.sh: {', '.join(self.wrapped_verifiers)}")
        if self.wrapped_answers:
            lines.append(f"  answer.md wrapped in solution/solve.sh: {', '.join(self.wrapped_answers)}")
        if self.harbor_check is None:
            lines.append(f"  Harbor format check skipped: {self.harbor_note}")
        else:
            lines.append(f"  Harbor format check: {len(self.tasks)} task(s) valid ({self.harbor_note})")
        return "\n".join(lines)


def _read_task_toml(task_folder: Path) -> dict:
    try:
        return tomllib.loads((task_folder / "task.toml").read_text())
    except (OSError, ValueError) as exc:
        raise PublishError(f"{task_folder / 'task.toml'}: {exc}") from exc


def _task_workdir(data: Mapping[str, Any]) -> str | None:
    environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    workdir = environment.get("workdir")
    return workdir if isinstance(workdir, str) and workdir else None


def separate_verifier_mode(data: Mapping[str, Any]) -> str | None:
    """Why the task's verifier runs in a separate environment under Harbor, or ``None`` in shared mode.

    Harbor 0.22.0's ``resolve_task_verifier_mode``: an explicit ``[verifier] environment_mode`` wins; without one, a
    ``[verifier.environment]`` table implies ``"separate"``; otherwise the verifier is shared.
    """
    verifier = data.get("verifier") if isinstance(data.get("verifier"), dict) else {}
    mode = verifier.get("environment_mode")
    if mode is not None:
        return '[verifier] environment_mode = "separate"' if mode == "separate" else None
    return "a [verifier.environment] table (which implies a separate verifier)" if "environment" in verifier else None


def answer_write_command(answer: str) -> str:
    """The shell command that leaves ``answer`` as ``answer.md`` in the current directory, byte for byte.

    The published ``solve.sh`` runs it in the task's working directory under Harbor's oracle, from a file, where the
    quoted answer's length is no concern. Gym's oracle copies ``/solution/answer.md`` there instead (an exec's command
    is one argument to the sandbox's shell, which Linux caps at 128 KiB on 4 KiB pages), so both leave the same bytes.
    """
    return f"printf '%s' {shlex.quote(answer)} > {ANSWER_FILE}"


def answer_solve_script(answer: str) -> str:
    """A ``solve.sh`` that leaves the reference answer as ``answer.md`` in the working directory."""
    return (
        "#!/usr/bin/env bash\n"
        "# Written by NeMo Gym from solution/answer.md: Harbor's oracle runs this in the task's working directory.\n"
        "set -eu\n"
        f"{answer_write_command(answer)}\n"
    )


def strip_gym_table(text: str) -> str:
    """``dataset.toml``'s text without its ``[gym]`` table: ``[gym]``, every ``[gym.*]`` and ``[[gym.*]]`` table with
    the lines under it, and a ``gym.<key>`` or ``gym = {...}`` entry of the root table (before any header; a key named
    ``gym`` under ``[dataset]`` or another table is Harbor's and stays). Every other line is kept as it is, so Harbor's
    own tables are untouched byte for byte. :func:`published_dataset_toml` checks the result parses to the original
    without ``gym`` before it is written."""
    kept: list[str] = []
    in_gym = False
    in_root = True
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        header = _TABLE_HEADER.match(stripped)
        if header is not None:
            name = header.group(1)
            in_gym = name == "gym" or name.startswith("gym.")
            in_root = False
        elif in_root and _GYM_KEY.match(stripped):
            continue
        if not in_gym:
            kept.append(line)
    return "".join(kept)


def published_dataset_toml(dataset_file: Path) -> str | None:
    """The text of ``dataset.toml`` as it is published, or ``None`` when nothing of Harbor's is in it.

    Unchanged without a ``[gym]`` table, else with the table stripped (:func:`strip_gym_table`) and checked: the result
    must parse to the original document minus ``gym``, so a file the line-based strip cannot handle (a ``[gym]`` header
    inside a multi-line string) is a :class:`PublishError` rather than a mangled manifest. What remains has to be a
    manifest Harbor reads, because ``harbor add`` takes the dataset branch whenever the file exists: a ``[dataset]``
    table whose ``name`` is ``org/name`` (:func:`is_harbor_name`, Harbor's own pattern) is required, whether or not the
    ``harbor`` package is installed to check the rest; a file that held the ``[gym]`` table alone is ``None``, not
    published, so the output is a plain folder of task folders.
    """
    try:
        text = dataset_file.read_text()
        parsed = tomllib.loads(text)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PublishError(f"{dataset_file}: {exc}") from exc
    expected = {key: value for key, value in parsed.items() if key != "gym"}
    if not expected:
        return None
    _check_harbor_manifest_shape(dataset_file, expected)
    if "gym" not in parsed:
        return text
    stripped = strip_gym_table(text)
    try:
        result = tomllib.loads(stripped)
    except ValueError as exc:
        result = None
        why = str(exc)
    else:
        changed = sorted(key for key in set(result) | set(expected) if result.get(key) != expected.get(key))
        why = f"Harbor's {', '.join(f'`{key}`' for key in changed)} would not survive it unchanged"
    if result != expected:
        raise PublishError(
            f"{dataset_file}: the [gym] table could not be stripped for publishing ({why}); write it as `[gym]` and "
            "`[gym.*]` tables on their own lines"
        )
    return stripped


def _check_harbor_manifest_shape(dataset_file: Path, manifest: Mapping[str, Any]) -> None:
    """Refuse a published ``dataset.toml`` Harbor's ``DatasetManifest`` would: no ``[dataset]`` table, or a ``name``
    that is not ``org/name``. Gym's own rule, so the refusal does not depend on the ``harbor`` package."""
    table = manifest.get("dataset")
    if not isinstance(table, dict):
        raise PublishError(
            f'{dataset_file}: the published dataset.toml needs a [dataset] table with name = "org/name" (Harbor\'s '
            "manifest; `harbor add` refuses the folder without it), or Gym's settings under [gym] alone, in which case "
            "no dataset.toml is published"
        )
    name = table.get("name")
    if not is_harbor_name(name):
        raise PublishError(
            f"{dataset_file}: [dataset].name {name!r} is not `org/name` (alphanumerics, hyphens, underscores and dots, "
            "no leading dot, no `..`), which Harbor's manifest requires"
        )


def _is_published(relative: str) -> bool:
    """Whether the task path at ``relative`` (posix, under the task folder) is one publish copies: what the digest
    covers (:func:`is_covered`), and the task root's ``.gitignore``, which Harbor's packager reads beyond those paths
    as its ignore rules at ``harbor add`` and ``harbor publish`` (``harbor task init`` writes one into every task,
    leaving ``node_modules/``, ``.venv/`` and ``.env`` behind at any depth), so that a task published as it is, with
    no shim and no ``solve.sh`` written for an ``answer.md``, packages to the source's hub content hash (a shim adds
    ``tests/test.sh`` and ``tests/.ng/``, and an ``answer.md`` task ``solution/solve.sh``, which Harbor packages too),
    unless the source holds a byte-code cache or editor leftover that its own ``.gitignore`` does not
    list: publish leaves those out (:func:`is_covered`), where Harbor's packager reads the ``.gitignore`` in place of
    its defaults and would package them. The root ``task_data.json`` is covered but left behind by the copy itself."""
    return relative == GITIGNORE or is_covered(relative)


def _check_no_links(task_folder: Path, name: str) -> None:
    """Refuse a task folder that is a symbolic link, or has one anywhere among the paths publish copies.

    The copy follows links, so a link would publish its target's bytes as a regular file, host files outside the dataset
    included, where the loader refuses a task file that escapes the folder and the renderer refuses every link. The
    folder itself first: the folder walk lists a linked task folder (so that the loader can refuse it, which it does),
    and ``copytree`` would otherwise follow the link and publish a whole task from outside the dataset. A link under
    a path publish leaves behind (a root ``.venv``'s interpreter) is not published, so it is not refused either; a
    linked root ``.gitignore`` is, since that file travels.
    """
    if task_folder.is_symlink():
        raise PublishError(
            f"task {name!r}: the task folder is a symbolic link, which publish does not follow (the loader refuses "
            "it too); copy the folder into the dataset"
        )
    for path in sorted(task_folder.rglob("*")):
        relative = path.relative_to(task_folder).as_posix()
        if path.is_symlink() and _is_published(relative):
            raise PublishError(
                f"task {name!r}: {relative} is a symbolic link, which publish does not follow; copy the file into the "
                "task (the renderer and the hub fetcher refuse links too)"
            )


def _harbor_task_name(config: Mapping[str, Any]) -> str | None:
    """The ``[task].name`` Harbor keys the task on, or ``None`` when the file has no ``[task]`` table."""
    task = config.get("task") if isinstance(config.get("task"), dict) else {}
    name = task.get("name")
    return name if isinstance(name, str) else None


def _publish_task(task_folder: Path, dataset_root: Path, destination: Path, report: PublishReport) -> str | None:
    """Write ``task_folder`` as a Harbor task at ``destination`` and return its ``[task].name`` (``None`` without one)."""
    name = task_folder.name
    config = _read_task_toml(task_folder)
    # The loader grades a multi-step task by test.sh alone (`python_verifier_for` is not consulted for one), so
    # neither is it here: the published task grades the way the source does.
    verifier = None if config.get("steps") else python_verifier_for(task_folder, dataset_root)
    if verifier is not None and reads_response(verifier):
        raise PublishError(
            f"task {name!r}: verify() in {verifier} reads `response`, which Harbor's test.sh cannot see; "
            "the task stays Gym-only (grade the workspace instead to publish it)"
        )
    if verifier is not None and (separate := separate_verifier_mode(config)) is not None:
        raise PublishError(
            f"task {name!r}: its Python verifier {verifier} would grade in a separate verifier environment under Harbor "
            f"({separate}), and Harbor uploads nothing into a separate verifier container, so the tests/test.sh shim "
            "could never reach it; the task stays Gym-only (grade in shared mode to publish it)"
        )
    # What the digest covers is copied (`is_covered`: the task's top-level files and its environment/, tests/,
    # solution/ and steps/ folders, the names the digest ignores left out at every depth), plus the task root's
    # `.gitignore`, nothing else (`_is_published`): a root `.env`, `.git/`, `.venv/` or scratch folder is the author's,
    # not the task's, while a hidden file below the root (`environment/.env`) is task content the sandbox receives.
    # That is the set Harbor's own packager reads, the root `.gitignore` being its ignore rules at `harbor add` and
    # `harbor publish` (a `harbor task init` task ships one that leaves node_modules/, .venv/ and .env behind at any
    # depth), so a task published as it is (no shim, no solve.sh written for an answer.md) packages to the source's hub
    # content hash (the shim and that solve.sh are files Harbor packages too), unless the source holds a byte-code
    # cache or editor leftover that its own `.gitignore` does not list: publish leaves those out, where Harbor's
    # packager reads the `.gitignore` in place of its defaults and would
    # package them. A template's root `.gitignore` reaches every rendered instance. The task's own tests/ is copied
    # as it is, tests/verifier.py included: the shim's test.sh runs the module there (`in_task_tests`), so a fixture it
    # reads by `__file__` is still beside it, and the published tests/ is the hub's helper shape (test.sh beside
    # verifier.py), graded by test.sh. A dataset-level module travels under tests/.ng/.
    # The task root's task_data.json stays behind (Harbor does not read it; the shim carries the fields under
    # tests/.ng/): only the root's, so a published task's tests/.ng/task_data.json survives a second publish and a
    # tests/ fixture of that name is kept.
    _check_no_links(task_folder, name)
    task_root = task_folder.resolve()
    in_task_tests = verifier is not None and verifier.parent == task_folder / "tests"

    def ignore(folder: str, names: list[str]) -> list[str]:
        relative = Path(folder).resolve().relative_to(task_root)
        at_root = relative == Path()
        return [n for n in names if not _is_published((relative / n).as_posix()) or (at_root and n == TASK_DATA_FILE)]

    shutil.copytree(task_folder, destination, symlinks=False, ignore=ignore)
    if verifier is not None:
        # write_shim creates tests/ when the task has none (and leaves it world-readable; a shipped one keeps
        # the author's mode): Harbor copies the folder into /tests as it is, modes included.
        shim = VerifierShim(
            verifier=verifier,
            in_task_tests=in_task_tests,
            workspace=_task_workdir(config),
            task_data=read_task_fields(task_folder),
        )
        write_shim(destination / "tests", shim)
        report.wrapped_verifiers.append(name)
    solution = task_folder / "solution"
    if (solution / ANSWER_FILE).is_file() and not (solution / SOLVE_SCRIPT).is_file():
        try:
            # The bytes, decoded: `read_text` would turn CR and CRLF line endings into LF, and the oracle leaves the
            # file byte for byte, so a byte-checking grader would pass the source and fail the published task.
            answer = (solution / ANSWER_FILE).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise PublishError(f"{solution / ANSWER_FILE}: {exc}") from exc
        script = destination / "solution" / SOLVE_SCRIPT
        script.write_text(answer_solve_script(answer))
        script.chmod(0o755)
        report.wrapped_answers.append(name)
    report.tasks.append(name)
    return _harbor_task_name(config)


def _check_distinct_names(names: Mapping[str, list[str]]) -> None:
    """Refuse a publish in which two task folders share a ``[task].name``: Harbor would keep one of them."""
    shared = {name: tasks for name, tasks in names.items() if len(tasks) > 1}
    if shared:
        detail = "; ".join(
            f"{name!r} is the [task].name of {', '.join(tasks)}" for name, tasks in sorted(shared.items())
        )
        raise PublishError(
            f"Nothing published: {detail}. Harbor keys a dataset on [task].name (`harbor add` keeps one folder per "
            "name), so give each task its own"
        )


def _check_task_data(task_folders: list[Path], dataset_root: Path) -> None:
    """Every task's fields against the dataset's ``TaskData``, the check prepare runs before a row is written.

    A ``task_data.py`` that does not import, a malformed ``task_data.json`` or a task whose fields the model refuses
    is a :class:`PublishError` naming the task, before anything is written: the published shim would otherwise carry
    a value that ``gym eval run`` and ``gym dataset validate`` refuse at prepare, and grade against it under Harbor.
    """
    try:
        schema = load_task_data_schema(dataset_root)
        for folder in task_folders:
            fields = read_task_fields(folder)
            if schema is not None:
                schema.validate(fields, task_id=folder.name)
    except TaskDataError as exc:
        raise PublishError(str(exc)) from exc


def _harbor_task_loader() -> tuple[Any, Any, str] | None:
    """Harbor's ``Task`` and ``DatasetManifest`` classes and its version, or ``None`` when the ``harbor`` package is
    not installed.

    Only a missing package is a skip. A module named ``harbor`` that is not Harbor (a folder of that name earlier on
    ``sys.path``, which happens when ``environment_servers/`` is on it) or a Harbor whose own imports fail is a
    :class:`PublishError` naming the import error and, for a shadow, the file ``import harbor`` resolved to: the check
    must not be skipped silently with a hint to install a package that is installed.
    """
    try:
        import harbor
        from harbor.models.dataset.manifest import DatasetManifest
        from harbor.models.task.task import Task
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "harbor":
            return None
        resolved = getattr(sys.modules.get("harbor"), "__file__", None)
        where = f"; `import harbor` resolves to {resolved}, which is not the Harbor package" if resolved else ""
        raise PublishError(
            f"The `harbor` package does not import ({type(exc).__name__}: {exc}{where}), so Harbor's format check "
            "could not run"
        ) from exc
    return Task, DatasetManifest, getattr(harbor, "__version__", None) or "installed"


def _harbor_problem(exc: Exception) -> str:
    """One line for a refusal by Harbor: a pydantic error's fields, else the message's first line."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = [
                f"{'.'.join(str(piece) for piece in item.get('loc', ())) or '(root)'}: {item.get('msg', 'invalid')}"
                for item in errors()
            ]
        except Exception:  # not pydantic's errors(); fall through to the message
            parts = []
        if parts:
            return f"{type(exc).__name__}: {'; '.join(parts)}"
    first = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {first[0] if first else exc}"


def harbor_format_check(
    task_folders: list[Path], dataset_toml: str | None = None
) -> tuple[dict[str, str] | None, str]:
    """Harbor's own loader over each folder and, when ``dataset_toml`` is given, its ``DatasetManifest`` over that
    text: ``(problems by task, note)``, the manifest's problem under ``dataset.toml``; problems ``None`` without the
    package."""
    loader = _harbor_task_loader()
    if loader is None:
        return None, INSTALL_HINT
    Task, DatasetManifest, version = loader
    problems: dict[str, str] = {}
    for folder in task_folders:
        try:
            Task(folder)
        except Exception as exc:  # Harbor raises pydantic, toml and FileNotFound errors alike
            problems[folder.name] = _harbor_problem(exc)
    if dataset_toml is not None:
        try:
            DatasetManifest.from_toml(dataset_toml)
        except Exception as exc:
            problems[DATASET_FILE] = _harbor_problem(exc)
    return problems, f"harbor {version}"


def publish_harbor(target: Path, output: Path) -> PublishReport:
    """Write ``target`` (a dataset folder, a task folder or a template folder) as Harbor tasks under ``output``.

    Every task's fields pass the dataset's ``TaskData`` first (:func:`_check_task_data`); the folders are written
    beside ``output``, Harbor's format check runs over them there, and they land in ``output`` at once, after every
    task published and passed; two folders sharing a ``[task].name`` are a :class:`PublishError` first
    (:func:`_check_distinct_names`), as is a task folder holding a symbolic link (:func:`_check_no_links`). A publish
    refused by Gym (a :class:`PublishError`) or by Harbor (a report with ``ok`` false, the refused folders named)
    leaves no ``output`` behind. Harbor's check is a step of every publish, not an option: it is skipped (and said so,
    with the install hint) only when the ``harbor`` package is not installed; a ``harbor`` that does not import is an
    error, and nothing is published. The check covers the published ``dataset.toml`` too (Harbor's ``DatasetManifest``
    over the stripped text, reported under its file name), after Gym's own rule on it (:func:`published_dataset_toml`:
    a ``[dataset]`` table with an ``org/name`` ``name``, or nothing of Harbor's left, in which case no ``dataset.toml``
    is published and the report says so).
    """
    source = Path(target).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise PublishError(f"{source} is not a directory")
    if output == source or source in output.parents:
        raise PublishError(f"The output folder {output} must be outside the dataset {source}")
    if output.exists() and not output.is_dir():
        raise PublishError(f"The output {output} exists and is not a folder; choose a new one")
    if output.is_dir() and any(output.iterdir()):
        raise PublishError(f"The output folder {output} exists and is not empty; choose a new one or empty it")
    with tempfile.TemporaryDirectory(prefix="nemo-gym-harbor-publish-") as scratch:
        if is_template_folder(source):
            source = render_template_folder(source, Path(scratch) / "rendered")
        if is_task_folder(source):
            task_folders = [source]
            dataset_root = source.parent
            dataset_file = None
        else:
            dataset_file = source / DATASET_FILE
            task_folders = iter_task_folders(source)
            dataset_root = source
            if not task_folders:
                raise PublishError(f"{source} is neither a task folder nor a folder of task folders")
        _check_task_data(task_folders, dataset_root)
        # Published beside the output first, checked there, and moved into place once every task succeeded and
        # passed Harbor's check: a task refused midway, by Gym or by Harbor, leaves no half-published output for the
        # next attempt to trip over. The staging folder goes whatever happens; after the move there is none.
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.parent / f".{output.name}.publish-{os.getpid()}-{secrets.token_hex(4)}"
        staging.mkdir()
        report = PublishReport(output=output)
        names: dict[str, list[str]] = {}
        try:
            for folder in task_folders:
                harbor_name = _publish_task(folder, dataset_root, staging / folder.name, report)
                if harbor_name is not None:
                    names.setdefault(harbor_name, []).append(folder.name)
            _check_distinct_names(names)
            dataset_toml = None
            if dataset_file is not None and dataset_file.is_file():
                dataset_toml = published_dataset_toml(dataset_file)
                if dataset_toml is None:
                    report.dataset_toml_note = (
                        "it holds Gym's [gym] table alone, and Harbor refuses a manifest without a [dataset] table; "
                        "the output is a plain folder of task folders"
                    )
                else:
                    (staging / DATASET_FILE).write_text(dataset_toml)
            try:
                report.harbor_check, report.harbor_note = harbor_format_check(
                    [staging / name for name in report.tasks], dataset_toml
                )
            except PublishError as exc:
                raise PublishError(f"{exc}; nothing was published") from exc
            if report.harbor_check:
                return report
            if output.is_dir():
                output.rmdir()  # empty, checked above
            os.replace(staging, output)
        except PythonVerifierError as exc:
            raise PublishError(str(exc)) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return report
