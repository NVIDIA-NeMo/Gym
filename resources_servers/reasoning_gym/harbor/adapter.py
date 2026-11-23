# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the ``environments/opencode_reasoning_gym`` pairing (Reasoning Gym rows graded by the reasoning_gym
Resources Server, answered by the local OpenCode CLI) into a parameterized Harbor dataset: one template folder plus
``tasks.jsonl``, one row per selected native row.

    python resources_servers/reasoning_gym/harbor/adapter.py --rows train_knights_knaves.jsonl \\
        --output-dir /path/to/dataset [--ids-file ids.txt] [--native-rows-out native.jsonl] \\
        [--id-map-out id_map.json] [--native-config environments/opencode_reasoning_gym/config.yaml]

What the conversion keeps from the native path:

- The instruction is the prompt the native agent hands the OpenCode CLI: the agent config's ``system_prompt``, two
  newlines, then the row's user message.
- The verifier grades with the Resources Server's own code: the answer extraction is copied out of
  ``resources_servers/reasoning_gym/app.py`` unchanged, at conversion time, into ``tests/reasoning_gym_grader.py``;
  scoring calls the ``reasoning-gym`` package at the server's locked version.
- The reference answer and the scorer's metadata live only in ``tasks.jsonl`` (``task_data``).

A task's id is ``rg-<n>``, ``n`` the 0-based line of the row in ``--rows`` zero-padded to 5 digits (row 98 is
``rg-00098``; the rows have no id field). Exit codes: 0 done, 2 a native setting or input the dataset cannot
reproduce. Output is deterministic.
"""

import argparse
import ast
import base64
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "task-template"
SERVER_APP = HERE.parent / "app.py"
REQUIREMENTS = HERE / "requirements.txt"
REPO_ROOT = HERE.parents[2]
DEFAULT_NATIVE_CONFIG = REPO_ROOT / "environments" / "opencode_reasoning_gym" / "config.yaml"
NATIVE_AGENT_INSTANCE = "opencode_reasoning_gym_agent"
NATIVE_AGENT_ENTRYPOINT = "legacy_app.py"
TASK_ID_PREFIX = "rg-"
GRADER_MODULE = "reasoning_gym_grader.py"
SERVER_CLASS = "ReasoningGymResourcesServer"

# Copied out of app.py verbatim: module constants and functions, then the server method that uses them (kept on a
# stand-in class of the same name, so the copied body runs unchanged).
GRADER_ASSIGNMENTS = ("_BOXED_MARKER", "_LATEX_TEXT_WRAPPER")
GRADER_FUNCTIONS = ("_matching_brace", "_last_boxed_content", "_normalize_latex_text")
GRADER_METHODS = ("_extract_answer_from_response",)
GRADER_IMPORTS = """\
import re
import textwrap
from typing import Any, Optional

from reasoning_gym.utils import extract_answer
"""

# Base image with uv and the compiler and headers the three source-only dependencies need (pycosat is C), pinned by
# digest to its linux/amd64 manifest, the platform OpenSandbox runs.
BASE_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.13-bookworm@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
)
# reasoning-gym 0.1.25 depends on these, which publish no wheel.
SOURCE_ONLY_PACKAGES = ("cellpylib", "pycosat", "zss")
REQUIREMENTS_IN_IMAGE = "/opt/nemo_gym/reasoning_gym_requirements.txt"


class AdapterError(ValueError):
    """A native setting or input the converted dataset cannot reproduce (exit 2)."""


def task_id_for_index(index: int) -> str:
    """The Harbor task id of the native row at 0-based line ``index``."""
    return f"{TASK_ID_PREFIX}{index:05d}"


def read_ids(path: Path) -> list[int]:
    """Native row indexes from a file, one per line; blank lines and ``#`` comments are ignored. Order is kept."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if not (line.isascii() and line.isdigit()):
            raise AdapterError(f"{path}: {line!r} is not a 0-based row index")
        ids.append(int(line))
    if len(set(ids)) != len(ids):
        raise AdapterError(f"{path}: duplicate ids")
    return ids


def native_system_prompt(config_path: Path) -> str:
    """The ``system_prompt`` the native agent instance prepends to every user message."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        agent = config[NATIVE_AGENT_INSTANCE]["responses_api_agents"]["opencode_agent"]
    except (KeyError, TypeError):
        raise AdapterError(f"{config_path}: no opencode_agent instance named {NATIVE_AGENT_INSTANCE}") from None
    if agent.get("entrypoint") != NATIVE_AGENT_ENTRYPOINT:
        raise AdapterError(
            f"{config_path}: the instruction copies the local CLI path ({NATIVE_AGENT_ENTRYPOINT}) only"
        )
    if agent.get("system_prompt_mode", "append") != "append":
        raise AdapterError(f"{config_path}: only system_prompt_mode 'append' folds into the instruction")
    server = (agent.get("resources_server") or {}).get("name")
    if "reasoning_gym" not in ((config.get(server) or {}).get("resources_servers") or {}):
        raise AdapterError(f"{config_path}: the native agent's resources server is not a reasoning_gym instance")
    prompt = agent.get("system_prompt") or ""
    if "{{" in prompt:
        raise AdapterError("system_prompt contains '{{', which the template renderer would read as a placeholder")
    return prompt


def file_sha256(path: Path) -> str:
    """The sha256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gym_commit() -> str:
    """The Gym commit the adapter runs from, or ``unknown`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def iter_rows(path: Path) -> Iterator[tuple[int, dict, str]]:
    """Stream ``(index, row, raw line)`` from a native rows JSONL file without loading it whole."""
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AdapterError(f"{path}:{index + 1}: not a JSON row ({error.msg})") from None
            yield index, row, line.rstrip("\n")


def task_data_for_row(row: dict) -> dict:
    """The task fields of one native row; refuses a row the dataset cannot reproduce."""
    messages = (row.get("responses_create_params") or {}).get("input")
    if not isinstance(messages, list) or [message.get("role") for message in messages] != ["user"]:
        raise AdapterError("expected exactly one user message per native row")
    content = messages[0].get("content")
    if not isinstance(content, str):
        raise AdapterError("expected a plain-text user message")
    extra = set(row["responses_create_params"]) - {"input"}
    if extra:
        raise AdapterError(f"responses_create_params carries {sorted(extra)}, which a task folder cannot carry")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("source_dataset"):
        raise AdapterError("metadata.source_dataset is missing: the native verify raises on such a row")
    if row.get("answer") is None:
        raise AdapterError("the row has no reference answer, so it has no oracle; leave it out with --ids-file")
    if not isinstance(row.get("question"), str):
        raise AdapterError("the row has no 'question' string, which the native verify requires")
    return {"user_message": content, "question": row["question"], "answer": row["answer"], "metadata": metadata}


def grader_source(app_path: Path = SERVER_APP) -> str:
    """``tests/reasoning_gym_grader.py``: the answer extraction of app.py, copied without changes."""
    source = app_path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    def segment(node: ast.AST) -> str:
        start = min([node.lineno, *(decorator.lineno for decorator in getattr(node, "decorator_list", []))])
        return "".join(lines[start - 1 : node.end_lineno])

    assignments = {
        target.id: node
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    server_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == SERVER_CLASS), None
    )
    methods = {
        node.name: node for node in (server_class.body if server_class else []) if isinstance(node, ast.FunctionDef)
    }
    missing = (
        [name for name in GRADER_ASSIGNMENTS if name not in assignments]
        + [name for name in GRADER_FUNCTIONS if name not in functions]
        + [name for name in GRADER_METHODS if name not in methods]
    )
    if missing:
        raise AdapterError(f"{app_path} no longer defines {missing}; update the adapter")
    parts = [
        "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n",
        "# SPDX-License-Identifier: Apache-2.0\n",
        f"# Vendored unchanged from resources_servers/reasoning_gym/app.py at {gym_commit()};"
        f" source sha256 {hashlib.sha256(source.encode()).hexdigest()}; license Apache-2.0\n",
        "# Generated by resources_servers/reasoning_gym/harbor/adapter.py: do not edit.\n",
        "# ruff: noqa\n",
        GRADER_IMPORTS,
        "\n\nclass ReasoningGymResourcesServer:\n",
        '    """Stand-in holding the server method the verifier calls."""\n\n',
        "\n".join(segment(methods[name]) for name in GRADER_METHODS),
        "\n\n",
    ]
    parts.extend("\n" + segment(assignments[name]) for name in GRADER_ASSIGNMENTS)
    parts.extend("\n\n" + segment(functions[name]) for name in GRADER_FUNCTIONS)
    return "".join(parts)


def dockerfile(requirements: Path = REQUIREMENTS) -> str:
    """``environment/Dockerfile``: the pinned base, then the hashed requirements unpacked and installed at seed."""
    packed = base64.b64encode(gzip.compress(requirements.read_bytes(), mtime=0)).decode()
    no_binary = " ".join(f"--no-binary {name}" for name in SOURCE_ONLY_PACKAGES)
    return (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        "# Generated by resources_servers/reasoning_gym/harbor/adapter.py from harbor/requirements.txt: do not edit.\n"
        "# Overlay mode: the base image is pulled and the RUN lines run at every seed; nothing is built or pushed.\n"
        "# The verifier imports reasoning-gym, installed with every dependency at the version and hash the\n"
        "# reasoning_gym server's uv.lock pins. Three of them ship no wheel and are built here (the base has gcc).\n"
        "# Every seed downloads from PyPI, so the sandbox needs that egress.\n"
        f"FROM {BASE_IMAGE}\n\n"
        "WORKDIR /workspace\n\n"
        f"RUN mkdir -p /opt/nemo_gym && echo {packed} | base64 -d | gunzip > {REQUIREMENTS_IN_IMAGE}\n"
        f"RUN uv pip install --system --no-cache --quiet --require-hashes --only-binary :all: {no_binary}"
        f" -r {REQUIREMENTS_IN_IMAGE}\n"
    )


def dataset_readme(*, rows: Path, row_count: int, task_count: int, ids_sha256: str, regenerate: str) -> str:
    """The dataset's root README.md: where its tasks came from. Gym copies no root README into a task."""
    return (
        "# Reasoning Gym (opencode_reasoning_gym) as Harbor tasks\n\n"
        "Generated by `resources_servers/reasoning_gym/harbor/adapter.py`; do not edit.\n\n"
        f"- Native rows file `{rows.name}`: {row_count} rows, sha256 `{file_sha256(rows)}`. Task `rg-<n:05d>` is\n"
        "  its 0-based row `n` (row 98 is `rg-00098`); another rows file (another seed, size or reasoning-gym\n"
        "  version) gives the same id to another problem.\n"
        f"- Tasks: {task_count}; ids file sha256 `{ids_sha256}`.\n"
        f"- Gym commit: `{gym_commit()}`.\n"
        f"- `tests/{GRADER_MODULE}` copied from `resources_servers/reasoning_gym/app.py`, sha256"
        f" `{file_sha256(SERVER_APP)}`.\n"
        f"- `environment/Dockerfile` installs `harbor/requirements.txt`, sha256 `{file_sha256(REQUIREMENTS)}`.\n"
        f"- Regenerate: `{regenerate}`\n"
    )


def convert(
    rows: Path,
    output_dir: Path,
    *,
    ids: list[int] | None = None,
    ids_sha256: str = "none (all rows)",
    native_config: Path = DEFAULT_NATIVE_CONFIG,
    native_rows_out: Path | None = None,
    id_map_out: Path | None = None,
    regenerate: str = "",
) -> int:
    """Write the parameterized dataset into ``output_dir`` (absent or empty); return the number of tasks."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AdapterError(f"{output_dir} is not empty; choose a new --output-dir")
    instruction = native_system_prompt(native_config)
    wanted = None if ids is None else set(ids)
    selected: dict[int, tuple[dict, str]] = {}
    row_count = 0
    for index, row, line in iter_rows(rows):
        row_count += 1
        if wanted is not None and index not in wanted:
            continue
        try:
            selected[index] = (task_data_for_row(row), line)
        except AdapterError as error:
            raise AdapterError(f"row {index}: {error}") from None
    order = sorted(selected) if ids is None else ids
    missing = [index for index in order if index not in selected]
    if missing:
        raise AdapterError(f"{len(missing)} ids name no row of {rows}: {missing[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        TEMPLATE_DIR, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    prefix = f"{instruction}\n\n" if instruction else ""
    (output_dir / "instruction.template.md").write_text(prefix + "{{ task_data.user_message }}", encoding="utf-8")
    (output_dir / "tests" / GRADER_MODULE).write_text(grader_source(), encoding="utf-8")
    (output_dir / "environment").mkdir(exist_ok=True)
    (output_dir / "environment" / "Dockerfile").write_text(dockerfile(), encoding="utf-8")
    (output_dir / "README.md").write_text(
        dataset_readme(
            rows=rows, row_count=row_count, task_count=len(order), ids_sha256=ids_sha256, regenerate=regenerate
        ),
        encoding="utf-8",
    )
    with (output_dir / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for index in order:
            record = {"task_id": task_id_for_index(index), "task_data": selected[index][0]}
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if native_rows_out is not None:
        native_rows_out.parent.mkdir(parents=True, exist_ok=True)
        with native_rows_out.open("w", encoding="utf-8") as stream:
            for index in order:
                stream.write(selected[index][1] + "\n")
    if id_map_out is not None:
        id_map = {str(position): task_id_for_index(index) for position, index in enumerate(order)}
        id_map_out.parent.mkdir(parents=True, exist_ok=True)
        id_map_out.write_text(json.dumps(id_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(order)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=Path, required=True, help="JSONL written by opencode_reasoning_gym/prepare.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Absent or empty folder for the dataset")
    parser.add_argument("--ids-file", type=Path, help="Convert only these 0-based row indexes, in this order")
    parser.add_argument("--native-rows-out", type=Path, help="Also write the selected raw rows here, in order")
    parser.add_argument("--id-map-out", type=Path, help="JSON {line in --native-rows-out: task id}")
    parser.add_argument("--native-config", type=Path, default=DEFAULT_NATIVE_CONFIG, help="Native pairing YAML")
    args = parser.parse_args(argv)
    # File names only: the README must not depend on where the inputs or the output live.
    regenerate = f"python resources_servers/reasoning_gym/harbor/adapter.py --rows {args.rows.name} --output-dir <dir>"
    if args.ids_file:
        regenerate += f" --ids-file {args.ids_file.name}"
    try:
        ids = read_ids(args.ids_file) if args.ids_file else None
        count = convert(
            args.rows,
            args.output_dir,
            ids=ids,
            ids_sha256=file_sha256(args.ids_file) if args.ids_file else "none (all rows)",
            native_config=args.native_config,
            native_rows_out=args.native_rows_out,
            id_map_out=args.id_map_out,
            regenerate=regenerate,
        )
    except AdapterError as error:
        print(f"adapter: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {count} tasks to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
