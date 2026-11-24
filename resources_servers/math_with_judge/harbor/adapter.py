# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the ``environments/opencode_math`` pairing (DAPO-Math-17k rows graded by math_with_judge, answered by
the local OpenCode CLI) into a parameterized Harbor dataset: one template folder plus ``tasks.jsonl``, one row per
selected native row.

    python resources_servers/math_with_judge/harbor/adapter.py --rows dapo17k_train.jsonl \\
        --output-dir /path/to/dataset [--ids-file ids.txt] [--native-rows-out native.jsonl] \\
        [--id-map-out id_map.json] [--native-config environments/opencode_math/config.yaml]

Scope: this adapter reproduces one native pairing, not every use of math_with_judge. It reads the native config
and refuses (exit 2) one it cannot reproduce: an agent other than ``legacy_app.py`` (the local CLI path whose
prompt it copies), ``system_prompt_mode`` other than ``append``, the LLM judge, format-tolerant answer
extraction, or an agent ``timeout``, ``thinking`` or ``opencode_version`` other than the Harbor arm's (the template's
``task.toml`` and ``configs/harbor_arm_harness.yaml``).

What the conversion keeps from the native path:

- The instruction is the prompt the native agent hands the OpenCode CLI: the agent config's ``system_prompt``, two
  newlines, then the row's user message (``OpenCodeAgent._run_opencode`` joins them). OpenCode 1.17.11's ``run``
  then wraps an argv prompt in double quotes and escapes the quotes inside it; that CLI artifact is not copied
  (the Harbor arm's ``argv_prompt_quoting`` reproduces it).
- The verifier grades with the Resources Server's own code. The functions it needs are copied out of
  ``resources_servers/math_with_judge/app.py`` unchanged, at conversion time, into
  ``tests/math_with_judge_grader.py``, with the native config's ``library_verifier_timeout_seconds``;
  ``tests/verifier.py`` repeats only ``verify``'s join of the response text and ``_verify_answer``'s control flow
  with the judge off.
- The expected answer lives only in ``tasks.jsonl`` (``task_data``), which only the verifier reads.

A task's id is ``dapo17k-<n>``, ``n`` the 0-based line of the row in ``--rows`` (the rows have no id field), so the
same id names the same row on both paths. ``--ids-file`` takes those indexes or the task ids. The dataset's root
``README.md`` (never copied into a task) records the rows file's sha256 and line count, so a later conversion can
tell whether the rows, and so what each id names, changed. Exit codes: 0 done, 2 an input or a native setting the
dataset cannot reproduce. Output is deterministic: converting twice gives identical bytes.
"""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import yaml

from nemo_gym.data.tasks.harbor import TaskDataError, load_task_data_schema


HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "task-template"
HARNESS_CONFIG = HERE / "configs" / "harbor_arm_harness.yaml"
SERVER_APP = HERE.parent / "app.py"
SERVER_APP_REPO_PATH = "resources_servers/math_with_judge/app.py"
ADAPTER_REPO_PATH = "resources_servers/math_with_judge/harbor/adapter.py"
REPO_ROOT = HERE.parents[2]
DEFAULT_NATIVE_CONFIG = REPO_ROOT / "environments" / "opencode_math" / "config.yaml"
NATIVE_AGENT_INSTANCE = "opencode_math_agent"
# The local OpenCode CLI path, whose prompt the instruction copies (sandbox sessions put system_prompt elsewhere).
NATIVE_AGENT_ENTRYPOINT = "legacy_app.py"
SERVER_CONFIG_CLASS = "LibraryJudgeMathResourcesServerConfig"
TASK_ID_PREFIX = "dapo17k-"
GRADER_MODULE = "math_with_judge_grader.py"

# Copied out of app.py verbatim: module-level functions, then methods of LibraryJudgeMathResourcesServer that they
# call (kept on a stand-in class of the same name, so the copied bodies run unchanged).
GRADER_FUNCTIONS = ("_extract_last_boxed_answer", "_run_math_verify", "_run_math_verify_in_subprocess")
GRADER_METHODS = ("_mute_output", "_strip_math_delimiters")
GRADER_IMPORTS = """\
import contextlib
from io import StringIO
from typing import Any, Optional

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
"""


class AdapterError(Exception):
    """An input or a native setting the dataset cannot reproduce; the CLI exits 2 with its message."""


def task_id_for_index(index: int) -> str:
    """The Harbor task id of the native row at 0-based line ``index``."""
    return f"{TASK_ID_PREFIX}{index:05d}"


def index_for_id(value: str) -> int:
    """The native row index an ids-file entry names: a 0-based line index, or a ``dapo17k-<index>`` task id."""
    digits = value[len(TASK_ID_PREFIX) :] if value.startswith(TASK_ID_PREFIX) else value
    if not (digits.isascii() and digits.isdigit()):
        raise AdapterError(f"not a row index or a {TASK_ID_PREFIX}<index> task id: {value!r}")
    return int(digits)


def read_ids(path: Path) -> list[int]:
    """Row indexes from an ids file, one per line; blank lines and ``#`` comments are ignored. Order is kept."""
    indexes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            indexes.append(index_for_id(line))
    if not indexes:
        raise AdapterError(f"{path}: no ids")
    if len(set(indexes)) != len(indexes):
        raise AdapterError(f"{path}: duplicate ids")
    return indexes


def load_native_config(config_path: Path) -> dict:
    """The native YAML as a mapping; malformed YAML or a non-mapping top level is an ``AdapterError``."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise AdapterError(f"{config_path}: not valid YAML ({error})") from None
    if not isinstance(config, dict):
        raise AdapterError(f"{config_path}: not a YAML mapping")
    return config


def _native_agent(config: dict, config_path: Path) -> dict:
    try:
        agent = config[NATIVE_AGENT_INSTANCE]["responses_api_agents"]["opencode_agent"]
    except (KeyError, TypeError):
        agent = None
    if not isinstance(agent, dict):
        raise AdapterError(f"{config_path}: no opencode_agent instance named {NATIVE_AGENT_INSTANCE}")
    return agent


def check_native_agent_settings(config_path: Path) -> None:
    """Refuse a native agent whose ``timeout``, ``thinking`` or ``opencode_version`` the Harbor arm does not use.

    The Harbor arm takes them from the template's ``task.toml`` (``[agent].timeout_sec``) and from
    ``configs/harbor_arm_harness.yaml``. An absent key is refused too: the agent's default may differ."""
    agent = _native_agent(load_native_config(config_path), config_path)
    harness = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))["opencode_agent"]["responses_api_agents"][
        "opencode_agent"
    ]
    harbor = {
        "timeout": tomllib.loads((TEMPLATE_DIR / "task.toml").read_text(encoding="utf-8"))["agent"]["timeout_sec"],
        "thinking": harness["thinking"],
        "opencode_version": harness["opencode_version"],
    }
    native = {key: agent.get(key) for key in harbor}
    if native != harbor:
        raise AdapterError(
            f"{config_path}: the Harbor arm runs the agent with {harbor}, but the native agent sets {native}"
        )


def native_system_prompt(config_path: Path) -> str:
    """The ``system_prompt`` the native agent instance prepends to every user message."""
    agent = _native_agent(load_native_config(config_path), config_path)
    if agent.get("entrypoint") != NATIVE_AGENT_ENTRYPOINT:
        raise AdapterError(
            f"{config_path}: the instruction copies the local CLI path ({NATIVE_AGENT_ENTRYPOINT}) only"
        )
    if agent.get("system_prompt_mode", "append") != "append":
        raise AdapterError(f"{config_path}: only system_prompt_mode 'append' folds into the instruction")
    return agent.get("system_prompt") or ""


def server_config_defaults(app_path: Path = SERVER_APP) -> dict:
    """The literal field defaults of the Resources Server's config class in app.py."""
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    config_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == SERVER_CONFIG_CLASS), None
    )
    if config_class is None:
        raise AdapterError(f"{app_path} no longer defines {SERVER_CONFIG_CLASS}; update the adapter")
    defaults = {}
    for node in config_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                defaults[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return defaults


def native_grading_settings(config_path: Path, app_path: Path = SERVER_APP) -> dict:
    """The grading settings of the Resources Server the native agent calls: its config over app.py's defaults.

    Refuses settings ``tests/verifier.py`` does not reproduce (the LLM judge, format-tolerant extraction)."""
    config = load_native_config(config_path)
    agent = _native_agent(config, config_path)
    server_ref = agent.get("resources_server")
    try:
        server = config[server_ref["name"]]["resources_servers"]["math_with_judge"]
    except (KeyError, TypeError):
        server = None
    if not isinstance(server, dict):
        raise AdapterError(f"{config_path}: the native agent's resources server is not a math_with_judge instance")
    settings = {**server_config_defaults(app_path), **server}
    unsupported = {
        key: settings.get(key)
        for key in ("should_use_judge", "format_tolerant_answer_extraction")
        if settings.get(key) is not False
    }
    if unsupported:
        raise AdapterError(f"{config_path}: the Harbor verifier grades with these off, but they are {unsupported}")
    timeout = settings.get("library_verifier_timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise AdapterError(
            f"{config_path}: library_verifier_timeout_seconds must be a positive number, got {timeout!r}"
        )
    return {"library_verifier_timeout_seconds": float(timeout)}


def file_sha256(path: Path) -> str:
    """The sha256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(HERE), *args], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def gym_commit() -> str:
    """The Gym commit the adapter runs from, ``unknown`` outside a git checkout.

    ``-dirty`` is appended when ``app.py`` or this adapter's folder differ from that commit (uncommitted or
    untracked changes), so the commit alone does not name the code that was vendored."""
    head = _git("rev-parse", "HEAD")
    if not head:
        return "unknown"
    changed = _git("status", "--porcelain", "--", str(SERVER_APP), str(HERE))
    return f"{head}-dirty" if changed is None or changed else head


def iter_native_rows(path: Path) -> Iterator[tuple[int, dict, str]]:
    """Stream ``(index, row, line)`` from a native rows JSONL file without loading it whole."""
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AdapterError(f"{path} line {index}: not JSON ({error})") from None
            if not isinstance(row, dict):
                raise AdapterError(f"{path} line {index}: not a JSON object")
            yield index, row, line.rstrip("\n")


def task_data_for_row(row: dict) -> dict:
    """The task fields of one native row: its single user message and the expected answer."""
    params = row.get("responses_create_params")
    if isinstance(params, dict):
        extra = sorted(set(params) - {"input"})
        if extra:
            raise AdapterError(
                f"responses_create_params carries {extra}, which a Harbor task cannot carry"
                " (per-task timeout, sampling)"
            )
    messages = params.get("input") if isinstance(params, dict) else None
    if (
        not isinstance(messages, list)
        or not all(isinstance(message, dict) for message in messages)
        or [message.get("role") for message in messages] != ["user"]
    ):
        raise AdapterError("expected exactly one user message per native row")
    content = messages[0].get("content")
    if not isinstance(content, str):
        raise AdapterError("expected a plain-text user message")
    if not isinstance(row.get("expected_answer"), str):
        raise AdapterError("the row has no string expected_answer")
    if row.get("question") != content:
        raise AdapterError("the row's question differs from its user message")
    return {"question": content, "expected_answer": row["expected_answer"]}


def grader_source(
    app_path: Path = SERVER_APP, *, library_verifier_timeout_seconds: float = 10.0, commit: str = "unknown"
) -> str:
    """``tests/math_with_judge_grader.py``: the grading functions of app.py, copied without changes, and the native
    config's verifier timeout."""
    source = app_path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    def segment(node: ast.AST) -> str:
        start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        return "".join(lines[start - 1 : node.end_lineno])

    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    server_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LibraryJudgeMathResourcesServer"
        ),
        None,
    )
    methods = {
        node.name: node for node in (server_class.body if server_class else []) if isinstance(node, ast.FunctionDef)
    }
    missing = [name for name in GRADER_FUNCTIONS if name not in functions] + [
        name for name in GRADER_METHODS if name not in methods
    ]
    if missing:
        raise AdapterError(f"{app_path} no longer defines {missing}; update the adapter")
    digest = hashlib.sha256(source.encode()).hexdigest()
    parts = [
        "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n",
        "# SPDX-License-Identifier: Apache-2.0\n",
        f"# Vendored unchanged from {SERVER_APP_REPO_PATH} at {commit}; source sha256 {digest}; license Apache-2.0\n",
        f"# Generated by {ADAPTER_REPO_PATH}: do not edit.\n",
        "# The functions and methods below are copied by source segment; the imports,\n",
        "# LIBRARY_VERIFIER_TIMEOUT_SECONDS and the stand-in class line are the adapter's glue.\n",
        "# ruff: noqa\n",
        GRADER_IMPORTS,
        "\n# library_verifier_timeout_seconds of the native config (app.py's default when it sets none).\n",
        f"LIBRARY_VERIFIER_TIMEOUT_SECONDS = {library_verifier_timeout_seconds!r}\n",
        "\n\nclass LibraryJudgeMathResourcesServer:\n",
        '    """Stand-in holding the server methods the copied functions call."""\n\n',
        "\n".join(segment(methods[name]) for name in GRADER_METHODS),
    ]
    for name in GRADER_FUNCTIONS:
        parts.append("\n\n" + segment(functions[name]))
    return "".join(parts)


def instruction_template(system_prompt: str) -> str:
    """``instruction.template.md``: exactly ``f"{system_prompt}\\n\\n{question}"``, with no trailing newline."""
    if "{{" in system_prompt:
        raise AdapterError("system_prompt contains '{{', which the template renderer would read as a placeholder")
    return f"{system_prompt}\n\n{{{{ task_data.question }}}}" if system_prompt else "{{ task_data.question }}"


def dataset_readme(
    *,
    rows: Path,
    row_count: int,
    task_count: int,
    ids_sha256: str,
    native_config: Path,
    commit: str,
    grader_sha256: str,
    regenerate: str,
) -> str:
    """The dataset's root README.md: where its tasks came from. Gym copies no root README into a task."""
    return (
        "# DAPO-Math-17k (math_with_judge) as Harbor tasks\n\n"
        f"Generated by `{ADAPTER_REPO_PATH}`; do not edit.\n\n"
        f"- Native rows file `{rows.name}` (`environments/opencode_math/prepare.py` output): {row_count} rows,\n"
        f"  sha256 `{file_sha256(rows)}`.\n"
        "  Task `dapo17k-<n>` is its 0-based row `n`; another rows file (for example from another Hugging Face\n"
        "  revision) can give the same id to another problem.\n"
        f"- Tasks: {task_count}; ids file sha256 `{ids_sha256}`.\n"
        f"- Native config `{native_config.name}` (system prompt and grading settings), sha256"
        f" `{file_sha256(native_config)}`.\n"
        f"- Gym commit: `{commit}`.\n"
        f"- `tests/{GRADER_MODULE}`, sha256 `{grader_sha256}`: vendored from `{SERVER_APP_REPO_PATH}`, sha256"
        f" `{file_sha256(SERVER_APP)}`.\n"
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
    """Write the parameterized dataset into ``output_dir`` (absent or empty); return the number of tasks.

    With ``ids`` only those rows are converted, in that order. ``native_rows_out`` also writes the same rows, in the
    same order, each line unchanged apart from its line ending (written as LF), for the native arm of a parity run;
    ``id_map_out`` writes ``{line in native_rows_out: task id}``, which pairs native rollouts (``_ng_task_index``)
    with Harbor tasks. Every row is validated against the template's ``task_data.py`` before anything is written.
    """
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise AdapterError(f"{output_dir} is not an empty folder; choose a new --output-dir")
    grading = native_grading_settings(native_config)
    check_native_agent_settings(native_config)
    instruction = instruction_template(native_system_prompt(native_config))
    wanted = None if ids is None else set(ids)
    selected: dict[int, tuple[dict, str]] = {}
    row_count = 0
    for index, row, line in iter_native_rows(rows):
        row_count += 1
        if wanted is not None and index not in wanted:
            continue
        try:
            selected[index] = (task_data_for_row(row), line)
        except AdapterError as error:
            raise AdapterError(f"{task_id_for_index(index)}: {error}") from None
    order = sorted(selected) if ids is None else ids
    missing = [index for index in order if index not in selected]
    if missing:
        raise AdapterError(f"{len(missing)} ids name no row of {rows.name}: {missing[:5]}")
    # Gym checks every row against the template's TaskData when it prepares the dataset; refuse here, before any
    # output, a row it would refuse there.
    try:
        schema = load_task_data_schema(TEMPLATE_DIR)
        if schema is None:
            raise AdapterError(f"{TEMPLATE_DIR} ships no task_data.py")
        for index in order:
            schema.validate(selected[index][0], task_id=task_id_for_index(index))
    except TaskDataError as error:
        raise AdapterError(str(error)) from None

    commit = gym_commit()
    grader = grader_source(commit=commit, **grading)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        TEMPLATE_DIR, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (output_dir / "instruction.template.md").write_text(instruction, encoding="utf-8")
    (output_dir / "tests" / GRADER_MODULE).write_text(grader, encoding="utf-8")
    (output_dir / "README.md").write_text(
        dataset_readme(
            rows=rows,
            row_count=row_count,
            task_count=len(order),
            ids_sha256=ids_sha256,
            native_config=native_config,
            commit=commit,
            grader_sha256=hashlib.sha256(grader.encode()).hexdigest(),
            regenerate=regenerate,
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
    parser.add_argument("--rows", type=Path, required=True, help="JSONL written by opencode_math/prepare.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Absent or empty folder for the dataset")
    parser.add_argument(
        "--ids-file", type=Path, help="Convert only these rows (0-based line indexes or dapo17k-<index>), in order"
    )
    parser.add_argument(
        "--native-rows-out",
        type=Path,
        help="Also write the selected rows here, in order, each line unchanged apart from its LF ending",
    )
    parser.add_argument("--id-map-out", type=Path, help="JSON {line in --native-rows-out: task id}")
    parser.add_argument("--native-config", type=Path, default=DEFAULT_NATIVE_CONFIG, help="Native pairing YAML")
    args = parser.parse_args(argv)
    # File names only: the README must not depend on where the inputs or the output live.
    regenerate = f"python {ADAPTER_REPO_PATH} --rows {args.rows.name} --output-dir <dir>"
    if args.ids_file:
        regenerate += f" --ids-file {args.ids_file.name}"
    if args.native_rows_out:
        regenerate += f" --native-rows-out {args.native_rows_out.name}"
    if args.id_map_out:
        regenerate += f" --id-map-out {args.id_map_out.name}"
    if args.native_config.resolve() != DEFAULT_NATIVE_CONFIG:
        regenerate += f" --native-config {args.native_config.name}"
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
    except (AdapterError, OSError, UnicodeDecodeError) as error:
        print(f"adapter: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {count} tasks to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
