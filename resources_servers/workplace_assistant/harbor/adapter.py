# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write workplace_assistant rows as a parameterized Harbor task dataset.

The output folder is ``task-template/`` plus one ``tasks.jsonl`` row per selected native row, with the Resources
Server's own tool code (``utils.py``, ``workplace_assistant_tools/``, ``csv_data/``) copied unchanged into
``environment/wa/`` (for the tools sidecar) and ``tests/wa/`` (for the verifier), and a ``README.md`` recording where
the dataset came from. ``gym eval run`` and ``gym dataset validate`` render one task folder per row.

    python resources_servers/workplace_assistant/harbor/adapter.py --rows validation.jsonl --output-dir <dir> \\
        [--ids-file ids.txt] [--native-rows-out native.jsonl] [--opencode-rows-out opencode.jsonl] \\
        [--id-map-out id_map.json]

Task ``wa-validation-0115`` is the row whose ``id`` is 115. ``--native-rows-out`` writes the selected raw lines, in
the same order; ``--opencode-rows-out`` writes the same rows in the form the native OpenCode config
(``harbor/configs/native_pairing.yaml``) runs (no ``tools``, no ``parallel_tool_calls``, no ``agent_ref``), so both
arms run the same tasks; ``--id-map-out`` writes
``{row id: task id}`` for ``python -m scripts.harbor_parity --native-key id --id-map``. ``--solution`` picks what the
oracle does: ``reference`` replays the row's ground-truth calls, ``none`` makes no call and ``wrong`` sends one email
no row asks for (negative controls).

Run the dataset with ``harbor/configs/harbor_arm.yaml``; see ``harbor/README.md``. Exit codes: 0 done, 2 an input, an
output path that cannot be written, or a native setting the dataset cannot reproduce (no dataset is written; a write
that fails partway through the dataset can leave a partial output folder). The output is deterministic.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from nemo_gym.data.tasks.harbor import TaskDataError, load_task_data_schema


HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
TEMPLATE_DIR = HERE / "task-template"
# The Resources Server files the tools and the grader need, copied under their package path.
VENDORED = ("utils.py", "workplace_assistant_tools", "csv_data")
VENDOR_PACKAGE = Path("resources_servers") / "workplace_assistant"
VENDOR_TARGETS = (Path("environment") / "wa", Path("tests") / "wa")
VENDOR_LICENSE = "Apache-2.0"
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
# A state-changing call no row's ground truth contains: the negative control's journal.
WRONG_CALLS = [
    {
        "name": "email_send_email",
        "arguments": json.dumps(
            {"recipient": "nobody@example.com", "subject": "negative control", "body": "This call is wrong."}
        ),
    }
]
# The request fields of a native row the dataset reproduces: the messages become the instruction, the tools are the
# sidecar's MCP tools, and temperature 1.0 is set for every request by the run configs (harbor/configs/). Both arms
# drop parallel_tool_calls, which OpenCode refuses. Any other field is a setting no task reproduces.
REPRODUCED_PARAMS = frozenset({"input", "tools", "parallel_tool_calls", "temperature"})
CONFIGURED_TEMPERATURE = 1.0
DEFAULT_PREFIX = "wa-validation"


class AdapterError(ValueError):
    """An input or native setting the converted dataset cannot reproduce (exit 2)."""


def task_id(row_id: int, prefix: str) -> str:
    """The Harbor task id of the native row whose ``id`` is ``row_id``."""
    return f"{prefix}-{row_id:04d}"


def read_ids(path: Path) -> list[int]:
    """Native row ids from a file, one per line; blank lines and ``#`` comments are ignored. Order is kept."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise AdapterError(f"{path.name}: not UTF-8 text") from None
    ids = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            raise AdapterError(f"{path.name}: {line!r} is not a row id") from None
    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    if duplicates:
        raise AdapterError(f"{path.name}: duplicate ids {duplicates}")
    return ids


def task_data(row: dict) -> dict:
    """The row's fields as the template reads them; refuses a row the task would not reproduce."""
    missing = [key for key in ("id", "category", "environment_name", "responses_create_params") if key not in row]
    if missing:
        raise AdapterError(f"row {row.get('id')}: no {missing}")
    if type(row["id"]) is not int:
        raise AdapterError(f"row id {row['id']!r} is not an integer; task ids are <prefix>-<row id>")
    params = row["responses_create_params"]
    if not isinstance(params, dict):
        raise AdapterError(f"row {row.get('id')}: responses_create_params is not an object")
    extra = sorted(set(params) - REPRODUCED_PARAMS)
    if extra:
        raise AdapterError(f"row {row.get('id')}: responses_create_params carries {extra}, which no task reproduces")
    if params.get("temperature", CONFIGURED_TEMPERATURE) != CONFIGURED_TEMPERATURE:
        raise AdapterError(
            f"row {row.get('id')}: temperature {params['temperature']}; the run configs set "
            f"{CONFIGURED_TEMPERATURE} for every request"
        )
    messages = params.get("input") or []
    if not all(isinstance(message, dict) for message in messages):
        raise AdapterError(f"row {row.get('id')}: input messages are not objects")
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user"]:
        raise AdapterError(f"row {row.get('id')}: expected a system and a user message, got {roles}")
    if not all(isinstance(message.get("content"), str) for message in messages):
        raise AdapterError(f"row {row.get('id')}: expected plain-text messages")
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, list) or not all(
        isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), str)
        for call in ground_truth
    ):
        raise AdapterError(
            f"row {row.get('id')}: ground_truth is not a list of {{name, arguments}} with JSON-text arguments"
        )
    return {
        "system_prompt": messages[0]["content"],
        "prompt": messages[1]["content"],
        "ground_truth": [{"name": call["name"], "arguments": call["arguments"]} for call in ground_truth],
        "id": row["id"],
        "category": row["category"],
        "environment_name": row["environment_name"],
    }


def native_opencode_row(row: dict) -> dict:
    """The row as OpenCode's sandbox session accepts it: the tools reach it over MCP, not in the request."""
    params = {key: value for key, value in row["responses_create_params"].items()}
    params.pop("tools", None)
    params.pop("parallel_tool_calls", None)
    return {
        key: (params if key == "responses_create_params" else value)
        for key, value in row.items()
        if key != "agent_ref"
    }


def select_rows(
    rows_path: Path, ids: list[int] | None, limit: int | None
) -> tuple[list[tuple[dict, bytes]], int, str]:
    """Stream the rows file; return the selected ``(row, raw line)`` pairs (in ``ids`` order, else file order), its
    line count and sha256. Refuses two selected rows with the same ``id``."""
    wanted = None if ids is None else set(ids)
    digest = hashlib.sha256()
    lines = 0
    selected: dict[int, tuple[dict, bytes]] = {}
    in_order: list[tuple[dict, bytes]] = []
    with rows_path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            lines += 1
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise AdapterError(f"{rows_path.name} line {lines}: not UTF-8 JSON") from None
            line = raw if raw.endswith(b"\n") else raw + b"\n"
            if wanted is None:
                if limit is None or len(in_order) < limit:
                    in_order.append((row, line))
            elif type(row.get("id")) is int and row["id"] in wanted:
                if row["id"] in selected:
                    raise AdapterError(f"{rows_path.name}: more than one row has id {row['id']}")
                selected[row["id"]] = (row, line)
    if wanted is not None:
        missing = [row_id for row_id in ids if row_id not in selected]
        if missing:
            raise AdapterError(f"ids not in {rows_path.name}: {missing}")
        in_order = [selected[row_id] for row_id in ids][:limit]
    seen: set[int] = set()
    for row, _ in in_order:
        row_id = row.get("id")
        if type(row_id) is int:
            if row_id in seen:
                raise AdapterError(f"{rows_path.name}: more than one row has id {row_id}")
            seen.add(row_id)
    return in_order, lines, digest.hexdigest()


def gym_commit() -> str:
    """The Gym commit the adapter runs from, or ``unknown`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def vendored_files() -> list[Path]:
    """The vendored files, relative to the server folder, sorted."""
    files = []
    for name in VENDORED:
        source = SERVER_DIR / name
        if source.is_dir():
            files.extend(
                path.relative_to(SERVER_DIR)
                for path in source.rglob("*")
                if path.is_file() and not IGNORED(str(path.parent), [path.name]) and "__pycache__" not in path.parts
            )
        else:
            files.append(Path(name))
    return sorted(files)


def vendoring_header(relative: Path, source: bytes, commit: str) -> bytes:
    """The provenance line put above a vendored Python file; the file's own bytes follow it unchanged."""
    return (
        f"# Vendored unchanged from {(Path('resources_servers') / 'workplace_assistant' / relative).as_posix()} at "
        f"{commit}; source sha256 {hashlib.sha256(source).hexdigest()}; license {VENDOR_LICENSE}\n"
    ).encode()


def dataset_readme(
    *, rows_name: str, rows_lines: int, rows_sha256: str, task_count: int, ids_note: str, commit: str, regenerate: str
) -> str:
    """The dataset's root README.md: where its tasks came from. Gym copies no root README into a task."""
    vendored = "".join(
        f"  - `{relative.as_posix()}` sha256 `{hashlib.sha256((SERVER_DIR / relative).read_bytes()).hexdigest()}`\n"
        for relative in vendored_files()
    )
    pinned = "".join(
        f"- `{name}` sha256 `{hashlib.sha256((TEMPLATE_DIR / name).read_bytes()).hexdigest()}` (written by"
        " `harbor/pin_requirements.py` from the Resources Server's `uv.lock`).\n"
        for name in ("environment/requirements.txt", "tests/requirements.txt")
    )
    return (
        "# workplace_assistant rows as Harbor tasks\n\n"
        "Generated by `resources_servers/workplace_assistant/harbor/adapter.py`; do not edit. Gym does not copy this\n"
        "file into the rendered tasks.\n\n"
        f"- Native rows file `{rows_name}`: {rows_lines} lines, sha256 `{rows_sha256}`.\n"
        f"- Tasks: {task_count}; {ids_note}.\n"
        f"- Gym commit: `{commit}`.\n"
        f"- Vendored unchanged (license {VENDOR_LICENSE}) from `resources_servers/workplace_assistant/` into\n"
        "  `environment/wa/resources_servers/workplace_assistant/` and "
        "`tests/wa/resources_servers/workplace_assistant/`.\n"
        "  Each Python copy starts with one `# Vendored unchanged from ...` line; "
        "the source's bytes follow it. Source\n"
        "  sha256:\n"
        f"{vendored}"
        f"{pinned}"
        "- Run it with `--config resources_servers/workplace_assistant/harbor/configs/harbor_arm.yaml` (and, for\n"
        "  `gym eval run`, `harbor_arm_harness.yaml`). The tasks verify in a separate sandbox "
        "from a journal the agent\n"
        "  can add to only by calling a tool; see `harbor/README.md`.\n"
        f"- Regenerate: `{regenerate}`\n"
    )


def write_dataset(
    rows: list[dict],
    output_dir: Path,
    *,
    prefix: str,
    solution: str,
    readme: str,
    commit: str,
) -> None:
    """Write the template, the vendored code, ``tasks.jsonl`` and ``README.md`` into ``output_dir``."""
    records = [{"task_id": task_id(row["id"], prefix), "task_data": task_data(row)} for row in rows]
    shutil.copytree(TEMPLATE_DIR, output_dir, ignore=IGNORED, dirs_exist_ok=True)
    for relative in vendored_files():
        source = (SERVER_DIR / relative).read_bytes()
        content = vendoring_header(relative, source, commit) + source if relative.suffix == ".py" else source
        for target in VENDOR_TARGETS:
            destination = output_dir / target / VENDOR_PACKAGE / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    if solution != "reference":
        calls = [] if solution == "none" else WRONG_CALLS
        (output_dir / "solution" / "calls.template.json").rename(output_dir / "solution" / "calls.json")
        (output_dir / "solution" / "calls.json").write_text(json.dumps(calls) + "\n")
    with (output_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=Path, required=True, help="native rows (JSONL), e.g. the HF validation split")
    parser.add_argument("--output-dir", type=Path, required=True, help="dataset folder to create (absent or empty)")
    parser.add_argument("--ids-file", type=Path, help="row ids to convert, one per line, in this order; # comments")
    parser.add_argument("--native-rows-out", type=Path, help="also write the selected raw lines, in the same order")
    parser.add_argument(
        "--opencode-rows-out",
        type=Path,
        help="also write the selected rows as the native OpenCode config (harbor/configs/native_pairing.yaml) runs them",
    )
    parser.add_argument("--id-map-out", type=Path, help="also write JSON {row id: task id} for harbor_parity --id-map")
    parser.add_argument("--limit", type=int, help="convert only the first N selected rows")
    parser.add_argument("--task-prefix", default=DEFAULT_PREFIX, help="task ids are <prefix>-<row id, 4 digits>")
    parser.add_argument("--solution", choices=["reference", "none", "wrong"], default="reference")
    args = parser.parse_args(argv)

    # Inputs by file name and the output as <dir>: the README must not depend on where they live.
    regenerate = f"python resources_servers/workplace_assistant/harbor/adapter.py --rows {args.rows.name}"
    regenerate += " --output-dir <dir>"
    if args.ids_file:
        regenerate += f" --ids-file {args.ids_file.name}"
    if args.native_rows_out:
        regenerate += " --native-rows-out <native rows>"
    if args.opencode_rows_out:
        regenerate += " --opencode-rows-out <opencode rows>"
    if args.id_map_out:
        regenerate += " --id-map-out <id map>"
    for flag, value, default in (
        ("--limit", args.limit, None),
        ("--task-prefix", args.task_prefix, DEFAULT_PREFIX),
        ("--solution", args.solution, "reference"),
    ):
        if value != default:
            regenerate += f" {flag} {value}"
    created = False
    try:
        if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            raise AdapterError(f"{args.output_dir} is not empty; choose a new --output-dir")
        if args.limit is not None and args.limit < 1:
            raise AdapterError(f"--limit {args.limit}: convert at least one row")
        ids = read_ids(args.ids_file) if args.ids_file else None
        selected, lines, rows_sha256 = select_rows(args.rows, ids, args.limit)
        rows = [row for row, _ in selected]
        if not rows:
            raise AdapterError(f"no rows selected from {args.rows.name}")
        # Each row as written, checked against the dataset's own TaskData the way Gym checks it at prepare.
        schema = load_task_data_schema(TEMPLATE_DIR)
        if schema is None:
            raise AdapterError(f"{TEMPLATE_DIR}/task_data.py is missing")
        for row in rows:
            schema.validate(task_data(row), task_id=task_id(row["id"], args.task_prefix))
        commit = gym_commit()
        ids_note = (
            f"ids file `{args.ids_file.name}`, sha256 `{hashlib.sha256(args.ids_file.read_bytes()).hexdigest()}`"
            if args.ids_file
            else "no ids file (rows in file order)"
        )
        readme = dataset_readme(
            rows_name=args.rows.name,
            rows_lines=lines,
            rows_sha256=rows_sha256,
            task_count=len(rows),
            ids_note=ids_note,
            commit=commit,
            regenerate=regenerate,
        )
        # The output folder and the side files first: a path that cannot be written stops the run before the dataset
        # exists.
        created = not args.output_dir.exists()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        side_files = {
            args.native_rows_out: b"".join(line for _, line in selected),
            args.opencode_rows_out: "".join(json.dumps(native_opencode_row(row)) + "\n" for row in rows).encode(),
            args.id_map_out: (
                json.dumps(
                    {str(row["id"]): task_id(row["id"], args.task_prefix) for row in rows}, indent=2, sort_keys=True
                )
                + "\n"
            ).encode(),
        }
        for path, content in side_files.items():
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    except (AdapterError, TaskDataError, OSError) as error:
        print(f"adapter: {error}", file=sys.stderr)
        if created and args.output_dir.is_dir() and not any(args.output_dir.iterdir()):
            args.output_dir.rmdir()
        return 2

    try:
        write_dataset(
            rows, args.output_dir, prefix=args.task_prefix, solution=args.solution, readme=readme, commit=commit
        )
    except OSError as error:
        print(f"adapter: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} tasks to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
