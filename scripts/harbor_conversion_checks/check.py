# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`check DATASET`: content checks of converted task folders that `gym dataset validate` does not make (G1).

Five checks, in order: `instruction` (instruction.md equals the native prompt), `hidden-values` (hidden native
values do not reach instruction.md or environment/), `hidden-template` (agent-visible template files do not render
hidden task_data fields), `secrets` and `local-paths`. Loading, unknown task.toml keys and skipped tasks are
`gym dataset validate`'s job.

Gym reads instruction.md in text mode, so a native prompt with CRLF or CR line endings reaches the Harbor model with
LF: a real difference that `--normalize-newlines` accepts and counts, so the log can record it.
"""

import argparse
import ast
import json
import re
import tomllib
from pathlib import Path

from scripts.harbor_conversion_checks.messages import (
    InputError,
    field_value,
    id_text,
    iter_jsonl,
    lf_newlines,
    message_text,
    read_id_list,
    task_folders,
)
from scripts.harbor_conversion_checks.scan import candidates, contents, layers, leaves


CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
SECRETS = [
    re.compile(pattern)
    for pattern in (
        rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}",
        rb"nvapi-",
        rb"hf_[A-Za-z]{30,}",
        rb"gh[pousr]_[A-Za-z0-9]{30,}",
        rb"AKIA[0-9A-Z]{16}",
        rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        rb"glpat-[A-Za-z0-9_-]{20,}",
        rb"github_pat_[A-Za-z0-9_]{30,}",
        rb"xox[baprs]-[A-Za-z0-9-]{10,}",
    )
]
FROM_LINE = re.compile(rb"^\s*FROM\s+(?:--\S+\s+)*(\S+)", re.MULTILINE | re.IGNORECASE)
# nemo_gym.data.tasks.harbor.parameterized: the placeholders a `.template` file is rendered with.
PLACEHOLDER = re.compile(rb"\{\{\s*(?:task_data\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
TEMPLATE_MARKER = ".template"
# Template folders the agent never sees: the oracle's solution and the verifier's tests.
TEMPLATE_HIDDEN_FOLDERS = {"solution", "tests"}
VISIBLE_CONSUMERS = {"prompt", "provenance"}
# Bases of a TaskData model that declare no field of the row.
FIELDLESS_BASES = {"pydantic.BaseModel", "typing.Generic"}
SHOWN = 20


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path, help="a task folder or a folder of them (a rendered template)")
    parser.add_argument("--native-rows", type=Path, help="the native JSONL rows the tasks came from")
    parser.add_argument("--native-id", help="dotted field naming a native row, or _index (0-based line)")
    parser.add_argument("--id-map", type=Path, help="JSON object {native id: task id}; default: the same id")
    parser.add_argument("--prefix-file", type=Path, help="text the adapter puts before the native prompt")
    parser.add_argument(
        "--normalize-newlines",
        action="store_true",
        help="read CRLF and CR in the native prompt as LF, as Gym reads instruction.md; affected tasks are counted",
    )
    parser.add_argument("--hidden", action="append", default=[], help="dotted native field the agent must not see")
    parser.add_argument(
        "--hidden-from",
        type=Path,
        help="the Resources Server's task_data.py: fields consumed_by neither 'prompt' nor 'provenance' are hidden",
    )
    parser.add_argument("--template", type=Path, help="the parameterized template the dataset was rendered from")
    parser.add_argument(
        "--template-hidden", action="append", default=[], help="task_data field the agent must not see"
    )
    parser.add_argument(
        "--allow-values",
        type=Path,
        help="hidden values reviewed as not a leak (one per line, # comments): their matches are listed, not failed",
    )
    parser.add_argument("--json", type=Path, help="write the report as JSON")


def run(args: argparse.Namespace) -> int:
    dataset = args.dataset
    tasks = _task_folders(dataset)
    if (args.hidden or args.hidden_from or args.native_id) and not args.native_rows:
        raise InputError("--hidden, --hidden-from and --native-id need --native-rows")
    if args.template_hidden and not args.template:
        raise InputError("--template-hidden needs --template")
    if args.normalize_newlines and not args.native_rows:
        raise InputError("--normalize-newlines needs --native-rows")
    allowed = set(read_id_list(args.allow_values)) if args.allow_values else set()
    rows, prefix = {}, ""
    if args.native_rows:
        if not args.native_id:
            raise InputError("--native-rows needs --native-id")
        rows = _native_rows(args.native_rows, args.native_id, args.id_map, set(tasks))
        prefix = _read_text(args.prefix_file) if args.prefix_file else ""
    hidden = list(dict.fromkeys(args.hidden + (_hidden_fields(args.hidden_from) if args.hidden_from else [])))
    explicit = set(args.hidden)
    base = dataset if dataset.is_dir() and not (dataset / "task.toml").is_file() else dataset.parent
    checks = [
        _check_instruction(tasks, rows, prefix, args.normalize_newlines)
        if args.native_rows
        else _skip("instruction", "no --native-rows"),
        _check_hidden_values(tasks, rows, hidden, explicit, prefix, base, allowed)
        if hidden
        else _skip("hidden-values", "no hidden fields"),
        _check_template(args.template, args.template_hidden)
        if args.template_hidden
        else _skip("hidden-template", "no --template-hidden"),
        *_check_files(dataset),
    ]
    for check in checks:
        print(f"{check['status']} {check['name']}: {check['detail']}")
        for failure in check["failures"][:SHOWN]:
            print(f"  {failure}")
        if len(check["failures"]) > SHOWN:
            print(f"  ... {len(check['failures']) - SHOWN} more")
        for match in check.get("allowed", [])[:SHOWN]:
            print(f"  allowed: {match}")
        if len(check.get("allowed", [])) > SHOWN:
            print(f"  ... {len(check['allowed']) - SHOWN} more allowed")
    counts = {status: sum(check["status"] == status for check in checks) for status in ("PASS", "FAIL", "SKIP")}
    print(f"check: {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    if args.json:
        args.json.write_text(json.dumps({"dataset": str(dataset), "checks": checks}, indent=1) + "\n")
    return 1 if counts["FAIL"] else 0


def _task_folders(dataset: Path) -> dict[str, Path]:
    if (dataset / "task.toml").is_file() and (dataset / "tasks.jsonl").is_file():
        raise InputError(
            f"{dataset} is a parameterized template: pass the rendered folder results/harbor/<taskset>/rendered that "
            "`gym dataset validate` writes (or the one under +harbor_output_root)"
        )
    tasks = {folder.name: folder for folder in task_folders(dataset)}
    if not tasks:
        raise InputError(f"{dataset}: no task folder (a folder with task.toml) found")
    return tasks


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


def _native_rows(path: Path, native_id: str, id_map_path: Path | None, task_ids: set[str]) -> dict[str, dict]:
    id_map = None
    if id_map_path:
        try:
            id_map = json.loads(_read_text(id_map_path))
        except json.JSONDecodeError as exc:
            raise InputError(f"--id-map {id_map_path}: not JSON ({exc})") from exc
        if not isinstance(id_map, dict):
            raise InputError(f"--id-map {id_map_path}: expected a JSON object of native id -> task id")
    rows: dict[str, dict] = {}
    for index, row in iter_jsonl(path):
        try:
            native = id_text(field_value(row, native_id, index))
        except KeyError:
            raise InputError(f"{path} line {index + 1}: no field {native_id}") from None
        task_id = id_map.get(native) if id_map is not None else native
        if task_id in task_ids:
            if task_id in rows:
                raise InputError(f"{path}: two rows name task {task_id}")
            rows[task_id] = row
    return rows


def _hidden_fields(task_data_py: Path) -> list[str]:
    """Fields of the `TaskData` model, inherited ones included, whose `json_schema_extra["consumed_by"]` lists
    neither "prompt" nor "provenance": what only the verifier (or metrics) reads. Provenance fields (ids, repository
    names) are meant to be visible."""
    fields = _class_fields(task_data_py.resolve(), "TaskData", frozenset())
    return [name for name, consumers in fields.items() if not VISIBLE_CONSUMERS & set(consumers)]


def _class_fields(path: Path, name: str, seen: frozenset) -> dict[str, list[str]]:
    """The annotated fields of class `name` as module `path` defines or imports it, with their `consumed_by`: those of
    its bases first (a field a subclass declares again keeps its place, as in pydantic). A base is followed through
    `from MODULE import NAME` to MODULE's file, found under a folder above `path` (or relative to it); a base that
    cannot be followed is an input error, not a model without fields."""
    if (path, name) in seen:
        raise InputError(f"{path}: class {name} inherits from itself")
    seen |= {(path, name)}
    try:
        tree = ast.parse(_read_text(path))
    except SyntaxError as exc:
        raise InputError(f"{path}: {exc}") from exc
    model = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name), None)
    if model is None:
        imported = next(
            (
                (node.module or "", node.level, alias.name)
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
                if (alias.asname or alias.name) == name
            ),
            None,
        )
        if imported is None:
            raise InputError(f"{path}: no class {name}, and no import of it")
        module, level, original = imported
        if not level and f"{module}.{original}" in FIELDLESS_BASES:
            return {}
        source = _module_file(path, module, level)
        if source is None:
            raise InputError(f"{path}: cannot find the module of {name}: name its hidden fields with --hidden")
        return _class_fields(source, original, seen)
    fields: dict[str, list[str]] = {}
    for base in reversed(model.bases):
        base = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(base, ast.Name):
            fields.update(_class_fields(path, base.id, seen))
        elif ast.unparse(base) not in FIELDLESS_BASES:
            raise InputError(
                f"{path}: cannot follow base {ast.unparse(base)} of {name}: name its fields with --hidden"
            )
    for node in model.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields[node.target.id] = _consumed_by(node.value)
    return fields


def _module_file(path: Path, module: str, level: int) -> Path | None:
    """The file of `module` as imported from `path`: relative to it, or under the nearest folder above it that has
    the module (the repository root for `resources_servers.<server>.task_data`)."""
    parts = module.split(".") if module else []
    if not parts or level > len(path.parents):
        return None
    roots = [path.parents[level - 1]] if level else list(path.parents)
    for root in roots:
        for candidate in (root.joinpath(*parts[:-1], f"{parts[-1]}.py"), root.joinpath(*parts, "__init__.py")):
            if candidate.is_file():
                return candidate
    return None


def _consumed_by(value: ast.expr | None) -> list[str]:
    if not isinstance(value, ast.Call):
        return []
    for keyword in value.keywords:
        if keyword.arg == "json_schema_extra" and isinstance(keyword.value, ast.Dict):
            for key, item in zip(keyword.value.keys, keyword.value.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "consumed_by"
                    and isinstance(item, (ast.List, ast.Tuple))
                ):
                    return [element.value for element in item.elts if isinstance(element, ast.Constant)]
    return []


def _skip(name: str, detail: str) -> dict:
    return {"name": name, "status": "SKIP", "detail": detail, "failures": []}


def _result(name: str, failures: list[str], passed: str) -> dict:
    status = "FAIL" if failures else "PASS"
    return {
        "name": name,
        "status": status,
        "detail": f"{len(failures)} failure(s)" if failures else passed,
        "failures": failures,
    }


def read_instruction(path: Path) -> str:
    """instruction.md as the agent sees it: leading canary lines and the blank lines after them dropped."""
    lines = path.read_text(encoding="utf-8").split("\n")
    removed = False
    while lines and CANARY_LINE.match(lines[0].strip()):
        lines.pop(0)
        removed = True
    while removed and lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _prompt_parts(row: dict) -> tuple[list[str], list[str]]:
    messages = (row.get("responses_create_params") or {}).get("input") or []
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    system = [
        message_text(m.get("content"))
        for m in messages
        if isinstance(m, dict) and m.get("role") in ("system", "developer")
    ]
    users = [message_text(m.get("content")) for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    return system, users


def _check_instruction(tasks: dict[str, Path], rows: dict[str, dict], prefix: str, normalize: bool) -> dict:
    failures, normalized = [], 0
    for task_id, folder in tasks.items():
        if task_id not in rows:
            failures.append(f"{task_id}: no native row")
            continue
        system, users = _prompt_parts(rows[task_id])
        if len(users) != 1:
            failures.append(f"{task_id}: the native row has {len(users)} user messages, not 1")
            continue
        if not (folder / "instruction.md").is_file():
            failures.append(f"{task_id}: no instruction.md")
            continue
        expected = prefix + "\n\n".join(system) + ("\n\n" if system else "") + users[0]
        actual = read_instruction(folder / "instruction.md")
        lf_expected = lf_newlines(expected).removesuffix("\n")
        expected, actual = expected.removesuffix("\n"), actual.removesuffix("\n")
        if actual != expected and lf_expected == actual:
            if normalize:
                normalized += 1
                continue
            failures.append(
                f"{task_id}: differs only in line endings (the native prompt has CR; Gym reads instruction.md as LF): "
                "record it as a difference and pass --normalize-newlines"
            )
            continue
        if actual != expected:
            at = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
            failures.append(
                f"{task_id}: differs at offset {at}: "
                f"instruction {actual[at : at + 40]!r} vs native {expected[at : at + 40]!r}"
            )
    result = _result("instruction", failures, f"{len(tasks)} task(s) equal their native prompt")
    if normalized:
        result["detail"] += f"; {normalized} only with CRLF/CR read as LF (a difference to record)"
    return result


def _environment_table(folder: Path) -> object:
    try:
        config = tomllib.loads((folder / "task.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        config = {}
    return config.get("environment") or {}


def _task_overrides(base: Path) -> tuple[dict, str | None]:
    """The `[gym.tasks]` table of `<base>/dataset.toml` (Gym's per-task overrides, keyed by task folder name), and a
    problem when the file has a shape Gym would not load, so that its overrides are not scanned."""
    path = base / "dataset.toml"
    if not path.is_file():
        return {}, None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return {}, f"{path.relative_to(base)}: cannot be read ({exc}): its overrides are not scanned"
    gym = data.get("gym") or {}
    tasks = gym.get("tasks", {}) if isinstance(gym, dict) else None
    if not isinstance(tasks, dict):
        return {}, f"{path.relative_to(base)}: [gym] or [gym.tasks] is not a table: its overrides are not scanned"
    return tasks, None


def effective_environment(environment: object, override: object) -> tuple[object, set[str], bool]:
    """task.toml's `[environment]` table with a dataset.toml `[gym.tasks."<id>"]` override applied as Gym's loader
    applies it (`apply_dataset_config`): the override's `environment.env` merges over the task's own `env` (a
    variable it names takes its value, an empty string included; none is removed) and every other key of its
    `environment` replaces the task's. Returns the table, the `env` variables the override set, and whether the
    override was applied: False when its shape is one the loader refuses or this does not model."""
    if override is None:
        return environment, set(), True
    table = override.get("environment", {}) if isinstance(override, dict) else None
    env = table.get("env") if isinstance(table, dict) else None
    own = environment.get("env") if isinstance(environment, dict) else None
    if (
        not isinstance(table, dict)
        or not isinstance(environment, dict)
        or (env is not None and not (isinstance(env, dict) and all(isinstance(v, str) for v in env.values())))
        or not isinstance(own or {}, dict)
    ):
        return environment, set(), False
    merged = dict(environment)
    for key, value in table.items():
        merged[key] = dict(own or {}) | value if key == "env" else value
    return merged, set(env or {}), True


def _environment_settings(folder: Path, environment: object) -> set[str]:
    """What configures the agent's own environment, visible by design: the effective `[environment]` table's
    `docker_image` and `workdir`, and the images `FROM` lines of environment/Dockerfile name. An image pinned by digest
    also counts without it (`<image>@sha256:<hex>` names `<image>`). The `env` variables and the other settings that
    reach the agent (`mcp_servers`, `skills_dir`, `healthcheck`) are scanned instead: an adapter could put a reference
    answer there."""
    settings = set()
    if isinstance(environment, dict):
        settings = {leaf.strip() for key in ("docker_image", "workdir") for leaf in leaves(environment.get(key))}
    dockerfile = folder / "environment" / "Dockerfile"
    if dockerfile.is_file():
        settings.update(image.decode(errors="replace") for image in FROM_LINE.findall(dockerfile.read_bytes()))
    return settings | {setting.partition("@sha256:")[0] for setting in settings}


def _check_hidden_values(
    tasks: dict[str, Path],
    rows: dict[str, dict],
    fields: list[str],
    explicit: set[str],
    prefix: str,
    base: Path,
    allowed: set[str],
) -> dict:
    failures, problems, scanned, short, allowed_matches = [], [], 0, 0, []
    absent: dict[str, int] = {}
    with_row = 0
    overrides, problem = _task_overrides(base)
    if problem:
        problems.append(problem)
    for task_id, folder in tasks.items():
        row = rows.get(task_id)
        if row is None:
            continue
        with_row += 1
        system, users = _prompt_parts(row)
        tools = (row.get("responses_create_params") or {}).get("tools") or []
        shown = "\n".join([prefix, *system, *users, *leaves(tools)])
        environment, overridden, applied = effective_environment(_environment_table(folder), overrides.get(task_id))
        if not applied:
            problems.append(f'dataset.toml [gym.tasks."{task_id}"]: a shape Gym does not load: not scanned')
        settings = _environment_settings(folder, environment)
        found: dict[str, str] = {}
        for name in fields:
            try:
                value = field_value(row, name, 0)
            except KeyError:
                absent[name] = absent.get(name, 0) + 1
                continue
            texts, too_short = candidates(value)
            short += too_short
            found.update((text, name) for text in texts if text not in shown and text not in settings)
        scanned += len(found)
        if not found:
            continue
        # A JSON file in environment/ holds a value escaped: quotes, backslashes, newlines and (by default) non-ASCII.
        forms = {
            text: {text.encode(), json.dumps(text)[1:-1].encode(), json.dumps(text, ensure_ascii=False)[1:-1].encode()}
            for text in found
        }
        # The effective `env` reaches the agent: each variable's name and value get the traversal files get. So do
        # the MCP servers, the skills folder and the healthcheck command, which the harbor Environment Server hands
        # the harness or writes into the agent's sandbox.
        task_toml = f"{(folder / 'task.toml').relative_to(base)} [environment]"
        override_toml = f'dataset.toml [gym.tasks."{task_id}".environment]'
        env = (environment.get("env") or {}) if isinstance(environment, dict) else {}
        if not isinstance(env, dict):
            env = {"": env}
        entries = []
        for key, value in env.items():
            source = f"{override_toml if key in overridden else task_toml}.env"
            entries += ([(f"{source} name", key)] if key else []) + [(source, leaf) for leaf in leaves(value)]
        override = (overrides.get(task_id) or {}).get("environment") if applied else None
        for key in ("mcp_servers", "skills_dir", "healthcheck"):
            source = f"{override_toml if isinstance(override, dict) and key in override else task_toml}.{key}"
            value = environment.get(key) if isinstance(environment, dict) else None
            entries += [(source, leaf) for leaf in leaves(value)]
        for entry_label, text in entries:
            for label, data in layers(entry_label, text.encode(errors="replace"), problems):
                failures += [
                    f"{label} contains {name} value {hidden[:60]!r}"
                    for hidden, name in sorted(found.items())
                    if any(form in data for form in forms[hidden])
                ]
        instructions = [folder / "instruction.md", *sorted((folder / "steps").glob("*/instruction.md"))]
        # A step's workdir/ is uploaded into the agent's working directory before its turn; its tests/ and
        # solution/ stay with the verifier and the oracle.
        staged = [folder / "environment", *sorted((folder / "steps").glob("*/workdir"))]
        visible = instructions + [p for root in staged for p in sorted(root.rglob("*")) if p.is_file()]
        for path in visible:
            if not path.is_file():
                continue
            for label, data in contents(path, base, problems):
                for text, name in sorted(found.items()):
                    if any(form in data for form in forms[text]):
                        line = f"{label} contains {name} value {text[:60]!r}"
                        reviewed = text in allowed and path not in instructions
                        (allowed_matches if reviewed else failures).append(line)
    typos = sorted(name for name in explicit if with_row and absent.get(name) == with_row)
    if typos:
        name = typos[0]
        raise InputError(
            f"--hidden {name}: no native row has field {name} "
            f"(a typo, or a nested path such as verifier_metadata.{name}?)"
        )
    everywhere = sorted(name for name, count in absent.items() if count == with_row)
    somewhere = sorted(name for name, count in absent.items() if count < with_row)
    note = f"; {short} short value(s) not scanned: use --template-hidden" if short else ""
    note += f"; absent from some rows: {', '.join(somewhere)}" if somewhere else ""
    note += f"; absent from every row: {', '.join(everywhere)}" if everywhere else ""
    result = _result(
        "hidden-values",
        failures + problems,
        f"none of {scanned} hidden value(s) of {len(rows)} row(s) in instruction.md or environment/",
    )
    result["detail"] += note
    if allowed_matches:
        result["detail"] += f"; {len(allowed_matches)} match(es) of --allow-values values, listed under 'allowed'"
        result["allowed"] = allowed_matches
    return result


def _is_template_file(name: str) -> bool:
    """Whether Gym renders the file: `<stem>.template<suffix>`, with a non-empty stem and a suffix that is empty or
    starts with a dot (nemo_gym.data.tasks.harbor.parameterized)."""
    stem, marker, suffix = name.rpartition(TEMPLATE_MARKER)
    return bool(marker and stem) and (not suffix or suffix.startswith("."))


def _check_template(template: Path, names: list[str]) -> dict:
    if not (template / "task.toml").is_file():
        raise InputError(f"--template {template}: no task.toml")
    hidden = {name.encode() for name in names}
    failures = []
    for path in sorted(p for p in template.rglob("*") if p.is_file() and _is_template_file(p.name)):
        parts = path.relative_to(template).parts
        if parts[0] in TEMPLATE_HIDDEN_FOLDERS or (
            parts[0] == "steps" and len(parts) > 2 and parts[2] in TEMPLATE_HIDDEN_FOLDERS
        ):
            continue
        # Gym's placeholder allows any whitespace, newlines included, inside the braces: match the whole file.
        data = path.read_bytes()
        for match in PLACEHOLDER.finditer(data):
            if match.group(1) in hidden:
                number = data.count(b"\n", 0, match.start()) + 1
                shown = b" ".join(match.group(0).split()).decode()
                failures.append(f"{path.relative_to(template)}:{number}: renders {shown}")
    return _result("hidden-template", failures, f"no agent-visible template file renders {', '.join(names)}")


def _check_files(dataset: Path) -> list[dict]:
    home, cwd = str(Path.home()), str(Path.cwd())
    # Home and the current directory count only as whole path components (/home/al is not in /home/alice). A home of
    # /root is left out: images name /root/ as their own fixed path.
    local = [cwd] if home in ("/", "/root") else [home, cwd]
    needles = {
        needle: re.compile(re.escape(needle.encode()) + rb"(?![A-Za-z0-9._-])") for needle in local if len(needle) > 1
    }
    for needle in ("/lustre/", "/raid/", "/scratch/"):
        needles.setdefault(needle, re.compile(re.escape(needle.encode())))
    secrets, paths, problems, files = [], [], [], 0
    base = dataset if dataset.is_dir() else dataset.parent
    for path in sorted(p for p in dataset.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        files += 1
        for label, data in contents(path, base, problems):
            if b"\0" in data[:8192]:
                continue
            for pattern in SECRETS:
                for match in pattern.finditer(data):
                    secrets.append(f"{label}: {match.group()[:12].decode(errors='replace')}...")
            for needle, pattern in needles.items():
                if pattern.search(data):
                    paths.append(f"{label}: {needle}")
    return [
        _result("secrets", secrets + problems, f"no secret pattern in {files} file(s)"),
        _result("local-paths", paths + problems, f"no local path in {files} file(s)"),
    ]
