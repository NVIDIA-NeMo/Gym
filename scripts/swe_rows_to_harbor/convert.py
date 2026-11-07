# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert SWE resources-server rows into Harbor task folders.

A row of ``scale_swe`` or ``swe_rebench`` becomes a task the ``harbor`` environment server runs the way the
resources server runs the row through ``single_agent_turn``:

- **Sandbox**: the row's own image. The server's seed setup (``pre_commands``, a git repository when the image has
  none, the anti-cheat history scrub, a committer identity, the base commit and the files already untracked) runs
  as the ``RUN`` line of a no-build Dockerfile, before the agent starts. Each step is non-fatal, as at the server.
- **Prompt**: ``instruction.md`` holds the row's user message byte for byte.
- **Patch**: a ``[[verifier.collect]]`` hook runs the server's capture command (``git add -N . && git diff <base>``)
  in the agent's sandbox once the agent is done and saves the patch under ``/logs/artifacts``.
- **Grading**: a fresh sandbox of the same image (separate verifier mode), where ``tests/test.sh`` runs
  ``tests/nemo_gym/grade.py``. That script calls the server's own grading functions, vendored from this checkout
  under the same relative paths (and, for ``swe_rebench``, the upstream log parsers the server fetches).
- **Limits**: the server's ``evaluation_timeout`` and sandbox resources, read from its config in this checkout.
  ``--evaluation-timeout-sec`` replaces the timeout with the one the compared run sets for every server.
- **Golden patch**: ``solution/solve.sh`` applies the row's patch, so ``gym dataset validate`` checks each task.

Rows are JSONL, either typed (``{"task_id": ..., "task_input": {...}}``) or flat; the source comes from the row's
``source_server`` or ``--source``. Run it from this checkout's environment::

    python scripts/swe_rows_to_harbor/convert.py rows.jsonl out/swe-tasks --dataset-name nemo-gym/swe-tasks

``--skip-unsupported`` leaves out rows of other sources and lists them in ``conversion.json``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
GRADER_SOURCE = Path(__file__).resolve().parent / "grade.py"

# Where the setup leaves what the capture hook reads; outside the repository, so the diff never sees it.
STATE_DIR = "/var/lib/nemo_gym"
ARTIFACTS_DIR = "/logs/artifacts/nemo_gym"
# MR 69's agent budget; set --agent-timeout-sec to the run's own.
DEFAULT_AGENT_TIMEOUT_SEC = 4800
# Room for the grader around the server's evaluation_timeout, which bounds the test run itself.
VERIFIER_TIMEOUT_MARGIN_SEC = 300
CAPTURE_TIMEOUT_SEC = 300
# The upstream log parsers need 3.10 (their signatures use `X | None`).
MIN_PYTHON = (3, 10)
# Gym drops leading lines like these from instruction.md, which would change the prompt.
_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_TASK_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def load_repo_module(relative: str) -> ModuleType:
    """Import a module of this checkout by path, so the converter reads exactly the code it vendors."""
    name = "swe_rows_to_harbor_" + relative.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def server_config(source: str) -> dict[str, Any]:
    """The source server's own settings, from ``resources_servers/<source>/configs/<source>.yaml``."""
    path = REPO_ROOT / "resources_servers" / source / "configs" / f"{source}.yaml"
    document = yaml.safe_load(path.read_text())
    return document[f"{source}_resources_server"]["resources_servers"][source]


@dataclass
class TaskPlan:
    """What one converted task needs, filled in by the source's adapter."""

    source: str
    instance_id: str
    image: str
    workdir: str
    prompt: str
    golden_patch: str
    setup_script: str
    capture_command: str
    grading_row: dict[str, Any]
    vendored: dict[str, Path]
    evaluation_timeout: int
    resources: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    extra_task_json: dict[str, Any] = field(default_factory=dict)


class SourceAdapter:
    """Shared seed and capture steps; ``resources_servers/swebench`` holds them for every SWE server."""

    name: str

    def __init__(self) -> None:
        self.config = server_config(self.name)
        if self.config.get("patch_capture_mode", "worktree") != "worktree":
            raise NotImplementedError(f"{self.name}: only the worktree patch capture is converted")
        patch_capture = load_repo_module("resources_servers/swebench/patch_capture.py")
        self.git_email = patch_capture.GIT_COMMIT_IDENTITY_EMAIL
        self.git_name = patch_capture.GIT_COMMIT_IDENTITY_NAME
        self.anti_cheat_script = (REPO_ROOT / "resources_servers/swebench/anti_cheat_setup.sh").read_text()

    def plan(self, row: dict[str, Any]) -> TaskPlan:
        raise NotImplementedError

    def resources(self) -> dict[str, Any]:
        return dict((self.config.get("sandbox_config") or {}).get("resources") or {})

    def anti_cheat_steps(self, workdir: str) -> list[str]:
        """``swebench/anti_cheat.py::apply_anti_cheat_setup``: upload the script into the workdir, then run it."""
        if not self.config.get("apply_anti_cheating", True):
            return []
        script_path = f"{workdir.rstrip('/')}/anti_cheat_setup.sh"
        command = (
            f"git reset --hard && WORKING_DIRECTORY={workdir} bash anti_cheat_setup.sh && rm -f anti_cheat_setup.sh"
        )
        return [
            f"cat > {shlex.quote(script_path)} <<'NEMO_GYM_ANTI_CHEAT_EOF'\n"
            f"{self.anti_cheat_script.rstrip(chr(10))}\nNEMO_GYM_ANTI_CHEAT_EOF",
            _non_fatal(f"cd {shlex.quote(workdir)} && timeout 600 bash -c {shlex.quote(command)}", "anti-cheat setup"),
        ]

    def identity_step(self, workdir: str) -> str:
        """``swebench/patch_capture.py::prepare_git_for_commits``."""
        wd = shlex.quote(workdir)
        email, name = shlex.quote(self.git_email), shlex.quote(self.git_name)
        command = (
            f"git config --global user.email {email} && git config --global user.name {name}"
            f" && git config --global --add safe.directory {wd}"
            f" ; git -C {wd} config user.email {email} ; git -C {wd} config user.name {name}"
        )
        return _non_fatal(f"timeout 60 bash -c {shlex.quote(command)}", "git identity setup")

    def pristine_step(self, workdir: str) -> str:
        """The server's ``_pristine_untracked_files``: what the diff must leave out."""
        return (
            f"git -C {shlex.quote(workdir)} ls-files --others --exclude-standard "
            f"> {STATE_DIR}/pristine_untracked.txt 2>/dev/null || true"
        )

    def capture_command(self, workdir: str, base_commit: str) -> str:
        """``swebench/patch_capture.py::capture_model_patch`` in worktree mode, run by the collect hook.

        Written to a temporary file and moved into place only when git succeeds: a failed capture leaves no
        patch file, which the grader reads as the server's capture failure (graded as an empty patch).
        """
        wd = shlex.quote(workdir)
        return (
            f"mkdir -p {ARTIFACTS_DIR}; cp {STATE_DIR}/pristine_untracked.txt {ARTIFACTS_DIR}/ 2>/dev/null; "
            f'base={base_commit} && [ -n "$base" ] '
            f'&& git -C {wd} add -N . && git -C {wd} --no-pager diff "$base" > /tmp/nemo_gym_model_patch.diff '
            f"&& mv /tmp/nemo_gym_model_patch.diff {ARTIFACTS_DIR}/model_patch.diff"
        )


class ScaleSWEAdapter(SourceAdapter):
    """``resources_servers/scale_swe/app.py``: seed in ``_seed_session``, grading in ``_verify``."""

    name = "scale_swe"

    def __init__(self) -> None:
        super().__init__()
        self.verification = load_repo_module("resources_servers/scale_swe/verification.py")

    def plan(self, row: dict[str, Any]) -> TaskPlan:
        workdir = row["workdir"]
        wd = shlex.quote(workdir)
        steps = [f"mkdir -p {STATE_DIR}"]
        pre = self.verification.clean_commands(row.get("pre_commands") or "")
        if pre:
            steps.append(f"( cd {wd} && {pre}\n) || echo '[nemo_gym] pre_commands failed (non-fatal)' >&2")
        init = (
            f"cd {wd} && git init -q "
            f"&& git config user.email nemo-gym@nvidia.com && git config user.name nemo-gym "
            f"&& git add -A && git commit -q -m 'nemo_gym: initial snapshot' --allow-empty"
        )
        steps.append(
            f"if ! git -C {wd} rev-parse --git-dir >/dev/null 2>&1; then\n"
            f"  {_non_fatal(f'( {init} )', 'git init')}\nfi"
        )
        steps += self.anti_cheat_steps(workdir)
        steps.append(self.identity_step(workdir))
        steps.append(f"git -C {wd} rev-parse HEAD > {STATE_DIR}/base_commit 2>/dev/null || true")
        steps.append(self.pristine_step(workdir))
        return TaskPlan(
            source=self.name,
            instance_id=row["instance_id"],
            image=row["image_url"],
            workdir=workdir,
            prompt=_prompt(row),
            golden_patch=row.get("patch") or "",
            setup_script=_setup_script(self.name, steps),
            capture_command=self.capture_command(workdir, f"$(cat {STATE_DIR}/base_commit 2>/dev/null)"),
            grading_row=_pick(
                row,
                "instance_id",
                "workdir",
                "pre_commands",
                "f2p_patch",
                "f2p_script",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
            ),
            vendored={
                "resources_servers/scale_swe/verification.py": REPO_ROOT
                / "resources_servers/scale_swe/verification.py"
            },
            evaluation_timeout=int(self.config.get("evaluation_timeout") or 1200),
            resources=self.resources(),
        )


class SWERebenchAdapter(SourceAdapter):
    """``resources_servers/swe_rebench/app.py``: seed in ``_seed_session``, grading in ``_verify``."""

    name = "swe_rebench"

    def __init__(self, parser_cache_dir: Path | None = None) -> None:
        super().__init__()
        self.verification = load_repo_module("resources_servers/swe_rebench/verification.py")
        self.log_parsers = load_repo_module("resources_servers/swe_rebench/log_parsers.py")
        app = (REPO_ROOT / "resources_servers/swe_rebench/app.py").read_text()
        # The JVM mirror environment the server gives both sandboxes (`JVM_MIRROR_ENV`), when enabled.
        self.env: dict[str, str] = {}
        if (self.config.get("sandbox_config") or {}).get("use_maven_mirror", True):
            match = re.search(r"^JVM_MIRROR_ENV = (\{.*?\n\})", app, flags=re.MULTILINE | re.DOTALL)
            if match is None:
                raise RuntimeError("resources_servers/swe_rebench/app.py no longer defines JVM_MIRROR_ENV")
            self.env = dict(ast.literal_eval(match.group(1)))
        cache_dir = Path(parser_cache_dir) if parser_cache_dir else self.log_parsers.default_cache_dir()
        self.parser_repo = cache_dir / "SWE-rebench-V2"
        # The server's own fetch: a shallow clone of upstream's default branch, once per cache.
        self.log_parsers._clone_locked(self.parser_repo)
        self.parser_commit = _git(self.parser_repo, "rev-parse", "HEAD")

    def plan(self, row: dict[str, Any]) -> TaskPlan:
        workdir = self.verification.repo_directory(row["repo"])
        steps = [f"mkdir -p {STATE_DIR}"]
        steps += self.anti_cheat_steps(workdir)
        steps.append(self.identity_step(workdir))
        steps.append(self.pristine_step(workdir))
        mirror = REPO_ROOT / "responses_api_agents/swe_agents/maven_mirror"
        vendored = {
            "resources_servers/swe_rebench/verification.py": REPO_ROOT
            / "resources_servers/swe_rebench/verification.py",
            "resources_servers/swe_rebench/log_parsers.py": REPO_ROOT / "resources_servers/swe_rebench/log_parsers.py",
        }
        for name in ("settings.xml", "init.gradle"):
            if (mirror / name).is_file():
                vendored[f"responses_api_agents/swe_agents/maven_mirror/{name}"] = mirror / name
        for relative in (
            "LICENSE",
            "lib/__init__.py",
            "lib/agent/__init__.py",
            "lib/agent/log_parsers.py",
            "lib/agent/swe_constants.py",
        ):
            vendored[f"swe_rebench_parsers/SWE-rebench-V2/{relative}"] = self.parser_repo / relative
        return TaskPlan(
            source=self.name,
            instance_id=row["instance_id"],
            image=row["image_name"],
            workdir=workdir,
            prompt=_prompt(row),
            golden_patch=row.get("patch") or "",
            setup_script=_setup_script(self.name, steps),
            capture_command=self.capture_command(workdir, shlex.quote(row["base_commit"])),
            grading_row=_pick(
                row,
                "instance_id",
                "repo",
                "base_commit",
                "test_patch",
                "install_config",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
            ),
            vendored=vendored,
            evaluation_timeout=int(self.config.get("evaluation_timeout") or 3600),
            resources=self.resources(),
            env=self.env,
            extra_task_json={
                "overwrite_maven_settings": bool(self.config.get("overwrite_maven_settings", True)),
                "log_parsers_upstream_commit": self.parser_commit,
            },
        )


ADAPTERS: dict[str, Callable[..., SourceAdapter]] = {"scale_swe": ScaleSWEAdapter, "swe_rebench": SWERebenchAdapter}


def _non_fatal(command: str, what: str) -> str:
    return f"{command} || echo '[nemo_gym] {what} failed (non-fatal)' >&2"


def _setup_script(source: str, steps: Iterable[str]) -> str:
    body = "\n".join(steps)
    return (
        "#!/bin/bash\n"
        f"# Seed setup of resources_servers/{source}, generated by scripts/swe_rows_to_harbor/convert.py.\n"
        "# Every step is non-fatal, as at the server.\n"
        f"{body}\n"
        "exit 0\n"
    )


def _pick(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys if key in row}


def _prompt(row: dict[str, Any]) -> str:
    """The row's single user message, which becomes ``instruction.md`` unchanged."""
    messages = (row.get("responses_create_params") or {}).get("input")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("expected responses_create_params.input to hold exactly one message")
    message = messages[0]
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError("expected one user message with string content")
    prompt = message["content"]
    first_line = prompt.split("\n", 1)[0].strip()
    if _CANARY_LINE.match(first_line):
        raise ValueError("the prompt opens with a line Gym treats as a canary marker and would drop")
    return prompt


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _test_script() -> str:
    major, minor = MIN_PYTHON
    return f"""#!/bin/bash
# Generated by scripts/swe_rows_to_harbor/convert.py: grades the agent's patch with the source server's code.
mkdir -p /logs/verifier
for candidate in python3 python /opt/conda/bin/python3 /usr/local/bin/python3; do
  if command -v "$candidate" >/dev/null 2>&1 \\
      && "$candidate" -c 'import sys; sys.exit(sys.version_info < ({major}, {minor}))' 2>/dev/null; then
    exec "$candidate" /tests/nemo_gym/grade.py
  fi
done
echo "[nemo_gym] no Python {major}.{minor}+ in this image; the task cannot be graded" >&2
exit 1
"""


def _solve_script(workdir: str) -> str:
    return (
        "#!/bin/bash\n"
        "# Applies the row's golden patch, for `gym dataset validate`.\n"
        f"cd {shlex.quote(workdir)} || exit 1\n"
        "git apply --reject --recount --ignore-space-change --whitespace=nowarn /solution/golden.patch\n"
    )


def _dockerfile(plan: TaskPlan) -> str:
    encoded = base64.b64encode(plan.setup_script.encode()).decode()
    return (
        f"# Generated by scripts/swe_rows_to_harbor/convert.py from {plan.source} {plan.instance_id}.\n"
        "# Gym runs it without a build: it starts the image and runs the RUN line in the sandbox before the\n"
        "# agent. The line runs environment/setup.sh, base64-encoded because such a Dockerfile cannot COPY.\n"
        f"FROM {plan.image}\n"
        f"RUN printf '%s' '{encoded}' | base64 -d > /tmp/nemo_gym_setup.sh "
        "&& bash /tmp/nemo_gym_setup.sh && rm -f /tmp/nemo_gym_setup.sh\n"
    )


def _sandbox_table(plan: TaskPlan) -> dict[str, Any]:
    table: dict[str, Any] = {"workdir": plan.workdir}
    if plan.resources.get("cpu") is not None:
        table["cpus"] = int(plan.resources["cpu"])
    if plan.resources.get("memory_mib") is not None:
        table["memory_mb"] = int(plan.resources["memory_mib"])
    if plan.resources.get("disk_gib") is not None:
        table["storage_mb"] = int(plan.resources["disk_gib"]) * 1024
    if plan.env:
        table["env"] = dict(plan.env)
    return table


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    """A TOML value for the few types the task files use; JSON string escapes are valid TOML ones."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value).replace("\x7f", "\\u007f")
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(key)} = {_toml_value(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"no TOML form for {type(value).__name__}")


def _toml(header: str, top: dict[str, Any], tables: list[tuple[str, dict[str, Any]]]) -> str:
    """A TOML document: a comment, top-level keys, then ``[table]`` sections (``[[name]]`` when named so)."""
    lines = [f"# {header}"]
    lines += [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in top.items()]
    for name, table in tables:
        lines += ["", name] + [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in table.items()]
    return "\n".join(lines) + "\n"


def _task_toml(plan: TaskPlan, agent_timeout_sec: int, metadata: dict[str, Any]) -> str:
    return _toml(
        f"Generated by scripts/swe_rows_to_harbor/convert.py from {plan.source} {plan.instance_id}.",
        {"schema_version": "1.4"},
        [
            ("[metadata]", metadata),
            # No docker_image here: Gym runs the Dockerfile's RUN line only when the task declares no image.
            ("[environment]", _sandbox_table(plan)),
            ("[agent]", {"timeout_sec": agent_timeout_sec}),
            (
                "[verifier]",
                {
                    "environment_mode": "separate",
                    "timeout_sec": plan.evaluation_timeout + VERIFIER_TIMEOUT_MARGIN_SEC,
                },
            ),
            # The server grades in a fresh sandbox of the plain image: no seed setup, only its eval script.
            ("[verifier.environment]", {"docker_image": plan.image, **_sandbox_table(plan)}),
            (
                "[[verifier.collect]]",
                {"service": "main", "timeout_sec": CAPTURE_TIMEOUT_SEC, "command": plan.capture_command},
            ),
        ],
    )


def write_task(plan: TaskPlan, task_dir: Path, *, agent_timeout_sec: int, metadata: dict[str, Any]) -> None:
    files = {
        "task.toml": _task_toml(plan, agent_timeout_sec, metadata),
        "instruction.md": plan.prompt,
        "environment/Dockerfile": _dockerfile(plan),
        "environment/setup.sh": plan.setup_script,
        "tests/test.sh": _test_script(),
        "tests/nemo_gym/task.json": json.dumps(
            {
                "source": plan.source,
                "evaluation_timeout": plan.evaluation_timeout,
                "row": plan.grading_row,
                **plan.extra_task_json,
            },
            indent=2,
        ),
    }
    if plan.golden_patch.strip():
        files["solution/solve.sh"] = _solve_script(plan.workdir)
        files["solution/golden.patch"] = plan.golden_patch
    for relative, content in files.items():
        path = task_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if relative.endswith(".sh"):
            path.chmod(0o755)
    shutil.copyfile(GRADER_SOURCE, task_dir / "tests/nemo_gym/grade.py")
    for relative, source_path in plan.vendored.items():
        target = task_dir / "tests/nemo_gym" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def read_rows(
    path: Path, default_source: str | None
) -> Iterable[tuple[int, str | None, dict[str, Any], dict[str, Any]]]:
    """``(line number, source, row fields, original task id)`` for each JSONL line."""
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            fields = record["task_input"] if "task_input" in record else record
            yield number, fields.get("source_server") or default_source, fields, record.get("task_id") or {}


def task_folder_name(source: str, instance_id: str) -> str:
    return _TASK_NAME.sub("-", f"{source}__{instance_id}")


def convert(
    input_path: Path,
    output_dir: Path,
    *,
    dataset_name: str,
    license_name: str = "TBD",
    split: str = "train",
    agent_timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC,
    evaluation_timeout_sec: int | None = None,
    default_source: str | None = None,
    skip_unsupported: bool = False,
    parser_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapters: dict[str, SourceAdapter] = {}
    converted, skipped = [], []
    for number, source, row, task_id in read_rows(input_path, default_source):
        if source not in ADAPTERS:
            if not skip_unsupported:
                raise ValueError(f"line {number}: source {source!r} is not supported; supported: {sorted(ADAPTERS)}")
            skipped.append({"line": number, "source": source, "instance_id": row.get("instance_id")})
            continue
        if source not in adapters:
            adapters[source] = SWERebenchAdapter(parser_cache_dir) if source == "swe_rebench" else ADAPTERS[source]()
        plan = adapters[source].plan(row)
        if evaluation_timeout_sec is not None:
            # A run can override each server's own timeout (MR 69 sets 1200 for every source).
            plan.evaluation_timeout = evaluation_timeout_sec
        name = task_folder_name(source, plan.instance_id)
        if (output_dir / name).exists():
            raise ValueError(f"line {number}: task {name} appears twice")
        metadata = {"source_server": source, "instance_id": plan.instance_id, "source_line": number}
        if task_id:
            metadata["source_task_id"] = dict(task_id)
        write_task(plan, output_dir / name, agent_timeout_sec=agent_timeout_sec, metadata=metadata)
        entry = {"task": name, "line": number, "source": source, "instance_id": plan.instance_id}
        if "\r" in plan.prompt:
            # Gym reads instruction.md as text, which turns CRLF into LF: the agent sees LF line endings.
            entry["note"] = "prompt has CR line endings; Gym's instruction.md reader turns them into LF"
        converted.append(entry)

    (output_dir / "dataset.toml").write_text(
        _toml(
            "Generated by scripts/swe_rows_to_harbor/convert.py; conversion.json lists the source rows.",
            {},
            [
                ("[dataset]", {"name": dataset_name, "version": "0.1.0"}),
                ("[gym]", {"type": split, "license": license_name, "domain": "coding"}),
            ],
        )
    )

    manifest = {
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "gym_commit": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "gym_dirty": bool(_git(REPO_ROOT, "status", "--porcelain")),
        "swe_rebench_log_parsers_commit": getattr(adapters.get("swe_rebench"), "parser_commit", None),
        "agent_timeout_sec": agent_timeout_sec,
        # None: each task keeps its server's own evaluation_timeout.
        "evaluation_timeout_sec": evaluation_timeout_sec,
        "converted": converted,
        "skipped": skipped,
    }
    (output_dir / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="JSONL rows, typed or flat")
    parser.add_argument("output_dir", type=Path, help="New dataset folder (must be empty or absent)")
    parser.add_argument("--dataset-name", required=True, help="Harbor's org/name, e.g. nemo-gym/swe-tasks")
    parser.add_argument("--license", dest="license_name", default="TBD", help="A Gym dataset license value")
    parser.add_argument("--split", default="train", choices=["train", "validation", "example", "benchmark"])
    parser.add_argument("--agent-timeout-sec", type=int, default=DEFAULT_AGENT_TIMEOUT_SEC)
    parser.add_argument(
        "--evaluation-timeout-sec",
        type=int,
        help="Bound on each task's test run, in place of its server's evaluation_timeout",
    )
    parser.add_argument(
        "--source", dest="default_source", choices=sorted(ADAPTERS), help="For rows without source_server"
    )
    parser.add_argument("--skip-unsupported", action="store_true", help="Leave out rows of other sources")
    parser.add_argument("--parser-cache-dir", type=Path, help="Where SWE-rebench's log parsers are cloned")
    args = parser.parse_args(argv)
    manifest = convert(
        args.input,
        args.output_dir,
        dataset_name=args.dataset_name,
        license_name=args.license_name,
        split=args.split,
        agent_timeout_sec=args.agent_timeout_sec,
        evaluation_timeout_sec=args.evaluation_timeout_sec,
        default_source=args.default_source,
        skip_unsupported=args.skip_unsupported,
        parser_cache_dir=args.parser_cache_dir,
    )
    print(f"converted {len(manifest['converted'])} tasks into {args.output_dir}; skipped {len(manifest['skipped'])}")
    for entry in manifest["skipped"]:
        print(f"  skipped line {entry['line']}: {entry['source']} {entry['instance_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
