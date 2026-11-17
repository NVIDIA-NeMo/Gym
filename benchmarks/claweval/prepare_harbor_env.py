# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write ClawEval task packages as Harbor task folders for Gym's `harbor` environment server.

A ClawEval task package is a folder holding ``task.yaml`` (the task id and category, the prompt, the fixture files
the agent sees and the grader files it must not), ``grader.py`` (the task's grader, a ``claw_eval`` grader class),
the fixtures and the grader's reference material. This script writes each package as a Harbor task folder into the
training entry ``environments/harbor/claweval_train/``: the prompt is the step's instruction, the fixtures go into
the step's workdir, and everything the grader needs goes under the step's ``tests/``, which the server uploads only
after the agent's session has closed. The verifier, ``judge_claweval.py``, runs the package's own grader with the
ClawEval package the tasks were graded with (``--claw-eval-source``, a checkout holding ``src/claw_eval``), so the
reward is the native ClawEval score. The OpenCode compaction plugins of ``benchmarks/agentic_vbench`` are staged
into every task's workdir for the entry's opt-in ``compaction_plugins.yaml``.

    python -m benchmarks.claweval.prepare_harbor_env --tasks-root <root> --claw-eval-source <checkout>

``--task-ids FILE`` (one task id per line) selects packages from the root. Environment: CLAWEVAL_TASKS_ROOT,
CLAWEVAL_SOURCE and CLAWEVAL_JUDGE_MODEL are the defaults of the flags of the same names.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from benchmarks.agentic_vbench.staging import OPENCODE_PLUGINS, OPENCODE_PLUGINS_DIR, OPENCODE_PLUGINS_WORKDIR


GYM_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
ENTRY_NAME = "claweval_train"
ENTRY_DIR = GYM_ROOT / "environments" / "harbor" / ENTRY_NAME
# The task's own typed field (`task_data.json`, typed by the entry's `task_data.py`).
TASK_DATA_FILE = "task_data.json"
# The files an entry commits beside its tasks: what the repository's .gitignore keeps under environments/harbor/.
ENTRY_FILES = ("README.md", "dataset.toml", "task_data.py")
ENTRY_SUFFIXES = (".yaml",)
PREPARED_RECORD = ".prepared.json"
TASKS_ROOT_ENV_VAR = "CLAWEVAL_TASKS_ROOT"
SOURCE_ENV_VAR = "CLAWEVAL_SOURCE"
JUDGE_MODEL_ENV_VAR = "CLAWEVAL_JUDGE_MODEL"
# The grader's LLM judge: the NVIDIA Inference Hub route the internal runs used. The key reaches the verifier
# through the `${NVINFERENCE_API_KEY}` template in task.toml, resolved from the server's `secrets` (the entry's
# overlay) or the server's environment.
DEFAULT_JUDGE_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_JUDGE_MODEL = "gcp/google/gemini-3-flash-preview"
JUDGE_KEY_VARIABLE = "NVINFERENCE_API_KEY"
VERIFIER_FILES = ("judge_claweval.py", "judge_transport.py")
CLAW_EVAL_PACKAGE = Path("src") / "claw_eval"
DEFAULT_AGENT_TIMEOUT_SEC = 1200
MIN_VERIFIER_TIMEOUT_SEC = 1800
# The image every task runs in: the Playwright Python image (a browser for the webpage tasks) plus ffmpeg and the
# Python packages the graders import. A `--docker-image` names a prebuilt image instead of building this file.
DOCKERFILE = """\
FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

RUN apt-get update \\
    && apt-get install -y --no-install-recommends ffmpeg git ripgrep unzip \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
        "httpx>=0.27,<1" "matplotlib>=3.9,<4" "numpy<3" \\
        "openai>=1.60,<3" "opencv-python-headless==4.13.*" \\
        "pillow>=10,<13" "pydantic>=2.10,<3" "pyyaml>=6,<7"

WORKDIR /workspace
RUN mkdir -p /workspace/fixtures /workspace/output /workspace/work
"""
TEST_SCRIPT = """\
#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier /logs/artifacts
python3 /tests/judge_claweval.py
find /workspace -maxdepth 2 -type f -not -path '/workspace/fixtures/*' -exec cp --parents '{}' /logs/artifacts/ \; 2>/dev/null || true
"""
AGENTS_MD = """\
# ClawEval task

The task's media is under `/workspace/fixtures`. Inspect a video by extracting frames with `ffmpeg` (for example
`ffmpeg -ss 12 -i /workspace/fixtures/video.mp4 -frames:v 1 /workspace/work/frame_12.png`) and reading the image
files; a webpage or animation can be rendered with the installed Chromium (`playwright`). Write the outputs where
the task asks for them, under `/workspace`.
"""
SOLVE_SH = "#!/bin/bash\n# ClawEval packages carry a reference answer for the judge, not a replayable solution.\n"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_digest(task_dir: Path, definition: dict[str, Any]) -> str:
    """Hash the package's scoring inputs (``task.yaml``, ``grader.py`` and the grader files) without its media.

    The reference media the grader reads (``local_grader_files``) enters by path and size, so a package that
    swaps a reference image of the same size keeps its digest; the staged verifier receives the files themselves.
    """
    paths = [task_dir / "task.yaml", task_dir / "grader.py"]
    paths.extend(task_dir / relative for relative in definition.get("sandbox_grader_files", []))
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.relative_to(task_dir.resolve())).encode())
        digest.update(b"\0")
        digest.update(_digest_file(path).encode())
        digest.update(b"\0")
    for relative in sorted(definition.get("local_grader_files", [])):
        path = (task_dir / relative).resolve(strict=True)
        digest.update(f"{relative}\0{path.stat().st_size}\0".encode())
    return digest.hexdigest()


def read_package(task_dir: Path) -> dict[str, Any]:
    """The package's definition and what the task folder is written from; a malformed package is a ``ValueError``."""
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.is_file():
        raise ValueError(f"{task_dir}: no task.yaml")
    definition = yaml.safe_load(task_yaml.read_text())
    if not isinstance(definition, dict):
        raise ValueError(f"{task_yaml}: not a mapping")
    task_id = definition.get("task_id")
    if not isinstance(task_id, str) or not task_id or task_id != task_dir.name:
        raise ValueError(f"{task_yaml}: task_id must be the package folder's name, got {task_id!r}")
    prompt = (definition.get("prompt") or {}).get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{task_yaml}: prompt.text is missing")
    if not (task_dir / "grader.py").is_file():
        raise ValueError(f"{task_dir}: no grader.py")
    for key in ("sandbox_files", "sandbox_grader_files", "local_grader_files"):
        for relative in definition.get(key) or []:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{task_yaml}: {key} entry {relative!r} leaves the package")
            if not (task_dir / path).exists():
                raise FileNotFoundError(task_dir / path)
    environment = definition.get("environment") or {}
    agent_timeout = int(environment.get("timeout_seconds", DEFAULT_AGENT_TIMEOUT_SEC))
    snapshot_timeout = int(environment.get("env_snapshot_timeout", 600))
    return {
        "task_id": task_id,
        "family": str(definition.get("category") or "unknown"),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "task_package_sha256": package_digest(task_dir, definition),
        "agent_timeout_sec": agent_timeout,
        "verifier_timeout_sec": max(MIN_VERIFIER_TIMEOUT_SEC, snapshot_timeout + 600),
        "definition": definition,
        "task_dir": task_dir,
    }


def select_packages(tasks_root: Path, task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """The packages under ``tasks_root`` (``<root>/<task_id>/task.yaml``), all or the listed ids, by id."""
    root = Path(tasks_root).resolve(strict=True)
    found = {path.parent.name: path.parent for path in sorted(root.glob("*/task.yaml"))}
    if not found:
        raise ValueError(f"{root}: no <task_id>/task.yaml packages")
    if task_ids is None:
        chosen = list(found)
    else:
        missing = sorted(set(task_ids) - set(found))
        if missing:
            raise ValueError(f"{root}: no package for task ids {missing[:5]}{'...' if len(missing) > 5 else ''}")
        chosen = sorted(dict.fromkeys(task_ids))
    return [read_package(found[task_id]) for task_id in chosen]


def task_toml(package: dict[str, Any], *, judge_base_url: str, judge_model: str, docker_image: str | None) -> str:
    environment = [
        "[environment]",
        *([f"docker_image = {json.dumps(docker_image)}"] if docker_image else []),
        "build_timeout_sec = 1200.0",
        "cpus = 4",
        "memory_mb = 16384",
        "storage_mb = 32768",
    ]
    return "\n".join(
        [
            'version = "1.0"',
            "",
            "[task]",
            f'name = "claweval/{package["task_id"]}"',
            "",
            "[metadata]",
            f"category = {json.dumps(package['family'])}",
            'source = "ClawEval task package"',
            "",
            *environment,
            "",
            "[[steps]]",
            'name = "solve"',
            "",
            "[steps.agent]",
            f"timeout_sec = {float(package['agent_timeout_sec']):.1f}",
            "",
            "[steps.verifier]",
            f"timeout_sec = {float(package['verifier_timeout_sec']):.1f}",
            "",
            # Whole-value host templates: the server resolves them from its `secrets`, then its environment.
            "[steps.verifier.env]",
            f'CLAW_EVAL_JUDGE_API_KEY = "${{{JUDGE_KEY_VARIABLE}}}"',
            f"CLAW_EVAL_JUDGE_BASE_URL = {json.dumps(judge_base_url)}",
            f"CLAW_EVAL_JUDGE_MODEL = {json.dumps(judge_model)}",
            "",
        ]
    )


def _place(source: Path, destination: Path, *, copy: bool) -> None:
    """Hard-link a fixture into the workdir (large videos, no second copy), or copy it when asked or when linking
    across filesystems or to a file another user owns fails."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve(strict=True)
    if not copy and source.stat().st_uid == os.geteuid():
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def write_task(
    package: dict[str, Any],
    destination: Path,
    *,
    claw_eval_source: Path,
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    docker_image: str | None = None,
    copy_fixtures: bool = False,
) -> dict[str, Any]:
    """Write one package at ``destination`` as a Harbor task folder; returns the task's ``task_data``.

    The agent's side of the task is the instruction, the fixtures and ``AGENTS.md`` in the workdir, and the
    staged OpenCode plugins. Everything that grades, ``task.yaml`` with its rubric and reference answer, the
    grader, its reference files and the ``claw_eval`` package, is under ``tests/``, which the server uploads
    to ``/tests`` only after the agent's session has closed.
    """
    if destination.exists():
        raise FileExistsError(destination)
    package_source = Path(claw_eval_source).resolve(strict=True) / CLAW_EVAL_PACKAGE
    if not package_source.is_dir():
        raise FileNotFoundError(package_source)
    source_dir = package["task_dir"]
    definition = package["definition"]
    step = destination / "steps" / "solve"
    workdir, tests, solution = step / "workdir", step / "tests", step / "solution"
    for directory in (destination / "environment", workdir, tests, solution):
        directory.mkdir(parents=True)

    (destination / "task.toml").write_text(
        task_toml(package, judge_base_url=judge_base_url, judge_model=judge_model, docker_image=docker_image)
    )
    (destination / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (step / "instruction.md").write_text(package["prompt"])
    (workdir / "AGENTS.md").write_text(AGENTS_MD)
    plugins = workdir / OPENCODE_PLUGINS_WORKDIR
    plugins.mkdir()
    for name in OPENCODE_PLUGINS:
        shutil.copyfile(OPENCODE_PLUGINS_DIR / name, plugins / name)
    for relative in definition.get("sandbox_files") or []:
        _place(source_dir / relative, workdir / relative, copy=copy_fixtures)

    (solution / "solve.sh").write_text(SOLVE_SH)
    (tests / "test.sh").write_text(TEST_SCRIPT)
    for name in VERIFIER_FILES:
        shutil.copyfile(BENCHMARK_DIR / name, tests / name)
    (tests / "task.yaml").write_text(yaml.safe_dump(definition, sort_keys=False, allow_unicode=True))
    shutil.copyfile(source_dir / "grader.py", tests / "grader.py")
    for key in ("sandbox_grader_files", "local_grader_files"):
        for relative in definition.get(key) or []:
            target = tests / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / relative, target)
    shutil.copytree(package_source, tests / "claw_eval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    task_data = {
        "family": package["family"],
        "prompt_sha256": package["prompt_sha256"],
        "task_package_sha256": package["task_package_sha256"],
    }
    (destination / TASK_DATA_FILE).write_text(json.dumps(task_data, indent=2) + "\n")
    return task_data


def clear_entry(entry_dir: Path) -> None:
    """Remove the task folders of a previous preparation; the entry's committed files stay."""
    for child in entry_dir.iterdir():
        if child.name in ENTRY_FILES or child.suffix in ENTRY_SUFFIXES or child.name.startswith("."):
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def prepare(
    tasks_root: Path,
    *,
    claw_eval_source: Path,
    entry_dir: Path = ENTRY_DIR,
    task_ids: list[str] | None = None,
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    docker_image: str | None = None,
    copy_fixtures: bool = False,
) -> list[dict[str, Any]]:
    """Write the selected packages into the entry, replacing the tasks of a previous preparation."""
    packages = select_packages(tasks_root, task_ids)
    entry_dir.mkdir(parents=True, exist_ok=True)
    clear_entry(entry_dir)
    written = []
    for package in packages:
        task_data = write_task(
            package,
            entry_dir / package["task_id"],
            claw_eval_source=claw_eval_source,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            docker_image=docker_image,
            copy_fixtures=copy_fixtures,
        )
        written.append({"task_id": package["task_id"], **task_data})
    (entry_dir / PREPARED_RECORD).write_text(
        json.dumps(
            {
                "tasks_root": str(Path(tasks_root).resolve()),
                "claw_eval_source": str(Path(claw_eval_source).resolve()),
                "judge_base_url": judge_base_url,
                "judge_model": judge_model,
                "docker_image": docker_image,
                "tasks": written,
            },
            indent=2,
        )
        + "\n"
    )
    return written


def _read_task_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines()]
    return [task_id for task_id in ids if task_id and not task_id.startswith("#")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-root", type=Path, default=os.environ.get(TASKS_ROOT_ENV_VAR))
    parser.add_argument("--claw-eval-source", type=Path, default=os.environ.get(SOURCE_ENV_VAR))
    parser.add_argument("--task-ids", type=Path, help="a file with one task id per line; default: every package")
    parser.add_argument("--entry", type=Path, default=ENTRY_DIR, help="the entry folder to write into")
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=os.environ.get(JUDGE_MODEL_ENV_VAR, DEFAULT_JUDGE_MODEL))
    parser.add_argument("--docker-image", default=None, help="a prebuilt task image instead of the Dockerfile")
    parser.add_argument("--copy-fixtures", action="store_true", help="copy fixtures instead of hard-linking them")
    args = parser.parse_args(argv)
    if args.tasks_root is None or args.claw_eval_source is None:
        parser.error(f"--tasks-root and --claw-eval-source are required ({TASKS_ROOT_ENV_VAR}, {SOURCE_ENV_VAR})")
    written = prepare(
        args.tasks_root,
        claw_eval_source=args.claw_eval_source,
        entry_dir=args.entry,
        task_ids=_read_task_ids(args.task_ids) if args.task_ids else None,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        docker_image=args.docker_image,
        copy_fixtures=args.copy_fixtures,
    )
    families = sorted({task["family"] for task in written})
    print(f"{len(written)} ClawEval tasks written to {args.entry} (families: {', '.join(families)})")


if __name__ == "__main__":
    main()
