# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned public LAB assets for the Legal Agent Bench server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import time
import tomllib
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]

LAB_SOURCE_REPOSITORY = "https://github.com/harveyai/harvey-labs"
LAB_SOURCE_REVISION = "f46ef86e4788545622db25dcffa3aebb7a139929"  # pragma: allowlist secret
LAB_SOURCE_ARCHIVE_URL = f"https://codeload.github.com/harveyai/harvey-labs/tar.gz/{LAB_SOURCE_REVISION}"
LAB_SOURCE_ARCHIVE_SHA256 = (
    "e45cbdf3236b22866e034bcc62fb23bf00ef2f2e49db7a0cd8a4b07dbae9212c"  # pragma: allowlist secret
)
LAB_SOURCE_ARCHIVE_ROOT = f"harvey-labs-{LAB_SOURCE_REVISION}"
EXPECTED_TASK_COUNT = 1_749
REQUIRED_SKILLS = ("docx", "pptx", "xlsx")
SMOKE_TASK_IDS = (
    "trusts-estates-private-client/compare-trust-documents-against-client-instructions",
    "corporate-ma/analyze-transition-services-agreement-markup",
    "healthcare-life-sciences/analyze-compliance-program-gaps",
    "employment-labor/analyze-reasonable-accommodation-request-under-ada-requirements",
    "litigation-dispute-resolution/categorize-document-production-set-by-relevance-and-privilege",
)

DEFAULT_TASKS_DIR = PACKAGE_DIR / "data" / "cache" / "harbor_tasks" / "legal_agent_bench"
DEFAULT_RUNTIME_TASKS_DIR = PACKAGE_DIR / "data" / "runtime" / "harbor_tasks" / "legal_agent_bench"
DEFAULT_SKILLS_DIR = PACKAGE_DIR / "data" / "cache" / "harness" / "skills"
INDEX_FILENAME = "all.jsonl"
DEFAULT_INDEX_FPATH = PACKAGE_DIR / "data" / "generated" / INDEX_FILENAME
CACHE_MARKER = ".nemo_gym_asset.json"
CACHE_FORMAT_VERSION = 4
REWARD_MODES = ("full_task", "criteria_pass_rate")
REWARD_MODE_ENV_KEY = "LEGAL_AGENT_BENCH_REWARD_MODE"
LAB_HARBOR_SOURCE_DIR = PACKAGE_DIR / "vendor" / "harvey_labs" / "lab_harbor"
TOOL_RUNNER_SOURCE = LAB_HARBOR_SOURCE_DIR / "container_tool_runner.py"
VERIFIER_TEMPLATE_SOURCES = {
    "tests/legal_agent_bench_verify.py": PACKAGE_DIR / "verifier.py",
    "tests/lab_harbor/__init__.py": LAB_HARBOR_SOURCE_DIR / "__init__.py",
    "tests/lab_harbor/judge.py": LAB_HARBOR_SOURCE_DIR / "judge.py",
    "tests/lab_harbor/scoring.py": LAB_HARBOR_SOURCE_DIR / "scoring.py",
}
REQUIRED_TASK_PATHS = (
    "instruction.md",
    "task.toml",
    "task.json",
    "environment/Dockerfile",
    "environment/harness/container_tool_runner.py",
    "tests/task.json",
    *VERIFIER_TEMPLATE_SOURCES,
    "tests/test.sh",
    "documents",
)

_DOCKERFILE = """FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    NODE_PATH=/usr/local/lib/node_modules

RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        bash ca-certificates coreutils curl file findutils fonts-liberation g++ \\
        gawk gcc git grep jq libreoffice nodejs npm pandoc poppler-utils procps \\
        qpdf ripgrep sed tesseract-ocr \\
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \\
    && python -m pip install \\
        "defusedxml>=0.7.1" "diff-match-patch>=20230430" "docxtpl>=0.19.0" \\
        "lxml>=5.0.0" "markitdown>=0.1.0" \\
        "openpyxl>=3.1.0" "pandas>=2.0.0" "pdf2image>=1.17.0" \\
        "pdfplumber>=0.10.0" "Pillow>=10.0.0" "pypdf>=4.0.0" \\
        "pypdfium2>=4.25.0" "pytesseract>=0.3.10" "python-docx>=1.1.0" \\
        "python-pptx>=0.6.23" "reportlab>=4.0.0" "xlcalculator>=0.5.0"

RUN npm install -g @marp-team/marp-cli docx pptxgenjs react react-dom react-icons sharp

COPY harness/container_tool_runner.py /opt/legal-agent-bench/container_tool_runner.py

WORKDIR /workspace/output
"""

_TEST_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
python /tests/legal_agent_bench_verify.py \\
  --task-json /tests/task.json \\
  --run-dir /logs/agent/artifacts/lab-run \\
  --verifier-dir /logs/verifier \\
  --reward-json /logs/verifier/reward.json
"""


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (NEMO_GYM_ROOT / path).resolve()


def flatten_task_id(source_id: str) -> str:
    parts = PurePosixPath(source_id).parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unexpected LAB task id: {source_id!r}")
    return "__".join(parts)


def discover_harness_skills(skills_dir: str | Path) -> list[str]:
    path = validate_harness_skills(skills_dir)
    return sorted(child.name for child in path.iterdir() if child.is_dir() and (child / "SKILL.md").is_file())


def validate_harness_skills(skills_dir: str | Path, *, require_marker: bool = True) -> Path:
    path = resolve_repo_path(skills_dir)
    missing = [name for name in REQUIRED_SKILLS if not (path / name / "SKILL.md").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Legal Agent Bench skills are incomplete in {path}; missing: {', '.join(missing)}. "
            "Run `python resources_servers/legal_agent_bench/prepare.py --asset skills`."
        )
    discovered = sorted(child.name for child in path.iterdir() if child.is_dir() and (child / "SKILL.md").is_file())
    if discovered != list(REQUIRED_SKILLS):
        raise ValueError(f"Expected exactly {list(REQUIRED_SKILLS)} in {path}, found {discovered}")
    if require_marker:
        _validate_marker(path, "skills")
    return path


def validate_harbor_tasks(tasks_dir: str | Path, *, require_marker: bool = True, pristine: bool = True) -> Path:
    path = resolve_repo_path(tasks_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Legal Agent Bench task cache does not exist: {path}")
    if require_marker:
        _validate_marker(path, "tasks")
    task_dirs = sorted(child for child in path.iterdir() if child.is_dir())
    if len(task_dirs) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_COUNT} Harbor tasks in {path}, found {len(task_dirs)}")
    invalid: list[str] = []
    source_ids: list[str] = []
    verifier_templates = {relpath: source.read_bytes() for relpath, source in VERIFIER_TEMPLATE_SOURCES.items()}
    tool_runner_template = TOOL_RUNNER_SOURCE.read_bytes()
    for task_dir in task_dirs:
        missing = [relpath for relpath in REQUIRED_TASK_PATHS if not (task_dir / relpath).exists()]
        if missing:
            invalid.append(f"{task_dir.name}: {', '.join(missing)}")
            continue
        task_toml = (task_dir / "task.toml").read_text(encoding="utf-8")
        try:
            parsed_toml = tomllib.loads(task_toml)
        except tomllib.TOMLDecodeError as exc:
            invalid.append(f"{task_dir.name}: invalid task.toml ({exc})")
            continue
        try:
            task_json = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
            _validate_task_json(task_json, task_dir / "task.json")
        except (OSError, ValueError) as exc:
            invalid.append(f"{task_dir.name}: invalid task.json ({exc})")
            continue
        source_id = (task_json.get("metadata") or {}).get("lab_task_id")
        if not source_id or flatten_task_id(str(source_id)) != task_dir.name:
            invalid.append(f"{task_dir.name}: missing or inconsistent lab_task_id metadata")
        else:
            source_ids.append(str(source_id))
        if (task_dir / "tests" / "task.json").read_bytes() != (task_dir / "task.json").read_bytes():
            invalid.append(f"{task_dir.name}: verifier task.json differs from source task.json")
        for relpath, template in verifier_templates.items():
            if (task_dir / relpath).read_bytes() != template:
                invalid.append(f"{task_dir.name}: {relpath} is stale")
        if (task_dir / "environment" / "harness" / "container_tool_runner.py").read_bytes() != tool_runner_template:
            invalid.append(f"{task_dir.name}: container tool runner is stale")
        if (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8") != _DOCKERFILE:
            invalid.append(f"{task_dir.name}: Dockerfile template is stale")
        if "lab_task_id:" not in (task_dir / "instruction.md").read_text(encoding="utf-8"):
            invalid.append(f"{task_dir.name}: instruction is missing lab_task_id marker")
        if pristine:
            verifier_env = parsed_toml.get("verifier", {}).get("env")
            if verifier_env or "JUDGE_API_KEY" in task_toml or "LAB_JUDGE_API_KEY" in task_toml:
                invalid.append(f"{task_dir.name}: pristine task.toml contains runtime verifier settings")
    if invalid:
        preview = "\n  - ".join(invalid[:10])
        raise FileNotFoundError(f"Invalid Legal Agent Bench task cache in {path}:\n  - {preview}")
    index_path = path / INDEX_FILENAME
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(
            f"Legal Agent Bench task index is missing from the prepared cache: {index_path}"
        ) from exc
    expected_index = _render_task_index(source_ids)
    if index_text != expected_index:
        raise ValueError(f"Legal Agent Bench task index is stale or non-deterministic: {index_path}")
    for source_id in SMOKE_TASK_IDS:
        if not (path / flatten_task_id(source_id)).is_dir():
            raise FileNotFoundError(f"Pinned smoke task is missing from {path}: {source_id}")
    return path


def prepare_assets(
    asset: str = "all",
    *,
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    skills_dir: str | Path = DEFAULT_SKILLS_DIR,
    force: bool = False,
    allow_download: bool = True,
) -> dict[str, Path]:
    if asset not in {"all", "tasks", "skills"}:
        raise ValueError(f"Unknown asset {asset!r}")
    targets = {
        "tasks": resolve_repo_path(tasks_dir),
        "skills": resolve_repo_path(skills_dir),
    }
    requested = [name for name in ("tasks", "skills") if asset in {"all", name}]
    validators: dict[str, Callable[[Path], Path]] = {
        "tasks": validate_harbor_tasks,
        "skills": validate_harness_skills,
    }
    prepared: dict[str, Path] = {}
    missing: list[str] = []
    for name in requested:
        if not force:
            try:
                prepared[name] = validators[name](targets[name])
                print(f"Using cached Legal Agent Bench {name} in {prepared[name]}", flush=True)
                continue
            except (FileNotFoundError, ValueError):
                pass
        missing.append(name)

    if not missing:
        if "tasks" in prepared:
            _publish_task_index(prepared["tasks"])
        return prepared
    if not allow_download:
        for name in missing:
            validators[name](targets[name])
        return prepared

    download_parent = targets[missing[0]].parent
    download_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=download_parent, prefix=".lab-source-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        archive_path = temp_dir / "lab-source.tar.gz"
        _download_source_archive(archive_path)
        extracted = temp_dir / "source"
        extracted.mkdir()
        _safe_extract_archive(archive_path, extracted)
        source_root = _validate_source_tree(extracted / LAB_SOURCE_ARCHIVE_ROOT)

        with ExitStack() as stack:
            staged: dict[str, Path] = {}
            for name in missing:
                targets[name].parent.mkdir(parents=True, exist_ok=True)
                stage_parent = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(
                            dir=targets[name].parent,
                            prefix=f".{targets[name].name}.prepare-",
                        )
                    )
                )
                stage = stage_parent / targets[name].name
                if name == "tasks":
                    _build_task_cache(source_root, stage)
                    validate_harbor_tasks(stage)
                else:
                    _build_skills_cache(source_root, stage)
                    validate_harness_skills(stage)
                staged[name] = stage

            for name in missing:
                _replace_directory(staged[name], targets[name])
                prepared[name] = validators[name](targets[name])
                print(f"Prepared Legal Agent Bench {name} in {prepared[name]}", flush=True)
    if "tasks" in prepared:
        _publish_task_index(prepared["tasks"])
    return prepared


def ensure_assets(
    *,
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    skills_dir: str | Path = DEFAULT_SKILLS_DIR,
    allow_download: bool = True,
) -> dict[str, Path]:
    return prepare_assets(
        "all",
        tasks_dir=tasks_dir,
        skills_dir=skills_dir,
        allow_download=allow_download,
    )


def hydrate_runtime_tasks(
    cache_dir: str | Path,
    runtime_dir: str | Path,
    *,
    verifier_env: dict[str, str],
    reward_mode: str,
    cache_is_validated: bool = False,
) -> Path:
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    cache = resolve_repo_path(cache_dir) if cache_is_validated else validate_harbor_tasks(cache_dir)
    runtime = resolve_repo_path(runtime_dir)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime.parent, prefix=f".{runtime.name}.hydrate-") as temp_dir_str:
        stage = Path(temp_dir_str) / runtime.name
        shutil.copytree(cache, stage, copy_function=_hardlink_or_copy)
        env = dict(verifier_env)
        env[REWARD_MODE_ENV_KEY] = reward_mode
        task_dirs = sorted(child for child in stage.iterdir() if child.is_dir())
        if len(task_dirs) != EXPECTED_TASK_COUNT:
            raise ValueError(f"Expected {EXPECTED_TASK_COUNT} hydrated tasks in {stage}, found {len(task_dirs)}")
        for task_dir in task_dirs:
            toml_path = task_dir / "task.toml"
            clean_toml = toml_path.read_text(encoding="utf-8")
            toml_path.unlink()  # break the hardlink before injecting runtime-only values
            toml_path.write_text(_replace_verifier_env(clean_toml, env), encoding="utf-8")
        _replace_directory(stage, runtime)
    return runtime


def _download_source_archive(output_path: Path) -> None:
    try:
        import requests
        from requests import RequestException
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:  # pragma: no cover - requirements install requests
        raise RuntimeError("Legal Agent Bench preparation requires the `requests` package") from exc

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    partial = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading pinned LAB source {LAB_SOURCE_REVISION} ...", flush=True)
    try:
        for attempt in range(1, 4):
            partial.unlink(missing_ok=True)
            digest = hashlib.sha256()
            downloaded = 0
            next_report = 64 * 1024 * 1024
            try:
                with session.get(LAB_SOURCE_ARCHIVE_URL, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length") or 0)
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            digest.update(chunk)
                            downloaded += len(chunk)
                            if downloaded >= next_report:
                                total_text = f"/{total // (1024 * 1024)} MiB" if total else ""
                                print(
                                    f"  downloaded {downloaded // (1024 * 1024)} MiB{total_text}",
                                    flush=True,
                                )
                                next_report += 64 * 1024 * 1024
                        handle.flush()
                        os.fsync(handle.fileno())
                actual = digest.hexdigest()
                if actual != LAB_SOURCE_ARCHIVE_SHA256:
                    raise ValueError(
                        f"LAB source archive checksum mismatch: expected {LAB_SOURCE_ARCHIVE_SHA256}, got {actual}"
                    )
                os.replace(partial, output_path)
                return
            except (RequestException, OSError, ValueError) as exc:
                partial.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                delay = 2 ** (attempt - 1)
                print(
                    f"  download attempt {attempt}/3 failed ({type(exc).__name__}); retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
    finally:
        session.close()


def _safe_extract_archive(archive_path: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise ValueError("LAB source archive is empty")
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if not path.parts or path.parts[0] != LAB_SOURCE_ARCHIVE_ROOT:
                    raise ValueError(f"Unexpected archive root: {member.name}")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ValueError(f"Unsupported archive entry type: {member.name}")
                if not member.isdir() and not member.isfile():
                    raise ValueError(f"Unsupported archive entry: {member.name}")
            archive.extractall(destination, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Invalid LAB source archive: {exc}") from exc


def _validate_source_tree(source_root: Path) -> Path:
    if source_root.name != LAB_SOURCE_ARCHIVE_ROOT or not source_root.is_dir():
        raise ValueError(f"LAB archive does not contain expected revision root {LAB_SOURCE_ARCHIVE_ROOT}")
    entries = _source_task_entries(source_root)
    if len(entries) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Pinned LAB source must contain {EXPECTED_TASK_COUNT} tasks, found {len(entries)}")
    source_ids = {source_id for source_id, _ in entries}
    missing_smoke = sorted(set(SMOKE_TASK_IDS) - source_ids)
    if missing_smoke:
        raise ValueError(f"Pinned LAB source is missing smoke tasks: {', '.join(missing_smoke)}")
    flattened = [flatten_task_id(source_id) for source_id in source_ids]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Flattening LAB source task ids produces a collision")
    for skill in REQUIRED_SKILLS:
        if not (source_root / "harness" / "skills" / skill / "SKILL.md").is_file():
            raise ValueError(f"Pinned LAB source is missing skill {skill}/SKILL.md")
    return source_root


def _source_task_entries(source_root: Path) -> list[tuple[str, Path]]:
    tasks_root = source_root / "tasks"
    entries: list[tuple[str, Path]] = []
    for task_json in sorted(tasks_root.rglob("task.json")):
        task_dir = task_json.parent
        source_id = task_dir.relative_to(tasks_root).as_posix()
        flatten_task_id(source_id)
        if not (task_dir / "documents").is_dir():
            raise ValueError(f"LAB task {source_id} has no documents directory")
        config = json.loads(task_json.read_text(encoding="utf-8"))
        _validate_task_json(config, task_json)
        entries.append((source_id, task_dir))
    return entries


def _validate_task_json(config: Any, path: Path) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a JSON object")
    for key in ("title", "instructions", "criteria"):
        if not config.get(key):
            raise ValueError(f"{path} is missing required field {key!r}")
    if not isinstance(config["criteria"], list):
        raise ValueError(f"{path} criteria must be a list")
    for index, criterion in enumerate(config["criteria"]):
        if not isinstance(criterion, dict) or not all(criterion.get(key) for key in ("id", "title", "match_criteria")):
            raise ValueError(f"{path} criterion {index} is invalid")


def _render_task_index(source_ids: Iterable[str]) -> str:
    rows = []
    for source_id in sorted(source_ids):
        row = {
            "agent_ref": {
                "name": "legal_agent_bench_harbor_agent",
                "type": "responses_api_agents",
            },
            "instance_id": f"legal_agent_bench::{flatten_task_id(source_id)}",
            "responses_create_params": {
                "input": [],
                "temperature": 1.0,
                "top_p": 0.95,
            },
        }
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return "".join(rows)


def _build_task_cache(source_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    missing_templates = [
        source for source in (*VERIFIER_TEMPLATE_SOURCES.values(), TOOL_RUNNER_SOURCE) if not source.is_file()
    ]
    if missing_templates:
        raise FileNotFoundError(f"Legal Agent Bench runtime templates are missing: {missing_templates}")

    source_entries = _source_task_entries(source_root)
    for source_id, source_task_dir in source_entries:
        task_name = flatten_task_id(source_id)
        task_dir = output_dir / task_name
        (task_dir / "environment" / "harness").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        shutil.copytree(source_task_dir / "documents", task_dir / "documents")

        config = json.loads((source_task_dir / "task.json").read_text(encoding="utf-8"))
        metadata = dict(config.get("metadata") or {})
        metadata.update(
            {
                "lab_task_id": source_id,
                "lab_source_repository": LAB_SOURCE_REPOSITORY,
                "lab_source_revision": LAB_SOURCE_REVISION,
            }
        )
        config["metadata"] = metadata
        task_json_text = json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        (task_dir / "task.json").write_text(task_json_text, encoding="utf-8")
        (task_dir / "tests" / "task.json").write_text(task_json_text, encoding="utf-8")
        (task_dir / "instruction.md").write_text(
            f"<!-- lab_task_id:{source_id} -->\n\n# {config['title']}\n\n{config['instructions']}\n",
            encoding="utf-8",
        )
        (task_dir / "task.toml").write_text(_task_toml(config, source_id), encoding="utf-8")
        (task_dir / "environment" / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
        shutil.copyfile(TOOL_RUNNER_SOURCE, task_dir / "environment" / "harness" / "container_tool_runner.py")
        for relpath, source in VERIFIER_TEMPLATE_SOURCES.items():
            target = task_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        test_script = task_dir / "tests" / "test.sh"
        test_script.write_text(_TEST_SCRIPT, encoding="utf-8")
        test_script.chmod(test_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (output_dir / INDEX_FILENAME).write_text(
        _render_task_index(source_id for source_id, _ in source_entries),
        encoding="utf-8",
    )
    _write_marker(output_dir, "tasks")


def _task_toml(config: dict[str, Any], source_id: str) -> str:
    metadata = {
        "lab_task_id": source_id,
        "title": config.get("title", ""),
        "work_type": config.get("work_type", ""),
        "difficulty": config.get("difficulty", ""),
        "seniority": config.get("seniority", ""),
        "tags": config.get("tags", []),
    }
    lines = ['version = "1.0"', "", "[metadata]"]
    for key, value in metadata.items():
        if value not in (None, "", []):
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "[agent]",
            "timeout_sec = 108000",
            "",
            "[verifier]",
            "timeout_sec = 1800",
            "",
            "[environment]",
            "build_timeout_sec = 1800",
            "cpus = 1",
            "memory_mb = 4096",
            "storage_mb = 10240",
            "allow_internet = true",
            "",
        ]
    )
    return "\n".join(lines)


def _build_skills_cache(source_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    source_skills = source_root / "harness" / "skills"
    for skill in REQUIRED_SKILLS:
        shutil.copytree(source_skills / skill, output_dir / skill)
    _write_marker(output_dir, "skills")


def _marker(kind: str) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "kind": kind,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "repository": LAB_SOURCE_REPOSITORY,
        "revision": LAB_SOURCE_REVISION,
        "archive_sha256": LAB_SOURCE_ARCHIVE_SHA256,
        "task_count": EXPECTED_TASK_COUNT,
        "skills": list(REQUIRED_SKILLS),
    }
    if kind == "tasks":
        marker.update(
            {
                "index_filename": INDEX_FILENAME,
                "dockerfile_sha256": hashlib.sha256(_DOCKERFILE.encode()).hexdigest(),
                "tool_runner_sha256": _file_sha256(TOOL_RUNNER_SOURCE),
                "verifier_files_sha256": {
                    relpath: _file_sha256(source) for relpath, source in VERIFIER_TEMPLATE_SOURCES.items()
                },
            }
        )
    return marker


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_marker(path: Path, kind: str) -> None:
    (path / CACHE_MARKER).write_text(json.dumps(_marker(kind), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_marker(path: Path, kind: str) -> None:
    marker_path = path / CACHE_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid Legal Agent Bench cache marker: {marker_path}") from exc
    if marker != _marker(kind):
        raise ValueError(f"Legal Agent Bench {kind} cache marker is stale: {marker_path}")


def _replace_verifier_env(toml: str, env: dict[str, str]) -> str:
    lines = toml.rstrip().splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() == "[verifier.env]":
            index += 1
            while index < len(lines) and not lines[index].lstrip().startswith("["):
                index += 1
            continue
        output.append(lines[index])
        index += 1
    block = ["[verifier.env]"] + [f"{key} = {json.dumps(value)}" for key, value in sorted(env.items())]
    return "\n".join(output).rstrip() + "\n\n" + "\n".join(block) + "\n"


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _replace_directory(source: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    try:
        source.rename(target)
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if backup.exists():
            backup.rename(target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _publish_task_index(tasks_dir: Path) -> Path:
    """Atomically publish the prepared index at Gym's stable dataset path."""
    source = tasks_dir / INDEX_FILENAME
    target = resolve_repo_path(DEFAULT_INDEX_FPATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=("all", "tasks", "skills"), default="all")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    prepare_assets(
        args.asset,
        tasks_dir=args.tasks_dir,
        skills_dir=args.skills_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
