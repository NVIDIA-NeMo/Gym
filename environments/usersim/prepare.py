# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the persona panels used by the NeMo UserSim environment."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


ENVIRONMENT_DIR = Path(__file__).parent
DATA_DIR = ENVIRONMENT_DIR / "data"
PERSONAS_CACHE_DIR = DATA_DIR / "personas"
TASK_SOURCE_FPATH = ENVIRONMENT_DIR.parents[1] / "resources_servers/usersim/data/example_source.jsonl"
TASKS_FPATH = DATA_DIR / "example.jsonl"
PREPARE_REQUIREMENTS_FPATH = ENVIRONMENT_DIR / "requirements.txt"
DEFAULT_PERSONAS_DATASET_VERSION = "0.0.2"
DEFAULT_PERSONAS_LOCALES = ("en_US",)
DEFAULT_PERSONAS_PANEL_SIZE = 1_000
USERSIM_REVISION = "3a928ef8b4f5f8e7740bde213606443ccf04e6b1"  # pragma: allowlist secret


def _panel_path(cache_dir: Path, version: str, locale: str) -> Path:
    return cache_dir / version / "panels" / f"{locale}.parquet"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parquet(path: Path) -> int:
    try:
        rows = pq.ParquetFile(path).metadata.num_rows
    except Exception as exc:
        raise RuntimeError(f"Persona dataset at {path} is not valid Parquet: {exc}") from exc
    if rows < 1:
        raise RuntimeError(f"Persona dataset at {path} contains no rows")
    return rows


def _prepare_panel(
    *,
    cache_dir: Path,
    version: str,
    locale: str,
    panel_size: int,
    usersim_executable: str | None,
    uv_executable: str,
    timeout_seconds: float,
) -> Path:
    destination = _panel_path(cache_dir, version, locale).resolve()
    manifest_path = destination.with_suffix(".manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            manifest = {}
        rows = _validate_parquet(destination)
        sha256 = _sha256_file(destination)
        if (
            manifest.get("locale") == locale
            and manifest.get("personas_dataset_version") == version
            and manifest.get("usersim_revision") == USERSIM_REVISION
            and manifest.get("panel_rows") == rows == panel_size
            and manifest.get("panel_sha256") == sha256
        ):
            print(f"Reusing prepared NeMo UserSim panel: {destination}")
            return destination

    if usersim_executable is None:
        executable = shutil.which(uv_executable)
        if executable is None:
            raise RuntimeError(f"{uv_executable!r} is not on PATH; it is required to prepare the UserSim panel.")
        command = [
            executable,
            "run",
            "--no-config",
            "--no-project",
            "--isolated",
            "--with-requirements",
            str(PREPARE_REQUIREMENTS_FPATH),
            "usersim",
        ]
    else:
        executable = shutil.which(usersim_executable)
        if executable is None:
            raise RuntimeError(f"{usersim_executable!r} is not on PATH.")
        command = [executable]

    temporary_destination = destination.with_suffix(".parquet.tmp")
    temporary_destination.unlink(missing_ok=True)
    command.extend(
        [
            "panel",
            "--locale",
            locale,
            "--num-personas",
            str(panel_size),
            "--out",
            str(temporary_destination),
        ]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="usersim-panel-") as working_dir:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_seconds,
                cwd=working_dir,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(f"Failed to prepare NeMo UserSim panel for {locale}: {stderr.strip() or exc}") from exc

    try:
        rows = _validate_parquet(temporary_destination)
        if rows != panel_size:
            raise RuntimeError(f"NeMo UserSim panel for {locale} contains {rows} rows; expected {panel_size}")
        os.replace(temporary_destination, destination)
    finally:
        temporary_destination.unlink(missing_ok=True)

    manifest = {
        "locale": locale,
        "personas_dataset_version": version,
        "usersim_revision": USERSIM_REVISION,
        "panel_sha256": _sha256_file(destination),
        "panel_size_bytes": destination.stat().st_size,
        "panel_rows": rows,
        "generator": "usersim panel",
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return destination


def _stable_index(size: int, *parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest, "big") % size


def _persona_from_row(row: dict[str, object]) -> dict[str, object]:
    persona = row.get("persona")
    if isinstance(persona, dict) and persona:
        return persona
    if isinstance(persona, str):
        try:
            decoded = json.loads(persona)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict) and decoded:
            return decoded
    if row:
        return row
    raise RuntimeError("Prepared UserSim panel contains an empty persona row")


def _materialize_tasks(*, cache_dir: Path, version: str, locales: tuple[str, ...] | list[str]) -> Path:
    panels: dict[str, tuple[list[dict[str, object]], dict[str, object]]] = {}
    for locale in locales:
        panel_path = _panel_path(cache_dir, version, locale)
        manifest = json.loads(panel_path.with_suffix(".manifest.json").read_text())
        personas = [_persona_from_row(row) for row in pq.read_table(panel_path).to_pylist()]
        panels[locale] = personas, manifest

    rows = [json.loads(line) for line in TASK_SOURCE_FPATH.read_text().splitlines() if line.strip()]
    if len(rows) != 14:
        raise RuntimeError(f"Expected 14 UserSim example task templates at {TASK_SOURCE_FPATH}; found {len(rows)}")

    materialized = []
    for row in rows:
        source = row["task_input"]
        locale = source["locale"]
        try:
            personas, manifest = panels[locale]
        except KeyError as error:
            raise RuntimeError(f"Task {row['task_id']} uses unprepared locale {locale!r}") from error
        seed = source["seed"]
        persona = personas[_stable_index(len(personas), seed, locale, "persona")]
        materialized.append(
            {
                "task_id": row["task_id"],
                "task_input": {
                    "scenario": {
                        "locale": locale,
                        "persona": persona,
                        "probe_type": source["probe_type"],
                        "theme": source["theme"],
                        "goal": source["goal"],
                        "probe_data": source.get("probe_data", {}),
                    },
                    "usersim_context": {
                        "locale": locale,
                        "seed": seed,
                        "personas_dataset_version": version,
                        "personas_panel_sha256": manifest["panel_sha256"],
                        "usersim_revision": USERSIM_REVISION,
                    },
                    "responses_create_params": source.get("responses_create_params", {}),
                },
            }
        )

    TASKS_FPATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_tasks = TASKS_FPATH.with_suffix(".jsonl.tmp")
    temporary_tasks.write_text("".join(f"{json.dumps(row, separators=(',', ':'))}\n" for row in materialized))
    os.replace(temporary_tasks, TASKS_FPATH)
    return TASKS_FPATH


def prepare(
    personas_cache_dir: str | Path = PERSONAS_CACHE_DIR,
    personas_dataset_version: str = DEFAULT_PERSONAS_DATASET_VERSION,
    personas_locales: list[str] | tuple[str, ...] = DEFAULT_PERSONAS_LOCALES,
    personas_panel_size: int = DEFAULT_PERSONAS_PANEL_SIZE,
    usersim_executable: str | None = None,
    uv_executable: str = "uv",
    usersim_panel_timeout_seconds: float = 3_600,
) -> Path:
    """Materialize UserSim persona panels and fully resolved example tasks."""
    cache_dir = Path(personas_cache_dir)
    for locale in personas_locales:
        _prepare_panel(
            cache_dir=cache_dir,
            version=personas_dataset_version,
            locale=locale,
            panel_size=personas_panel_size,
            usersim_executable=usersim_executable,
            uv_executable=uv_executable,
            timeout_seconds=usersim_panel_timeout_seconds,
        )

    tasks_path = _materialize_tasks(
        cache_dir=cache_dir,
        version=personas_dataset_version,
        locales=personas_locales,
    )
    print(f"Prepared NeMo UserSim scenario tasks at {tasks_path}")
    return tasks_path.absolute()


if __name__ == "__main__":
    prepare()
