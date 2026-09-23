# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare NeMo UserSim persona panels and Gym rows."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
PERSONAS_CACHE_DIR = DATA_DIR / "personas"
OUTPUT_FPATH = DATA_DIR / "usersim.jsonl"
EXAMPLE_FPATH = BENCHMARK_DIR.parents[1] / "resources_servers/usersim/data/example.jsonl"
DEFAULT_PERSONAS_DATASET_VERSION = "0.0.2"
DEFAULT_PERSONAS_LOCALES = ("en_US",)
DEFAULT_PERSONAS_PANEL_SIZE = 1_000


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
    usersim_executable: str,
    timeout_seconds: float,
) -> Path:
    destination = _panel_path(cache_dir, version, locale)
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
            and manifest.get("panel_rows") == rows == panel_size
            and manifest.get("panel_sha256") == sha256
        ):
            print(f"Reusing prepared NeMo UserSim panel: {destination}")
            return destination

    executable = shutil.which(usersim_executable)
    if executable is None:
        raise RuntimeError(
            f"{usersim_executable!r} is not on PATH. Install the pinned NeMo UserSim package "
            "before running `gym eval prepare --benchmark usersim`."
        )
    temporary_destination = destination.with_suffix(".parquet.tmp")
    temporary_destination.unlink(missing_ok=True)
    command = [
        executable,
        "panel",
        "--locale",
        locale,
        "--num-personas",
        str(panel_size),
        "--out",
        str(temporary_destination),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
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
        "panel_sha256": _sha256_file(destination),
        "panel_size_bytes": destination.stat().st_size,
        "panel_rows": rows,
        "generator": "usersim panel",
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return destination


def prepare(
    personas_cache_dir: str | Path = PERSONAS_CACHE_DIR,
    personas_dataset_version: str = DEFAULT_PERSONAS_DATASET_VERSION,
    personas_locales: list[str] | tuple[str, ...] = DEFAULT_PERSONAS_LOCALES,
    personas_panel_size: int = DEFAULT_PERSONAS_PANEL_SIZE,
    usersim_executable: str = "usersim",
    usersim_panel_timeout_seconds: float = 3_600,
) -> Path:
    """Materialize UserSim persona panels and the benchmark JSONL."""
    cache_dir = Path(personas_cache_dir)
    for locale in personas_locales:
        _prepare_panel(
            cache_dir=cache_dir,
            version=personas_dataset_version,
            locale=locale,
            panel_size=personas_panel_size,
            usersim_executable=usersim_executable,
            timeout_seconds=usersim_panel_timeout_seconds,
        )

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT_FPATH.with_suffix(".jsonl.tmp")
    rows = []
    for line in EXAMPLE_FPATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["task_id"]["taskset"] = "usersim:benchmark"
        rows.append(json.dumps(row, separators=(",", ":")))
    temporary_output.write_text("\n".join(rows) + "\n")
    os.replace(temporary_output, OUTPUT_FPATH)
    print(f"Prepared NeMo UserSim benchmark rows at {OUTPUT_FPATH}")
    return OUTPUT_FPATH.absolute()


if __name__ == "__main__":
    prepare()
