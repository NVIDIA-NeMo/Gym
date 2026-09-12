# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned Nemotron Personas assets and Gym rows for NeMo-Sim."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from nemo_gym.global_config import get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
PERSONAS_CACHE_DIR = DATA_DIR / "personas"
OUTPUT_FPATH = DATA_DIR / "nemo_sim.jsonl"
EXAMPLE_FPATH = BENCHMARK_DIR.parents[1] / "resources_servers/nemo_sim/data/example.jsonl"
NEMOTRON_PERSONAS_TEAM = "nvidia/nemotron-personas"
NEMOTRON_PERSONAS_DATASET_PREFIX = "nemotron-personas-dataset-"
DEFAULT_PERSONAS_DATASET_VERSION = "0.0.2"
DEFAULT_PERSONAS_LOCALES = ("en_US",)


def _resource(locale: str) -> str:
    dataset_name = f"{NEMOTRON_PERSONAS_DATASET_PREFIX}{locale.lower()}"
    return f"{NEMOTRON_PERSONAS_TEAM}/{dataset_name}"


def _source_path(cache_dir: Path, version: str, locale: str) -> Path:
    return cache_dir / version / "source" / f"{locale}.parquet"


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


def _ngc_environment() -> dict[str, str]:
    global_config = get_global_config_dict()
    environment = os.environ.copy()
    api_key = global_config.get("ngc_cli_api_key")
    org = global_config.get("ngc_cli_org")
    if api_key:
        environment["NGC_CLI_API_KEY"] = str(api_key)
    if org:
        environment["NGC_CLI_ORG"] = str(org)
    return environment


def _download_personas(
    *,
    cache_dir: Path,
    version: str,
    locale: str,
    ngc_executable: str,
    timeout_seconds: float,
) -> Path:
    destination = _source_path(cache_dir, version, locale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Reusing prepared persona dataset: {destination}")
    else:
        executable = shutil.which(ngc_executable)
        if executable is None:
            raise RuntimeError(
                f"{ngc_executable!r} is not on PATH. Install the NGC CLI, then configure "
                "ngc_cli_api_key and ngc_cli_org in env.yaml or run `ngc config set`."
            )
        versioned_resource = f"{_resource(locale)}:{version}"
        print(f"Downloading {versioned_resource}")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_dir:
            command = [
                executable,
                "registry",
                "resource",
                "download-version",
                versioned_resource,
                "--dest",
                temporary_dir,
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=timeout_seconds,
                    env=_ngc_environment(),
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                raise RuntimeError(f"Failed to download {versioned_resource}: {stderr.strip() or exc}") from exc

            parquet_files = list(Path(temporary_dir).rglob("*.parquet"))
            if len(parquet_files) != 1:
                raise RuntimeError(f"Expected one Parquet file in {versioned_resource}, found {len(parquet_files)}")
            temporary_destination = destination.with_suffix(".parquet.tmp")
            shutil.copyfile(parquet_files[0], temporary_destination)
            _validate_parquet(temporary_destination)
            os.replace(temporary_destination, destination)

    rows = _validate_parquet(destination)
    manifest = {
        "locale": locale,
        "resource": _resource(locale),
        "version": version,
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "rows": rows,
    }
    manifest_path = destination.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return destination


def prepare(
    personas_cache_dir: str | Path = PERSONAS_CACHE_DIR,
    personas_dataset_version: str = DEFAULT_PERSONAS_DATASET_VERSION,
    personas_locales: list[str] | tuple[str, ...] = DEFAULT_PERSONAS_LOCALES,
    ngc_executable: str = "ngc",
    personas_download_timeout_seconds: float = 3_600,
) -> Path:
    """Download pinned persona sources and materialize the benchmark JSONL."""
    cache_dir = Path(personas_cache_dir)
    for locale in personas_locales:
        _download_personas(
            cache_dir=cache_dir,
            version=personas_dataset_version,
            locale=locale,
            ngc_executable=ngc_executable,
            timeout_seconds=personas_download_timeout_seconds,
        )

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT_FPATH.with_suffix(".jsonl.tmp")
    temporary_output.write_bytes(EXAMPLE_FPATH.read_bytes())
    os.replace(temporary_output, OUTPUT_FPATH)
    print(f"Prepared NeMo-Sim benchmark rows at {OUTPUT_FPATH}")
    return OUTPUT_FPATH.absolute()


if __name__ == "__main__":
    prepare()
