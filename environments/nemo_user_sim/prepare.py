# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize canonical NeMo UserSim episode inputs."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ENVIRONMENT_DIR = Path(__file__).parent
DATA_DIR = ENVIRONMENT_DIR / "data"
TASKS_FPATH = DATA_DIR / "nemo_user_sim.jsonl"
PREPARE_REQUIREMENTS_FPATH = ENVIRONMENT_DIR / "requirements.txt"
USERSIM_REVISION = "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee"  # pragma: allowlist secret
NEMOTRON_PERSONAS_VERSION = "0.0.2"
NEMOTRON_PERSONAS_SHA256 = (
    "0341192b00a376cf5643d98cb244e596529030fb3011694ca6ab381f149d3ae8"  # pragma: allowlist secret
)
NEMOTRON_PERSONAS_RESOURCE = "nvidia/nemotron-personas/nemotron-personas-dataset-en_us:0.0.2"
NEMOTRON_PERSONAS_DOWNLOAD_COMMAND = f'ngc registry resource download-version "{NEMOTRON_PERSONAS_RESOURCE}"'
_MATERIALIZE_SCRIPT = """
import json
import sys
from pathlib import Path

repository_root = sys.argv[1]
sys.path.insert(0, repository_root)
from environments.nemo_user_sim.prepare import _download_persona_asset, _probe_seed
from usersim.cli._ngc import ensure_ngc_cli, ensure_ngc_org, has_ngc_key
from usersim.engine.core.probes import known_probes
from usersim.engine.external import materialize_episode_inputs

locale = sys.argv[2]
seed = int(sys.argv[3])
managed_assets_path = Path(sys.argv[4])
models_path = Path(sys.argv[5])
destination = Path(sys.argv[6])
persona_asset_path = managed_assets_path / "datasets" / f"{locale}.parquet"

if not persona_asset_path.is_file():
    if not has_ngc_key():
        raise RuntimeError(
            f"{persona_asset_path} is not cached. Set NGC_CLI_API_KEY and rerun "
            "`gym eval prepare --config environments/nemo_user_sim/config.yaml`."
        )
    ngc_executable = ensure_ngc_cli()
    if ensure_ngc_org() is None:
        raise RuntimeError(
            "Could not determine an NGC organization. Set NGC_CLI_ORG or run `ngc config set`, then rerun."
        )
    _download_persona_asset(
        ngc_executable=ngc_executable,
        locale=locale,
        managed_assets_path=managed_assets_path,
    )

rows = []
for probe in known_probes():
    [row] = materialize_episode_inputs(
        locale=locale,
        num_rows=1,
        probe_mix={probe: 1.0},
        random_seed=_probe_seed(seed, probe),
        models_path=models_path,
    )
    rows.append(row)
destination.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\\n" for row in rows))
"""


class _PersonaAssetMissingError(RuntimeError):
    """Raised when the pinned persona asset has not been downloaded yet."""


def _probe_seed(seed: int, probe: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{probe}".encode()).digest()[:4], "big")


def _task_id(row: dict[str, object]) -> str:
    task_id = row.get("trajectory_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Each UserSim row must have a non-empty string trajectory_id")
    return task_id


def _models_config() -> str:
    # Materialization has no runtime endpoint configuration. Keep UserSim's
    # canonical aliases unresolved so model-specific probes select their
    # provider-neutral form instead of inventing an identity for the policy.
    model_names = (
        "user_model",
        "assistant_model",
        "api_response_model",
        "judge_model",
        "summary_model",
        "evaluator_model",
    )
    specs = ",\n".join(
        f'  {{ alias = {json.dumps(alias)}, model = {json.dumps(alias)}, provider = "openai" }}'
        for alias in model_names
    )
    return f"models = [\n{specs},\n]\n"


def _validate_provenance(row: dict[str, object]) -> None:
    provenance = row.get("usersim_provenance")
    if isinstance(provenance, str):
        provenance = json.loads(provenance)
    if not isinstance(provenance, dict):
        raise ValueError("Each UserSim row must contain usersim_provenance")
    if provenance.get("code_sha") != USERSIM_REVISION:
        raise ValueError("Generated UserSim row does not record the pinned UserSim revision")
    if provenance.get("nemotron_personas_version") != NEMOTRON_PERSONAS_VERSION:
        raise ValueError("Generated UserSim row does not record the pinned Nemotron-Personas version")


def _managed_assets_path() -> Path:
    if configured_path := os.environ.get("DATA_DESIGNER_MANAGED_ASSETS_PATH"):
        return Path(configured_path).expanduser()
    if data_designer_home := os.environ.get("DATA_DESIGNER_HOME"):
        return Path(data_designer_home).expanduser() / "managed-assets"
    return Path.home() / ".data-designer" / "managed-assets"


def _download_persona_asset(*, ngc_executable: Path, locale: str, managed_assets_path: Path) -> Path:
    if locale != "en_US":
        raise ValueError("NeMo UserSim validation preparation currently supports only locale='en_US'")
    destination = managed_assets_path / "datasets" / f"{locale}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nemotron-personas-") as download_dir:
        command = [
            str(ngc_executable),
            "registry",
            "resource",
            "download-version",
            NEMOTRON_PERSONAS_RESOURCE,
            "--dest",
            download_dir,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            raise RuntimeError(f"Failed to download Nemotron-Personas: {stderr.strip() or exc}") from exc
        downloaded_assets = list(Path(download_dir).rglob(f"{locale}.parquet"))
        if len(downloaded_assets) != 1:
            raise RuntimeError(
                f"Expected one {locale}.parquet in the downloaded Nemotron-Personas resource, "
                f"found {len(downloaded_assets)}."
            )
        downloaded_asset = downloaded_assets[0]
        digest = hashlib.sha256()
        with downloaded_asset.open("rb") as asset:
            for chunk in iter(lambda: asset.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != NEMOTRON_PERSONAS_SHA256:
            raise RuntimeError(
                f"Downloaded Nemotron-Personas {NEMOTRON_PERSONAS_VERSION} asset has SHA-256 "
                f"{digest.hexdigest()}, expected {NEMOTRON_PERSONAS_SHA256}."
            )
        temporary_destination = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(downloaded_asset, temporary_destination)
            os.replace(temporary_destination, destination)
        finally:
            temporary_destination.unlink(missing_ok=True)
    return destination


def _validate_persona_asset(locale: str) -> Path:
    if locale != "en_US":
        raise ValueError("NeMo UserSim validation preparation currently supports only locale='en_US'")
    managed_assets_path = _managed_assets_path()
    asset_path = managed_assets_path / "datasets" / f"{locale}.parquet"
    if not asset_path.is_file():
        raise _PersonaAssetMissingError(
            f"Pinned Nemotron-Personas {NEMOTRON_PERSONAS_VERSION} asset {asset_path} is missing. "
            "Set NGC_CLI_API_KEY and rerun preparation. "
            f"To download it manually, run `{NEMOTRON_PERSONAS_DOWNLOAD_COMMAND}`, then place its "
            f"`en_US.parquet` at {asset_path}."
        )
    digest = hashlib.sha256()
    with asset_path.open("rb") as asset:
        for chunk in iter(lambda: asset.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != NEMOTRON_PERSONAS_SHA256:
        raise RuntimeError(
            f"Pinned Nemotron-Personas {NEMOTRON_PERSONAS_VERSION} asset {asset_path} "
            f"has SHA-256 {digest.hexdigest()}, expected {NEMOTRON_PERSONAS_SHA256}. "
            f"Download the pinned resource with `{NEMOTRON_PERSONAS_DOWNLOAD_COMMAND}`, then place its "
            f"`en_US.parquet` at {asset_path}."
        )
    return managed_assets_path


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_tasks(tasks: list[dict[str, object]]) -> None:
    temporary_tasks: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{TASKS_FPATH.name}.",
            suffix=".tmp",
            dir=TASKS_FPATH.parent,
            delete=False,
        ) as output:
            temporary_tasks = Path(output.name)
            output.write("".join(f"{json.dumps(row, separators=(',', ':'), allow_nan=False)}\n" for row in tasks))
        os.replace(temporary_tasks, TASKS_FPATH)
    finally:
        if temporary_tasks is not None:
            temporary_tasks.unlink(missing_ok=True)


def prepare(
    locale: str = "en_US",
    random_seed: int = 1042,
    uv_executable: str = "uv",
    timeout_seconds: float = 3_600,
) -> Path:
    """Materialize one canonical resolved row for each registered UserSim probe."""
    executable = shutil.which(uv_executable)
    if executable is None:
        raise RuntimeError(f"{uv_executable!r} is not on PATH; it is required to prepare UserSim inputs.")
    try:
        managed_assets_path = _validate_persona_asset(locale)
    except _PersonaAssetMissingError:
        managed_assets_path = _managed_assets_path()
    TASKS_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="usersim-materialize-") as working_dir:
        resolved_path = Path(working_dir) / "resolved.jsonl"
        models_path = Path(working_dir) / "models.toml"
        models_path.write_text(_models_config())
        command = [
            executable,
            "run",
            "--no-config",
            "--no-project",
            "--isolated",
            "--with-requirements",
            str(PREPARE_REQUIREMENTS_FPATH),
            "python",
            "-c",
            _MATERIALIZE_SCRIPT,
            str(ENVIRONMENT_DIR.parents[1]),
            locale,
            str(random_seed),
            str(managed_assets_path),
            str(models_path),
            str(resolved_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
                env={
                    **os.environ,
                    "DATA_DESIGNER_MANAGED_ASSETS_PATH": str(managed_assets_path),
                    "USERSIM_CODE_SHA": USERSIM_REVISION,
                    "USERSIM_NEMOTRON_PERSONAS_VERSION": NEMOTRON_PERSONAS_VERSION,
                },
                timeout=timeout_seconds,
                cwd=working_dir,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            raise RuntimeError(f"Failed to materialize NeMo UserSim inputs: {stderr.strip() or exc}") from exc
        _validate_persona_asset(locale)
        resolved_rows = [json.loads(line) for line in resolved_path.read_text().splitlines() if line.strip()]

    tasks = []
    task_ids: set[str] = set()
    for row in resolved_rows:
        _validate_provenance(row)
        usersim_config = row.get("usersim_config")
        if isinstance(usersim_config, dict):
            usersim_config.pop("assets_dir", None)
        task_id = _task_id(row)
        if task_id in task_ids:
            raise ValueError(f"Duplicate UserSim trajectory_id: {task_id}")
        task_ids.add(task_id)
        tasks.append({"task_id": task_id, "resolved_row": _json_safe(row)})
    _write_tasks(tasks)
    print(f"Prepared {len(tasks)} NeMo UserSim tasks at {TASKS_FPATH}")
    return TASKS_FPATH.absolute()


if __name__ == "__main__":
    prepare()
