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
NEMOTRON_PERSONAS_DOWNLOAD_COMMAND = (
    'ngc registry resource download-version "nvidia/nemotron-personas/nemotron-personas-dataset-en_us:0.0.2"'
)
_MATERIALIZE_SCRIPT = """
import json
import sys
from pathlib import Path

repository_root = sys.argv[1]
sys.path.insert(0, repository_root)
from environments.nemo_user_sim.prepare import _probe_seed
from usersim.engine.core.probes import known_probes
from usersim.engine.external import materialize_episode_inputs

locale, seed, models_path, destination = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
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


def _validate_persona_asset(locale: str) -> Path:
    if locale != "en_US":
        raise ValueError("NeMo UserSim validation preparation currently supports only locale='en_US'")
    managed_assets_path = _managed_assets_path()
    asset_path = managed_assets_path / "datasets" / f"{locale}.parquet"
    if asset_path.is_file():
        digest = hashlib.sha256()
        with asset_path.open("rb") as asset:
            for chunk in iter(lambda: asset.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() == NEMOTRON_PERSONAS_SHA256:
            return managed_assets_path
        problem = f"has SHA-256 {digest.hexdigest()}, expected {NEMOTRON_PERSONAS_SHA256}"
    else:
        problem = "is missing"
    raise RuntimeError(
        f"Pinned Nemotron-Personas {NEMOTRON_PERSONAS_VERSION} asset {asset_path} {problem}. "
        f"Download the pinned resource with `{NEMOTRON_PERSONAS_DOWNLOAD_COMMAND}`, then place its "
        f"`en_US.parquet` at {asset_path}."
    )


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
    managed_assets_path = _validate_persona_asset(locale)
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
