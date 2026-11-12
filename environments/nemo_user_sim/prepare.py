# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize canonical NeMo UserSim episode inputs."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path


ENVIRONMENT_DIR = Path(__file__).parent
DATA_DIR = ENVIRONMENT_DIR / "data"
TASKS_FPATH = DATA_DIR / "nemo_user_sim.jsonl"
PREPARE_PROJECT_FPATH = ENVIRONMENT_DIR / "pyproject.toml"


def _pinned_usersim_revision() -> str:
    """Return the UserSim commit this project pins, in UserSim's 12-character form."""
    match = re.search(r"UserSim\.git@([0-9a-f]{40})", PREPARE_PROJECT_FPATH.read_text())
    if match is None:
        raise RuntimeError(f"{PREPARE_PROJECT_FPATH} does not pin UserSim to a git commit")
    return match.group(1)[:12]


USERSIM_REVISION = _pinned_usersim_revision()
# Each locale is a separate Nemotron-Personas NGC resource, downloaded only when preparation asks for that
# locale. The digests are of each resource's <locale>.parquet, as NGC lists them
# (`ngc registry resource info <resource>:<version> --files`).
NEMOTRON_PERSONAS_VERSIONS: dict[str, str] = {
    "de_DE": "0.0.1",
    "en_GB": "0.0.1",
    "en_IN": "0.0.2",
    "en_SG": "0.0.1",
    "en_US": "0.0.2",
    "fr_FR": "0.0.1",
    "hi_Deva_IN": "0.0.2",
    "hi_Latn_IN": "0.0.2",
    "ja_JP": "0.0.2",
    "ko_KR": "0.0.1",
    "pt_BR": "0.0.1",
}
NEMOTRON_PERSONAS_SHA256: dict[str, str] = {
    "de_DE": "9277e53b3e20f54d7c24f6bea42ea6ea418cb46ff9df1a5e54078cf887fdacfa",  # pragma: allowlist secret
    "en_GB": "a9509a5df6c55338d8e6511bb4ca8d8a199b0cbbf5bb04bea85b7b23bc83e302",  # pragma: allowlist secret
    "en_IN": "051baf57cc0bfc4ef900fbb26f91f6a42242708aae1fa1961be472c780be2ff8",  # pragma: allowlist secret
    "en_SG": "357e8c503f1b2d669fadf3048c280bf22da77e4c568d193a951ffda3260e3db0",  # pragma: allowlist secret
    "en_US": "0341192b00a376cf5643d98cb244e596529030fb3011694ca6ab381f149d3ae8",  # pragma: allowlist secret
    "fr_FR": "df8254706e050787b9f1fa481c3f9c505c791edf1f18c28c51f425160355c946",  # pragma: allowlist secret
    "hi_Deva_IN": "f370339c0123009dfe4c49858e06acb3b25fcff2d4b4eebd38dd844c17146825",  # pragma: allowlist secret
    "hi_Latn_IN": "7418bd528ab1b50815baa66cbe61ca7398a5df6749cbfca7c4ed97afec9c6092",  # pragma: allowlist secret
    "ja_JP": "9ec84e130b887014e285984b780d31b32665edbc980793e90986d93c709351d2",  # pragma: allowlist secret
    "ko_KR": "8128b83c300c0f9f128580f6a5f0aafadf9b87c4fb9a6fff7ad8c141be320332",  # pragma: allowlist secret
    "pt_BR": "b69dfd37feef5939de467976d2520a8ed6427ce9baddc045e7b79131ee1b89c0",  # pragma: allowlist secret
}
# Validation locales prepared when none are requested: three scripts and four languages, 56 tasks.
DEFAULT_LOCALES = ("en_US", "de_DE", "ja_JP", "hi_Deva_IN")
# Probes the pinned UserSim has no bank for in a locale. Preparation skips these pairs.
UNSUPPORTED_PROBE_LOCALES = frozenset({("financial_services", "en_SG"), ("financial_services", "pt_BR")})
_MATERIALIZE_SCRIPT = """
import json
import sys
from pathlib import Path

repository_root = sys.argv[1]
sys.path.insert(0, repository_root)
from environments.nemo_user_sim.prepare import UNSUPPORTED_PROBE_LOCALES, _download_persona_asset, _probe_seed
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
    if (probe, locale) in UNSUPPORTED_PROBE_LOCALES:
        continue
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


def _persona_resource(locale: str) -> str:
    version = NEMOTRON_PERSONAS_VERSIONS[locale]
    return f"nvidia/nemotron-personas/nemotron-personas-dataset-{locale.lower()}:{version}"


def _persona_download_command(locale: str) -> str:
    return f'ngc registry resource download-version "{_persona_resource(locale)}"'


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset:
        for chunk in iter(lambda: asset.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_provenance(row: dict[str, object], locale: str) -> None:
    provenance = row.get("usersim_provenance")
    if isinstance(provenance, str):
        provenance = json.loads(provenance)
    if not isinstance(provenance, dict):
        raise ValueError("Each UserSim row must contain usersim_provenance")
    if row.get("locale") != locale:
        raise ValueError(f"Generated UserSim row has locale {row.get('locale')!r}, expected {locale!r}")
    if provenance.get("code_sha") != USERSIM_REVISION:
        raise ValueError("Generated UserSim row does not record the pinned UserSim revision")
    if provenance.get("nemotron_personas_version") != NEMOTRON_PERSONAS_VERSIONS[locale]:
        raise ValueError(f"Generated UserSim row does not record the pinned {locale} Nemotron-Personas version")


def _managed_assets_path() -> Path:
    if configured_path := os.environ.get("DATA_DESIGNER_MANAGED_ASSETS_PATH"):
        return Path(configured_path).expanduser()
    if data_designer_home := os.environ.get("DATA_DESIGNER_HOME"):
        return Path(data_designer_home).expanduser() / "managed-assets"
    return Path.home() / ".data-designer" / "managed-assets"


def _download_persona_asset(*, ngc_executable: Path, locale: str, managed_assets_path: Path) -> Path:
    version = NEMOTRON_PERSONAS_VERSIONS[locale]
    expected_sha256 = NEMOTRON_PERSONAS_SHA256[locale]
    destination = managed_assets_path / "datasets" / f"{locale}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nemotron-personas-") as download_dir:
        command = [
            str(ngc_executable),
            "registry",
            "resource",
            "download-version",
            _persona_resource(locale),
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
        actual_sha256 = _sha256_file(downloaded_asset)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Downloaded Nemotron-Personas {locale} {version} asset has SHA-256 "
                f"{actual_sha256}, expected {expected_sha256}."
            )
        temporary_destination = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(downloaded_asset, temporary_destination)
            os.replace(temporary_destination, destination)
        finally:
            temporary_destination.unlink(missing_ok=True)
    return destination


def _validate_persona_asset(locale: str) -> Path:
    version = NEMOTRON_PERSONAS_VERSIONS[locale]
    expected_sha256 = NEMOTRON_PERSONAS_SHA256[locale]
    managed_assets_path = _managed_assets_path()
    asset_path = managed_assets_path / "datasets" / f"{locale}.parquet"
    if not asset_path.is_file():
        raise _PersonaAssetMissingError(
            f"Pinned Nemotron-Personas {locale} {version} asset {asset_path} is missing. "
            "Set NGC_CLI_API_KEY and rerun preparation. "
            f"To download it manually, run `{_persona_download_command(locale)}`, then place its "
            f"`{locale}.parquet` at {asset_path}."
        )
    actual_sha256 = _sha256_file(asset_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Pinned Nemotron-Personas {locale} {version} asset {asset_path} "
            f"has SHA-256 {actual_sha256}, expected {expected_sha256}. "
            f"Download the pinned resource with `{_persona_download_command(locale)}`, then place its "
            f"`{locale}.parquet` at {asset_path}."
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
    locales: Sequence[str] = DEFAULT_LOCALES,
    random_seed: int = 1042,
    uv_executable: str = "uv",
    timeout_seconds: float = 3_600,
) -> Path:
    """Materialize one canonical resolved row for each registered UserSim probe in each locale.

    Each locale needs its own Nemotron-Personas download (0.3 to 4.5 GB), so only the locales asked for are
    prepared. The default spans three scripts and downloads about 10 GB the first time; select others with
    ``++prepare_script_args.locales='[en_US]'``.
    """
    locales = list(dict.fromkeys(locales))
    unknown = [locale for locale in locales if locale not in NEMOTRON_PERSONAS_VERSIONS]
    if not locales or unknown:
        raise ValueError(f"Unsupported locales {unknown or locales}; choose from {sorted(NEMOTRON_PERSONAS_VERSIONS)}")
    executable = shutil.which(uv_executable)
    if executable is None:
        raise RuntimeError(f"{uv_executable!r} is not on PATH; it is required to prepare UserSim inputs.")
    TASKS_FPATH.parent.mkdir(parents=True, exist_ok=True)
    resolved_rows: list[tuple[str, dict[str, object]]] = []
    with tempfile.TemporaryDirectory(prefix="usersim-materialize-") as working_dir:
        models_path = Path(working_dir) / "models.toml"
        models_path.write_text(_models_config())
        for locale in locales:
            resolved_path = _materialize_locale(
                locale,
                executable=executable,
                random_seed=random_seed,
                models_path=models_path,
                working_dir=Path(working_dir),
                timeout_seconds=timeout_seconds,
            )
            resolved_rows.extend(
                (locale, json.loads(line)) for line in resolved_path.read_text().splitlines() if line.strip()
            )

    tasks = []
    task_ids: set[str] = set()
    for locale, row in resolved_rows:
        _validate_provenance(row, locale)
        usersim_config = row.get("usersim_config")
        if isinstance(usersim_config, dict):
            usersim_config.pop("assets_dir", None)
        task_id = _task_id(row)
        if task_id in task_ids:
            raise ValueError(f"Duplicate UserSim trajectory_id: {task_id}")
        task_ids.add(task_id)
        tasks.append({"task_id": task_id, "resolved_row": _json_safe(row)})
    _write_tasks(tasks)
    print(f"Prepared {len(tasks)} NeMo UserSim tasks for {', '.join(locales)} at {TASKS_FPATH}")
    return TASKS_FPATH.absolute()


def _materialize_locale(
    locale: str,
    *,
    executable: str,
    random_seed: int,
    models_path: Path,
    working_dir: Path,
    timeout_seconds: float,
) -> Path:
    """Write one locale's resolved rows to a file in ``working_dir`` and return its path."""
    try:
        managed_assets_path = _validate_persona_asset(locale)
    except _PersonaAssetMissingError:
        managed_assets_path = _managed_assets_path()
    resolved_path = working_dir / f"resolved-{locale}.jsonl"
    command = [
        executable,
        "run",
        "--locked",
        "--project",
        str(ENVIRONMENT_DIR),
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
                # Never sync this preparation-only project into an enclosing Gym/trainer venv.
                "UV_PROJECT_ENVIRONMENT": str(working_dir / ".venv"),
                "DATA_DESIGNER_MANAGED_ASSETS_PATH": str(managed_assets_path),
                "USERSIM_CODE_SHA": USERSIM_REVISION,
                "USERSIM_NEMOTRON_PERSONAS_VERSION": NEMOTRON_PERSONAS_VERSIONS[locale],
            },
            timeout=timeout_seconds,
            cwd=working_dir,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(f"Failed to materialize NeMo UserSim {locale} inputs: {stderr.strip() or exc}") from exc
    _validate_persona_asset(locale)
    return resolved_path


if __name__ == "__main__":
    prepare()
