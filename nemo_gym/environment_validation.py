# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline validation for manifest-backed environments and benchmarks."""

from __future__ import annotations

import ast
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import orjson
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, _resolve_under_cwd_or_install
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import (
    DatasetKind,
    EnvironmentKind,
    EnvironmentManifest,
    ManifestDataset,
    dump_manifest,
    load_manifest,
)
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility


class CatalogEntry(Protocol):
    name: str
    kind: EnvironmentKind | str
    path: Path
    config_path: Path | None
    manifest_path: Path | None


class EnvironmentValidationError(ConfigError):
    """A manifest-backed workload is not ready to run."""


@dataclass(frozen=True)
class ResolvedComponent:
    role: str
    name: str
    implementation: str
    boundary: str


@dataclass(frozen=True)
class DatasetValidation:
    name: str
    type: str
    path: str
    rows: int
    prompt_config: str | None = None


@dataclass(frozen=True)
class EnvironmentValidationReport:
    name: str
    version: str
    kind: str
    integration_profile: str
    manifest_path: str
    config_path: str
    components: tuple[ResolvedComponent, ...]
    datasets: tuple[DatasetValidation, ...]
    synchronized_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedComposition:
    resources_server: str | None
    agent_server: str | None
    model_server: str | None
    datasets: tuple[ManifestDataset, ...]
    rollout_driver: str | None
    grading_mode: str | None
    components: tuple[ResolvedComponent, ...]


def _implementation_name(server: Any) -> str:
    return next(iter(getattr(server, server.SERVER_TYPE)))


def _kind_value(kind: EnvironmentKind | str) -> str:
    return str(getattr(kind, "value", kind))


def _manifest_dataset(dataset: Any) -> ManifestDataset:
    values = dataset.model_dump(mode="json", exclude_none=True)
    return ManifestDataset.model_validate(
        {
            key: values[key]
            for key in ("name", "type", "jsonl_fpath", "prepare_script", "prompt_config", "num_repeats")
            if key in values
        }
    )


def _component_root(config_path: Path) -> Path | None:
    return next(
        (parent.parent for parent in config_path.resolve().parents if parent.name in {"environments", "benchmarks"}),
        None,
    )


@contextmanager
def _with_component_root(config_path: Path):
    root = _component_root(config_path)
    if root is None:
        yield
        return
    original = os.environ.get(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME)
    roots = [str(root), *([original] if original else [])]
    os.environ[NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME] = os.pathsep.join(roots)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, None)
        else:
            os.environ[NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME] = original


def resolve_composition(config_path: Path) -> ResolvedComposition:
    """Resolve Gym wiring without opening ports, starting services, or loading data."""
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [str(config_path.resolve())]},
    )
    parser = GlobalConfigDictParser()
    with _with_component_root(config_path):
        resolved = parser.parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=initial,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
    servers = parser.filter_for_server_instance_configs(resolved)
    by_instance = {server.name: server for server in servers}
    agents = [server for server in servers if server.SERVER_TYPE == "responses_api_agents"]
    dataset_agents = [server for server in agents if server.datasets]
    if len(dataset_agents) != 1:
        names = ", ".join(server.name for server in dataset_agents) or "none"
        raise EnvironmentValidationError(
            "A manifest-backed config must resolve exactly one agent instance with datasets; "
            f"found {len(dataset_agents)} ({names})."
        )
    selected_agent = dataset_agents[0]

    agent_server = _implementation_name(selected_agent)
    agent_config = selected_agent.get_inner_run_server_config_dict()
    resources_ref = agent_config.get("resources_server") or {}
    model_ref = agent_config.get("model_server") or {}
    resources_instance = by_instance.get(resources_ref.get("name")) if isinstance(resources_ref, DictConfig) else None

    resources_server = _implementation_name(resources_instance) if resources_instance is not None else None
    model_server = model_ref.get("name") if isinstance(model_ref, DictConfig) else None
    datasets = tuple(_manifest_dataset(dataset) for dataset in (selected_agent.datasets or []))
    grading_mode = None
    if resources_instance is not None:
        grading_mode = resources_instance.get_inner_run_server_config_dict().get("grading_mode")

    components: list[ResolvedComponent] = []
    if resources_instance is not None:
        components.append(
            ResolvedComponent(
                role="resources_server",
                name=resources_instance.name,
                implementation=resources_server or "",
                boundary="resources_servers",
            )
        )
    components.append(
        ResolvedComponent(
            role="agent_server",
            name=selected_agent.name,
            implementation=agent_server,
            boundary="responses_api_agents",
        )
    )
    if model_server:
        model_instance = by_instance.get(model_server)
        model_implementation = (
            _implementation_name(model_instance) if model_instance is not None else str(model_server)
        )
        if model_implementation == "dummy_model":
            model_implementation = "runtime-selected"
        components.append(
            ResolvedComponent(
                role="model_server",
                name=str(model_server),
                implementation=model_implementation,
                boundary="responses_api_models",
            )
        )

    return ResolvedComposition(
        resources_server=resources_server,
        agent_server=agent_server,
        model_server=str(model_server) if model_server else None,
        datasets=datasets,
        rollout_driver=resolved.get("rollout_collection_driver"),
        grading_mode=str(grading_mode) if grading_mode is not None else None,
        components=tuple(components),
    )


def _mirror_values(composition: ResolvedComposition) -> dict[str, Any]:
    return {
        "resources_server": composition.resources_server,
        "agent_server": composition.agent_server,
        "model_server": composition.model_server,
        "datasets": list(composition.datasets) or None,
        "rollout_driver": composition.rollout_driver,
        "grading_mode": composition.grading_mode,
    }


def _mirror_differences(manifest: EnvironmentManifest, composition: ResolvedComposition) -> dict[str, Any]:
    return {
        field: resolved
        for field, resolved in _mirror_values(composition).items()
        if getattr(manifest, field) != resolved
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _render_value(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True)


def _resolve_asset_path(raw_path: str) -> Path:
    return _resolve_under_cwd_or_install(raw_path)


def _parse_python_file(path: Path, label: str) -> ast.Module:
    if not path.is_file():
        raise EnvironmentValidationError(f"{label} was not found: {path}")
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise EnvironmentValidationError(f"Could not parse {label.lower()} '{path}': {error}") from error


def _is_path_annotation(annotation: ast.expr | None) -> bool:
    return (
        (isinstance(annotation, ast.Name) and annotation.id == "Path")
        or (isinstance(annotation, ast.Attribute) and annotation.attr == "Path")
        or (isinstance(annotation, ast.Constant) and annotation.value in {"Path", "pathlib.Path"})
    )


def _validate_prepare_script(path: Path) -> None:
    tree = _parse_python_file(path, "Dataset prepare script")
    prepare = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare"),
        None,
    )
    if prepare is None:
        raise EnvironmentValidationError(f"Dataset prepare script '{path}' must define synchronous prepare().")
    arguments = [*prepare.args.posonlyargs, *prepare.args.args, *prepare.args.kwonlyargs]
    arguments.extend(argument for argument in (prepare.args.vararg, prepare.args.kwarg) if argument is not None)
    if any(argument.annotation is None for argument in arguments) or not _is_path_annotation(prepare.returns):
        raise EnvironmentValidationError(
            f"Dataset prepare() in '{path}' must annotate every argument and return pathlib.Path."
        )


def _validate_rollout_driver(reference: str) -> None:
    module_name, function_name = reference.split(":", 1)
    relative = Path(*module_name.split("."))
    candidates = (
        _resolve_under_cwd_or_install(relative.with_suffix(".py")),
        _resolve_under_cwd_or_install(relative / "__init__.py"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    tree = _parse_python_file(path, "Rollout driver module")
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name for node in tree.body
    ):
        raise EnvironmentValidationError(f"Rollout driver '{reference}' was not found in '{path}'.")


def _load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EnvironmentValidationError(f"Dataset file was not found: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError as error:
                    raise EnvironmentValidationError(
                        f"Malformed JSON in '{path}' at line {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise EnvironmentValidationError(
                        f"Dataset '{path}' line {line_number} must contain a JSON object."
                    )
                rows.append(row)
    except OSError as error:
        raise EnvironmentValidationError(f"Could not read dataset '{path}': {error}") from error
    if not rows:
        raise EnvironmentValidationError(f"Dataset '{path}' is empty.")
    return rows


def _validate_dataset(dataset: ManifestDataset) -> DatasetValidation:
    data_path = _resolve_asset_path(dataset.jsonl_fpath)
    rows = _load_dataset_rows(data_path)
    prompt_path: Path | None = None

    if dataset.type == DatasetKind.BENCHMARK:
        prepare_path = _resolve_asset_path(dataset.prepare_script)
        _validate_prepare_script(prepare_path)
        prompt_path = _resolve_asset_path(dataset.prompt_config)
        try:
            prompt = load_prompt_config(str(prompt_path))
            validate_prompt_compatibility(rows, prompt)
            for index, row in enumerate(rows):
                rows[index] = apply_prompt_to_row(row, prompt)
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:
            raise EnvironmentValidationError(
                f"Could not materialize benchmark dataset '{dataset.name}': {error}"
            ) from error

    for index, row in enumerate(rows, 1):
        try:
            BaseRunRequest.model_validate(row)
        except ValidationError as error:
            issue = error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )[0]
            location = ".".join(str(part) for part in issue["loc"]) or "row"
            raise EnvironmentValidationError(
                f"Dataset '{dataset.name}' row {index} is not a valid rollout input at {location}: {issue['msg']}"
            ) from error

    return DatasetValidation(
        name=dataset.name,
        type=dataset.type.value,
        path=str(data_path),
        rows=len(rows),
        prompt_config=str(prompt_path) if prompt_path else None,
    )


def _validate_benchmark_prompt_contract(manifest: EnvironmentManifest) -> None:
    if manifest.kind != EnvironmentKind.BENCHMARK:
        return
    benchmark_datasets = [dataset for dataset in manifest.datasets if dataset.type == DatasetKind.BENCHMARK]
    mismatched = [
        dataset.name for dataset in benchmark_datasets if dataset.prompt_config != manifest.standard_prompt_config
    ]
    if mismatched:
        raise EnvironmentValidationError(
            "Benchmark dataset prompt_config must match standard_prompt_config for: " + ", ".join(mismatched)
        )


def _write_manifest_atomically(path: Path, manifest: EnvironmentManifest) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(dump_manifest(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_environment(entry: CatalogEntry, *, sync: bool = False) -> EnvironmentValidationReport:
    """Validate one catalog entry and return its resolved, machine-readable composition."""
    if entry.manifest_path is None:
        raise EnvironmentValidationError(
            f"'{entry.name}' is a legacy {_kind_value(entry.kind)} with no manifest. "
            "It remains runnable, but the onboarding checks require manifest.yaml."
        )
    if entry.config_path is None:
        raise EnvironmentValidationError(f"'{entry.name}' has no config.yaml beside its manifest.")

    manifest = load_manifest(entry.manifest_path)
    entry_kind = _kind_value(entry.kind)
    if manifest.name != entry.name or manifest.kind.value != entry_kind:
        raise EnvironmentValidationError(
            f"Manifest identity {manifest.kind.value}/{manifest.name} does not match "
            f"catalog entry {entry_kind}/{entry.name}."
        )

    composition = resolve_composition(entry.config_path)
    differences = _mirror_differences(manifest, composition)
    synchronized: tuple[str, ...] = ()
    if differences:
        if not sync:
            details = "\n".join(
                f"  - {field}: manifest={_render_value(getattr(manifest, field))}, config={_render_value(value)}"
                for field, value in differences.items()
            )
            raise EnvironmentValidationError(
                "Manifest composition is stale. Config remains authoritative for mirrored fields:\n"
                f"{details}\nRun `gym env validate {entry.name} --sync` to update only these fields."
            )
        updated = manifest.model_dump(mode="json", exclude_none=False)
        updated.update(differences)
        try:
            manifest = EnvironmentManifest.model_validate(updated)
        except ValidationError as error:
            issue = error.errors(include_url=False, include_context=False, include_input=False)[0]
            location = ".".join(str(part) for part in issue["loc"]) or "manifest"
            raise EnvironmentValidationError(
                f"Resolved config cannot satisfy the manifest at {location}: {issue['msg']}"
            ) from error
        synchronized = tuple(differences)

    with _with_component_root(entry.config_path):
        _validate_benchmark_prompt_contract(manifest)
        if manifest.rollout_driver:
            _validate_rollout_driver(manifest.rollout_driver)
        dataset_reports = tuple(_validate_dataset(dataset) for dataset in manifest.datasets)
    if synchronized:
        _write_manifest_atomically(entry.manifest_path, manifest)
    return EnvironmentValidationReport(
        name=manifest.name,
        version=manifest.version,
        kind=manifest.kind.value,
        integration_profile=manifest.integration_profile.value,
        manifest_path=str(entry.manifest_path.resolve()),
        config_path=str(entry.config_path.resolve()),
        components=composition.components,
        datasets=dataset_reports,
        synchronized_fields=synchronized,
    )
