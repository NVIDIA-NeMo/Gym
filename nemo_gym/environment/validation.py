# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static manifest validation for manifest-backed workloads."""

from __future__ import annotations

import ast
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError
from yaml import YAMLError

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, _resolve_under_cwd_or_install
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.manifest import (
    DatasetKind,
    EnvironmentKind,
    EnvironmentManifest,
    ManifestDataset,
    dump_manifest,
    load_manifest,
)
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility


class EnvironmentValidationError(ConfigError):
    """A manifest-backed workload failed static manifest validation."""


@dataclass(frozen=True)
class ResolvedComponent:
    role: str
    name: str
    implementation: str
    boundary: str
    entrypoint: str | None = None


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
    declared_profile: str
    inferred_profile: str
    profile_evidence: str
    manifest_path: str
    config_path: str
    components: tuple[ResolvedComponent, ...]
    datasets: tuple[DatasetValidation, ...]
    rollout_driver: str | None = None
    grading_mode: str | None = None
    synchronized_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

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


def _manifest_dataset(dataset: Any) -> ManifestDataset:
    values = dataset.model_dump(mode="json", exclude_none=True)
    return ManifestDataset.model_validate(
        {
            key: values[key]
            for key in ("name", "type", "jsonl_fpath", "prepare_script", "prompt_config", "num_repeats")
            if key in values
        }
    )


def _catalog_directory(path: Path) -> Path | None:
    return next(
        (parent for parent in path.resolve().parents if parent.name in {"environments", "benchmarks"}),
        None,
    )


def _component_root(config_path: Path) -> Path | None:
    catalog_directory = _catalog_directory(config_path)
    return catalog_directory.parent if catalog_directory else None


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


def _select_dataset_agent(agents: list[Any]) -> Any:
    dataset_agents = [server for server in agents if server.datasets]
    if len(dataset_agents) == 1:
        return dataset_agents[0]

    available = ", ".join(f"{server.name} ({_implementation_name(server)})" for server in dataset_agents) or "none"
    raise EnvironmentValidationError(
        f"Workload config must define exactly one dataset-bearing agent instance; found: {available}."
    )


def _resolve_manifest_composition(config_path: Path) -> ResolvedComposition:
    """Resolve manifest wiring without probing or materializing runtime services."""
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
    selected_agent = _select_dataset_agent(agents)

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
                entrypoint=str(resources_instance.get_inner_run_server_config().entrypoint),
            )
        )
    components.append(
        ResolvedComponent(
            role="agent_server",
            name=selected_agent.name,
            implementation=agent_server,
            boundary="responses_api_agents",
            entrypoint=str(selected_agent.get_inner_run_server_config().entrypoint),
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
                entrypoint=(
                    str(model_instance.get_inner_run_server_config().entrypoint)
                    if model_instance is not None
                    else None
                ),
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


def _parse_python_file(path: Path, label: str) -> ast.Module:
    if not path.is_file():
        raise EnvironmentValidationError(f"{label} was not found: {path}")
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise EnvironmentValidationError(f"Could not parse {label.lower()} '{path}': {error}") from error


def _server_class(tree: ast.Module) -> ast.ClassDef | None:
    class_names = {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_webserver"
        and isinstance(node.func.value, ast.Name)
    }
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in class_names]
    return matches[0] if len(matches) == 1 else None


def _contains_not_implemented_raise(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(exception, ast.Name) and exception.id == "NotImplementedError":
            return True
    return False


def _only_raises_not_implemented(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(method.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body.pop(0)
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    exception = body[0].exc.func if isinstance(body[0].exc, ast.Call) else body[0].exc
    return isinstance(exception, ast.Name) and exception.id == "NotImplementedError"


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _only_delegates_to_super(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(method.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body.pop(0)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    if isinstance(value, ast.Await):
        value = value.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == method.name
        and isinstance(value.func.value, ast.Call)
        and isinstance(value.func.value.func, ast.Name)
        and value.func.value.func.id == "super"
    )


def _infer_profile(composition: ResolvedComposition) -> tuple[str, str]:
    if composition.rollout_driver:
        return "external-rollout-driver", "rollout_collection_driver is configured"
    if composition.agent_server == "simple_agent":
        return "custom-gym-verifier", "selected agent implementation is simple_agent"

    agent = next((component for component in composition.components if component.role == "agent_server"), None)
    if agent is None or not agent.entrypoint:
        return "unknown", "selected agent has no inspectable entrypoint"

    entrypoint = Path(agent.entrypoint)
    if not entrypoint.is_absolute():
        entrypoint = _resolve_under_cwd_or_install(Path(agent.boundary) / agent.implementation / entrypoint)
    try:
        tree = _parse_python_file(entrypoint, "Agent entrypoint")
    except EnvironmentValidationError:
        return "unknown", f"agent entrypoint could not be inspected: {entrypoint}"

    server_class = _server_class(tree)
    if server_class is None:
        return "unknown", "agent server class could not be identified"

    inherits_simple_agent = any(_base_name(base) == "SimpleAgent" for base in server_class.bases)
    responses = next(
        (
            node
            for node in server_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "responses"
        ),
        None,
    )
    if responses is not None:
        if _only_delegates_to_super(responses):
            return "unknown", "agent responses() still delegates to the default Gym behavior"
        if _only_raises_not_implemented(responses):
            return "external-agent-loop", "agent responses() raises NotImplementedError"
        if _contains_not_implemented_raise(responses):
            return "unknown", "agent responses() behavior depends on runtime configuration"
        return "custom-gym-agent-loop", "agent overrides responses() with measured behavior"

    if inherits_simple_agent:
        return "custom-gym-verifier", "agent inherits the default SimpleAgent responses() behavior"
    return "unknown", "agent server does not declare an inspectable responses() behavior"


def _profile_warnings(declared: str, inferred: str, evidence: str) -> tuple[str, ...]:
    if inferred == "unknown":
        return (f"Could not infer integration_profile ({evidence}); declared profile is {declared!r}.",)
    if inferred != declared:
        return (
            f"Declared integration_profile {declared!r} does not match inferred profile {inferred!r} ({evidence}).",
        )
    return ()


def _validate_prepare_script(path: Path) -> None:
    tree = _parse_python_file(path, "Dataset prepare script")
    if not any(isinstance(node, ast.FunctionDef) and node.name == "prepare" for node in tree.body):
        raise EnvironmentValidationError(f"Dataset prepare script '{path}' must define synchronous prepare().")


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


def _iter_dataset_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise EnvironmentValidationError(f"Dataset file was not found: {path}")
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
                yield line_number, row
    except (OSError, UnicodeError) as error:
        raise EnvironmentValidationError(f"Could not read dataset '{path}': {error}") from error


def _validate_dataset(
    dataset: ManifestDataset,
    *,
    standard_prompt_config: str | None,
) -> DatasetValidation:
    data_path = _resolve_under_cwd_or_install(dataset.jsonl_fpath)
    prompt_path: Path | None = None
    prompt = None

    if dataset.type == DatasetKind.BENCHMARK:
        if dataset.prepare_script is None:
            raise EnvironmentValidationError(f"Benchmark dataset '{dataset.name}' has no prepare script.")
        prepare_path = _resolve_under_cwd_or_install(dataset.prepare_script)
        _validate_prepare_script(prepare_path)
        prompt_config = dataset.prompt_config or standard_prompt_config
        if prompt_config is None:
            raise EnvironmentValidationError(
                f"Benchmark dataset '{dataset.name}' has no prompt_config and the manifest has no "
                "standard_prompt_config."
            )
        prompt_path = _resolve_under_cwd_or_install(prompt_config)
        try:
            prompt = load_prompt_config(str(prompt_path))
        except (OSError, UnicodeError, YAMLError, ValueError, KeyError, AttributeError, TypeError) as error:
            raise EnvironmentValidationError(
                f"Could not materialize benchmark dataset '{dataset.name}': {error}"
            ) from error

    row_count = 0
    for line_number, row in _iter_dataset_rows(data_path):
        row_count += 1
        if prompt is not None:
            try:
                validate_prompt_compatibility([row], prompt)
                row = apply_prompt_to_row(row, prompt)
            except (ValueError, KeyError, AttributeError, TypeError) as error:
                raise EnvironmentValidationError(
                    f"Could not materialize benchmark dataset '{dataset.name}' at row {line_number}: {error}"
                ) from error
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
                f"Dataset '{dataset.name}' row {line_number} is not a valid rollout input at {location}: "
                f"{issue['msg']}"
            ) from error
    if row_count == 0:
        raise EnvironmentValidationError(f"Dataset '{data_path}' is empty.")

    return DatasetValidation(
        name=dataset.name,
        type=dataset.type.value,
        path=str(data_path),
        rows=row_count,
        prompt_config=str(prompt_path) if prompt_path else None,
    )


def _validate_benchmark_prompt_contract(manifest: EnvironmentManifest) -> None:
    if manifest.kind != EnvironmentKind.BENCHMARK:
        return
    benchmark_datasets = [dataset for dataset in manifest.datasets if dataset.type == DatasetKind.BENCHMARK]
    mismatched = [
        dataset.name
        for dataset in benchmark_datasets
        if dataset.prompt_config is not None and dataset.prompt_config != manifest.standard_prompt_config
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


def _validate_manifest_location(manifest_path: Path, manifest: EnvironmentManifest) -> None:
    catalog_directory = _catalog_directory(manifest_path)
    catalog_name = (
        manifest_path.parent.relative_to(catalog_directory).as_posix()
        if catalog_directory is not None
        else manifest_path.parent.name
    )
    if manifest.name != catalog_name:
        raise EnvironmentValidationError(
            f"Manifest identity name '{manifest.name}' does not match its catalog path '{catalog_name}'."
        )

    catalog_kind = catalog_directory.name if catalog_directory is not None else None
    expected_kind = {
        "environments": EnvironmentKind.ENVIRONMENT,
        "benchmarks": EnvironmentKind.BENCHMARK,
    }.get(catalog_kind)
    if expected_kind is not None and manifest.kind != expected_kind:
        raise EnvironmentValidationError(
            f"Manifest kind '{manifest.kind.value}' does not match its location under '{catalog_kind}/'."
        )


def validate_environment(
    manifest_path: str | Path,
    config_path: str | Path | None = None,
    *,
    sync: bool = False,
) -> EnvironmentValidationReport:
    """Validate a manifest without probing services or runtime output paths."""
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_config_path = (
        resolved_manifest_path.with_name("config.yaml")
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    manifest = load_manifest(resolved_manifest_path)
    _validate_manifest_location(resolved_manifest_path, manifest)
    if not resolved_config_path.is_file():
        raise EnvironmentValidationError(f"Gym config was not found: {resolved_config_path}")

    composition = _resolve_manifest_composition(resolved_config_path)
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
                f"{details}\nRun static manifest validation with sync enabled to update only these fields."
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

    with _with_component_root(resolved_config_path):
        inferred_profile, profile_evidence = _infer_profile(composition)
        _validate_benchmark_prompt_contract(manifest)
        if manifest.rollout_driver:
            _validate_rollout_driver(manifest.rollout_driver)
        dataset_reports = tuple(
            _validate_dataset(dataset, standard_prompt_config=manifest.standard_prompt_config)
            for dataset in manifest.datasets
        )
    if synchronized:
        _write_manifest_atomically(resolved_manifest_path, manifest)
    return EnvironmentValidationReport(
        name=manifest.name,
        version=manifest.version,
        kind=manifest.kind.value,
        declared_profile=manifest.integration_profile.value,
        inferred_profile=inferred_profile,
        profile_evidence=profile_evidence,
        manifest_path=str(resolved_manifest_path),
        config_path=str(resolved_config_path),
        components=composition.components,
        datasets=dataset_reports,
        rollout_driver=composition.rollout_driver,
        grading_mode=composition.grading_mode,
        synchronized_fields=synchronized,
        warnings=_profile_warnings(
            manifest.integration_profile.value,
            inferred_profile,
            profile_evidence,
        ),
    )
