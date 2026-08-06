# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate manifest drafts and an environment migration inventory.

This is deliberately a repository-maintenance script, not a ``gym`` command. It
only adds adjacent ``manifest.yaml`` drafts and a deterministic inventory; it
does not rewrite a runnable config or participate in runtime config loading.

Drafts are intentionally invalid until a maintainer replaces every
``TODO_REQUIRED`` value. This prevents generated guesses from being mistaken for
authored catalog metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


# Make ``nemo_gym`` importable when the script runs without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_gym.discovery import iter_server_configs
from nemo_gym.environment_inventory import (
    MIGRATION_DRAFT_HEADER,
    MIGRATION_INVENTORY_PATH,
    MIGRATION_TODO_PREFIX,
    RunnableUnit,
    discover_runnable_units,
    is_generated_migration_draft,
)
from nemo_gym.environment_validation import infer_integration_profile, resolve_composition_mirror
from nemo_gym.global_config import GlobalConfigDictParserConfig, StaticValidationConfigParser
from nemo_gym.repository_io import atomic_write_text
from nemo_gym.server_metadata import ServerMetadata


INVENTORY_SCHEMA_VERSION = 1
TODO_PREFIX = MIGRATION_TODO_PREFIX
DEFAULT_INVENTORY_PATH = MIGRATION_INVENTORY_PATH
_DRAFT_HEADER = MIGRATION_DRAFT_HEADER


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def resolve_unit_config(unit: RunnableUnit, repo_root: Path) -> DictConfig:
    """Resolve one workload with deterministic ports and no service startup."""

    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.create({"config_paths": [str(unit.config_path)]}),
    )
    with _working_directory(repo_root):
        return StaticValidationConfigParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=initial,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
            )
        )


def extract_server_metadata(config: Mapping[str, Any] | DictConfig) -> ServerMetadata:
    """Carry over the legacy ``ServerMetadata`` fields from a resolved config."""

    if isinstance(config, DictConfig):
        plain = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    else:
        plain = config
    resources = [
        server_config for group, _name, server_config in iter_server_configs(plain) if group == "resources_servers"
    ]
    if len(resources) != 1:
        raise ValueError(f"expected one resources server for metadata, found {len(resources)}")
    resource = resources[0]
    return ServerMetadata(
        domain=str(resource["domain"]) if resource.get("domain") else None,
        description=str(resource["description"]) if resource.get("description") else None,
        verified=bool(resource.get("verified", False)),
        verified_url=str(resource["verified_url"]) if resource.get("verified_url") else None,
        value=str(resource["value"]) if resource.get("value") else None,
    )


def _todo(field_name: str) -> str:
    return f"{TODO_PREFIX}: replace {field_name}"


def draft_manifest(
    unit: RunnableUnit,
    config: Mapping[str, Any] | DictConfig,
) -> tuple[dict[str, Any], ServerMetadata, tuple[str, ...]]:
    """Build a first draft from config-owned fields and explicit authored TODOs."""

    profile = infer_integration_profile(config)
    mirror = resolve_composition_mirror(config)
    metadata = extract_server_metadata(config)
    required_by_profile = {
        "stock-loop": ("resources_server", "agent_server", "model_server", "datasets"),
        "measured-loop": ("resources_server", "agent_server", "model_server", "datasets"),
        "external-loop": ("resources_server", "agent_server", "datasets"),
        "custom-driver": ("resources_server", "datasets", "rollout_driver"),
    }
    missing_composition = [
        field_name for field_name in required_by_profile[profile] if not getattr(mirror, field_name)
    ]
    if missing_composition:
        raise ValueError("config composition is ambiguous or incomplete: " + ", ".join(missing_composition))

    authored_todos = [
        "version",
        "modality",
        "licensing",
        "authors",
        "reward.range",
        "reward.higher_is_better",
        "determinism",
        "requires",
        "provides",
    ]
    domain = metadata.domain
    if not domain:
        authored_todos.append("domain")
        domain = _todo("domain")
    description = metadata.description
    if not description:
        authored_todos.append("description")
        description = _todo("description")

    draft: dict[str, Any] = {
        "name": unit.name,
        "version": _todo("semantic version"),
        "kind": unit.kind,
        "integration_profile": profile,
        "domain": domain,
        "description": description,
        "modality": _todo("modality"),
        "licensing": _todo("SPDX identifier or access classification"),
        "authors": [_todo("author or owning team")],
        "reward": {
            "range": [_todo("reward minimum"), _todo("reward maximum")],
            "higher_is_better": _todo("whether higher reward is better"),
        },
        "determinism": _todo("seeded, stochastic, or unknown"),
    }
    for field_name, value in mirror.to_manifest_update().items():
        if value is None:
            continue
        if field_name == "datasets":
            draft[field_name] = [dataset.model_dump(mode="json", exclude_none=True) for dataset in value]
        else:
            draft[field_name] = value
    if unit.kind == "benchmark":
        draft["canonical_split"] = _todo("canonical benchmark split")
        draft["standard_prompt_config"] = _todo("standard prompt config")
        authored_todos.extend(("canonical_split", "standard_prompt_config"))
    draft["requires"] = [_todo("required capability")]
    draft["provides"] = [_todo("provided capability")]
    return draft, metadata, tuple(sorted(authored_todos))


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _is_generated_draft(path: Path) -> bool:
    return is_generated_migration_draft(path)


def _render_draft(draft: Mapping[str, Any]) -> str:
    header = f"{_DRAFT_HEADER}\n# This draft is intentionally invalid until every TODO_REQUIRED value is replaced.\n"
    return header + yaml.safe_dump(dict(draft), sort_keys=False, allow_unicode=True)


def migrate_environment_manifests(
    repo_root: Path,
    *,
    inventory_path: Path = DEFAULT_INVENTORY_PATH,
    write: bool = False,
    inventory_only: bool = False,
    units: Sequence[RunnableUnit] | None = None,
    draft_units: Sequence[RunnableUnit] | None = None,
) -> dict[str, Any]:
    """Draft legacy manifests and return the deterministic migration inventory."""

    if write and inventory_only:
        raise ValueError("write and inventory_only are mutually exclusive")

    root = repo_root.resolve()
    selected_units = tuple(units) if units is not None else discover_runnable_units(root)
    selected_draft_configs = (
        None if draft_units is None else frozenset(unit.config_path.resolve() for unit in draft_units)
    )
    records: list[dict[str, Any]] = []
    exceptions: list[dict[str, str]] = []

    for unit in sorted(
        selected_units,
        key=lambda item: (item.kind, item.name.casefold(), item.registry, item.config_path.as_posix()),
    ):
        base_record: dict[str, Any] = {
            "name": unit.name,
            "kind": unit.kind,
            "registry": unit.registry,
            "config_path": _relative(unit.config_path, root),
            "manifest_path": _relative(unit.manifest_path, root),
        }
        existing_draft = unit.manifest_path.exists() and _is_generated_draft(unit.manifest_path)
        if unit.manifest_path.exists() and not existing_draft:
            records.append({**base_record, "status": "already-manifest"})
            continue
        if unit.blocker:
            exception = {**base_record, "reason": unit.blocker}
            records.append({**exception, "status": "exception"})
            exceptions.append(exception)
            continue
        try:
            resolved = resolve_unit_config(unit, root)
            draft, metadata, authored_todos = draft_manifest(unit, resolved)
            rendered_draft = _render_draft(draft)
        except Exception as error:
            reason = f"{type(error).__name__}: {str(error).splitlines()[0]}"
            exception = {**base_record, "reason": reason}
            records.append({**exception, "status": "exception"})
            exceptions.append(exception)
            continue

        write_this_draft = write and (
            selected_draft_configs is None or unit.config_path.resolve() in selected_draft_configs
        )
        status = "drafted" if write_this_draft or existing_draft else "planned"
        if write_this_draft and not existing_draft:
            # Existing manifests were handled above. Never overwrite a unit if
            # another process created its manifest during config resolution.
            if unit.manifest_path.exists():
                status = "already-manifest"
            else:
                atomic_write_text(unit.manifest_path, rendered_draft, create_parent=True)
        records.append(
            {
                **base_record,
                "status": status,
                "integration_profile": draft["integration_profile"],
                "authored_todos": list(authored_todos),
                "legacy_metadata": {
                    key: value for key, value in metadata.to_dict().items() if key not in {"verified", "verified_url"}
                },
            }
        )

    summary = {
        "total": len(records),
        "drafted": sum(record["status"] == "drafted" for record in records),
        "planned": sum(record["status"] == "planned" for record in records),
        "already_manifest": sum(record["status"] == "already-manifest" for record in records),
        "exceptions": len(exceptions),
    }
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "summary": summary,
        "units": records,
        "exceptions": exceptions,
    }
    if write or inventory_only:
        destination = inventory_path if inventory_path.is_absolute() else root / inventory_path
        atomic_write_text(
            destination,
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            create_parent=True,
        )
    return inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Draft manifest.yaml files and a tracked exception inventory without modifying runnable configs.")
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--write",
        action="store_true",
        help="Write one non-overwriting manifest draft selected with --config, plus the inventory.",
    )
    output_mode.add_argument(
        "--inventory-only",
        action="store_true",
        help="Write only the deterministic inventory; do not create manifest drafts.",
    )
    output_mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if the checked-in inventory differs from a fresh, non-mutating discovery pass.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Repository-relative config path for the single unit selected by --write.",
    )
    parser.add_argument(
        "--fail-on-exceptions",
        action="store_true",
        help="Return a non-zero status when the tracked inventory contains an exception.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    draft_units: Sequence[RunnableUnit] | None = None
    if args.write:
        if args.config is None:
            parser.error("--write requires --config so draft generation is limited to one reviewed unit")
        root = args.repo_root.resolve()
        selected_config = args.config if args.config.is_absolute() else root / args.config
        matches = tuple(
            unit for unit in discover_runnable_units(root) if unit.config_path.resolve() == selected_config.resolve()
        )
        if len(matches) != 1:
            parser.error(f"--config must identify exactly one discovered runnable unit; found {len(matches)}")
        draft_units = matches
    elif args.config is not None:
        parser.error("--config is only valid with --write")

    inventory = migrate_environment_manifests(
        args.repo_root,
        inventory_path=args.inventory,
        write=args.write,
        inventory_only=args.inventory_only,
        draft_units=draft_units,
    )
    print(json.dumps(inventory["summary"], sort_keys=True))
    if args.check:
        root = args.repo_root.resolve()
        destination = args.inventory if args.inventory.is_absolute() else root / args.inventory
        expected = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        try:
            actual = destination.read_text(encoding="utf-8")
        except OSError as error:
            print(f"Could not read tracked migration inventory '{destination}': {error}", file=sys.stderr)
            return 1
        if actual != expected:
            print(
                "Tracked migration inventory is stale; run "
                "`uv run python scripts/migrate_environment_manifests.py --inventory-only`.",
                file=sys.stderr,
            )
            return 1
    return 1 if args.fail_on_exceptions and inventory["exceptions"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
