# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight CLI rendering for the environment catalog."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

import rich
from omegaconf import OmegaConf
from rich.table import Table

from nemo_gym.cli.utils import exit_unknown_component, print_no_matches, print_rich_table
from nemo_gym.environment_catalog import (
    CatalogFilters,
    EnvironmentCatalog,
    EnvironmentCatalogEntry,
    discover_environment_catalog,
    discover_exact_environment_catalog,
)
from nemo_gym.environment_manifest import load_manifest


@dataclass(frozen=True)
class CatalogCommandConfig:
    component_name: str | None = None
    query: str | None = None
    catalog_domain: str | None = None
    catalog_kind: str | None = None
    catalog_modality: str | None = None
    catalog_licensing: str | None = None
    catalog_status: str | None = None
    catalog_lifecycle: str | None = None
    catalog_requires: list[str] = field(default_factory=list)
    catalog_facts: bool = False
    json_format: bool = False


def _command_config() -> CatalogCommandConfig:
    tokens = []
    for token in sys.argv[1:]:
        if token.startswith("++"):
            token = token[2:]
        elif token.startswith("+"):
            token = token[1:]
        tokens.append(token)
    raw = OmegaConf.to_container(OmegaConf.from_dotlist(tokens), resolve=True)
    values = raw if isinstance(raw, dict) else {}
    requirements = values.get("catalog_requires") or []
    if isinstance(requirements, str):
        requirements = [requirements]
    return CatalogCommandConfig(
        component_name=values.get("component_name"),
        query=values.get("query"),
        catalog_domain=values.get("catalog_domain"),
        catalog_kind=values.get("catalog_kind"),
        catalog_modality=values.get("catalog_modality"),
        catalog_licensing=values.get("catalog_licensing"),
        catalog_status=values.get("catalog_status"),
        catalog_lifecycle=values.get("catalog_lifecycle"),
        catalog_requires=[str(item) for item in requirements],
        catalog_facts=bool(values.get("catalog_facts", False)),
        json_format=bool(values.get("json", False)),
    )


def _catalog_filters(config: CatalogCommandConfig) -> CatalogFilters:
    if config.catalog_kind not in {None, "environment", "benchmark"}:
        rich.print("[red]Error:[/red] --kind must be 'environment' or 'benchmark'.")
        raise SystemExit(1)
    return CatalogFilters(
        name=config.component_name,
        query=config.query,
        domain=config.catalog_domain,
        kind=config.catalog_kind,
        modality=config.catalog_modality,
        licensing=config.catalog_licensing,
        lifecycle=config.catalog_lifecycle,
        status=config.catalog_status,
        required_capabilities=frozenset(config.catalog_requires),
    )


def _entry_facts(entry: EnvironmentCatalogEntry) -> dict[str, Any]:
    payload = entry.to_json_dict()
    if entry.manifest_path is not None:
        payload["manifest"] = load_manifest(entry.manifest_path).model_dump(mode="json", exclude_none=True)
    return payload


def _render_catalog_issues(catalog: EnvironmentCatalog, *, verbose: bool = False) -> None:
    deferred_codes = {"migration-draft", "ambiguous-legacy-resource"}
    deferred = [issue for issue in catalog.issues if issue.code in deferred_codes]
    for issue in catalog.issues:
        if not verbose and issue.code in deferred_codes:
            continue
        rich.print(f"[yellow]Warning:[/yellow] {issue.message}")
    if deferred and not verbose:
        drafts = sum(issue.code == "migration-draft" for issue in deferred)
        ambiguous = sum(issue.code == "ambiguous-legacy-resource" for issue in deferred)
        parts = []
        if drafts:
            parts.append(f"{drafts} generated migration draft{'s' if drafts != 1 else ''}")
        if ambiguous:
            parts.append(f"{ambiguous} ambiguous legacy component{'s' if ambiguous != 1 else ''}")
        rich.print(f"[yellow]Migration diagnostics:[/yellow] {', '.join(parts)} omitted; pass --facts for details.")
    for limitation in catalog.filter_limitations:
        names = f": {', '.join(limitation.entry_names)}" if verbose else ""
        rich.print(
            f"[yellow]Filter limitation:[/yellow] {limitation.count} no-manifest entr"
            f"{'y' if limitation.count == 1 else 'ies'} could not be filtered by {limitation.field}{names}"
        )


def list_environment_catalog() -> None:
    """List the manifest registry unioned with legacy runnable units."""

    config = _command_config()
    filters = _catalog_filters(config)
    catalog = (
        discover_exact_environment_catalog(config.component_name, filters.kind).filtered(filters)
        if config.component_name
        else discover_environment_catalog(filters)
    )

    if config.component_name and len(catalog.entries) > 1:
        kinds = ", ".join(entry.kind for entry in catalog.entries)
        rich.print(
            f"[red]Error:[/red] Environment name '{config.component_name}' is ambiguous across kinds "
            f"({kinds}); pass --kind."
        )
        raise SystemExit(1)
    if config.component_name and not catalog.entries:
        all_names = [entry.name for entry in discover_environment_catalog().entries]
        exit_unknown_component(config.component_name, all_names, "environment")

    if config.json_format:
        payload = catalog.to_json_dict()
        if config.catalog_facts:
            payload["entries"] = [_entry_facts(entry) for entry in catalog.entries]
        else:
            issue_counts: dict[str, int] = {}
            for issue in catalog.issues:
                issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
            payload["issue_counts"] = dict(sorted(issue_counts.items()))
            payload["issues"] = []
        print(json.dumps(payload, sort_keys=True))
        return

    if not catalog.entries:
        print_no_matches("environments", config.query)
        _render_catalog_issues(catalog, verbose=config.catalog_facts)
        return

    if config.component_name:
        entry = catalog.entries[0]
        rich.print(f"The [bold]{entry.name}[/bold] {entry.kind} ({entry.status})")
        if entry.description:
            rich.print(entry.description)
        facts = _entry_facts(entry) if config.catalog_facts else entry.to_json_dict()
        table = Table(title="Catalog facts", show_header=False)
        table.add_column("Field")
        table.add_column("Value")
        for key, value in facts.items():
            if value not in (None, [], {}, ()):
                rendered = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
                table.add_row(key, rendered)
        print_rich_table(table)
        _render_catalog_issues(catalog, verbose=config.catalog_facts)
        return

    title = (
        f"Environments matching '{config.query}' ({len(catalog.entries)})"
        if config.query
        else f"Available environments in NeMo Gym ({len(catalog.entries)})"
    )
    table = Table(title=title)
    for column in ("Name", "Kind", "Status", "Domain", "Modality", "License", "Profile", "Description"):
        table.add_column(column)
    for entry in catalog.entries:
        status = f"{entry.status} / deprecated" if entry.lifecycle == "deprecated" else entry.status
        table.add_row(
            entry.name,
            entry.kind,
            status,
            entry.domain or "",
            entry.modality or "",
            entry.licensing or "",
            entry.integration_profile or "",
            entry.description or "",
        )
    print_rich_table(table)
    rich.print(
        f"Manifest coverage: {catalog.coverage.with_manifest}/{catalog.coverage.total} "
        f"({catalog.coverage.percent:.1f}%); {catalog.coverage.without_manifest} no-manifest."
    )
    _render_catalog_issues(catalog, verbose=config.catalog_facts)


__all__ = ["list_environment_catalog"]
