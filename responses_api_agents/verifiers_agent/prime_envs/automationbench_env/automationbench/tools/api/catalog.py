# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Load declarative API catalogs and expose their request schemas."""

import json
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from automationbench.tools.api.validation import check_schema

SCHEMAS_DIR = Path(__file__).parent / "schemas"


class CatalogEndpoint(BaseModel):
    """One catalog endpoint, including simulator-only routing metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    method: str
    description: str = ""
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    request: dict[str, Any] | None = None
    response: Any = None
    implemented: bool = True
    internal_path: dict[str, Any] = Field(default_factory=dict, alias="internalPath")
    fixed_params: dict[str, Any] = Field(default_factory=dict, alias="fixedParams")
    hidden_params: set[str] = Field(default_factory=set, alias="hiddenParams")
    fixed_body: dict[str, Any] = Field(default_factory=dict, alias="fixedBody")
    forbidden_body: set[str] = Field(default_factory=set, alias="forbiddenBody")
    body_aliases: dict[str, str] = Field(default_factory=dict, alias="bodyAliases")


class Catalog(BaseModel):
    """The complete on-disk format consumed by the API contract."""

    model_config = ConfigDict(extra="forbid")

    api: str
    endpoints: tuple[CatalogEndpoint, ...]


@cache
def load_catalogs() -> dict[str, tuple[CatalogEndpoint, ...]]:
    """Load every catalog once and reject malformed metadata or request schemas."""
    catalogs = {}
    for schema_file in SCHEMAS_DIR.glob("*.jsonc"):
        # Catalog comments are full lines, so stripping them preserves JSON string
        # contents and endpoint descriptions used for discovery.
        source = "\n".join(
            line
            for line in schema_file.read_text().splitlines()
            if not line.lstrip().startswith("//")
        )
        catalog = Catalog.model_validate(json.loads(source))
        for endpoint in catalog.endpoints:
            if endpoint.request is not None:
                check_schema(endpoint.request)
        catalogs[catalog.api] = catalog.endpoints
    return catalogs


def _property_schema(metadata: dict[str, Any]) -> dict[str, Any]:
    """Translate path and query parameter metadata into ordinary JSON Schema."""
    schema = {key: value for key, value in metadata.items() if key not in {"location", "required"}}
    if schema.get("type") == "int":
        schema["type"] = "integer"
    if isinstance(schema.get("properties"), dict):
        nested_required = [
            name for name, value in schema["properties"].items() if value.get("required")
        ]
        schema["properties"] = {
            name: _property_schema(value) for name, value in schema["properties"].items()
        }
        if nested_required:
            schema["required"] = nested_required
    if isinstance(schema.get("items"), dict):
        schema["items"] = _property_schema(schema["items"])
    return schema


def schema_from_parameters(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the object schema exposed for path or query parameters."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {name: _property_schema(value) for name, value in properties.items()},
        "additionalProperties": False,
    }
    required = [name for name, value in properties.items() if value.get("required")]
    if required:
        schema["required"] = required
    return schema


__all__ = ["CatalogEndpoint", "SCHEMAS_DIR", "load_catalogs", "schema_from_parameters"]
