# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Task-scoped discovery and execution for simulated APIs."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult, TextContent

from automationbench.schema.world import WorldState
from automationbench.tools.api.catalog import (
    CatalogEndpoint,
    load_catalogs,
    schema_from_parameters,
)
from automationbench.tools.api.validation import validation_errors
from automationbench.utils.bm25 import BM25Scorer
from automationbench.utils.routing import Router

SERVICE_ALIASES = {"openai": "chatgpt"}


def _search_document(api: str, endpoint: CatalogEndpoint) -> str:
    parts = [api, endpoint.id, endpoint.description]
    parts.extend(metadata.get("description", "") for metadata in endpoint.parameters.values())
    return " ".join(filter(None, parts))


def _has_failed_operation(payload: Any) -> bool:
    if isinstance(payload, dict):
        return payload.get("didSucceed") is False or any(
            _has_failed_operation(value) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(_has_failed_operation(value) for value in payload)
    return False


def _is_application_error(payload: Any) -> bool:
    """Recognize failure conventions used across the simulated service APIs."""
    if not isinstance(payload, dict):
        return False
    code = payload.get("code")
    return bool(
        payload.get("error")
        or payload.get("errors")
        or "Fault" in payload
        or payload.get("ok") is False
        or payload.get("success") is False
        or (isinstance(code, (int, float)) and not isinstance(code, bool) and code >= 400)
        or _has_failed_operation(payload)
    )


def _result(payload: Any, *, is_error: bool = False) -> CallToolResult:
    structured = payload if isinstance(payload, dict) else {"result": payload}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=structured,
        isError=is_error,
    )


def _error(
    error_type: str,
    message: str,
    endpoint: str | None = None,
) -> CallToolResult:
    detail: dict[str, Any] = {"type": error_type, "message": message}
    if endpoint:
        detail["endpoint"] = endpoint
    return _result({"error": detail}, is_error=True)


@dataclass(frozen=True)
class Endpoint:
    """One executable catalog entry and its model-facing input contract."""

    id: str
    service: str
    method: str
    path: str
    description: str
    path_schema: dict[str, Any]
    params_schema: dict[str, Any]
    body_schema: dict[str, Any] | None
    response: Any
    router: Router
    internal_path: dict[str, Any]
    fixed_params: dict[str, Any]
    fixed_body: dict[str, Any]
    forbidden_body: set[str]
    body_aliases: dict[str, str]


def _endpoint_input_schema(endpoint: Endpoint) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "endpoint": {
            "type": "string",
            "const": endpoint.id,
            "description": endpoint.description,
        }
    }
    required = ["endpoint"]
    for name, schema in (
        ("path", endpoint.path_schema),
        ("params", endpoint.params_schema),
        ("body", endpoint.body_schema),
    ):
        if schema is None or (
            not schema.get("properties") and schema.get("additionalProperties") is False
        ):
            continue
        properties[name] = schema
        if name == "body" or schema.get("required"):
            required.append(name)
    return {
        "type": "object",
        "title": endpoint.id,
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class APIContract:
    """Discover and execute connected, simulator-backed API endpoints."""

    def __init__(self, world: WorldState):
        self.world = world
        allowed = world.meta.allowed_services
        self.allowed_services = (
            set(allowed) if allowed is not None else set(WorldState.model_fields) - {"meta"}
        )
        self.endpoints: dict[str, Endpoint] = {}
        self._documents: list[str] = []
        self._document_endpoints: list[str] = []

        for api, catalog in load_catalogs().items():
            service = SERVICE_ALIASES.get(api, api)
            if service not in self.allowed_services:
                continue
            router = self._load_router(service)
            for specification in catalog:
                endpoint = self._build_endpoint(service, specification, router)
                if endpoint is None:
                    continue
                if endpoint.id in self.endpoints:
                    raise ValueError(f"Duplicate endpoint ID: {endpoint.id}")
                self.endpoints[endpoint.id] = endpoint
                self._documents.append(_search_document(api, specification))
                self._document_endpoints.append(endpoint.id)

        self._scorer = BM25Scorer(self._documents)
        self.connected_services = sorted({endpoint.service for endpoint in self.endpoints.values()})

    @staticmethod
    def _load_router(service: str) -> Router:
        module = importlib.import_module(f"automationbench.tools.api.routes.{service}")
        router = getattr(module, f"route_{service}")
        if not isinstance(router, Router):
            raise TypeError(f"{service} does not expose a structured Router")
        return router

    @staticmethod
    def _build_endpoint(
        service: str,
        specification: CatalogEndpoint,
        router: Router,
    ) -> Endpoint | None:
        if not specification.implemented:
            return None
        if (
            router.resolve(
                specification.id,
                specification.method,
                specification.path,
            )
            is None
        ):
            return None

        # Internal path values identify the task's simulated workspace and are
        # supplied by the catalog, not by the model. ``hiddenParams`` covers
        # other adapter-only parameters that likewise stay out of the contract.
        hidden_parameters = set(specification.internal_path) | specification.hidden_params
        parameters = {
            name: metadata
            for name, metadata in specification.parameters.items()
            if name not in hidden_parameters
        }
        body_parameters = {
            name for name, metadata in parameters.items() if metadata.get("location") == "body"
        }
        if body_parameters:
            names = ", ".join(sorted(body_parameters))
            raise ValueError(
                f"{specification.id} has body parameters outside request schema: {names}"
            )

        placeholders = set(re.findall(r"{([^}]+)}", specification.path))
        path_parameters = {
            name: metadata
            for name, metadata in parameters.items()
            if metadata.get("location") == "path"
        }
        query_parameters = {
            name: metadata
            for name, metadata in parameters.items()
            if metadata.get("location") == "query"
        }
        declared_path = set(path_parameters) | set(specification.internal_path)
        if placeholders != declared_path:
            raise ValueError(f"{specification.id} path fields do not match {specification.path}")

        return Endpoint(
            id=specification.id,
            service=service,
            method=specification.method,
            path=specification.path,
            description=specification.description,
            path_schema=schema_from_parameters(path_parameters),
            params_schema=schema_from_parameters(query_parameters),
            body_schema=specification.request,
            response=specification.response,
            router=router,
            internal_path=specification.internal_path,
            fixed_params=specification.fixed_params,
            fixed_body=specification.fixed_body,
            forbidden_body=specification.forbidden_body,
            body_aliases=specification.body_aliases,
        )

    def fetch_schema(self) -> dict[str, Any]:
        """Return a parser-friendly schema with one branch per executable endpoint."""
        endpoint_ids = sorted(self.endpoints)
        body_types = sorted(
            {
                body_type
                for endpoint in self.endpoints.values()
                if endpoint.body_schema is not None
                if isinstance(body_type := endpoint.body_schema.get("type"), str)
            }
        )
        if not body_types:
            body_types = ["object"]
        body_type: str | list[str] = body_types[0] if len(body_types) == 1 else body_types
        schema: dict[str, Any] = {
            "title": "api_fetchArguments",
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    **({"enum": endpoint_ids} if endpoint_ids else {}),
                },
                "path": {"type": "object"},
                "params": {"type": "object"},
                "body": {"type": body_type},
            },
            "required": ["endpoint"],
            "additionalProperties": False,
        }
        if not endpoint_ids:
            schema["not"] = {}
            return schema

        # Keep the endpoint-specific validation below the root object so tool
        # parsers can discover each argument's type before selecting a branch.
        schema["allOf"] = [
            {
                "oneOf": [
                    _endpoint_input_schema(self.endpoints[endpoint_id])
                    for endpoint_id in endpoint_ids
                ]
            }
        ]
        return schema

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        results = []
        for index in self._scorer.top_k(query, k=top_k):
            endpoint = self.endpoints[self._document_endpoints[index]]
            result: dict[str, Any] = {
                "endpoint": endpoint.id,
                "description": endpoint.description,
                "path": endpoint.path_schema,
                "params": endpoint.params_schema,
                "response": endpoint.response,
            }
            if endpoint.body_schema is not None:
                result["body"] = endpoint.body_schema
            results.append(result)

        payload: dict[str, Any] = {
            "connected_services": self.connected_services,
            "results": results,
            "count": len(results),
        }
        if "google_sheets" in self.allowed_services:
            payload["resources"] = {
                "google_sheets": {
                    "spreadsheets": [
                        {"spreadsheetId": item.id, "title": item.title}
                        for item in self.world.google_sheets.spreadsheets
                    ],
                    "worksheets": [
                        {
                            "spreadsheetId": item.spreadsheet_id,
                            "sheetId": item.id,
                            "title": item.title,
                        }
                        for item in self.world.google_sheets.worksheets
                    ],
                }
            }
        return payload

    def execute(
        self,
        endpoint_id: str,
        path: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> CallToolResult:
        endpoint = self.endpoints.get(endpoint_id)
        if endpoint is None:
            return _error(
                "unknown_endpoint",
                f"{endpoint_id!r} is not available in this workspace; use api_search first.",
            )

        values = {
            "path": {} if path is None else path,
            "params": {} if params is None else params,
            "body": {} if body is None else body,
        }
        # Path and query parameters are named fields. Request bodies follow the
        # endpoint schema and can therefore be objects or arrays.
        for name in ("path", "params"):
            value = values[name]
            if not isinstance(value, dict):
                return _error(
                    "invalid_request",
                    f"{name} must be object",
                    endpoint.id,
                )

        errors = validation_errors(endpoint.path_schema, values["path"], "path")
        errors.extend(validation_errors(endpoint.params_schema, values["params"], "params"))
        if endpoint.body_schema is None:
            if values["body"]:
                errors.append("this endpoint does not accept a body")
        else:
            errors.extend(validation_errors(endpoint.body_schema, values["body"], "body"))
        if isinstance(values["body"], dict):
            for name in endpoint.forbidden_body & values["body"].keys():
                errors.append(f"body.{name} selects a different endpoint operation")
        if errors:
            return _error("invalid_request", "; ".join(errors), endpoint.id)

        # Endpoint-local metadata bridges differences between the public catalog
        # and a simulator adapter. Aliases rename public fields, while fixed
        # values select an operation that shares a route with other endpoints.
        if isinstance(values["body"], dict):
            request_body = {
                endpoint.body_aliases.get(name, name): value
                for name, value in values["body"].items()
            }
            request_body.update(endpoint.fixed_body)
        else:
            request_body = values["body"]
        params_dict = {**values["params"], **endpoint.fixed_params}
        internal_path = endpoint.path
        for name, value in {
            **values["path"],
            **endpoint.internal_path,
        }.items():
            internal_path = internal_path.replace("{" + name + "}", str(value))

        # The endpoint ID selects the operation. Some identifiers are full URIs,
        # but interpolation must not make their slashes select another handler.
        declared_route = endpoint.router.resolve(endpoint.id, endpoint.method, endpoint.path)
        resolved_route = endpoint.router.resolve(endpoint.id, endpoint.method, internal_path)
        if (
            declared_route is None
            or resolved_route is None
            or resolved_route.handler is not declared_route.handler
        ):
            return _error(
                "invalid_request",
                "path parameters do not resolve to the selected endpoint",
                endpoint.id,
            )

        try:
            response = endpoint.router.execute(
                endpoint.id,
                self.world,
                endpoint.method,
                internal_path,
                params_dict,
                request_body,
            )
            payload = json.loads(response)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return _error("execution_error", str(error), endpoint.id)
        return _result(payload, is_error=_is_application_error(payload))


__all__ = ["APIContract", "Endpoint"]
