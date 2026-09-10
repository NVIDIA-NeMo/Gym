# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Generic route dispatcher for AutomationBench's simulated APIs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from automationbench.schema.world import WorldState


@dataclass(frozen=True)
class RouteMatch:
    """A resolved route and the path captures to pass to its handler."""

    handler: Callable
    ids: tuple[str, ...]


class Router:
    """Resolve and execute one service's endpoint route table."""

    def __init__(
        self,
        routes: list[tuple[str, str, str]],
        handlers: dict[str, Callable],
        endpoints: dict[str, str] | None = None,
    ) -> None:
        endpoint_handlers = endpoints or {}
        referenced_handlers = {key for _, _, key in routes} | set(endpoint_handlers.values())
        missing = referenced_handlers - handlers.keys()
        if missing:
            # Fail when the route module is imported instead of advertising a
            # path that can only return a runtime 404.
            raise ValueError(f"Routes have no handlers: {', '.join(sorted(missing))}")
        self._routes = tuple(
            (method.upper(), re.compile(pattern), handlers[handler_key])
            for method, pattern, handler_key in routes
        )
        self._endpoints = (
            {
                endpoint_id: handlers[handler_key]
                for endpoint_id, handler_key in endpoint_handlers.items()
            }
            if endpoints is not None
            else None
        )

    def match(self, method: str, path: str) -> RouteMatch | None:
        """Return the handler and path captures for a concrete or templated path."""
        method_upper = method.upper()
        for route_method, pattern, handler in self._routes:
            if route_method != method_upper:
                continue
            match = pattern.match(path)
            if match:
                return RouteMatch(handler, match.groups())
        return None

    def resolve(self, endpoint_id: str, method: str, path: str) -> RouteMatch | None:
        """Resolve a catalog endpoint to the adapter that will execute it.

        Most REST endpoints are unambiguous from their method and path. Protocols
        such as GraphQL put many operations behind one path, so those routers also
        register canonical endpoint IDs. Keeping that distinction in the router
        prevents catalog discovery from treating every documented GraphQL operation
        as implemented merely because the shared transport path exists.
        """
        if self._endpoints is not None:
            handler = self._endpoints.get(endpoint_id)
            transport = self.match(method, path)
            return RouteMatch(handler, transport.ids) if handler and transport else None
        return self.match(method, path)

    def execute(
        self,
        endpoint_id: str,
        world: WorldState,
        method: str,
        path: str,
        params: dict[str, Any],
        body_dict: Any,
    ) -> str:
        """Execute a canonical endpoint through its registered route adapter."""
        match = self.resolve(endpoint_id, method, path)
        if match:
            return match.handler(world, list(match.ids), params, body_dict)
        return json.dumps({"error": {"code": 404, "message": f"No handler for {endpoint_id}"}})


def make_router(
    routes: list[tuple[str, str, str]],
    handlers: dict[str, Callable],
    endpoints: dict[str, str] | None = None,
) -> Router:
    """Build a router while preserving the route modules' compact table syntax."""
    return Router(routes, handlers, endpoints)
