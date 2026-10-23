# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``boundary.json``: what ``if-boundary.mjs`` applies to every model call, compiled from the RuntimeSpec only.

Python resolves everything that depends on the row; the plugin applies it to the request OpenCode actually sends:

- ``tool_names`` / ``targets``: the per-row selection and the inventory check (``effectiveToolNames`` in the plugin);
- ``deny``: the permission denies the plugin's ``config`` hook appends;
- ``descriptions``: the effective description of each tool (the catalog interface's, overridden by the row's);
- ``parameter_descriptions``: the row's parameter descriptions as parsed JSON-pointer paths (``describeParameters``);
- ``historical_rejected_calls``: rejected model calls witnessed in a continuation prefix (``prefix_model_calls``).

The staged bytes are evidence format (``boundary_sha256`` in the seed record, ``config_sha256`` in the plugin's load
receipt): keep the keys and their order.

The row's literal system text is not here: at seed the row has no ``responses_create_params``. The plugin reads it
from the instructions file the OpenCode agent writes (see ``if-boundary.mjs``).
"""

from collections.abc import Mapping
from copy import deepcopy

from pydantic import JsonValue

from resources_servers.agentic_if.opencode_bundle.features import check_supported
from resources_servers.agentic_if.task_variants.catalog import BUILTIN_TOOLS
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec


BOUNDARY_VERSION = 1
"""The ``boundary.json`` format version the plugin accepts."""

PROVIDER_ID = "nemo_gym"
"""The OpenCode provider the OpenCode agent configures for the policy model."""

OPENCODE_EXTRA_TOOLS = frozenset({"webfetch", "websearch", "skill", "question", "lsp", "plan_enter", "plan_exit"})
"""Built-ins of the pinned OpenCode registry beyond the selectable catalog; denied whenever a row selects tools.

An explicit selection is not permission to enable every other built-in. Unknown future tools fail the boundary's
inventory check rather than silently becoming usable.
"""


def parameter_targets(descriptions: Mapping[str, str]) -> list[dict[str, JsonValue]]:
    """Parse a tool's parameter-description keys into schema paths, in the row's order.

    Keys are top-level property names or JSON pointers (RFC 6901) to nested schema objects. The plugin walks each path
    on the schema OpenCode sends (``parameterSchema``) and raises when the target does not exist; a duplicate target is
    refused here, at compile time.
    """
    seen: set[tuple[str, ...]] = set()
    targets = []
    for key, text in descriptions.items():
        path = (
            tuple(part.replace("~1", "/").replace("~0", "~") for part in key[1:].split("/"))
            if key.startswith("/")
            else ("properties", key)
        )
        if path in seen:
            raise ValueError(f"duplicate parameter target: {key}")
        seen.add(path)
        targets.append({"target": key, "path": list(path), "description": text})
    return targets


def prefix_model_calls(continuation: ContinuationSpec | None) -> dict[str, dict[str, JsonValue]]:
    """Rejected model calls witnessed in a continuation prefix, by call ID; never inferred.

    Each receipt is the ``nemo_gym_model_call`` metadata of the prefix's tool part:
    ``{"function": {"name", "arguments"}, "available_names": [...]}``.
    """
    result: dict[str, dict[str, JsonValue]] = {}
    if continuation is None:
        return result
    for message in continuation.prefix_export["messages"]:
        for part in message.get("parts", []):
            if part.get("type") != "tool":
                continue
            metadata = part.get("state", {}).get("metadata") or {}
            receipt = metadata.get("nemo_gym_model_call")
            if receipt is None:
                if (
                    part.get("tool") == "invalid"
                    or (metadata.get("nemo_gym_profile") or {}).get("dispatch_rejected") is True
                ):
                    raise ValueError("rejected prefix requires captured call evidence")
                continue
            identity = part.get("callID")
            function = receipt.get("function", {}) if isinstance(receipt, dict) else {}
            names = receipt.get("available_names") if isinstance(receipt, dict) else None
            if (
                not isinstance(identity, str)
                or not identity
                or identity in result
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
                or not isinstance(function.get("arguments"), str)
                or not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)
                or len(names) != len(set(names))
            ):
                raise ValueError("invalid or duplicate rejected-call evidence in prefix")
            result[identity] = deepcopy(receipt)
    return result


def compile_boundary(spec: RuntimeSpec) -> dict[str, JsonValue]:
    """The boundary plugin's configuration for one row; raises for unsupported or invalid rows."""
    check_supported(spec)
    tools = spec.tools
    selected = None if tools.tool_names is None else list(tools.tool_names)
    descriptions: dict[str, JsonValue] = {}
    for tool, definition in installed_interfaces(tools.tool_interfaces).items():
        if "description" in definition:
            descriptions[tool] = definition["description"]
    descriptions.update(tools.tool_descriptions)
    return {
        "boundary_version": BOUNDARY_VERSION,
        "provider_id": PROVIDER_ID,
        "tool_names": selected,
        "targets": sorted(
            set(tools.tool_names or {}) | set(tools.tool_descriptions) | set(tools.tool_parameter_descriptions)
        ),
        "deny": [] if selected is None else sorted((BUILTIN_TOOLS | OPENCODE_EXTRA_TOOLS) - set(selected)),
        "descriptions": descriptions,
        "parameter_descriptions": {
            tool: parameter_targets(texts) for tool, texts in tools.tool_parameter_descriptions.items()
        },
        # Only rows that select tools present rejected calls; omitted selection keeps OpenCode's history unchanged.
        "historical_rejected_calls": {} if selected is None else prefix_model_calls(spec.continuation),
    }
