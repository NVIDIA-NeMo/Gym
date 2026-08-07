# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn a server's config (and optionally one dataset row) into a runnable sandbox.

Every sandbox-backed server declares ``sandbox_provider``; that part is uniform.
How each one arrives at a :class:`~nemo_gym.sandbox.SandboxSpec` is not — some
spell the whole spec out in YAML, others build it in Python because it depends
on the task row. This module supplies the shared, declarative half and a pair of
opt-in escape hatches for the rest, so tooling (notably ``gym sandbox debug``)
can reconstruct a server's sandbox without importing the server itself.

A server opts in through a ``sandbox_task`` block on its config::

    sandbox_task:
      image_from_row: image_name                     # row field holding the image
      id_from_row: instance_id                       # row field naming the task
      spec_resolver: my_pkg.hooks:spec_for_row       # (row, config) -> SandboxSpec
      exec_wrapper:  my_pkg.hooks:wrap               # (command, **kwargs) -> str

Every key is optional. With none of them set, the spec comes from the config's
``sandbox_spec`` mapping alone, which already covers servers that keep their
sandbox fully declarative.

Hook targets must be importable from the CLI's environment, so point them at a
dependency-light module rather than at a server ``app.py`` that imports Ray or a
benchmark harness at module scope.
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from nemo_gym.sandbox.providers import SandboxResources, SandboxSpec
from nemo_gym.sandbox.utils import rewrite_image


# Row fields consulted, in order, when `sandbox_task.id_from_row` is unset.
DEFAULT_ID_FIELDS = ("instance_id", "task_id", "uuid", "id")

# Keys `spec_from_mapping` understands. `image_rewrites` is consumed rather than
# forwarded: it rewrites `image` instead of being a SandboxSpec field.
_SPEC_FIELDS = (
    "image",
    "ttl_s",
    "ready_timeout_s",
    "workdir",
    "env",
    "files",
    "metadata",
    "resources",
    "entrypoint",
    "provider_options",
)


class SandboxHookError(RuntimeError):
    """Raised when a ``sandbox_task`` hook cannot be loaded or run."""


def load_hook(reference: str) -> Callable[..., Any]:
    """Import a ``"module.path:attribute"`` hook reference.

    Args:
        reference: Dotted module path and attribute separated by ``:``.

    Returns:
        The referenced callable.

    Raises:
        SandboxHookError: If the reference is malformed, the module or attribute
            does not exist, or the attribute is not callable. Import failures
            keep the original exception as ``__cause__``; the common cause is a
            hook that lives in a module importing heavy server-only dependencies.
    """
    module_path, separator, attribute = reference.partition(":")
    if not separator or not module_path or not attribute:
        raise SandboxHookError(f"Sandbox hook {reference!r} must be of the form 'module.path:attribute'")

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise SandboxHookError(
            f"Sandbox hook {reference!r} could not be imported: {e}. Hooks must be importable from the "
            f"environment running them, so point them at a dependency-light module rather than a server "
            f"entrypoint that imports its harness at module scope."
        ) from e

    try:
        hook = getattr(module, attribute)
    except AttributeError as e:
        raise SandboxHookError(f"Sandbox hook {reference!r} not found: {module_path} has no {attribute!r}") from e

    if not callable(hook):
        raise SandboxHookError(f"Sandbox hook {reference!r} is not callable (got {type(hook).__name__})")
    return hook


def spec_from_mapping(mapping: Mapping[str, Any] | None) -> SandboxSpec:
    """Build a :class:`SandboxSpec` from a YAML ``sandbox_spec`` mapping.

    Any ``image_rewrites`` entry is applied to ``image`` and then dropped, so a
    deployment can point at a registry mirror without the caller knowing. Unknown
    keys fail loudly instead of being silently ignored — a typo in a spec key is
    otherwise invisible until the sandbox behaves unexpectedly.

    Args:
        mapping: The raw ``sandbox_spec`` mapping, or ``None`` for an empty spec.

    Returns:
        The equivalent ``SandboxSpec``.

    Raises:
        ValueError: If the mapping holds keys that are not spec fields.
    """
    remaining = dict(mapping or {})
    image = rewrite_image(remaining.pop("image", None), list(remaining.pop("image_rewrites", []) or []))

    spec = SandboxSpec(
        image=image,
        ttl_s=remaining.pop("ttl_s", None),
        ready_timeout_s=remaining.pop("ready_timeout_s", None),
        workdir=remaining.pop("workdir", None),
        env=dict(remaining.pop("env", {}) or {}),
        files=dict(remaining.pop("files", {}) or {}),
        metadata=dict(remaining.pop("metadata", {}) or {}),
        resources=SandboxResources.from_mapping(remaining.pop("resources", {}) or {}),
        entrypoint=remaining.pop("entrypoint", None),
        provider_options=dict(remaining.pop("provider_options", {}) or {}),
    )
    if remaining:
        known = ", ".join(sorted((*_SPEC_FIELDS, "image_rewrites")))
        raise ValueError(f"Unknown sandbox_spec keys: {', '.join(sorted(remaining))}. Known keys: {known}")
    return spec


def coerce_spec(value: Any) -> SandboxSpec:
    """Accept either a ``SandboxSpec`` or a raw mapping from a ``spec_resolver``."""
    if isinstance(value, SandboxSpec):
        return value
    if isinstance(value, Mapping):
        return spec_from_mapping(value)
    raise SandboxHookError(f"spec_resolver must return a SandboxSpec or a mapping, got {type(value).__name__}")


def task_id_for_row(row: Mapping[str, Any], *, id_from_row: str | None = None) -> str | None:
    """Return a row's task identifier.

    Looks at ``id_from_row`` when the server names the field, otherwise tries the
    conventional ones in :data:`DEFAULT_ID_FIELDS`. Nested lookups such as
    ``verifier_metadata.task_id`` are supported with dotted paths.
    """
    fields = (id_from_row,) if id_from_row else DEFAULT_ID_FIELDS
    for field in fields:
        value = _lookup(row, field)
        if value is not None and str(value):
            return str(value)
    return None


def _lookup(row: Mapping[str, Any], field: str | None) -> Any:
    """Read a possibly-dotted path out of a row, returning ``None`` if absent."""
    if not field:
        return None
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def resolve_spec(
    *,
    sandbox_spec: Mapping[str, Any] | None,
    sandbox_task: Mapping[str, Any] | None,
    row: Mapping[str, Any] | None,
    server_config: Mapping[str, Any] | None = None,
) -> tuple[SandboxSpec, str]:
    """Resolve the spec a server would use, optionally bound to one dataset row.

    A ``spec_resolver`` hook wins when the server declares one, since a server
    that computes its spec in Python is the authority on it. Otherwise the
    declarative ``sandbox_spec`` mapping is used, with ``image_from_row``
    filling in an image the config does not carry.

    Args:
        sandbox_spec: The server's ``sandbox_spec`` mapping, if any.
        sandbox_task: The server's ``sandbox_task`` hook block, if any.
        row: The dataset row to bind, or ``None`` to resolve the server's
            configured sandbox on its own.
        server_config: The full server config block, passed through to the hook.

    Returns:
        ``(spec, source)`` where ``source`` names where the spec came from, for
        display by ``--dry-run`` and for the persisted trace.
    """
    hooks = dict(sandbox_task or {})

    resolver_ref = hooks.get("spec_resolver")
    if resolver_ref:
        resolver = load_hook(str(resolver_ref))
        try:
            resolved = resolver(row, dict(server_config or {}))
        except Exception as e:
            raise SandboxHookError(f"spec_resolver {resolver_ref!r} failed: {e}") from e
        return coerce_spec(resolved), f"spec_resolver ({resolver_ref})"

    spec = spec_from_mapping(sandbox_spec)
    if spec.image is None and row is not None:
        image = _lookup(row, hooks.get("image_from_row"))
        if image:
            rewrites = list((sandbox_spec or {}).get("image_rewrites", []) or [])
            return replace_image(spec, rewrite_image(str(image), rewrites)), "row field"

    return spec, "sandbox_spec"


def replace_image(spec: SandboxSpec, image: str | None) -> SandboxSpec:
    """Return ``spec`` with a different image (``SandboxSpec`` is frozen)."""
    return SandboxSpec(
        image=image,
        ttl_s=spec.ttl_s,
        ready_timeout_s=spec.ready_timeout_s,
        workdir=spec.workdir,
        env=dict(spec.env),
        files=dict(spec.files),
        metadata=dict(spec.metadata),
        resources=spec.resources,
        entrypoint=list(spec.entrypoint) if spec.entrypoint else None,
        provider_options=dict(spec.provider_options),
    )


def wrap_command(
    command: str,
    *,
    sandbox_task: Mapping[str, Any] | None,
    exec_kwargs: Mapping[str, Any] | None = None,
    bare: bool = False,
) -> tuple[str, bool]:
    """Apply a server's ``exec_wrapper`` so a command runs as the server runs it.

    Servers routinely prepare the shell before their real command — activating a
    conda env, relaxing apt's sandbox, fixing ``PATH``. Reproducing that is the
    difference between debugging the environment the rollout sees and debugging a
    different one. ``bare`` skips the wrapper, which is what you want when the
    wrapper itself is the suspect.

    Returns:
        ``(command, wrapped)`` — ``wrapped`` is ``False`` when no wrapper applied.
    """
    wrapper_ref = (sandbox_task or {}).get("exec_wrapper")
    if bare or not wrapper_ref:
        return command, False

    wrapper = load_hook(str(wrapper_ref))
    try:
        return str(wrapper(command, **dict(exec_kwargs or {}))), True
    except Exception as e:
        raise SandboxHookError(f"exec_wrapper {wrapper_ref!r} failed: {e}") from e
