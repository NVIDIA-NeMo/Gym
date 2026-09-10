# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Shared utility for creating tool wrappers with stripped parameters."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import get_type_hints


def _create_tool_wrapper(func: Callable, args_to_skip: list[str]) -> Callable:
    """Expose only agent-supplied arguments without changing function execution.

    Tool schema generators inspect both ``__signature__`` and ``__annotations__``.
    Simulator-owned values such as ``world`` must therefore be removed from both
    views, while the wrapper still forwards the complete call assembled by the
    harness to the original function.
    """
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    parameters = [
        parameter for name, parameter in signature.parameters.items() if name not in args_to_skip
    ]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Keep the two introspection surfaces in lockstep so schema generation cannot
    # publish an annotation for an argument that is absent from the signature.
    wrapper.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
    wrapper.__annotations__ = {
        name: annotation for name, annotation in hints.items() if name not in args_to_skip
    }
    return wrapper
