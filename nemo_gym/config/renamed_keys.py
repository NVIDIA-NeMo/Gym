# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Accept renamed config keys and arguments under their old names, with a warning, until ``RENAMED_KEYS_REMOVED_IN``.

The config parser renames old keys in each config source before merging, so an old key overrides the new key from a
config of lower precedence, and the other way round, as any key does. Setting both keys in one config to different
values is an error, because neither one can win silently.

Only keys are renamed. A reference to an old key from elsewhere, such as ``${old_key.dir}``, ``${inherit_from:...}``,
``_copy`` or ``_delete_key``, still names the old key and has to be updated by hand.
"""

import functools
import inspect
import logging
import warnings
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.config_types import ConfigError


logger = logging.getLogger(__name__)

# The nemo_gym release that stops accepting the old names.
RENAMED_KEYS_REMOVED_IN = "0.10.0"

# Top-level config keys renamed: old key -> new key.
RENAMED_TOP_LEVEL_KEYS: dict[str, str] = {}
# Keys renamed inside each server instance of a server type, such as `responses_api_agents`:
# server type -> {old key: new key}.
RENAMED_SERVER_INSTANCE_KEYS: dict[str, dict[str, str]] = {}

_Function = TypeVar("_Function", bound=Callable[..., Any])


# A ConfigError, so the command line reports it without a traceback; a ValueError, so a Pydantic validator that raises
# it reports a validation error.
class ConflictingRenamedKeyError(ConfigError, ValueError):
    """A config sets a renamed key under both its old and its new name, to different values."""


def rename_deprecated_keys(config: MutableMapping, renamed: Mapping[str, str], *, where: str = "") -> None:
    """Move each old key in ``config`` to its new name in place, warning for each key moved.

    If ``config`` sets both names to equal values, the new key keeps its value; to different values, this raises
    ``ConflictingRenamedKeyError``. An OmegaConf key left unset (``???``) is a required value not given yet, so the
    other name fills it, and an unset old key moves to the new name as unset. ``where`` is the dotted path of ``config`` in the global config, used in messages.
    """
    prefix = f"{where}." if where else ""
    for old, new in renamed.items():
        # keys(), because `in` on a DictConfig is False for a key left unset.
        present = config.keys()
        if old not in present:
            continue
        # Unresolved, so an interpolation moves as written and a reference to another source does not fail here.
        if isinstance(config, DictConfig):
            values = OmegaConf.to_container(OmegaConf.masked_copy(config, [old, new]), resolve=False)
        else:
            values = config
        unset = {
            key
            for key in (old, new)
            if key in present and isinstance(config, DictConfig) and OmegaConf.is_missing(config, key)
        }
        if new in present and not unset and values[new] != values[old]:
            raise ConflictingRenamedKeyError(
                f"Config sets both '{prefix}{old}' and '{prefix}{new}' to different values. '{old}' was renamed "
                f"'{new}'; remove '{prefix}{old}' and keep only '{prefix}{new}'."
            )
        message = (
            f"Config key '{prefix}{old}' was renamed '{prefix}{new}'; the old key will be removed in nemo_gym "
            f"{RENAMED_KEYS_REMOVED_IN}. Set '{prefix}{new}' instead."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        # Python hides DeprecationWarning outside __main__ by default, and a config author needs to see this one.
        logger.warning(message)
        if new not in present or new in unset:
            config[new] = values[old]
        del config[old]


def with_renamed_keys(data: Any, renamed: Mapping[str, str]) -> Any:
    """Return ``data`` with its old keys moved to their new names, for a Pydantic ``mode="before"`` validator.

    Leaves the caller's mapping untouched.
    """
    if isinstance(data, Mapping) and any(old in data for old in renamed):
        data = dict(data)
        rename_deprecated_keys(data, renamed)
    return data


def renamed_attribute(old: str, new: str, *, owner: str) -> property:
    """A read-only property under a field's old name that warns and returns the field."""

    def getter(self: Any) -> Any:
        warnings.warn(
            f"{owner}.{old} was renamed {new}; the old name will be removed in nemo_gym {RENAMED_KEYS_REMOVED_IN}. "
            f"Use {owner}.{new} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new)

    return property(getter, doc=f"Deprecated name of ``{new}``.")


def renamed_keyword(old: str, new: str) -> Callable[[_Function], _Function]:
    """Accept the keyword argument ``new`` under its old name ``old``, with a warning.

    Passing both names is a ``TypeError``, as passing one argument twice is.
    """

    def decorate(function: _Function) -> _Function:
        def rename(kwargs: dict[str, Any]) -> None:
            if old not in kwargs:
                return
            if new in kwargs:
                raise TypeError(
                    f"{function.__qualname__}() got both '{old}' and '{new}'. '{old}' was renamed '{new}'; "
                    f"pass only '{new}'."
                )
            warnings.warn(
                f"{function.__qualname__}() argument '{old}' was renamed '{new}'; the old name will be removed in "
                f"nemo_gym {RENAMED_KEYS_REMOVED_IN}. Pass '{new}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(old)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rename(kwargs)
            return function(*args, **kwargs)

        if inspect.iscoroutinefunction(function):
            # The wrapper returns the coroutine, so callers await it as before; this keeps it detectable as async.
            inspect.markcoroutinefunction(wrapper)
        return wrapper  # type: ignore[return-value]

    return decorate
