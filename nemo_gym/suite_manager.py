# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Named environment suites: saved lists of environment names that expand to ``config_paths``.

Production runners sweep dozens of environments at once; today that means hand-maintaining 80+
comma-separated ``config_paths``. A *suite* is a named list of environment names (resolved against
the live ``environments/`` directory) so the same sweep becomes ``--suite ultra_v3``.

Suites live in YAML files of the form::

    description: Ultra V3 evaluation suite
    environments:
      - gpqa
      - workplace_assistant

Built-in suites ship under ``nemo_gym/suites/``; user suites live under
``~/.config/nemo_gym/suites/`` and shadow a built-in of the same name. Resolution only reads config
files (via the registry); it never resolves interpolations or starts servers.
"""

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from nemo_gym.registry import (
    ENVIRONMENTS_DIR,
    EnvironmentNotFoundError,
    resolve_environment_config_paths,
)


BUILTIN_SUITES_DIR = Path(__file__).parent / "suites"
USER_SUITES_DIR = Path.home() / ".config" / "nemo_gym" / "suites"
SUITE_FILE_SUFFIX = ".yaml"


class SuiteNotFoundError(ValueError):
    """A suite was referenced by a name that is not registered."""


class EmptySuiteError(ValueError):
    """A suite was found but lists no environments."""


@dataclass(frozen=True)
class Suite:
    """A discovered suite: its name, the environments it contains, and where it came from."""

    name: str
    environments: Tuple[str, ...]
    description: Optional[str]
    path: Path
    is_builtin: bool


def _read_suite(path: Path, is_builtin: bool) -> Optional[Suite]:
    """Best-effort parse of a suite file. Returns ``None`` if it is not a suite definition.

    A file is a suite iff it is a mapping with an ``environments`` list. Non-string entries are
    dropped so a malformed line can't smuggle a non-name into resolution.
    """
    try:
        container = OmegaConf.to_container(OmegaConf.load(path), resolve=False, throw_on_missing=False)
    except Exception:
        return None

    if not isinstance(container, dict):
        return None
    environments = container.get("environments")
    if not isinstance(environments, list):
        return None

    description = container.get("description")
    return Suite(
        name=path.stem,
        environments=tuple(e for e in environments if isinstance(e, str)),
        description=description if isinstance(description, str) else None,
        path=path,
        is_builtin=is_builtin,
    )


def _read_suites_dir(suites_dir: Path, is_builtin: bool) -> Dict[str, Suite]:
    suites: Dict[str, Suite] = {}
    if not suites_dir.is_dir():
        return suites
    for path in sorted(suites_dir.glob(f"*{SUITE_FILE_SUFFIX}")):
        suite = _read_suite(path, is_builtin=is_builtin)
        if suite is not None:
            suites[suite.name] = suite
    return suites


def discover_suites(
    builtin_dir: Path = BUILTIN_SUITES_DIR,
    user_dir: Path = USER_SUITES_DIR,
) -> Dict[str, Suite]:
    """Map suite name -> :class:`Suite`. A user suite shadows a built-in of the same name."""
    suites = _read_suites_dir(builtin_dir, is_builtin=True)
    suites.update(_read_suites_dir(user_dir, is_builtin=False))
    return suites


def resolve_suite_config_paths(
    name: str,
    builtin_dir: Path = BUILTIN_SUITES_DIR,
    user_dir: Path = USER_SUITES_DIR,
    environments_dir: Path = ENVIRONMENTS_DIR,
) -> List[str]:
    """Expand suite ``name`` into the ordered, de-duplicated ``config_paths`` to load.

    Raises :class:`SuiteNotFoundError` (with a "did you mean?" hint) for an unknown suite,
    :class:`EmptySuiteError` for a suite with no environments, and
    :class:`EnvironmentNotFoundError` (annotated with the suite name) if a listed environment is
    not registered.
    """
    suites = discover_suites(builtin_dir, user_dir)
    suite = suites.get(name)
    if suite is None:
        available = sorted(suites)
        suggestions = get_close_matches(name, available, n=3, cutoff=0.6)
        if suggestions:
            hint = "Did you mean: " + ", ".join(repr(s) for s in suggestions) + "?"
        else:
            hint = "Available suites: " + (", ".join(repr(s) for s in available) or "(none)")
        raise SuiteNotFoundError(f"No suite named '{name}'.\n{hint}")

    if not suite.environments:
        raise EmptySuiteError(f"Suite '{name}' lists no environments ({suite.path}).")

    config_paths: List[str] = []
    for environment in suite.environments:
        try:
            resolved = resolve_environment_config_paths(environment, environments_dir)
        except EnvironmentNotFoundError as error:
            raise EnvironmentNotFoundError(f"Suite '{name}' references an unknown environment.\n{error}") from error
        config_paths.extend(resolved)

    # Preserve order while dropping duplicates (an environment listed twice, or shared paths).
    return list(dict.fromkeys(config_paths))
