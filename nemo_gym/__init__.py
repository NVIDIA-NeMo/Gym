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
import sys
from os import environ
from pathlib import Path
from typing import Callable, List, Optional, Union


# /path/to/dir/Gym (PARENT_DIR)
# |- cache (CACHE_DIR)
# |- results (RESULTS_DIR)
# |- nemo_gym (ROOT_DIR)
# |- responses_api_models
# |- responses_api_agents
# ...
ROOT_DIR = Path(__file__).absolute().parent
PARENT_DIR = ROOT_DIR.parent

# Editable install: PARENT_DIR is the repo root (has pyproject.toml)
# Wheel install: PARENT_DIR is site-packages/ so use cwd instead
_is_editable_install = (PARENT_DIR / "pyproject.toml").exists()
WORKING_DIR = PARENT_DIR if _is_editable_install else Path.cwd()

CACHE_DIR = WORKING_DIR / "cache"
RESULTS_DIR = WORKING_DIR / "results"


def _parse_extra_roots(value: str) -> List[Path]:
    return [Path(p).expanduser() for p in value.split(":") if p]


def _parse_bool_env_var(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


NEMO_GYM_ROOT_ENV_VAR = "NEMO_GYM_ROOT"
NEMO_GYM_EXTRA_ROOTS_ENV_VAR = "NEMO_GYM_EXTRA_ROOTS"
NEMO_GYM_ALLOW_ROOT_OVERRIDE_ENV_VAR = "NEMO_GYM_ALLOW_ROOT_OVERRIDE"

# Final-rung override for the artifact resolver. Defaults to PARENT_DIR so that
# wheel and editable installs both behave as before when no env var is set.
NEMO_GYM_ROOT_DIR = (
    Path(environ[NEMO_GYM_ROOT_ENV_VAR]).expanduser() if environ.get(NEMO_GYM_ROOT_ENV_VAR) else PARENT_DIR
)

# Extra plugin roots searched between cwd and NEMO_GYM_ROOT_DIR. See
# the "Developing External Artifacts" section in the Configuration Reference docs.
#
# These three constants are read at runtime by ``get_artifact_roots`` and
# ``resolve_artifact`` (and by ``benchmarks.py`` via ``nemo_gym.ALLOW_ROOT_OVERRIDE``).
# To override them in a test, monkeypatch the module attribute (``nemo_gym.EXTRA_ROOTS``,
# etc.) — re-importing the binding (``from nemo_gym import EXTRA_ROOTS``) captures the
# value at import time and won't see later patches.
EXTRA_ROOTS: List[Path] = _parse_extra_roots(environ.get(NEMO_GYM_EXTRA_ROOTS_ENV_VAR, ""))

# When False (default), an artifact present in more than one root raises
# ArtifactCollisionError instead of silently using the earliest one. Set
# NEMO_GYM_ALLOW_ROOT_OVERRIDE=1 to opt back into earliest-wins precedence
# (cwd > NEMO_GYM_EXTRA_ROOTS > NEMO_GYM_ROOT_DIR).
ALLOW_ROOT_OVERRIDE: bool = _parse_bool_env_var(environ.get(NEMO_GYM_ALLOW_ROOT_OVERRIDE_ENV_VAR, ""))


class ArtifactCollisionError(RuntimeError):
    """Raised when the same artifact is defined by multiple roots and override is not allowed."""


def get_artifact_roots() -> List[Path]:
    """Ordered list of roots searched by ``resolve_artifact`` and benchmark discovery."""
    return [Path.cwd(), *EXTRA_ROOTS, NEMO_GYM_ROOT_DIR]


def _format_collision(rel_path: Path, candidates: List[Path]) -> str:
    formatted = "\n  ".join(str(c) for c in candidates)
    return (
        f"Artifact {str(rel_path)!r} is defined in multiple roots:\n  {formatted}\n"
        f"Set {NEMO_GYM_ALLOW_ROOT_OVERRIDE_ENV_VAR}=1 to use the earliest "
        f"(cwd > {NEMO_GYM_EXTRA_ROOTS_ENV_VAR} > {NEMO_GYM_ROOT_ENV_VAR}), "
        f"or remove the duplicate."
    )


def resolve_artifact(
    rel_path: Union[str, Path],
    *,
    validator: Optional[Callable[[Path], bool]] = None,
) -> Path:
    """Resolve a Gym artifact path against the ordered artifact roots.

    Absolute paths are returned unchanged. Relative paths are searched against
    ``get_artifact_roots()`` (``[cwd, *EXTRA_ROOTS, NEMO_GYM_ROOT_DIR]``).

    By default, if more than one root contains the artifact, ``ArtifactCollisionError``
    is raised so the user disambiguates explicitly. Set ``NEMO_GYM_ALLOW_ROOT_OVERRIDE=1``
    (or set ``ALLOW_ROOT_OVERRIDE`` at runtime) to fall back to earliest-wins precedence.
    Roots that resolve to the same canonical filesystem path (for example, when
    cwd equals ``NEMO_GYM_ROOT_DIR``) are deduped before counting as collisions.

    If nothing matches, the path under the final root is returned so the caller
    can produce a "not found" error against the canonical default location.

    ``validator`` overrides the default ``Path.exists`` check; pass it when a
    candidate is "valid" only if specific marker files are present (for example,
    a server directory must contain ``requirements.txt`` or ``pyproject.toml``).
    """
    p = Path(rel_path)
    if p.is_absolute():
        return p
    is_valid = validator if validator is not None else Path.exists
    roots = get_artifact_roots()
    matches: List[Path] = []
    seen_resolved: set[Path] = set()
    for root in roots:
        candidate = root / p
        if not is_valid(candidate):
            continue
        resolved = candidate.resolve()
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        matches.append(candidate)
    if not matches:
        return roots[-1] / p
    if len(matches) > 1 and not ALLOW_ROOT_OVERRIDE:
        raise ArtifactCollisionError(_format_collision(p, matches))
    return matches[0]


def _augment_sys_path(extra_roots: List[Path], root_dir: Path, parent_dir: Path) -> None:
    """Append every artifact root to ``sys.path`` so plugin ``prepare.py`` modules import.

    Order matches ``resolve_artifact``: extras first, then ``NEMO_GYM_ROOT_DIR`` (if it
    differs from ``PARENT_DIR``), then ``PARENT_DIR`` last. Earlier ``sys.path`` entries
    win for ``importlib.import_module`` lookups, so an extras-root
    ``benchmarks/foo/prepare.py`` shadows an in-Gym one when ``ALLOW_ROOT_OVERRIDE`` is on,
    matching the artifact resolver's earliest-wins precedence.

    With no extras and no ``NEMO_GYM_ROOT`` override the call collapses to a single
    ``append(parent_dir)``, identical to the pre-Phase-1 behavior.
    """
    paths: List[str] = []
    for extra_root in extra_roots:
        paths.append(str(extra_root))
    if root_dir.resolve() != parent_dir.resolve():
        paths.append(str(root_dir))
    paths.append(str(parent_dir))
    for p in paths:
        if p not in sys.path:
            sys.path.append(p)


_augment_sys_path(EXTRA_ROOTS, NEMO_GYM_ROOT_DIR, PARENT_DIR)

# TODO: Maybe eventually we want an override for OMP_NUM_THREADS ?

# Turn off HF tokenizers paralellism
environ["TOKENIZERS_PARALLELISM"] = "false"

# Huggingface related caching directory overrides to local folders.
# Only override if not already set by the user.
if "HF_DATASETS_CACHE" not in environ:
    environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "huggingface")
if "HF_HOME" not in environ:
    environ["HF_HOME"] = str(CACHE_DIR / "huggingface")


OLD_PRINT = print


def print_always_flushes(*args, **kwargs) -> None:
    kwargs["flush"] = True
    OLD_PRINT(*args, **kwargs)


__builtins__["print"] = print_always_flushes


from nemo_gym.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)
