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

sys.path.append(str(PARENT_DIR))


def _resolve_under_cwd_or_install(path) -> Path:
    """Resolve a possibly-relative path for *reading* a built-in or user-supplied file.

    Absolute paths are returned unchanged. A relative path is tried first under the current working
    directory (the user's project), then under the Gym install root (``PARENT_DIR``) where built-in
    assets live in both editable and wheel installs. This mirrors ``config_paths`` resolution, so a
    repo-relative path like ``resources_servers/<env>/data/example.jsonl`` resolves by name from any
    cwd. If neither exists the cwd candidate is returned so error messages point at the user's cwd.

    Use this for read paths only — never for write targets (e.g. metrics written next to a dataset),
    which must stay relative to the user's writable cwd rather than the install root.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    install_candidate = PARENT_DIR / p
    if install_candidate.exists():
        return install_candidate
    return cwd_candidate


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
