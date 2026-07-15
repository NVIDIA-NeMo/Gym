# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Registry of model servers under ``responses_api_models/<name>/``.

Maps each model dir to the config flavors it ships (``configs/<flavor>.yaml``), so they can be
enumerated by the token passed to ``--model-type`` (see :attr:`ModelEntry.model_types`). Reads the
directory tree only; never loads a config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from nemo_gym import PARENT_DIR
from nemo_gym.discovery import discover_components


MODELS_SUBDIR = "responses_api_models"
MODELS_DIR = PARENT_DIR / MODELS_SUBDIR
MODEL_CONFIGS_SUBDIR = "configs"


@dataclass(frozen=True)
class ModelEntry:
    """A discovered model server: its name, where it lives, and its config flavors."""

    name: str
    path: Path
    config_paths: Tuple[Path, ...]  # flavor config files, sorted (a model always ships at least one)

    @property
    def variants(self) -> Dict[str, Path]:
        """Map flavor name (config filename stem) -> config path."""
        return {path.stem: path for path in self.config_paths}

    @property
    def model_types(self) -> List[str]:
        """The tokens accepted by ``--model-type``: ``<name>`` for the flavor named after the model (the
        default the selector resolves), ``<name>/<flavor>`` for the rest."""
        return [self.name if stem == self.name else f"{self.name}/{stem}" for stem in sorted(self.variants)]


def _discover_models_in_dir(models_dir: Path) -> Dict[str, ModelEntry]:
    """Map model name -> :class:`ModelEntry` for every model dir under one ``responses_api_models/`` dir.

    The name is the directory name. A directory is a model iff it ships at least one ``configs/*.yaml`` —
    the config a user passes to ``--model-type`` (a config-less dir has nothing to select, so it isn't
    listed). Returns an empty dict if the directory is missing.
    """
    models: Dict[str, ModelEntry] = {}
    if not models_dir.is_dir():
        return models

    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        configs_dir = child / MODEL_CONFIGS_SUBDIR
        config_files = tuple(sorted(configs_dir.glob("*.yaml"))) if configs_dir.is_dir() else ()
        if not config_files:
            continue
        models[child.name] = ModelEntry(name=child.name, path=child, config_paths=config_files)

    return models


def discover_models(search_dirs: Optional[Union[Path, Sequence[Path]]] = None) -> Dict[str, ModelEntry]:
    """Map model name -> :class:`ModelEntry` for every discoverable model server.

    Scans the ``responses_api_models/`` subdir of every :func:`~nemo_gym.discovery.component_search_roots`
    root (``search_dirs`` + cwd + built-ins), merged so user models shadow same-named built-ins.
    ``search_dirs`` is one dir or a list.
    """
    return discover_components(MODELS_SUBDIR, _discover_models_in_dir, search_dirs)
