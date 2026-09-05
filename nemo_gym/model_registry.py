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
"""Registry of model backends under ``model_backends/<name>/``.

Maps each model dir to the config flavors it ships (``configs/<flavor>.yaml``), so they can be
enumerated by the token passed to ``--model-type`` (see :attr:`ModelEntry.model_types`). Reads the
directory tree only; never loads a config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from nemo_gym import PARENT_DIR, component_search_roots
from nemo_gym._config_aliases import LEGACY_MODEL_BACKENDS_SUBDIR, MODEL_BACKENDS_SUBDIR
from nemo_gym.discovery import merge_by_name


MODELS_SUBDIR = MODEL_BACKENDS_SUBDIR
MODELS_DIR = PARENT_DIR / MODELS_SUBDIR
MODEL_CONFIGS_SUBDIR = "configs"


@dataclass(frozen=True)
class ModelEntry:
    """A discovered model server: its name, group, and config file."""

    name: str
    model_group: str
    config_path: Path


def _discover_models_in_dir(models_dir: Path) -> Dict[str, ModelEntry]:
    """Map model name -> :class:`ModelEntry` for every config flavor under one ``model_backends/`` dir.

    One entry per ``configs/<flavor>.yaml``: ``<dir>`` for the flavor named after the model, ``<dir>/<flavor>``
    for the rest. A config-less dir contributes nothing. Returns an empty dict if the directory is missing.
    """
    models: Dict[str, ModelEntry] = {}
    if not models_dir.is_dir():
        return models

    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        configs_dir = child / MODEL_CONFIGS_SUBDIR
        config_files = sorted(configs_dir.glob("*.yaml")) if configs_dir.is_dir() else []
        for config in config_files:
            name = child.name if config.stem == child.name else f"{child.name}/{config.stem}"
            models[name] = ModelEntry(name=name, model_group=child.name, config_path=config)

    return models


def discover_models() -> Dict[str, ModelEntry]:
    """Map model name -> :class:`ModelEntry` for every discoverable model server.

    Scans the canonical ``model_backends/`` subdir of every :func:`~nemo_gym.component_search_roots`
    root (``NEMO_GYM_EXTRA_ROOTS`` + cwd + built-ins). For a transition period it also discovers
    third-party backends in the legacy ``responses_api_models/`` subdir. Root precedence is preserved,
    and a canonical backend shadows a legacy backend in the same root.
    """
    per_root = []
    for root in component_search_roots():
        per_root.append(
            merge_by_name(
                [
                    _discover_models_in_dir(root / MODELS_SUBDIR),
                    _discover_models_in_dir(root / LEGACY_MODEL_BACKENDS_SUBDIR),
                ]
            )
        )
    return merge_by_name(per_root)
