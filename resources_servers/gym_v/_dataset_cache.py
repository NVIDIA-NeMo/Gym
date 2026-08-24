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
from __future__ import annotations

import functools
import importlib
import logging
import threading
from collections import OrderedDict
from typing import Any


logger = logging.getLogger(__name__)

_REASONING_GYM_ENV_CLASSES: list[tuple[str, str]] = [
    ("gym_v.envs.single_turn.algorithmic.game_of_life", "GameOfLifeEnv"),
    ("gym_v.envs.single_turn.logic.mini_sudoku", "MiniSudokuEnv"),
    ("gym_v.envs.single_turn.puzzles.tower_of_hanoi", "TowerOfHanoiEnv"),
    ("gym_v.envs.single_turn.puzzles.maze_qa", "MazeQAEnv"),
    ("gym_v.envs.single_turn.arc.arc_agi", "ArcAgiEnv"),
]


def _hashable(value: Any) -> Any:
    """Convert nested kwargs into a stable hashable cache key."""

    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_hashable(item) for item in sorted(value, key=repr))
    if hasattr(value, "tolist"):
        return _hashable(value.tolist())
    return value


def _dataset_kwargs_for(env: object) -> Any:
    if hasattr(env, "_dataset_kwargs"):
        return getattr(env, "_dataset_kwargs")
    return getattr(env, "dataset_kwargs", {})


def _patch_make_dataset(cls: type, maxsize: int) -> bool:
    # Soft-skip env classes that no longer define _make_dataset on the class
    # itself (e.g. MazeQAEnv in gym-v@d47a022 inlines its dataset construction
    # in reset() instead of factoring out _make_dataset). Such envs don't
    # benefit from the per-class cache, but they still work correctly — and
    # the parent plan only requires the cache to fire on classes that
    # actually expose _make_dataset.
    original = cls.__dict__.get("_make_dataset")
    if original is None:
        logger.debug(
            "Skipping cache patch for %s.%s: class does not define _make_dataset",
            cls.__module__,
            cls.__qualname__,
        )
        return False
    if getattr(original, "_gym_v_cached", False):
        return False

    cache: OrderedDict[Any, Any] = OrderedDict()
    lock = threading.Lock()

    @functools.wraps(original)
    def patched(self: object, *args: Any, **kwargs: Any) -> Any:
        # The seed argument changes the dataset when _dataset_kwargs does not
        # already specify one, so method args are part of the cache key.
        key = (
            cls.__module__,
            cls.__qualname__,
            _hashable(_dataset_kwargs_for(self)),
            _hashable(args),
            _hashable(kwargs),
        )
        with lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

        dataset = original(self, *args, **kwargs)
        with lock:
            cache[key] = dataset
            cache.move_to_end(key)
            while len(cache) > maxsize:
                cache.popitem(last=False)
        return dataset

    patched._gym_v_cached = True  # type: ignore[attr-defined]
    patched._gym_v_original = original  # type: ignore[attr-defined]
    patched._gym_v_cache = cache  # type: ignore[attr-defined]
    cls._make_dataset = patched
    return True


def install_dataset_cache(maxsize: int = 32) -> int:
    """Install per-class dataset caches on reasoning-gym-backed Gym-V envs.

    Returns the number of classes patched. The function is idempotent.
    """

    installed = 0
    for module_path, cls_name in _REASONING_GYM_ENV_CLASSES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError) as exc:
            logger.debug("Skipping cache patch for %s.%s: %s", module_path, cls_name, exc)
            continue

        if _patch_make_dataset(cls, maxsize=maxsize):
            installed += 1
            logger.info("Installed dataset cache on %s.%s", module_path, cls_name)

    logger.info(
        "Installed Gym-V dataset cache on %d/%d configured classes.",
        installed,
        len(_REASONING_GYM_ENV_CLASSES),
    )
    return installed
