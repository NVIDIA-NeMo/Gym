# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load environment-owned dataset hooks selected by trusted training configuration."""

import importlib
import sys
from functools import cache
from pathlib import Path
from types import ModuleType


@cache
def load_dataset_hooks(module_name: str) -> ModuleType:
    """Load validation, sampling, and sampled-input hooks from an installed environment.

    The module comes from operator configuration, never a task row. Resources
    components live beside the installed ``nemo_gym`` package in a Gym checkout.
    Hooks must not import a training framework or render during module import.
    """
    if not module_name or any(not part.isidentifier() for part in module_name.split(".")):
        raise ValueError("Dataset hooks require a dotted Python module name")
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.append(root)
    module = importlib.import_module(module_name)
    for name in ("validate_data_config", "dataloader_kwargs", "materialize_examples"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Dataset hooks module {module_name!r} must export {name}()")
    return module
