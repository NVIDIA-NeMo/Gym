# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Built-in model and inference backends for NeMo Gym."""

import sys
from pathlib import Path

from nemo_gym import component_search_roots


# Keep this regular package extensible. A plain ``__init__.py`` would make Python ignore
# ``model_backends/`` directories in higher-priority plugin roots even though Gym's registry
# discovers them there.
__path__ = [
    str(backend_dir)
    for root in component_search_roots(sys_path=[Path(entry) for entry in sys.path if entry])
    if (backend_dir := root / "model_backends").is_dir()
]
