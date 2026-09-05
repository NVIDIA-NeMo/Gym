# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility namespace for model backends moved to :mod:`model_backends`.

New integrations should import from ``model_backends``. The internal configuration wire key remains
``responses_api_models`` and is intentionally unaffected by this filesystem compatibility package.
"""

import sys
from pathlib import Path

from nemo_gym import component_search_roots
from nemo_gym._config_aliases import LEGACY_MODEL_BACKENDS_SUBDIR, MODEL_BACKENDS_SUBDIR


# Match discovery precedence for legacy imports: earlier roots win, and canonical wins over
# legacy within one root. This also keeps pre-MB-1553 third-party backends importable rather
# than limiting the shim to Gym's built-in canonical tree.
__path__ = [
    str(backend_dir)
    for root in component_search_roots(sys_path=[Path(entry) for entry in sys.path if entry])
    for subdir in (MODEL_BACKENDS_SUBDIR, LEGACY_MODEL_BACKENDS_SUBDIR)
    if (backend_dir := root / subdir).is_dir()
]
