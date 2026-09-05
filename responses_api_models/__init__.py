# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility namespace for model backends moved to :mod:`model_backends`.

New integrations should import from ``model_backends``. The internal configuration wire key remains
``responses_api_models`` and is intentionally unaffected by this filesystem compatibility package.
"""

from model_backends import __path__ as _canonical_backend_paths


__path__ = list(_canonical_backend_paths)
