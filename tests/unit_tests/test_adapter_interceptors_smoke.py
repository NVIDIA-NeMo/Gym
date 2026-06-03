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
"""Smoke gate: every builtin interceptor resolves and instantiates.

For every name in ``InterceptorRegistry.available()`` the class is resolved
(import-correctness) and instantiated with ``config={}``. Interceptors that
require non-default kwargs raise ``TypeError`` and are recorded as
``requires_config`` rather than failed.
"""

from __future__ import annotations

import logging

from nemo_gym.adapters import InterceptorRegistry


logger = logging.getLogger(__name__)


def test_all_builtins_resolve_and_instantiable_or_require_config(caplog) -> None:
    caplog.set_level(logging.INFO)

    names = InterceptorRegistry.available()
    # Subset check — framework-level minimum, follow-on PRs add more.
    assert "endpoint" in names and "logging" in names, f"framework builtins missing from {names}"
    assert len(names) >= 2, f"Expected at least 2 builtins, got {len(names)}: {names}"

    instantiates: list[str] = []
    requires_config: list[tuple[str, str]] = []

    for name in names:
        cls = InterceptorRegistry.resolve_class(name)
        assert isinstance(cls, type), f"{name!r} did not resolve to a class (got {cls!r})"

        try:
            InterceptorRegistry.create(name, config={})
        except TypeError as exc:
            requires_config.append((name, str(exc)))
        else:
            instantiates.append(name)

    summary = (
        f"{len(instantiates)}/{len(names)} instantiate with empty config; "
        f"{len(requires_config)}/{len(names)} require config"
    )
    print(summary)
    print("  empty-config OK:", sorted(instantiates))
    print("  require config:")
    for n, msg in sorted(requires_config):
        print(f"    {n}: {msg}")

    # Sanity: at least one of each side. If every interceptor took empty
    # config the contract would be too loose; if none did, the registry
    # is probably broken.
    assert instantiates, "no interceptor accepted empty config — registry likely broken"
    assert requires_config, "every interceptor accepted empty config — likely loss of required-arg validation"
