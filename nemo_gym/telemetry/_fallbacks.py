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
"""The single import point for nemo-lens instrumentation primitives.

Every instrumentation site in ``nemo_gym`` imports ``managed_span`` / ``span_cm`` /
``trace_fn`` / ``is_span_group_enabled`` / ``safe_set_span_attributes`` from here, never
from ``nemo.lens`` directly. This module resolves them once:

* nemo-lens installed -> the **real** implementations from ``nemo.lens.helpers`` and
  ``nemo.lens.state``.
* nemo-lens absent -> local no-op shims mirroring ``nemo.lens.fallbacks``.

This differs from NeMo-RL, whose ``_fallbacks.py`` re-exports the *no-op*
``nemo.lens.fallbacks`` even when lens is installed, and so needs a second
``try: from nemo.lens.helpers import ...`` at every call site. Resolving once here keeps
Gym's call sites to a single import and removes the chance of a site binding the no-op
while lens is present.

The no-op branch below is one of the four places named in
``kb/knowledge/conventions/fallback-sync.md``. It mirrors ``nemo/lens/fallbacks.py`` at
commit ``b85578fc``; when a signature changes there, change it here in the same PR.
``tests/unit_tests/telemetry/test_fallbacks.py`` asserts the two agree parameter-for-
parameter whenever lens is importable.
"""

try:
    from nemo.lens.helpers import (  # noqa: F401
        managed_span,
        safe_set_span_attributes,
        span_cm,
        trace_fn,
    )
    from nemo.lens.state import is_span_group_enabled  # noqa: F401

    NEMO_LENS_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by test_fallbacks.py via a stubbed importer
    from contextlib import contextmanager

    NEMO_LENS_AVAILABLE = False

    def trace_fn(group, name, tracer=None):
        """No-op decorator — returns the function unchanged."""

        def decorator(func):
            return func

        return decorator

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        """No-op context manager — yields None."""
        yield None

    @contextmanager
    def span_cm(name, tracer=None, record_exception=True, **attributes):
        """No-op context manager — yields None."""
        yield None

    def is_span_group_enabled(group):
        """Always returns False."""
        return False

    def safe_set_span_attributes(span, attributes, redact_keys=None):
        """No-op."""
        pass
