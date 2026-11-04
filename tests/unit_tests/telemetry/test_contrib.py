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
"""Gym's boundary onto nemo-lens's propagation and FastAPI helpers.

Both ``inject_trace_context`` and ``instrument_fastapi_app`` are defensive by design: they
must never be the reason a request fails to send or a server fails to start. That means
the exercised path in an environment without the telemetry extra is the ``except`` branch
— proving these degrade safely is exactly what needs covering, lens or not.
"""

from fastapi import FastAPI

from nemo_gym.telemetry.contrib import inject_trace_context, instrument_fastapi_app
from tests.unit_tests.telemetry.conftest import requires_lens


def test_inject_trace_context_is_a_noop_without_a_recording_span():
    headers = {}
    inject_trace_context(headers)
    assert headers == {} or "traceparent" not in headers


def test_inject_trace_context_never_raises_when_lens_is_unavailable(monkeypatch):
    """`nemo.lens.propagation` is unimportable without the telemetry extra.

    The ``import nemo.lens.propagation`` inside ``inject_trace_context`` then raises
    ``ImportError``, which must be swallowed rather than propagated.
    """
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name.startswith("nemo.lens"):
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    headers = {"existing": "value"}
    inject_trace_context(headers)
    assert headers == {"existing": "value"}


def test_instrument_fastapi_app_returns_false_when_lens_is_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name.startswith("nemo.lens"):
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    app = FastAPI()
    assert instrument_fastapi_app(app) is False


def test_instrument_fastapi_app_returns_false_rather_than_raising_on_bad_input():
    """Any failure inside the instrumentor — not just an absent import — is swallowed."""
    assert instrument_fastapi_app(object()) is False


@requires_lens
def test_instrument_fastapi_app_returns_true_when_lens_is_available():
    app = FastAPI()
    assert instrument_fastapi_app(app) is True
