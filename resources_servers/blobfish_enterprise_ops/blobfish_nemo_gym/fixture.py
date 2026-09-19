# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""Verifier fixture for NeMo Gym's manifest onboarding checks.

``gym env test <environment>`` imports the resources server's ``app.py``, takes its
``VERIFIER_FIXTURE`` and exercises the scorer in-process against ``tests/verifier_cases.jsonl``:
a full-reward case that must pin the top of the declared reward range, a zero-reward case that
must pin the bottom, a malformed case that must raise, and, for seeded environments, a
determinism case that must reproduce on a fresh server. The cases script episodes on the
self-contained fixture suite in ``tests/fixtures/mini_suite``, so the check needs no download,
and they are scored through the same code as ``/verify``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .server import BlobfishVerifier, ScriptedEpisode


__all__ = ["verifier_fixture"]


def verifier_fixture(tests_dir: str | Path) -> Any:
    """The fixture NeMo Gym main expects, or ``None`` on releases without ``nemo_gym.verifier_fixture``."""
    try:
        from nemo_gym.verifier_fixture import VerifierFixture
    except ImportError:
        return None
    tests_dir = Path(tests_dir)
    suite_dir = tests_dir / "fixtures" / "mini_suite"
    return VerifierFixture(
        server_factory=lambda: BlobfishVerifier(suite_dir),
        request_model=ScriptedEpisode,
        cases_path=tests_dir / "verifier_cases.jsonl",
        reseed=_reseed,
    )


def _reseed(verifier: BlobfishVerifier, episode: ScriptedEpisode) -> None:
    """Every episode already starts from the package seed; drop any open one to make that explicit."""
    verifier.reset()
