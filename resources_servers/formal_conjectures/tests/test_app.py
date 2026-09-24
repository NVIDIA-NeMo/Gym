# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercises ``FormalConjecturesVerifier.verify`` against real request/response shapes.

``VERIFIER_FIXTURE`` as declared in ``app.py`` uses the bare ``FormalConjecturesVerifier``
class as its ``server_factory``. That class has no ``__init__``: it is a mixin that only gets
``config``/``_sandbox_client``/``_toolchain`` from ``FormalConjecturesResourcesServer.
model_post_init``, which a fixture-constructed instance never runs. Calling it directly would
raise ``AttributeError`` on ``self.config`` before a single case could execute. So this module
builds its own fixture with a working ``server_factory`` instead of using ``VERIFIER_FIXTURE``
as-is; the fixture's ``request_model``/``cases_path`` are reused unchanged.

The fake sandbox is driven entirely by markers embedded in the submitted Lean code (the
``response`` text each JSONL case declares) rather than a mock configured per test function,
because ``VerifierFixture`` calls ``server_factory()`` fresh per case with no per-case hook
available for ``full_reward``/``zero_reward``/``malformed`` cases (only ``determinism`` cases
get ``reseed``, and this fixture declares none).
"""

import asyncio
from dataclasses import replace
from typing import Any, Dict

from ..app import (
    VERIFIER_FIXTURE,
    FormalConjecturesResourcesServerConfig,
    FormalConjecturesVerifier,
)


class _FakeSandboxClient:
    """Returns canned compiler output selected by a marker in the submitted code.

    Mirrors what a real Lean sandbox would say for each situation, so the fixture cases
    exercise the exact same ``verify()`` branches a real compile would hit.
    """

    async def execute_lean4(self, code: str, timeout: float) -> Dict[str, Any]:
        if "MARKER_TIMEOUT" in code:
            return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}
        if "MARKER_COMPILE_ERROR" in code:
            return {"process_status": "failed", "stdout": "", "stderr": "error: unknown identifier 'bogus'"}
        if "sorry" in code:
            # A real `#print axioms` on an unfilled proof reports `sorryAx` in the list.
            return {"process_status": "completed", "stdout": "'thm' depends on axioms: [sorryAx]", "stderr": ""}
        return {
            "process_status": "completed",
            "stdout": "'thm' depends on axioms: [propext, Classical.choice]",
            "stderr": "",
        }


class _FixtureServer(FormalConjecturesVerifier):
    """A ``FormalConjecturesVerifier`` with the dependencies ``model_post_init`` normally sets."""

    async def _check_toolchain_once(self, expected) -> None:
        return None


def _build_fixture_server() -> _FixtureServer:
    server = _FixtureServer()
    server.config = FormalConjecturesResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="formal_conjectures",
        check_lean_version=False,
    )
    server._sandbox_client = _FakeSandboxClient()
    return server


FIXTURE = replace(VERIFIER_FIXTURE, server_factory=_build_fixture_server)


def test_verifier_fixture() -> None:
    from nemo_gym.verifier_fixture import exercise_verifier_fixture

    asyncio.run(
        exercise_verifier_fixture(
            FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )
