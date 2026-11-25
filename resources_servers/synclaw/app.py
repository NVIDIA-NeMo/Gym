# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import logging
import multiprocessing as mp
import sys
import threading
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import PrivateAttr
from synclaw.core.db import preload_building_block_search, session, stage_prepared_search_database
from synclaw.rollout.loop import call_local_tool
from synclaw.rollout.policy import submission_identical_to_target, submission_similarity_to_target
from synclaw.rollout.prompts import build_initial_messages
from synclaw.tools.synthesis import SynthesisTools

from nemo_gym.core import NeMoGymEasyInputMessage
from nemo_gym.server_runtime import SimpleResourcesServer
from nemo_gym.verification import VerifierFixture
from resources_servers.synclaw.schema import (
    SYNCLAW_SUBMISSION_KEY,
    SynClawResourcesServerConfig,
    SynClawSeedSessionRequest,
    SynClawSeedSessionResponse,
    SynClawVerifyRequest,
    SynClawVerifyResponse,
)


LOG = logging.getLogger(__name__)

# Set by _init_tool_worker; untouched by the main process.
_WORKER_TOOLS: SynthesisTools | None = None


def _init_tool_worker(db_path: Path) -> None:
    global _WORKER_TOOLS
    _WORKER_TOOLS = SynthesisTools(db_path)
    _target_tools.cache_clear()


@lru_cache(maxsize=1024)
def _target_tools(db_path: Path, target_smiles: str) -> SynthesisTools:
    # Cache stock exclusions per target without mutating tools shared by other tasks.
    return SynthesisTools(db_path, hidden_target_smiles=target_smiles)


def _run_tool_in_worker(
    name: str,
    arguments: dict[str, Any],
    target_smiles: str | None,
    hide_target_building_block: bool = False,
) -> dict[str, Any]:
    if _WORKER_TOOLS is not None:
        tools = _WORKER_TOOLS
        if hide_target_building_block:
            if not target_smiles:
                raise ValueError("Target filtering requires the task target_smiles")
            tools = _target_tools(tools.db_path, target_smiles)
        return call_local_tool(
            tools=tools,
            name=name,
            arguments=arguments,
            target_smiles=target_smiles,
        )
    else:
        raise RuntimeError("SynClaw tool worker is not initialized")


class SynClawVerifier:
    """Scores a submission. It reads only the submission and the target, so it needs no database."""

    config: SynClawResourcesServerConfig

    async def verify(self, body: SynClawVerifyRequest) -> SynClawVerifyResponse:
        submission = (body.response.model_extra or {}).get(SYNCLAW_SUBMISSION_KEY)
        submitted = submission is not None
        accepted = bool((submission or {}).get("accepted"))

        similarity = submission_similarity_to_target(
            submission=submission,
            target_smiles=body.target_smiles,
        )
        similarity = float(similarity or 0)
        is_exact = submission_identical_to_target(submission, body.target_smiles) is True

        alpha = self.config.max_similarity_reward
        reward = 1.0 if is_exact else alpha * similarity * accepted

        return SynClawVerifyResponse(
            **(body.model_dump() | {"evaluation_completed": True}),
            reward=reward,
            submitted=submitted,
            accepted=accepted,
            similarity=similarity,
            exact_reconstruction=is_exact,
            target_hidden_from_stock=self.config.hide_target_building_block,
        )


class SynClawResourcesServer(SynClawVerifier, SimpleResourcesServer):
    config: SynClawResourcesServerConfig

    _db_path: Path = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._db_path = self.config.db_path
        if self.config.db_staging_dir is not None:
            self._db_path = stage_prepared_search_database(
                self.config.db_path, destination_directory=self.config.db_staging_dir
            )
            LOG.info("Staged SynClaw database at %s", self._db_path)
        # Load the ~1.4 GB building-block fingerprint index once here, so forked tool workers share it.
        # This also fails startup if the catalog's search index is missing or stale.
        with session(self._db_path) as conn:
            preload_building_block_search(conn)
        self._semaphore = asyncio.Semaphore(value=self.config.max_concurrent_tool_calls)
        if threading.active_count() > 1:
            LOG.warning("Forking SynClaw tool workers while %d threads run", threading.active_count())
        self._pool = self._new_pool()
        atexit.register(lambda: self._pool.shutdown())

    def _new_pool(self) -> Executor:
        pool_kwargs = dict(
            max_workers=self.config.max_concurrent_tool_calls,
            initializer=_init_tool_worker,
            initargs=(self._db_path,),
        )
        if not sys.platform.startswith("linux"):
            return ThreadPoolExecutor(thread_name_prefix="synclaw-tool", **pool_kwargs)
        pool = ProcessPoolExecutor(mp_context=mp.get_context("fork"), **pool_kwargs)
        # A fork pool starts every worker on its first submit. Do it now, before the web server starts threads
        # whose locks a forked child could inherit while held.
        pool.submit(int).result()
        return pool

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.call_tool)
        return app

    async def seed_session(
        self,
        request: Request,
        body: SynClawSeedSessionRequest,
    ) -> SynClawSeedSessionResponse:
        messages = build_initial_messages(
            task={"system_prompt": body.system_prompt, "question": body.question},
            config=body.config,
        )
        response_input = [NeMoGymEasyInputMessage.model_validate(message) for message in messages]
        return SynClawSeedSessionResponse(input=response_input)

    async def call_tool(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        extra = body.pop("extra_tool_call_arguments", {})
        target_smiles = extra.get("target_smiles")

        loop = asyncio.get_running_loop()
        async with self._semaphore:
            pool = self._pool
            try:
                return await loop.run_in_executor(
                    pool,
                    _run_tool_in_worker,
                    path,
                    body,
                    target_smiles,
                    self.config.hide_target_building_block,
                )
            except BrokenProcessPool:
                # A dead worker breaks the whole pool. This call still fails, but later ones get a new pool.
                if self._pool is pool:
                    LOG.error("SynClaw tool pool broke; starting a new one")
                    pool.shutdown(wait=False)
                    self._pool = self._new_pool()
                raise


class _FixtureVerifier(SynClawVerifier):
    """The verifier alone, with default scoring settings, for `VERIFIER_FIXTURE`."""

    def __init__(self) -> None:
        self.config = SynClawResourcesServerConfig.model_construct()


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_FixtureVerifier,
    request_model=SynClawVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    SynClawResourcesServer.run_webserver()
