# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChemCoTBench-V2 scoring through its pinned upstream parser and layer evaluators."""

import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.chemcotbench.scoring_pool import ScoringPool, ScoringWorkerError
from resources_servers.chemcotbench.setup_molopt import ensure_molopt_runtime
from resources_servers.chemcotbench.setup_upstream import DATA_REVISION, REPO_REVISION, ensure_data, ensure_repository
from resources_servers.chemcotbench.task_data import TaskData
from resources_servers.chemcotbench.verifier_fixture import create_server, invoke


THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", re.I | re.S)
THINK_END = re.compile(r"</(?:think|thinking)\s*>", re.I)


class ChemCoTBenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    repo_path: str | None = None
    repo_revision: str = REPO_REVISION
    data_dir: str | None = None
    data_revision: str = DATA_REVISION
    enable_molopt: bool = True
    molopt_python: str | None = None
    oracle_dir: str | None = None
    run_layer3: bool = True
    max_concurrency: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=120, gt=0)


class ChemCoTBenchVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class ChemCoTBenchVerifyResponse(BaseVerifyResponse, ChemCoTBenchVerifyRequest):
    predicted_answer: str | float | int | None = None
    layer1_correct: bool = False
    layer2_state_score: float | None = None
    layer3_type1: bool | None = None
    layer3_type2: bool | None = None
    layer3_type2_matched: int | None = None
    layer3_type2_total: int | None = None
    layer3_step_score: float | None = None
    optimization_metrics: dict | None = None
    layer1_mae: float | None = None
    layer1_tanimoto: float | None = None
    layer1_fts: float | None = None
    layer1_ndcg: float | None = None
    layer1_mrr: float | None = None
    parse_ok: bool | None = None
    scoring_error: str | None = None


class ChemCoTBenchResourcesServer(SimpleResourcesServer):
    ray_enabled = False

    config: ChemCoTBenchResourcesServerConfig
    _repo: Path = PrivateAttr()
    _data: Path | None = PrivateAttr(default=None)
    _pool: ScoringPool = PrivateAttr()
    _molopt_python: Path | None = PrivateAttr(default=None)
    _oracle_dir: Path | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        self._repo = ensure_repository(self.config.repo_path, self.config.repo_revision)
        if self.config.run_layer3:
            self._data = ensure_data(self.config.data_dir, self.config.data_revision)
        if self.config.enable_molopt:
            self._molopt_python, self._oracle_dir = ensure_molopt_runtime(
                self.config.molopt_python, self.config.oracle_dir
            )
        self._pool = ScoringPool(self.config.max_concurrency)
        super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.close()

        app.router.lifespan_context = lifespan
        return app

    async def close(self) -> None:
        """Reap cached scoring workers when the server or a direct caller is done."""
        await self._pool.close()

    async def verify(self, body: ChemCoTBenchVerifyRequest) -> ChemCoTBenchVerifyResponse:
        text = THINK_END.split(THINK_BLOCK.sub("", body.response.output_text or ""))[-1].strip()
        if not text:
            return ChemCoTBenchVerifyResponse(**body.model_dump(), reward=0.0, parse_ok=False)
        is_molopt = body.verifier_metadata.task_family == "mol_opt"
        if is_molopt and self._molopt_python is None:
            return ChemCoTBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                scoring_error="molopt_disabled",
                failure_reason="Enable enable_molopt to score molecular optimization tasks",
            )
        python = str(self._molopt_python) if is_molopt else sys.executable
        command = [python, str(Path(__file__).with_name("worker.py")), "--repo", str(self._repo), "--serve"]
        if self._data is not None:
            command.extend(["--data-dir", str(self._data)])
        payload = {
            "metadata": body.verifier_metadata.model_dump(),
            "generation": text,
            "run_layer3": self.config.run_layer3,
        }
        try:
            result = await self._pool.score(
                command,
                payload,
                cwd=self._oracle_dir if is_molopt else None,
                timeout=self.config.timeout_seconds,
            )
            return ChemCoTBenchVerifyResponse(**body.model_dump(), **result)
        except TimeoutError:
            return ChemCoTBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                scoring_error="timeout",
                failure_reason="ChemCoTBench upstream scoring timed out",
            )
        except (ScoringWorkerError, OSError) as error:
            return ChemCoTBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                scoring_error=error.code if isinstance(error, ScoringWorkerError) else "upstream_error",
                failure_reason=str(error)[-2000:],
            )
        except (ValueError, TypeError) as error:
            return ChemCoTBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                scoring_error="invalid_result",
                failure_reason=f"Invalid upstream scorer result: {type(error).__name__}",
            )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    invoke=invoke,
    request_model=ChemCoTBenchVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "fixture_cases.jsonl",
)


if __name__ == "__main__":
    ChemCoTBenchResourcesServer.run_webserver()
