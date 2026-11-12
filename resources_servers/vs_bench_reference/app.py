# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke the unmodified author parser and isolated perception scorer.

Source contract: https://github.com/zelaix/VS-Bench, MIT license, revision
2c3531cdf91d1b620b4a9015126a242eb305a4d0, agents/prompt_agent.py and
eval/strategic_reasoning.py and eval/perception.py. Gameplay is not enabled.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, PrivateAttr

from benchmarks.vs_bench.prepare import AUTHOR_REVISION, DATASET_REVISION, GAMES, validate_source
from benchmarks.vs_bench.reference_identity import validate_author_checkout
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture


class VSBenchReferenceConfig(BaseResourcesServerConfig):
    author_root: str
    author_python: str
    bwrap_path: str | None = None
    dataset_root: str | None = None
    author_venv: str | None = None
    author_python_root: str | None = None
    max_scorers: int = 8
    scorer_timeout_seconds: float = 60


class VSBenchReasoningRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]
    author_answer: str | None = None


class VSBenchReasoningResponse(VSBenchReasoningRequest, BaseVerifyResponse):
    action: int | None = None
    parse_outcome: str | None = None
    author_source_variant: str


class VSBenchReferenceServer(SimpleResourcesServer):
    config: VSBenchReferenceConfig
    _scorer_semaphore: asyncio.Semaphore = PrivateAttr()
    _author_source_variant: str = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.max_scorers < 1 or self.config.scorer_timeout_seconds <= 0:
            raise ValueError("Scorer concurrency and timeout must be positive")
        self._author_source_variant = validate_author_checkout(self.config.author_root, self.config.author_python)
        sandbox_fields = (
            self.config.bwrap_path,
            self.config.dataset_root,
            self.config.author_venv,
            self.config.author_python_root,
        )
        if any(sandbox_fields):
            if not all(sandbox_fields) or any(not Path(path).is_absolute() for path in sandbox_fields if path):
                raise ValueError("Perception requires all absolute sandbox runtime paths")
            if not Path(self.config.bwrap_path).is_file():
                raise ValueError("Bubblewrap executable is missing")
            validate_source(Path(self.config.dataset_root), ("perception",), GAMES)
        self._scorer_semaphore = asyncio.Semaphore(self.config.max_scorers)

    async def score_perception(self, metadata: dict[str, Any], answer: str) -> float:
        if not all(
            (self.config.bwrap_path, self.config.dataset_root, self.config.author_venv, self.config.author_python_root)
        ):
            raise ValueError("Perception sandbox is not configured")
        game, index = metadata.get("game"), metadata.get("row_index")
        limit = 300 if game == "knights_archers_zombies" else 400
        if game not in GAMES or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < limit:
            raise ValueError("Invalid perception game or row")
        adapter = Path(__file__).resolve().parents[2] / "benchmarks/vs_bench"
        env = {
            "PATH": os.defpath,
            "BWRAP": self.config.bwrap_path,
            "AUTHOR_ROOT": self.config.author_root,
            "DATASET_ROOT": self.config.dataset_root,
            "AUTHOR_VENV": self.config.author_venv,
            "AUTHOR_PYTHON_ROOT": self.config.author_python_root,
            "ADAPTER_ROOT": str(adapter),
            "VS_BENCH_GAME": game,
        }
        async with self._scorer_semaphore:
            process = await asyncio.create_subprocess_exec(
                "/bin/bash",
                str(adapter / "score_in_bwrap.sh"),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(json.dumps({"game": game, "row_index": index, "answer": answer}).encode()),
                    timeout=self.config.scorer_timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise
        if process.returncode:
            raise RuntimeError(
                f"Isolated author perception scorer failed for {game} row {index} "
                f"(exit {process.returncode}); the track is incomplete"
            )
        result = json.loads(stdout)
        return float(result["reward"])

    async def score_attempt(self, metadata: dict[str, Any], answer: str) -> dict[str, Any]:
        if metadata.get("track") != "reasoning":
            raise ValueError("Only reference reasoning parsing is supported")
        if metadata.get("author_revision") != AUTHOR_REVISION:
            raise ValueError("Wrong reference revision")
        if metadata.get("dataset_revision") != DATASET_REVISION:
            raise ValueError("Wrong dataset revision")
        worker = Path(__file__).resolve().parents[2] / "benchmarks/vs_bench/reference_reasoning.py"
        payload = {
            "game": metadata["game"],
            "answer": answer,
            "legal_actions": metadata["next_legal_actions"],
            "next_action": metadata["next_action"],
        }
        env = {
            "PATH": os.defpath,
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MPLBACKEND": "Agg",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        async with self._scorer_semaphore:
            process = await asyncio.create_subprocess_exec(
                self.config.author_python,
                str(worker),
                self.config.author_root,
                cwd=self.config.author_root,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(json.dumps(payload).encode()), timeout=self.config.scorer_timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise
        if process.returncode:
            raise RuntimeError("Unmodified VS-Bench reasoning parser failed")
        return json.loads(stdout)

    async def verify(self, body: VSBenchReasoningRequest) -> VSBenchReasoningResponse:
        metadata = body.verifier_metadata
        if metadata.get("track") == "perception":
            if (
                metadata.get("author_revision") != AUTHOR_REVISION
                or metadata.get("dataset_revision") != DATASET_REVISION
            ):
                raise ValueError("Wrong reference revision")
            answer = body.author_answer if body.author_answer is not None else body.response.output_text
            return VSBenchReasoningResponse(
                **body.model_dump(),
                reward=await self.score_perception(metadata, answer),
                author_source_variant=self._author_source_variant,
            )
        result = await self.score_attempt(body.verifier_metadata, body.response.output_text)
        if result["outcome"] == "retryable_format_error":
            # The upstream agent retries invalid answers. A single-shot scorer
            # must not silently score this as a final model failure.
            raise ValueError("Reference action is invalid; the author agent retry loop is required")
        return VSBenchReasoningResponse(
            **body.model_dump(),
            reward=float(result["correct"]),
            action=result["action"],
            parse_outcome=result["outcome"],
            author_source_variant=self._author_source_variant,
        )


def _fixture_server() -> VSBenchReferenceServer:
    """Use the real upstream parser with explicitly provisioned source/runtime."""
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    config = VSBenchReferenceConfig(
        name="vs_bench_reference_fixture",
        host="127.0.0.1",
        port=8080,
        entrypoint="",
        author_root=os.environ["VS_BENCH_AUTHOR_ROOT"],
        author_python=os.environ["VS_BENCH_AUTHOR_PYTHON"],
    )
    return VSBenchReferenceServer(config=config, server_client=client)


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_server,
    request_model=VSBenchReasoningRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    VSBenchReferenceServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VSBenchReferenceServer.run_webserver()
