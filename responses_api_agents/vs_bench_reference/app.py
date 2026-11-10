# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge Gym model calls to the original VS-Bench reasoning agent.

The author's MIT-licensed code is invoked from a separate pinned checkout,
https://github.com/zelaix/VS-Bench at 2c3531cdf91d1b620b4a9015126a242eb305a4d0.
This prototype supports reasoning and invokes the original perception scorers.
"""

import asyncio
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

from aiohttp import ClientError
from fastapi import Body, Request, Response
from pydantic import ConfigDict, PrivateAttr

from benchmarks.vs_bench.prepare import AUTHOR_REVISION, DATASET_REVISION, GAMES
from benchmarks.vs_bench.reference_identity import validate_author_checkout
from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class VSBenchAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef | None = None
    author_root: str
    author_python: str
    reasoning_model: bool = True
    temperature: float = 1.0
    max_response_tokens: int
    max_reasoning_tokens: int
    max_errors: int = 20
    max_author_processes: int = 32
    child_message_timeout_seconds: float = 600


class VSBenchAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class VSBenchAgentResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]
    action: int | None = None
    author_model_calls: int
    author_exception_types: list[str]
    author_source_variant: str


def author_chat_to_gym(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        parts = []
        for item in message["content"]:
            if item["type"] == "text":
                parts.append({"type": "input_text", "text": item["text"]})
            elif item["type"] == "image_url":
                image = item["image_url"]
                parts.append({"type": "input_image", "image_url": image["url"], "detail": image["detail"]})
            else:
                raise ValueError("Unexpected author prompt content")
        result.append({"role": message["role"], "content": parts})
    return result


async def write_frame(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    writer.write(struct.pack("!I", len(payload)) + payload)
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    length = struct.unpack("!I", await reader.readexactly(4))[0]
    return json.loads(await reader.readexactly(length))


class VSBenchReferenceAgent(SimpleResponsesAPIAgent):
    config: VSBenchAgentConfig
    _author_processes: asyncio.Semaphore = PrivateAttr()
    _author_source_variant: str = PrivateAttr()

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Report author scores per game and a macro only at full coverage."""
        scores: dict[str, dict[int, float]] = {game: {} for game in GAMES}
        track = None
        source_variant = None
        for task in tasks:
            if len(task) != 1:
                raise ValueError("Reference evaluation expects one rollout per question")
            result = task[0]
            variant = result.get("author_source_variant")
            if variant not in ("upstream", "kaz_numpy_labels_v1") or (
                source_variant is not None and source_variant != variant
            ):
                raise ValueError("Missing or mixed VS-Bench source variant")
            source_variant = variant
            metadata = result.get("verifier_metadata") or {}
            row_track = metadata.get("track")
            if row_track not in ("reasoning", "perception") or (track is not None and row_track != track):
                raise ValueError("A score aggregate must contain one benchmark track")
            track = row_track
            game = metadata.get("game")
            index = metadata.get("row_index")
            reward = result.get("reward")
            limit = 300 if track == "perception" and game == "knights_archers_zombies" else 400
            if game not in scores or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < limit:
                raise ValueError("Invalid game or row identity")
            if index in scores[game] or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                raise ValueError("Duplicate question or invalid reward")
            if track == "reasoning" and reward not in (0, 0.0, 1, 1.0):
                raise ValueError("Invalid reasoning reward")
            scores[game][index] = float(reward)
        if track is None:
            return {}
        metric = "accuracy" if track == "reasoning" else "score"
        metrics: dict[str, Any] = {}
        for game, values in scores.items():
            metrics[f"vs_bench/{track}/{game}/count"] = len(values)
            if values:
                metrics[f"vs_bench/{track}/{game}/{metric}"] = sum(values.values()) / len(values)
        complete = all(
            set(values) == set(range(300 if track == "perception" and game == "knights_archers_zombies" else 400))
            for game, values in scores.items()
        )
        metrics[f"vs_bench/{track}/full_coverage"] = complete
        if complete:
            metrics[f"vs_bench/{track}/macro_{metric}"] = sum(sum(v.values()) / len(v) for v in scores.values()) / len(
                scores
            )
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = ("vs_bench/reasoning/macro_accuracy", "vs_bench/perception/macro_score")
        return {key: agent_metrics[key] for key in keys if key in agent_metrics}

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._author_source_variant = validate_author_checkout(self.config.author_root, self.config.author_python)
        if (
            min(
                self.config.max_author_processes,
                self.config.max_errors,
                self.config.max_response_tokens,
                self.config.max_reasoning_tokens,
            )
            < 1
        ):
            raise ValueError("Process concurrency, token limits and author attempt limit must be positive")
        self._author_processes = asyncio.Semaphore(self.config.max_author_processes)

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise NotImplementedError("The author agent owns its calls through /run")

    async def run(self, request: Request, body: VSBenchAgentRunRequest) -> VSBenchAgentResponse:
        meta = body.verifier_metadata
        if (
            (meta.get("author_revision"), meta.get("dataset_revision"))
            != (
                AUTHOR_REVISION,
                DATASET_REVISION,
            )
            or meta.get("game") not in GAMES
            or meta.get("track") not in ("reasoning", "perception")
        ):
            raise ValueError("Wrong benchmark track or pinned source")
        source_prompt = dict(meta["source_prompt_text"])
        payload = body.responses_create_params.model_dump(mode="json", exclude_none=True)
        images = []
        for message in payload["input"]:
            for part in message["content"]:
                if part["type"] == "input_image":
                    url = part["image_url"]
                    prefix = "data:image/png;base64,"
                    if not isinstance(url, str) or not url.startswith(prefix):
                        raise ValueError("Expected pinned dataset's embedded image")
                    images.append(url[len(prefix) :])
        if not images:
            raise ValueError("Source image missing")
        source_prompt["image_path"] = images
        task = {
            "game": meta["game"],
            "track": meta["track"],
            "reasoning_model": self.config.reasoning_model,
            "temperature": self.config.temperature,
            "max_response_tokens": self.config.max_response_tokens,
            "max_reasoning_tokens": self.config.max_reasoning_tokens,
            "max_errors": self.config.max_errors,
            "dataset_row": {"prompt": source_prompt},
        }
        if meta["track"] == "reasoning":
            task["dataset_row"].update(next_action=meta["next_action"], next_legal_actions=meta["next_legal_actions"])
        worker = Path(__file__).resolve().parents[2] / "benchmarks/vs_bench/reference_reasoning_agent.py"
        env = {
            "PATH": os.defpath,
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MPLBACKEND": "Agg",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        async with self._author_processes:
            process = await asyncio.create_subprocess_exec(
                self.config.author_python,
                str(worker),
                self.config.author_root,
                cwd=self.config.author_root,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            last_response = None
            last_call_failed = False
            cookies = request.cookies
            try:
                await write_frame(process.stdin, task)
                while True:
                    frame = await asyncio.wait_for(
                        read_frame(process.stdout), self.config.child_message_timeout_seconds
                    )
                    if frame["type"] == "complete":
                        break
                    if frame["type"] != "model_request":
                        raise RuntimeError("Author process failed")
                    author_call = frame["request"]
                    model_call = dict(payload)
                    model_call["input"] = author_chat_to_gym(author_call["messages"])
                    model_call["temperature"] = author_call["temperature"]
                    model_call["max_output_tokens"] = author_call["max_completion_tokens"]
                    try:
                        reply = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=self.url_path_for_run("/v1/responses", body),
                            json=NeMoGymResponseCreateParamsNonStreaming.model_validate(model_call),
                            cookies=cookies,
                        )
                        await raise_for_status(reply)
                        last_response = NeMoGymResponse.model_validate(await get_response_json(reply))
                        cookies = reply.cookies
                    except (ClientError, asyncio.TimeoutError):
                        last_call_failed = True
                        await write_frame(process.stdin, {"type": "model_error"})
                        continue
                    last_call_failed = False
                    if last_response.status == "incomplete":
                        details = last_response.incomplete_details
                        if details is None or details.reason != "max_output_tokens":
                            raise RuntimeError("Unknown incomplete model response")
                        finish_reason = "length"
                    elif last_response.status in (None, "completed"):
                        finish_reason = "stop"
                    else:
                        raise RuntimeError("Model response did not complete")
                    usage = last_response.usage
                    await write_frame(
                        process.stdin,
                        {
                            "type": "model_response",
                            "finish_reason": finish_reason,
                            "content": last_response.output_text,
                            "prompt_tokens": usage.input_tokens if usage else None,
                            "completion_tokens": usage.output_tokens if usage else None,
                        },
                    )
                if await process.wait() != 0 or last_response is None or last_call_failed:
                    raise RuntimeError("Author agent did not complete a model call")
                if meta["track"] == "perception":
                    if self.config.resources_server is None or not isinstance(frame.get("answer"), str):
                        raise RuntimeError("Perception requires a scorer and a completed author answer")
                    scorer_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=self.url_path_for_run("/verify", body),
                        json={
                            **body.model_dump(mode="json"),
                            "response": last_response.model_dump(mode="json"),
                            "author_answer": frame["answer"],
                        },
                        cookies=cookies,
                    )
                    await raise_for_status(scorer_response)
                    scored = await get_response_json(scorer_response)
                    if scored.get("author_source_variant") != self._author_source_variant:
                        raise ValueError("Agent and perception scorer source variants differ")
                    score = scored["reward"]
                else:
                    score = float(frame["correct"])
                return VSBenchAgentResponse(
                    **body.model_dump(),
                    response=last_response,
                    reward=score,
                    action=frame.get("action"),
                    author_model_calls=frame["model_calls"],
                    author_exception_types=frame["exception_types"],
                    author_source_variant=self._author_source_variant,
                )
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()


if __name__ == "__main__":
    VSBenchReferenceAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VSBenchReferenceAgent.run_webserver()
