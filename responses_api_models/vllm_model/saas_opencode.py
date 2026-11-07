# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://gitlab-master.nvidia.com/pritamb/saas-bench/-/blob/ad48579c1af66bb7100501f45460bb3e9ec5eaae/integrations/gym/responses_api_models/vllm_model/saas_opencode.py (Apache-2.0).
"""SaaS adapter for Gym's chat/ reasoning convention; OpenCode stays unmodified."""

import asyncio
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

from fastapi import Body, HTTPException, Request
from pydantic import PrivateAttr

from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    current_rollout_id,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


class SaasOpenCodeModel(VLLMModel):
    ray_enabled = False
    _protocol_failures: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context):
        source = Path(os.environ["SAAS_BENCH_ROOT"]).resolve(strict=True)
        if not (source / "saas_bench/opencode/protocol.py").is_file():
            raise ValueError("SaaS native model protocol implementation is missing")
        sys.path.insert(0, str(source))
        super().model_post_init(context)

    async def retain_rejection(self, identity: str, key: str, payload: dict, error: str, *, attempt: int) -> None:
        """Keep rejected provider replies without feeding them back as tool actions."""
        from nemo_gym.config import get_global_config_dict

        directory = get_global_config_dict().get("model_call_capture_dir")
        if not directory:
            return

        def write():
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            path = root / ("saas_protocol_" + hashlib.sha256(identity.encode()).hexdigest() + ".jsonl")
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(
                    json.dumps(
                        {
                            "rollout_id": identity,
                            "request_fingerprint": key,
                            "provider_attempt": attempt,
                            "error": error,
                            "provider_response": payload,
                        }
                    )
                    + "\n"
                )

        await asyncio.to_thread(write)

    async def chat_completions(
        self,
        request: Request,
        body: NeMoGymChatCompletionCreateParamsNonStreaming = Body(),  # noqa: B008 -- FastAPI dependency
    ) -> NeMoGymChatCompletion:
        from saas_bench.opencode.protocol import ModelProtocolError, native_message

        identity = current_rollout_id() or (
            request.scope.get("session", {}).get("session_id", "unknown") if request is not None else "test"
        )
        key = identity + ":" + hashlib.sha256(body.model_dump_json(exclude_unset=True).encode()).hexdigest()
        if self._protocol_failures.get(key, 0) >= 3:
            raise HTTPException(
                422,
                "Model violated the native tool protocol; retry budget exhausted",
            )
        while True:
            response = await super().chat_completions(request, body)
            payload = response.model_dump(exclude_unset=True)
            try:
                for choice in payload["choices"]:
                    choice["message"] = native_message(
                        choice["message"],
                        tools_available=bool(body.tools),
                        finish_reason=choice.get("finish_reason"),
                    )
            except ModelProtocolError as exc:
                count = self._protocol_failures.get(key, 0) + 1
                await self.retain_rejection(
                    identity,
                    key,
                    response.model_dump(exclude_unset=True),
                    str(exc),
                    attempt=count,
                )
                self._protocol_failures[key] = count
                if len(self._protocol_failures) > 4096:
                    self._protocol_failures.pop(next(iter(self._protocol_failures)))
                if count >= 3:
                    raise HTTPException(
                        422,
                        "Model violated the native tool protocol; retry budget exhausted "
                        f"after {count} provider attempts: {exc}",
                    ) from exc
                # Gym may already have sent HTTP-200 SSE keepalives. Regenerate
                # here: a later SSE error cannot trigger native HTTP retries.
                # Keep the same request, sampling and per-call output budget.
                continue
            self._protocol_failures.pop(key, None)
            return NeMoGymChatCompletion.model_validate(payload)


if __name__ == "__main__":
    SaasOpenCodeModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SaasOpenCodeModel.run_webserver()
