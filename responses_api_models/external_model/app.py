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
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# Adapted from https://github.com/coreweave/inference-cookbook/blob/41f991fef0d45430d20cfd8bc142c4bfc5070369/libs/cwtrain/cwtrain/integrations/nemo_rl/gym_model.py
# Also adapted from https://github.com/coreweave/inference-cookbook/blob/41f991fef0d45430d20cfd8bc142c4bfc5070369/libs/cwtrain/cwtrain/integrations/nvext.py
# Original license: Apache-2.0.


"""Training adapter for external endpoints that accept exact token-ID completions."""

import asyncio
import hashlib
from pathlib import Path
from typing import Literal, cast

from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import Field, model_validator
from yarl import URL

from nemo_gym.client import NeMoGymAsyncOpenAI
from responses_api_models.external_model.conversion import (
    CONTEXT_FULL,
    ContextFull,
    Tokenizer,
    chat_response,
    completion_request,
    is_context_overflow,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class ExternalModelConfig(VLLMModelConfig):
    """External token endpoint, local trainer tokenizer, and confirmed checkpoint record."""

    tokenizer_path: str
    max_model_len: int = Field(gt=0)
    cache_salt_file: str
    api_key_file: str | None = None
    # The file is resolved in the server process, never in the serialized configuration.
    api_key: Literal["unused"] = "unused"
    seeds: bool = True
    return_token_id_information: Literal[True] = True
    uses_reasoning_parser: bool = True
    preserve_reasoning_in_assistant_content: Literal[True] = True
    use_completions_api: Literal[False] = False
    is_responses_native: Literal[False] = False
    supply_prefix_token_ids: Literal[False] = False
    continue_final_assistant_message: Literal[False] = False

    @model_validator(mode="after")
    def validate_external_settings(self) -> "ExternalModelConfig":
        """Refuse modes that cannot preserve the external completion contract."""
        if self.endpoint_file or self.generation_cut_control_url:
            raise ValueError("external_model requires fixed endpoints without generation-cut controls")
        if self.extra_body:
            raise ValueError("external_model does not support extra_body overrides")
        if any(key.lower() == "authorization" for key in self.default_headers):
            raise ValueError("external_model credentials must come from api_key_file")
        return self


def _bad_request(url: str, content: bytes) -> ClientResponseError:
    info = RequestInfo(URL(url), "POST", CIMultiDictProxy(CIMultiDict()), URL(url))
    error = ClientResponseError(info, (), status=400, message="Bad Request")
    error.response_content = content
    return error


def served_cache_salt(path: str) -> str:
    """Read the confirmed checkpoint identity; refuse missing or empty records.

    Uses cwtrain's nvext.as_cache_salt convention (Apache-2.0):
    https://github.com/coreweave/inference-cookbook/blob/41f991fef0d45430d20cfd8bc142c4bfc5070369/libs/cwtrain/cwtrain/integrations/nvext.py
    """
    try:
        identity = Path(path).read_text().strip()
    except OSError as error:
        raise RuntimeError("Cannot read served checkpoint identity; sampling must remain paused") from error
    if not identity:
        raise RuntimeError("Served checkpoint identity is empty; sampling must remain paused")
    if len(identity) > 128 or set("@/\\\0").intersection(identity):
        return hashlib.sha256(identity.encode()).hexdigest()
    return identity


class ExternalModelClient(NeMoGymAsyncOpenAI):
    """Use Gym's aiohttp transport with a locally rendered token-ID prompt."""

    tokenizer: object
    max_model_len: int
    cache_salt_file: str
    api_key_file: str | None = None
    seeds: bool = True

    async def create_chat_completion(self, *, _resend_after_disconnect: bool = True, **kwargs: object) -> dict:
        """Read checkpoint identity and credentials immediately before external generation."""
        url = self.base_url.rstrip("/") + "/completions"
        salt = await asyncio.to_thread(served_cache_salt, self.cache_salt_file)
        try:
            body = await asyncio.to_thread(
                completion_request,
                kwargs,
                cast(Tokenizer, self.tokenizer),
                max_model_len=self.max_model_len,
                cache_salt=salt,
                seed=self.seeds,
            )
        except ContextFull:
            raise _bad_request(url, CONTEXT_FULL) from None
        # Resolve the key into a short-lived client, leaving server config and pooled clients clean.
        key = (await asyncio.to_thread(Path(self.api_key_file).read_text)).strip() if self.api_key_file else "unused"
        if not key:
            raise RuntimeError("api_key_file must hold a nonempty inference key")
        transport = self.model_copy(update={"api_key": key})
        request = {"url": url, "json": body, "_resend_after_disconnect": _resend_after_disconnect}
        response = await transport._request(method="POST", **request)
        try:
            await transport._raise_for_status(response, request)
        except ClientResponseError as error:
            if is_context_overflow(error.status, error.response_content.decode(errors="replace")):
                error.response_content = CONTEXT_FULL
            raise
        result = await transport._read_response_json(response)
        # Catch a checkpoint publication racing a request; never train on an unconfirmed version.
        if await asyncio.to_thread(served_cache_salt, self.cache_salt_file) != salt:
            raise RuntimeError("Served checkpoint changed during generation; discard the rollout")
        return chat_response(result, body["prompt"], kwargs.get("tools"))


class ExternalModel(VLLMModel):
    """Serve Gym Responses and Chat requests through an external token completions endpoint."""

    config: ExternalModelConfig

    def _post_init(self) -> None:
        super()._post_init()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, local_files_only=True)
        self._clients = [
            ExternalModelClient(
                **client.model_dump(),
                tokenizer=tokenizer,
                max_model_len=self.config.max_model_len,
                cache_salt_file=self.config.cache_salt_file,
                api_key_file=self.config.api_key_file,
                seeds=self.config.seeds,
            )
            for client in self._clients
        ]
        if self._external_capture_handler is not None:
            raise ValueError("external_model uses inline training metadata; external staging is unsupported")


if __name__ == "__main__":
    ExternalModel.run_webserver()
