# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model server with exact-prefix context-compaction support."""

import logging
from typing import Any, ClassVar, Dict, List, Optional

from fastapi import Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_streaming import (
    sanitize_streaming_responses_body,
    synthesize_responses_failure_sse,
    synthesize_responses_sse,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


LOG = logging.getLogger("nemo_gym.vllm_model_with_compaction")


class VLLMContextCompactionResponseCreateParams(NeMoGymResponseCreateParamsNonStreaming):
    """Responses request carrying vLLM's exact physical-prefix control."""

    required_prefix_token_ids: Optional[List[int]] = None


class VLLMContextCompactionChatCompletionCreateParams(NeMoGymChatCompletionCreateParamsNonStreaming):
    """Chat Completions counterpart of the exact-prefix extension."""

    required_prefix_token_ids: Optional[List[int]] = None


def _validate_context_compaction_params(body: Dict[str, Any]) -> VLLMContextCompactionResponseCreateParams:
    """Validate the dedicated server's extended Responses request."""

    try:
        return VLLMContextCompactionResponseCreateParams.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]) from exc


class VLLMModelWithCompaction(VLLMModel):
    """Dedicated vLLM adapter for context-compacted generation."""

    non_generating_model_routes: ClassVar[frozenset[tuple[str, str]]] = frozenset({("POST", "/tokenize")})
    _TOKENIZE_CHAT_FIELDS = (*VLLMModel._TOKENIZE_CHAT_FIELDS, "required_prefix_token_ids")

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/tokenize")(self.tokenize)
        return app

    @classmethod
    def _get_tokenize_chat_body(cls, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Reject post-generation retokenization in this exact-token adapter."""

        raise RuntimeError(
            "The context-compaction model requested generation-consumed prompt and sampled token IDs "
            "from vLLM (return_token_ids=True), but the response contained neither prompt_token_ids nor "
            "choice.token_ids. Refusing to reconstruct on-policy training evidence with a separate "
            "/tokenize request."
        )

    @classmethod
    def _get_context_guard_tokenize_body(cls, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build the pre-generation context-guard request using the shared field selection."""

        return super()._get_tokenize_chat_body(body_dict)

    async def responses_dispatch(self, request: Request, body: dict = Body()):
        """Serve the compaction extension on the standard `/v1/responses` path."""

        if not body.get("stream"):
            params = _validate_context_compaction_params(body)
            return await self._invoke_responses(request, params)

        cleaned, namespace_map = sanitize_streaming_responses_body(body)
        params = _validate_context_compaction_params(cleaned)
        try:
            response = await self._invoke_responses(request, params)
            response_json = response.model_dump(mode="json") if isinstance(response, BaseModel) else dict(response)
        except Exception as exc:
            LOG.exception("responses() failed while serving a streaming /v1/responses request")
            return StreamingResponse(
                synthesize_responses_failure_sse(str(exc)),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            synthesize_responses_sse(response_json, namespace_map),
            media_type="text/event-stream",
        )

    async def tokenize(
        self,
        request: Request,
        body: VLLMContextCompactionResponseCreateParams = Body(),
    ) -> Dict[str, List[int]]:
        """Tokenize an admitted request for context-limit guard evaluation."""

        _, chat_params = self._context_compaction_chat_params(body)
        body_dict = chat_params.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)
        tokenize_body = self._get_context_guard_tokenize_body(body_dict)
        result = await self._resolve_client(request).create_tokenize(**tokenize_body)
        return {
            "tokens": self._require_token_id_list(
                result.get("tokens"),
                f"{self.config.name}.tokenize.tokens",
            )
        }

    def _context_compaction_chat_params(
        self,
        body: VLLMContextCompactionResponseCreateParams,
    ) -> tuple[NeMoGymResponseCreateParamsNonStreaming, VLLMContextCompactionChatCompletionCreateParams]:
        """Convert semantic input while retaining vLLM's exact-prefix control."""

        required_prefix_token_ids = body.required_prefix_token_ids
        standard_body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            body.model_dump(exclude={"required_prefix_token_ids"})
        )
        standard_chat_params = self._converter.responses_to_chat_completion_create_params(standard_body)
        chat_params = VLLMContextCompactionChatCompletionCreateParams.model_validate(
            standard_chat_params.model_dump(exclude_unset=True)
            | {"required_prefix_token_ids": required_prefix_token_ids}
        )
        return standard_body, chat_params

    async def responses(
        self,
        request: Request,
        body: VLLMContextCompactionResponseCreateParams = Body(),
    ) -> NeMoGymResponse:
        """Generate from compacted context while preserving an exact token prefix."""

        if self.config.is_responses_native:
            raise NotImplementedError("Context compaction currently requires the vLLM Chat Completions adapter")

        standard_body, chat_params = self._context_compaction_chat_params(body)
        standard_body.model = self.config.model
        chat_completion_response = await self.chat_completions(request, chat_params)
        return self._converter.chat_completion_to_response(
            responses_create_params=standard_body,
            chat_completion=chat_completion_response,
        )


if __name__ == "__main__":
    VLLMModelWithCompaction.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModelWithCompaction.run_webserver()  # noqa: F401
