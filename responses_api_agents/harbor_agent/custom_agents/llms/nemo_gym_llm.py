import asyncio
from pathlib import Path
from typing import Any, Literal

import requests

from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.models.metric import UsageInfo


class NemoGymLLM(BaseLLM):
    """LLM backend that calls NeMo Gym model servers via chat completions."""

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "placeholder",
        temperature: float = 1.0,
        collect_rollout_details: bool = False,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "default"]
        | None = None,
        model_info: dict[str, Any] | None = None,
        timeout_sec: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._temperature = temperature
        self._collect_rollout_details = collect_rollout_details
        self._reasoning_effort = reasoning_effort
        self._model_info = model_info or {}
        self._timeout_sec = timeout_sec

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        **kwargs: Any,
    ) -> LLMResponse:
        messages = message_history + [{"role": "user", "content": prompt}]

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = self._reasoning_effort
        if self._collect_rollout_details:
            payload["logprobs"] = True

        payload.update(self._to_jsonable(kwargs))

        try:
            response_dict = await asyncio.to_thread(self._post_chat_completions, payload)
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.text if e.response is not None else ""
            except Exception:
                body = ""

            combined = f"{str(e)} {body}".lower()
            if "context length" in combined or "context_length_exceeded" in combined:
                raise ContextLengthExceededError from e
            raise

        choices = self._response_get(response_dict, "choices", [])
        choice = choices[0] if choices else {}
        message = self._response_get(choice, "message", {}) if choice else {}
        content = self._response_get(message, "content", "") or ""
        reasoning_content = self._response_get(message, "reasoning_content", None)

        if self._response_get(choice, "finish_reason") == "length":
            raise OutputLengthExceededError(
                f"Model {self._model_name} hit max_tokens limit. "
                "Response was truncated. Consider increasing max_tokens if possible.",
                truncated_response=content,
            )

        usage = self._extract_usage_info(response_dict)
        prompt_token_ids = None
        completion_token_ids = None
        logprobs = None
        if self._collect_rollout_details:
            prompt_token_ids, completion_token_ids = self._extract_token_ids(response_dict)
            logprobs = self._extract_logprobs(response_dict)

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            usage=usage,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            logprobs=logprobs,
        )

    def get_model_context_limit(self) -> int:
        max_input_tokens = self._model_info.get("max_input_tokens")
        if isinstance(max_input_tokens, int) and max_input_tokens > 0:
            return max_input_tokens

        max_tokens = self._model_info.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens

        return 1000000

    def get_model_output_limit(self) -> int | None:
        max_output_tokens = self._model_info.get("max_output_tokens")
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            return max_output_tokens
        return None

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self._chat_completions_endpoint()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        return response.json()

    def _chat_completions_endpoint(self) -> str:
        """Build a chat completions endpoint that tolerates base URLs with/without /v1."""
        if self._api_base.endswith("/v1"):
            return f"{self._api_base}/chat/completions"
        return f"{self._api_base}/v1/chat/completions"

    def _extract_token_ids(self, response: dict[str, Any]) -> tuple[list[int] | None, list[int] | None]:
        choices = self._response_get(response, "choices", [])
        choice = choices[0] if choices else {}
        message = self._response_get(choice, "message", {}) if choice else {}

        prompt_token_ids = self._response_get(response, "prompt_token_ids", None)
        if prompt_token_ids is None and isinstance(message, dict):
            prompt_token_ids = message.get("prompt_token_ids")

        completion_token_ids = None
        provider_specific_fields = self._response_get(choice, "provider_specific_fields", {})
        if isinstance(provider_specific_fields, dict):
            completion_token_ids = provider_specific_fields.get("token_ids")
        if completion_token_ids is None and isinstance(choice, dict):
            completion_token_ids = choice.get("token_ids")
        if completion_token_ids is None and isinstance(message, dict):
            completion_token_ids = message.get("generation_token_ids")

        return (
            self._normalize_token_ids(prompt_token_ids),
            self._normalize_token_ids(completion_token_ids),
        )

    def _extract_logprobs(self, response: dict[str, Any]) -> list[float] | None:
        choices = self._response_get(response, "choices", [])
        if not choices:
            return None

        choice = choices[0]
        logprobs_data = self._response_get(choice, "logprobs")
        if isinstance(logprobs_data, dict):
            content = logprobs_data.get("content", [])
            extracted = [
                token_data["logprob"]
                for token_data in content
                if isinstance(token_data, dict) and "logprob" in token_data
            ]
            if extracted:
                return extracted

        message = self._response_get(choice, "message", {})
        if isinstance(message, dict):
            generation_log_probs = message.get("generation_log_probs")
            if isinstance(generation_log_probs, list):
                return [
                    float(lp) for lp in generation_log_probs if isinstance(lp, (int, float))
                ] or None

        return None

    def _extract_usage_info(self, response: dict[str, Any]) -> UsageInfo | None:
        usage = self._response_get(response, "usage")
        if not isinstance(usage, dict):
            return None

        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        cache_tokens = (
            prompt_tokens_details.get("cached_tokens", 0)
            if isinstance(prompt_tokens_details, dict)
            else 0
        ) or 0

        return UsageInfo(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            cache_tokens=int(cache_tokens),
            cost_usd=0.0,
        )

    def _response_get(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _normalize_token_ids(self, token_ids: Any) -> list[int] | None:
        if not isinstance(token_ids, list):
            return None

        normalized: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, int):
                normalized.append(token_id)
                continue
            if isinstance(token_id, str):
                stripped = token_id.removeprefix("token_id:")
                if stripped.isdigit():
                    normalized.append(int(stripped))
                    continue
            return None

        return normalized or None

    def _to_jsonable(self, value: Any) -> Any:
        """Recursively convert values into JSON-serializable structures."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(v) for v in value]
        return value
