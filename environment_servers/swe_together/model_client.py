# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auxiliary models use the same asynchronous Gym transport as other servers."""

import asyncio
from dataclasses import dataclass
from time import monotonic, time

from aiohttp import ClientResponseError

from nemo_gym.client import ServerClient, get_response_json, raise_for_status


ERROR_BODY_CHARS = 8000


def http_error_fields(error: BaseException | None) -> dict:
    """Return the status and bounded body of the nearest HTTP error in the cause chain of ``error``.

    A wrapper such as ``User simulator request failed`` hides the model server's answer, which carries the
    upstream status and the provider's reason, for example a gateway timeout.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ClientResponseError):
            content = getattr(error, "response_content", b"")
            body = content.decode(errors="replace") if isinstance(content, bytes) else str(content or "")
            return {"status_code": error.status, "body": body[:ERROR_BODY_CHARS]}
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return {}


def error_evidence(error: BaseException) -> dict:
    """Describe a failure with its direct cause and the nearest HTTP status and body behind it."""
    evidence = {"type": type(error).__name__, "message": str(error)[:2000]}
    if error.__cause__ is not None:
        evidence["cause"] = f"{type(error.__cause__).__name__}: {error.__cause__}"[:2000]
    return evidence | http_error_fields(error)


@dataclass(frozen=True)
class ModelMessage:
    content: str
    tool_calls: list[dict] | None = None


class AuxiliaryModel:
    def __init__(self, client: ServerClient, name: str, *, cookies: dict[str, str], temperature: float):
        self.client = client
        self.name = name
        self.cookies = cookies
        self.temperature = temperature
        self.calls = []

    async def call(
        self,
        prompt: str,
        message_history: list | None = None,
        tools: list | None = None,
        tool_choice: str | None = None,
    ):
        body = {
            "model": self.name,
            "messages": (message_history or []) + [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        record = {"request": body, "attempt": 1, "started_at_unix_seconds": time()}
        self.calls.append(record)
        started = monotonic()
        try:
            response = await self.client.post(self.name, "/v1/chat/completions", json=body, cookies=self.cookies)
            await raise_for_status(response)
            data = await get_response_json(response)
            message = data["choices"][0]["message"]
            record["response"] = data
            return ModelMessage(content=message.get("content") or "", tool_calls=message.get("tool_calls"))
        except asyncio.CancelledError:
            record["cancelled"] = True
            raise
        except Exception as error:
            record["error"] = str(error)
            record.update(http_error_fields(error))
            raise
        finally:
            record["elapsed_seconds"] = monotonic() - started
