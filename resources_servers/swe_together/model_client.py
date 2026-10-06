# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auxiliary models use the same asynchronous Gym transport as other servers."""

import asyncio
from types import SimpleNamespace

from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


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
        for attempt in range(3):
            record = {"request": body, "attempt": attempt + 1}
            self.calls.append(record)
            try:
                response = await self.client.post(self.name, "/v1/chat/completions", json=body, cookies=self.cookies)
                await raise_for_status(response)
                data = await get_response_json(response)
                message = data["choices"][0]["message"]
                record["response"] = data
                return SimpleNamespace(content=message.get("content") or "", tool_calls=message.get("tool_calls"))
            except Exception as error:
                record["error"] = str(error)
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)
