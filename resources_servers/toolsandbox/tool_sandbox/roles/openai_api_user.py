# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# For licensing see accompanying LICENSE file.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""User simulator role against any OpenAI-compatible chat endpoint."""
from __future__ import annotations

import logging
from logging import getLogger
from typing import Dict, Iterable, List, Literal, Optional, Union, cast

from openai import NOT_GIVEN, AsyncOpenAI, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tool
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.openai_api import (
    OpenAIRoleConfig,
    _sampling_kwargs,
    openai_retry,
)

LOGGER = getLogger(__name__)


class OpenAIAPIUser(BaseRole):
    """Simulated user driven by any OpenAI-compatible chat endpoint."""

    role_type: RoleType = RoleType.USER

    def __init__(self, config: OpenAIRoleConfig) -> None:
        self._config = config
        self.model_name = config.model
        self.openai_client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "not-used",
        )
        self._sampling = _sampling_kwargs(config)

    async def teardown(self) -> None:
        await self.openai_client.close()

    async def respond(self, ending_index: Optional[int] = None) -> None:
        """Reply to the agent, or terminate the conversation via ``end_conversation``."""
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return

        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven]
        if messages[-1].sender == RoleType.AGENT:
            openai_tools = cast(
                Iterable[ChatCompletionToolParam],
                [convert_to_openai_tool(tool) for tool in available_tools.values()],
            )
        else:
            openai_tools = NOT_GIVEN

        openai_messages = self._to_openai_messages(messages)
        LOGGER.debug("User-sim model input (last msg): %s", openai_messages[-1])
        response = await self._model_inference(openai_messages, openai_tools)
        openai_response_message = response.choices[0].message

        response_messages: List[Message] = []
        if not openai_response_message.tool_calls:
            assert openai_response_message.content is not None
            response_messages.append(
                Message(
                    sender=self.role_type,
                    recipient=RoleType.AGENT,
                    content=openai_response_message.content,
                )
            )
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=None,
                        ),
                    )
                )
        self.add_messages(response_messages)

    @openai_retry
    async def _model_inference(
        self,
        openai_messages: list[dict[Literal["role", "content"], str]],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        with all_logging_disabled(logging.INFO):
            return await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,
                **self._sampling,
            )

    @staticmethod
    def _to_openai_messages(
        messages: List[Message],
    ) -> List[Dict[Literal["role", "content"], str]]:
        """From the user simulator's perspective, agent dialog is the OpenAI "user" role."""
        openai_messages: List[Dict[Literal["role", "content"], str]] = []
        for message in messages:
            if message.sender == RoleType.SYSTEM and message.recipient == RoleType.USER:
                openai_messages.append({"role": "system", "content": message.content})
            elif message.sender == RoleType.AGENT and message.recipient == RoleType.USER:
                openai_messages.append({"role": "user", "content": message.content})
            elif message.sender == RoleType.USER and message.recipient == RoleType.AGENT:
                openai_messages.append({"role": "assistant", "content": message.content})
            elif (
                message.sender == RoleType.USER
                and message.recipient == RoleType.EXECUTION_ENVIRONMENT
            ) or (
                message.sender == RoleType.EXECUTION_ENVIRONMENT
                and message.recipient == RoleType.USER
            ):
                continue
            else:
                raise ValueError(
                    f"Unrecognized sender recipient pair {(message.sender, message.recipient)}"
                )
        return openai_messages
