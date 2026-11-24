# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional vLLM 0.32.0 plugin for the Super 3.5 non-streaming protocol.

The prompt already opens the thinking block. Only the first closing delimiter
separates reasoning from action content. Every character after that boundary,
including repeated or malformed tags, belongs to the model's output.

This independently implemented splitter does not repair actions or delimiters.
It retains the native reasoning-only rstrip presentation rule; retain original
completion token IDs separately when auditing raw-generation integrity.

Load by module name (``--reasoning-parser-plugin super35_reasoning``), with this
directory on PYTHONPATH. vLLM imports plugins more than once; module imports
reuse Python's cache without overwriting an existing parser registration.
"""

from typing import TYPE_CHECKING, NoReturn

from vllm.reasoning import ReasoningParserManager


if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


NativeNemotronV3 = ReasoningParserManager.get_reasoning_parser("nemotron_v3")


class NemotronV3LosslessNonstream(NativeNemotronV3):
    def extract_reasoning(self, model_output: str, request: "ChatCompletionRequest") -> tuple[str | None, str | None]:
        if getattr(request, "stream", False):
            raise ValueError("This frozen benchmark protocol requires stream=False")
        kwargs = getattr(request, "chat_template_kwargs", None) or {}
        if kwargs.get("enable_thinking") is False or kwargs.get("force_nonempty_content"):
            raise ValueError("This frozen benchmark protocol requires thinking enabled")
        self._streaming_count_valid = False
        before, delimiter, after = model_output.partition("</think>")
        return (before.rstrip() or None), (after if delimiter else None)

    def extract_reasoning_streaming(self, *args: object, **kwargs: object) -> NoReturn:
        raise ValueError("This frozen benchmark protocol requires stream=False")


ReasoningParserManager.register_module(
    name="nemotron_v3_lossless_nonstream",
    module=NemotronV3LosslessNonstream,
    force=False,
)
