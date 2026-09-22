# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip unused streaming argument conversion when parsing complete Inkling output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.inkling_tool_parser import InklingEngineToolParser


if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


if vllm.__version__.split("+", 1)[0] != "0.29.0":
    raise RuntimeError(
        "Inkling complete-response plugins require the validated vLLM 0.29.0 parser API. "
        "Use benchmarks/nemotron_3.5_super/vllm_configs/inkling_small.sh for stock parsers."
    )


class InklingCompleteToolParser(InklingEngineToolParser):
    """Keep upstream parsing and streaming behavior, omitting unused complete-response deltas."""

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        engine = self._parser_engine
        previous = engine._stream_arg_deltas
        engine._stream_arg_deltas = False
        try:
            return super().extract_tool_calls(model_output, request)
        finally:
            engine._stream_arg_deltas = previous


ToolParserManager.register_module(name="inkling_complete_fast", module=InklingCompleteToolParser)
