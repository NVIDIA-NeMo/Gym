# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip unused streaming argument conversion when parsing complete Inkling output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager


if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


logger = init_logger(__name__)

if vllm.__version__.split("+", 1)[0] == "0.29.0":
    from vllm.tool_parsers.inkling_tool_parser import InklingEngineToolParser

    class InklingCompleteToolParser(InklingEngineToolParser):
        """Keep upstream parsing and streaming behavior, omitting unused complete-response deltas."""

        def extract_tool_calls(
            self, model_output: str, request: ChatCompletionRequest
        ) -> ExtractedToolCallInformation:
            engine = self._parser_engine
            previous = engine._stream_arg_deltas
            engine._stream_arg_deltas = False
            try:
                return super().extract_tool_calls(model_output, request)
            finally:
                engine._stream_arg_deltas = previous

    ToolParserManager.register_module(name="inkling_complete_fast", module=InklingCompleteToolParser)
else:
    # The optimization depends on vLLM 0.29.0 internals; other versions get the stock parser, unmodified.
    logger.warning(
        "vLLM %s is not the validated 0.29.0; inkling_complete_fast uses the stock parser.", vllm.__version__
    )
    ToolParserManager.register_module(
        name="inkling_complete_fast", module=ToolParserManager.get_tool_parser("inkling")
    )
