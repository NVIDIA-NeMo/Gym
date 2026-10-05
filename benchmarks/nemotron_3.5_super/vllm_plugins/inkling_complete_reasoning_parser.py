# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip unused tool argument conversion during complete-response reasoning accounting."""

from collections.abc import Sequence

import vllm
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager


logger = init_logger(__name__)

if vllm.__version__.split("+", 1)[0] == "0.29.0":
    from vllm.reasoning.inkling_reasoning_parser import InklingParserReasoningAdapter

    @ReasoningParserManager.register_module("inkling_count_fast")
    class InklingCompleteReasoningCounter(InklingParserReasoningAdapter):
        """Preserve reasoning-token counts without constructing unused tool argument deltas."""

        def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
            if self._streaming_count_valid or not token_ids:
                return super().count_reasoning_tokens(token_ids)
            if self._counting_parser_engine is None:
                self._counting_parser_engine = self._parser_engine_cls(
                    self.model_tokenizer, **self._parser_engine_kwargs
                )
            engine = self._counting_parser_engine
            previous = engine._stream_arg_deltas
            engine._stream_arg_deltas = False
            try:
                return super().count_reasoning_tokens(token_ids)
            finally:
                engine._stream_arg_deltas = previous
else:
    # The optimization depends on vLLM 0.29.0 internals; other versions get the stock parser, unmodified.
    logger.warning("vLLM %s is not the validated 0.29.0; inkling_count_fast uses the stock parser.", vllm.__version__)
    ReasoningParserManager.register_module(
        name="inkling_count_fast", module=ReasoningParserManager.get_reasoning_parser("inkling")
    )
