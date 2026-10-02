# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip unused tool argument conversion during complete-response reasoning accounting."""

from collections.abc import Sequence

import vllm
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.inkling_reasoning_parser import InklingParserReasoningAdapter


if vllm.__version__.split("+", 1)[0] != "0.29.0":
    raise RuntimeError(
        "Inkling complete-response plugins require the validated vLLM 0.29.0 parser API. "
        "Use benchmarks/nemotron_3.5_super/vllm_configs/inkling_small.sh for stock parsers."
    )


@ReasoningParserManager.register_module("inkling_count_fast")
class InklingCompleteReasoningCounter(InklingParserReasoningAdapter):
    """Preserve reasoning-token counts without constructing unused tool argument deltas."""

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if self._streaming_count_valid or not token_ids:
            return super().count_reasoning_tokens(token_ids)
        if self._counting_parser_engine is None:
            self._counting_parser_engine = self._parser_engine_cls(self.model_tokenizer, **self._parser_engine_kwargs)
        engine = self._counting_parser_engine
        previous = engine._stream_arg_deltas
        engine._stream_arg_deltas = False
        try:
            return super().count_reasoning_tokens(token_ids)
        finally:
            engine._stream_arg_deltas = previous
