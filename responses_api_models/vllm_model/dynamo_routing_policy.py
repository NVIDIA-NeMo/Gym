# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import threading
import traceback
from typing import Any, Dict, List, Optional
import time

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from responses_api_models.vllm_model.routing_policy import RoutingPolicy

logger = logging.getLogger(__name__)


class DynamoKvRoutingPolicy(RoutingPolicy):
    """Cache-aware routing policy backed by Dynamo's standalone KvRouter.

    Constructor kwargs:
        model_name: HuggingFace model name.
        block_size: KV cache block size.
        overlap_score_weight: Weight for KV overlap in scoring.
        router_temperature: Softmax temperature.
        router_max_tree_size: Max radix tree nodes before pruning.
        router_ttl_secs: TTL for entries in approximate mode.
        router_prune_target_ratio: Prune target ratio.
    """

    def __init__(
        self,
        clients: List[NeMoGymAsyncOpenAI],
        model_name: str,
        block_size: int = 16,
        overlap_score_weight: float = 5.0,
        router_temperature: float = 0,
        router_max_tree_size: int = 1048576,
        router_ttl_secs: float = 120.0,
        router_prune_target_ratio: float = 0.8,
        **kwargs: Any,
    ) -> None:
        from dynamo._core import KvRouter, KvRouterConfig
        from transformers import AutoTokenizer

        self._num_clients = len(clients)
        self._block_size = block_size
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._kv_config_kwargs = dict(
            use_kv_events=False,
            overlap_score_weight=overlap_score_weight,
            router_temperature=router_temperature,
            router_max_tree_size=router_max_tree_size,
            router_ttl_secs=router_ttl_secs,
            router_prune_target_ratio=router_prune_target_ratio,
        )

        config = KvRouterConfig(**self._kv_config_kwargs)
        self._router = KvRouter.standalone(block_size, self._num_clients, config)
        self.request_count = 0

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="dynamo-kv-gym"
        )
        self._loop_thread.start()

        logger.info(
            "DynamoKvRoutingPolicy initialized: %d clients, block_size=%d, model=%s",
            self._num_clients, block_size, model_name,
        )

    def _run_async(self, coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
        async def _wrap():
            return await coro_fn(*args, **kwargs)

        future = asyncio.run_coroutine_threadsafe(_wrap(), self._loop)
        return future.result()

    def _tokenize_request(self, request_body: Dict[str, Any]) -> list[int]:
        """Convert a chat completion request body to token IDs.

        Expects a trimmed request_body (via _trim_request_body) containing
        only ``messages`` (with chat-only keys) and optionally ``tools``.
        """
        messages = request_body.get("messages", [])
        tools = request_body.get("tools")

        for m in messages:
            if m.get("content") is None:
                m["content"] = ""

        try:
            token_ids = self._tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, add_generation_prompt=True,
            )
        except Exception as e:
            text = " ".join(
                m.get("content", "") for m in messages
                if isinstance(m.get("content"), str)
            )
            token_ids = self._tokenizer.encode(text)
            print(f"Encountered error applying the chat template: {e}")
            traceback.print_exc()
            print(f"MESSAGES: {messages}")
            print(f"TOOLS: {tools}")
        return token_ids

    _CHAT_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}

    @classmethod
    def _clean_tool_calls(cls, tool_calls: list) -> list:
        """Parse tool_call arguments from JSON strings to dicts.

        The chat template iterates arguments with |items, so they must be dicts.
        """
        cleaned = []
        for tc in tool_calls:
            tc = dict(tc)
            if "function" in tc:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                tc["function"] = fn
            elif isinstance(tc.get("arguments"), str):
                tc["arguments"] = json.loads(tc["arguments"])
            cleaned.append(tc)
        return cleaned

    @classmethod
    def _trim_request_body(cls, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only the fields needed for tokenization: messages and tools."""
        trimmed: Dict[str, Any] = {}
        if "messages" in request_body:
            clean_messages = []
            for m in request_body["messages"]:
                msg = {k: v for k, v in m.items() if k in cls._CHAT_KEYS}
                if "tool_calls" in msg and msg["tool_calls"]:
                    msg["tool_calls"] = cls._clean_tool_calls(msg["tool_calls"])
                clean_messages.append(msg)
            trimmed["messages"] = clean_messages
        if "tools" in request_body:
            trimmed["tools"] = request_body["tools"]
        return trimmed

    async def select_client(
        self,
        *,
        request_body: Dict[str, Any],
        request_id: str,
        session_id: Optional[str] = None,
    ) -> int:
        start_preprocess = time.perf_counter()
        request_body = self._trim_request_body(request_body)
        token_ids = await asyncio.to_thread(self._tokenize_request, request_body)
        end_preprocess = time.perf_counter()
        print(f"Took {end_preprocess-start_preprocess}s to preprocess request")

        temp_config_dict = None
        if self.request_count < self._num_clients:
            temp_config_dict = {"overlap_score_weight": 0.0}
            self.request_count += 1

        start_routing = time.perf_counter()
        worker_id, _dp_rank, _overlap = self._run_async(
            self._router.best_worker, token_ids, request_id=request_id, router_config_override=temp_config_dict
        )
        end_routing = time.perf_counter()
        print(f"Took {end_routing-start_routing}s to route request")

        # Dynamo worker IDs are 1-based; client indices are 0-based.
        client_idx = worker_id - 1
        print(f"Selected {client_idx} for request {request_id}")
        return client_idx

    def on_prefill_complete(self, request_id: str) -> None:
        print(f"Prefill complete for request {request_id}")
        self._run_async(self._router.mark_prefill_complete, request_id)

    def on_generation_complete(self, request_id: str) -> None:
        print(f"Generation complete for request {request_id}")
        self._run_async(self._router.free, request_id)

    def on_weights_updated(self) -> None:
        from dynamo._core import KvRouter, KvRouterConfig

        config = KvRouterConfig(**self._kv_config_kwargs)
        self._router = KvRouter.standalone(
            self._block_size, self._num_clients, config,
        )
        self.request_count = 0
        print("DynamoKvRoutingPolicy: radix tree reset after weight update")

