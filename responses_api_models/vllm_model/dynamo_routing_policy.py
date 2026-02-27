# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

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
        overlap_score_weight: float = 1.0,
        router_temperature: float = 0.0,
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
        """Convert a chat completion request body to token IDs."""
        messages = request_body.get("messages", [])
        try:
            token_ids = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: concatenate text content and encode directly
            text = " ".join(
                m.get("content", "") for m in messages
                if isinstance(m.get("content"), str)
            )
            token_ids = self._tokenizer.encode(text)
        return token_ids

    def select_client(
        self,
        *,
        request_body: Dict[str, Any],
        request_id: str,
        session_id: Optional[str] = None,
    ) -> int:
        token_ids = self._tokenize_request(request_body)

        worker_id, _dp_rank, _overlap = self._run_async(
            self._router.best_worker, token_ids, request_id=request_id,
        )

        # Dynamo worker IDs are 1-based; client indices are 0-based.
        client_idx = worker_id - 1
        return client_idx

    def on_prefill_complete(self, request_id: str) -> None:
        self._run_async(self._router.mark_prefill_complete, request_id)

    def on_generation_complete(self, request_id: str) -> None:
        self._run_async(self._router.free, request_id)

    def on_weights_updated(self) -> None:
        from dynamo._core import KvRouter, KvRouterConfig

        config = KvRouterConfig(**self._kv_config_kwargs)
        self._router = KvRouter.standalone(
            self._block_size, self._num_clients, config,
        )
        logger.info("DynamoKvRoutingPolicy: radix tree reset after weight update")
