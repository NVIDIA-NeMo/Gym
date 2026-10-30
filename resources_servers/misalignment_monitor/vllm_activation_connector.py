# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MisalignmentActivationConnector._submit_async_write adapts ExampleHiddenStatesConnector._submit_async_write from
# vLLM v0.29.0 (Apache-2.0, Copyright contributors to the vLLM project):
# https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
"""A vLLM plugin that scores the serving model's own activations for the misalignment monitor's layer 0.

It extends vLLM's ``ExampleHiddenStatesConnector``. When a request finishes, it scores the residual stream at one
layer with a Jacobian lens or a linear probe and writes a small JSON result, whose path vLLM returns in
``kv_transfer_params``. README.md lists the serve flags. It imports nothing from Gym, so vLLM can load it alone.
"""

import fcntl
import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    PendingSave,
    extract_from_kv_cache,
)
from vllm.logger import init_logger

from resources_servers.misalignment_monitor.activation_readout import (
    concept_token_ids,
    jspace_hits,
    jspace_reading,
    probe_score,
)


logger = init_logger(__name__)


@dataclass
class ScoredSave(PendingSave):
    # The probe reads the last prompt token: the model's state just before it acts on this turn.
    num_prompt_tokens: int = 0


class MisalignmentActivationConnector(ExampleHiddenStatesConnector):
    """Scores each finished request's activations and writes the result where vLLM would write them."""

    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.get_from_extra_config
        self._mode = extra("mode", "jspace")
        if self._mode not in ("jspace", "probe"):
            raise ValueError(f"mode must be 'jspace' or 'probe', got {self._mode!r}")
        layers = self._vllm_config.speculative_config.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids
        if len(layers) != 1:
            raise ValueError(f"Configure exactly one eagle_aux_hidden_state_layer_ids entry, got {layers}")
        self._layer = int(layers[0])
        # Long prompts are mostly context already read on earlier turns, so only their end is scored.
        self._max_positions = int(extra("max_positions", 4096))
        self._readout: dict[str, Any] | None = None

    # ==============================
    # Scheduler side
    # ==============================

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        meta = super().build_connector_meta(scheduler_output)
        # The results are JSON, so they get a .json name, and the lock file follows it.
        for req_id, filename in meta.new_req_filenames.items():
            json_name = filename.removesuffix(".safetensors") + ".json"
            meta.new_req_filenames[req_id] = self._request_filenames[req_id] = json_name
        return meta

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        delay, params = super().request_finished(request, block_ids)
        pending = self._pending_saves[request.request_id]
        # Generated tokens are the model acting, so they are always read. The last one was never an input.
        self._pending_saves[request.request_id] = ScoredSave(
            req_id=pending.req_id,
            filename=pending.filename,
            token_ids=torch.tensor(list(request.all_token_ids)[:-1]),
            block_ids=pending.block_ids,
            num_prompt_tokens=request.num_prompt_tokens,
        )
        return delay, {"misalignment_activations_path": params["hidden_states_path"]}

    # ==============================
    # Worker side
    # ==============================

    def _build_readout(self) -> dict[str, Any]:
        import transformers

        extra = self._kv_transfer_config.get_from_extra_config
        model_config = self._vllm_config.model_config
        device = self._kv_cache.device
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_config.tokenizer, trust_remote_code=model_config.trust_remote_code
        )
        readout: dict[str, Any] = {"tokenizer": tokenizer}
        if self._mode == "probe":
            from safetensors.torch import load_file

            probe_dir = Path(extra("probe_path"))
            probe_config = json.loads((probe_dir / "config.json").read_text())
            if int(probe_config["layer"]) != self._layer:
                raise ValueError(f"The probe reads layer {probe_config['layer']}, but vLLM extracts {self._layer}")
            readout["probe"] = {
                k: v.float().to(device) for k, v in load_file(str(probe_dir / "probe.safetensors")).items()
            }
            readout["temperature"] = float(probe_config.get("temperature", 1.0))
            return readout

        jacobians = torch.load(extra("jlens_path"), map_location="cpu", weights_only=True)["J"]
        # The lens indexes residual blocks, and aux layer l is the output of block l - 1.
        readout["jacobian"] = jacobians[self._layer - 1].float().to(device)
        names = [extra("unembed_tensor", "lm_head.weight"), extra("final_norm_tensor", "model.norm.weight")]
        weights = _checkpoint_tensors(model_config.model, [*names, "model.embed_tokens.weight"])
        # Models with tied embeddings have no separate lm_head.
        readout["unembed"] = weights.get(names[0], weights.get("model.embed_tokens.weight")).to(device)
        readout["norm"] = weights[names[1]].float().to(device)
        readout["eps"] = float(getattr(model_config.hf_text_config, "rms_norm_eps", 1e-6))
        readout["concepts"] = concept_token_ids(tokenizer, list(extra("concepts")))
        readout["concept_ids"] = torch.tensor(sorted(readout["concepts"]), device=device)
        readout["top_k"] = int(extra("top_k", 10))
        return readout

    def _lens_logits(self, residual: torch.Tensor) -> torch.Tensor:
        readout = self._readout
        transported = residual @ readout["jacobian"].T
        # The model's final RMSNorm, then its unembedding.
        normed = transported * torch.rsqrt(transported.pow(2).mean(-1, keepdim=True) + readout["eps"])
        return (normed * readout["norm"]).to(readout["unembed"].dtype) @ readout["unembed"].T

    def _submit_async_write(self, pending: PendingSave) -> None:
        """Copy the request's activations out of the cache, then score them in a writer thread."""
        if not self._is_tp_rank_zero:
            return
        lock_fd = self._lock_fds.pop(pending.req_id, None)
        if lock_fd is None and self.use_lock:
            lock_fd = os.open(pending.filename + ".lock", os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        os.makedirs(os.path.dirname(pending.filename), exist_ok=True)
        residual, first, copied = None, 0, None
        try:
            if self._readout is None:
                self._readout = self._build_readout()
            num_tokens = pending.token_ids.shape[0]
            first = max(0, num_tokens - self._max_positions)
            blocks = torch.tensor(pending.block_ids, dtype=torch.long)
            slots = (
                torch.arange(self._block_size).reshape(1, -1) + blocks.reshape(-1, 1) * self._block_size
            ).flatten()
            stream = self._get_copy_stream()
            ready = torch.cuda.Event()
            ready.record()
            stream.wait_event(ready)
            with torch.cuda.stream(stream):
                window = slots[first:num_tokens].to(self._kv_cache.device, non_blocking=True)
                # Indexing copies, so the cache blocks can be freed once ``copied`` completes.
                residual = extract_from_kv_cache(self._kv_cache, window, num_tokens - first)[:, 0, :]
            copied = torch.cuda.Event()
            copied.record(stream)
            self._req_copy_events[pending.req_id] = copied
        except Exception:
            # A scoring failure must not stop serving. The writer still answers whoever waits on the lock.
            logger.exception("Misalignment activation extraction failed for %s", pending.req_id)
        future = self._executor.submit(self._write_result, pending, residual, first, copied, lock_fd)
        self._req_futures[pending.req_id] = future
        future.add_done_callback(partial(self._on_write_done, pending.req_id))

    def _write_result(
        self,
        pending: PendingSave,
        residual: torch.Tensor | None,
        first: int,
        copied: torch.cuda.Event | None,
        lock_fd: int | None,
    ) -> None:
        try:
            try:
                if residual is None:
                    raise RuntimeError("the activations could not be extracted, see the vLLM log")
                copied.synchronize()
                with torch.cuda.stream(self._get_copy_stream()):
                    reading = self._read(pending, residual.float(), first)
            except Exception as e:
                logger.exception("Misalignment activation scoring failed for %s", pending.req_id)
                reading = {"mode": self._mode, "flagged": False, "error": f"{type(e).__name__}: {e}"}
            reading |= {"layer": self._layer, "tokens": int(pending.token_ids.shape[0])}
            with open(pending.filename, "w") as f:
                json.dump(reading, f)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)

    def _read(self, pending: PendingSave, residual: torch.Tensor, first: int) -> dict[str, Any]:
        readout = self._readout
        if self._mode == "probe":
            position = getattr(pending, "num_prompt_tokens", pending.token_ids.shape[0]) - 1
            if position < first:
                return {"mode": "probe", "flagged": False, "position": position}
            score = probe_score(residual[position - first], readout["probe"], readout["temperature"])
            threshold = float(self._kv_transfer_config.get_from_extra_config("probe_threshold", 0.5))
            return {"mode": "probe", "score": score, "flagged": score > threshold, "position": position}
        hits = jspace_hits(
            residual, self._lens_logits, readout["concept_ids"], top_k=readout["top_k"], first_position=first
        )
        token_ids = pending.token_ids.tolist()
        return jspace_reading(
            hits,
            readout["concepts"],
            lambda position: readout["tokenizer"].decode(token_ids[max(0, position - 16) : position + 1]),
        )


def _checkpoint_tensors(model: str, names: list[str]) -> dict[str, torch.Tensor]:
    """Read the named tensors from a local or already-downloaded safetensors checkpoint."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(model) if Path(model).is_dir() else Path(snapshot_download(model, local_files_only=True))
    index = root / "model.safetensors.index.json"
    shard_of = json.loads(index.read_text())["weight_map"] if index.exists() else {}
    tensors = {}
    for name in names:
        shard = shard_of.get(name, "model.safetensors")
        if not (root / shard).exists():
            continue
        with safe_open(root / shard, framework="pt") as f:
            if name in f.keys():
                tensors[name] = f.get_tensor(name)
    return tensors
