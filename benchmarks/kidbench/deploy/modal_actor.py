# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve the KIDBench multi-turn child actor on Modal.

The actor is the model that plays a child probing the system under test. KIDBench's own
actor is Gemma-4-31B-it with its refusal direction ablated, and upstream deliberately does
not release that checkpoint — a refusal-ablated model is itself a safety-sensitive
artifact. Running a stock instruct model in its place makes the actor refuse to stay in
character on exactly the prompts that matter most, which flatters every model under test.

This deploys a published third-party ablation of the same base model so the actor behaves
closer to the paper's. It is **not** the paper's checkpoint: see
``ACTOR_MODEL_PROVENANCE`` in ``benchmarks/kidbench/upstream_spec.py`` for what differs
and what that costs.

Deploy::

    modal deploy benchmarks/kidbench/deploy/modal_actor.py

The served endpoint is OpenAI-compatible at ``<url>/v1`` and sits behind Modal proxy auth,
so requests need a workspace proxy token as the bearer credential.
"""

from __future__ import annotations

import modal


APP_NAME = "kidbench-gemma4-31b-ara-actor"

#: The actor checkpoint, pinned to an exact commit. A floating `main` would let the
#: served weights change under a rerun, which would silently make two runs incomparable —
#: the actor is the instrument, not a detail.
MODEL_ID = "wangzhang/gemma-4-31B-it-abliterated"
MODEL_REVISION = "d7431183c443443024debb08b222cbcf2e424da7"

#: The ablated repo ships the weights, config and tokenizer but omits
#: ``processor_config.json``. Its config still declares Gemma 4's vision tower — and the
#: tower's weights really are in the checkpoint — so vLLM looks for a feature extractor,
#: does not find one, and the engine dies before it serves anything. The file is taken
#: from the base model it was ablated from, pinned to its own commit, and dropped into the
#: snapshot. Nothing about the weights changes.
BASE_MODEL_ID = "google/gemma-4-31B-it"
BASE_MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
PROCESSOR_FILE = "processor_config.json"

MODEL_DIR = "/models/actor"

#: 62.6 GB of bf16 weights. Two H100s leave roughly 49 GB per card for KV cache, which is
#: ample for the actor's short turns; the head counts (32 attention, 16 KV) divide evenly
#: by two, so tensor parallelism needs no padding.
GPU_CONFIG = "H100:2"
TENSOR_PARALLEL_SIZE = 2

#: Upstream gives the actor an 8,192-token output budget, and its prompt carries the full
#: persona plus the whole conversation so far. By turn five that history alone can pass
#: 8k tokens, so anything near 16k leaves no room for the reply and vLLM rejects the
#: request outright. Sized well clear of that rather than trimming upstream's budget.
MAX_MODEL_LEN = 65536

VLLM_PORT = 8000
MINUTES = 60

# A CUDA *devel* image, not a slim one: vLLM's torch.compile path shells out to nvcc, and
# on a runtime-only image the engine gets all the way through loading 62.6 GB of weights
# before dying on "Could not find nvcc".
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    # Pinned: vLLM's Gemma 4 support and its attention-backend defaults both move between
    # releases, and the actor's behaviour is part of the measurement. flashinfer is left
    # out deliberately — its build pulls a yanked apache-tvm-ffi pin, and vLLM's default
    # backend is fine for this workload.
    .pip_install(
        "vllm==0.29.0",
        "hf_xet",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_V1": "1"})
)

hf_cache = modal.Volume.from_name("kidbench-actor-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("kidbench-actor-vllm-cache", create_if_missing=True)
model_vol = modal.Volume.from_name("kidbench-actor-model", create_if_missing=True)

app = modal.App(APP_NAME)


def _materialize_model() -> str:
    """Assemble a complete model directory and return its path.

    Idempotent: a snapshot already on the volume is reused, so only the first container
    pays the 62.6 GB download.
    """
    import json
    import os

    from huggingface_hub import hf_hub_download, snapshot_download

    marker = os.path.join(MODEL_DIR, ".complete")
    if os.path.exists(marker):
        return MODEL_DIR

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        # The eval artefacts on that repo are documentation, not weights.
        ignore_patterns=["eval/*", "*.md"],
    )

    processor = hf_hub_download(
        repo_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        filename=PROCESSOR_FILE,
    )
    with open(processor, encoding="utf-8") as source:
        config = json.load(source)
    with open(os.path.join(MODEL_DIR, PROCESSOR_FILE), "w", encoding="utf-8") as target:
        json.dump(config, target, indent=2)

    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(f"{MODEL_ID}@{MODEL_REVISION}\n{BASE_MODEL_ID}@{BASE_MODEL_REVISION}:{PROCESSOR_FILE}\n")
    model_vol.commit()
    return MODEL_DIR


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    # Weights alone take several minutes to load; a cold start that times out mid-load
    # leaves the run retrying against a server that was always going to come up.
    scaledown_window=20 * MINUTES,
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/models": model_vol,
    },
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES, requires_proxy_auth=True)
def serve() -> None:
    import subprocess

    model_path = _materialize_model()

    subprocess.Popen(
        [
            "vllm",
            "serve",
            model_path,
            # Clients address it by the Hub id, not the container path, so the model name
            # a rollout records stays meaningful outside this deployment.
            "--served-model-name",
            MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--tensor-parallel-size",
            str(TENSOR_PARALLEL_SIZE),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--dtype",
            "bfloat16",
            # The checkpoint carries Gemma 4's vision tower. The actor is text-only, so
            # refusing image inputs keeps the multimodal preprocessor off the hot path
            # and stops it reserving KV cache for image tokens that never arrive.
            "--limit-mm-per-prompt",
            '{"image": 0, "audio": 0}',
        ]
    )
