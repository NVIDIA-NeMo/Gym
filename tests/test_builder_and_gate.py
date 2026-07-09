# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the builder, the NeMo-RL projection, the inline-token adapter,
and the gate.

The projection tests run a copy of NeMo-RL's contiguity assertion, so the
consumer's requirement is enforced here on the producer side.
"""

from __future__ import annotations

import json

import pytest

from nemo_gym.observability.capture_reader import LocalCaptureReader
from nemo_gym.observability.capture_store import LocalJsonlCaptureStore
from nemo_gym.observability.records import ModelCallRecord, TokenEntry
from nemo_gym.trajectory.builder import (
    assert_nemo_rl_contiguity,
    build_trajectories,
    project_chain_to_response_output,
    project_main_chain_response,
)
from nemo_gym.trajectory.registry import get_builder
from nemo_gym.trajectory.sources import (
    CaptureTokenSource,
    InlineItemsTokenSource,
    classify_auxiliary,
)

from .test_capture_and_strategies import (  # fixtures
    G1,
    G2,
    G3,
    P0,
    RID,
    T1,
    T2,
    branching_rollout,
    linear_rollout,
)


def _seed_store(tmp_path, entries, aux_step=False):
    store = LocalJsonlCaptureStore(str(tmp_path))
    for e in entries:
        store.append(
            ModelCallRecord(
                rollout_id=RID,
                request_id=e.request_id,
                model="policy",
                request={},
                response={
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": f"gen-{e.request_id}", "annotations": []}],
                        }
                    ]
                },
            )
        )
        store.append(
            TokenEntry(
                rollout_id=RID,
                request_id=e.request_id,
                prompt_token_ids=e.prompt_token_ids,
                generation_token_ids=e.generation_token_ids,
                generation_log_probs=e.generation_log_probs,
                model="policy",
            )
        )
    if aux_step:
        store.append(
            ModelCallRecord(
                rollout_id=RID,
                request_id="aux0",
                model="policy",
                request={"messages": [{"role": "system", "content": "You are a title generator"}]},
                response={"output": []},
            )
        )
        store.append(
            TokenEntry(
                rollout_id=RID,
                request_id="aux0",
                prompt_token_ids=[7, 7],
                generation_token_ids=[8],
                generation_log_probs=[-0.2],
                model="policy",
            )
        )
    return store


def test_build_trajectories_end_to_end(tmp_path):
    store = _seed_store(tmp_path, branching_rollout(), aux_step=True)
    reader = LocalCaptureReader(store)
    steps = [r for r in reader.records(RID, kinds={"model_call"}) if isinstance(r, ModelCallRecord)]
    trajs = build_trajectories(
        RID,
        CaptureTokenSource(reader),
        model_call_records=steps,
        builder="prefix_merging",
        reward=1.0,
        is_resolved=True,
        policy_model="policy",
    )
    by_id = {t.chain_id: t for t in trajs}
    assert set(by_id) == {"main", "branch-0"}
    main = by_id["main"]
    assert main.token_ids == P0 + G1 + T1 + G2 + T2 + G3
    assert main.reward == 1.0 and main.provenance["n_aux_excluded"] == 1
    assert sum(main.loss_mask) == len(G1) + len(G2) + len(G3)


def test_projection_is_contiguous_and_matches_consumer_contract(tmp_path):
    store = _seed_store(tmp_path, branching_rollout())
    reader = LocalCaptureReader(store)
    steps = [r for r in reader.records(RID, kinds={"model_call"}) if isinstance(r, ModelCallRecord)]
    entries = CaptureTokenSource(reader).entries(RID)
    out = get_builder("prefix_merging")(entries)
    proj = project_main_chain_response(RID, out.chains, {s.request_id: s.response for s in steps})
    assert_nemo_rl_contiguity(proj)  # main chain: PASS
    assert proj["ng_projection"]["n_branches"] == 1  # branch held back
    items = proj["output"]
    assert items[0]["prompt_token_ids"] == P0
    assert items[1]["prompt_token_ids"] == P0 + G1 + T1  # cumulative + interstitial
    # semantic content survives for batch_decode/logging
    assert items[0]["content"][0]["text"].startswith("gen-")


def test_forest_as_one_list_fails_contiguity():
    """Shows why only the main chain is delivered today: NeMo-RL's consumer
    cannot ingest a branch inline."""
    out = get_builder("prefix_merging")(branching_rollout())
    all_items = []
    for c in out.chains:
        all_items += project_chain_to_response_output(c, {})
    with pytest.raises(AssertionError, match="contiguity"):
        assert_nemo_rl_contiguity({"output": all_items})


def test_inline_shim_equivalence(tmp_path):
    """InlineItemsTokenSource (from an inline response) yields the same entries
    as CaptureTokenSource on the same rollout."""
    entries = linear_rollout()
    store = _seed_store(tmp_path, entries)
    cap = CaptureTokenSource(LocalCaptureReader(store)).entries(RID)
    inline_resp = {
        "output": [
            {
                "type": "message",
                "prompt_token_ids": e.prompt_token_ids,
                "generation_token_ids": e.generation_token_ids,
                "generation_log_probs": e.generation_log_probs,
            }
            for e in entries
        ]
    }
    inline = InlineItemsTokenSource(RID, inline_resp).entries(RID)
    key = lambda e: (e.prompt_token_ids, e.generation_token_ids, e.generation_log_probs)
    assert [key(e) for e in cap] == [key(e) for e in inline]


def test_classify_auxiliary_rules():
    steps = [
        ModelCallRecord(rollout_id=RID, request_id="a", model="policy", request={}, response={}),
        ModelCallRecord(rollout_id=RID, request_id="b", model="other-model", request={}, response={}),
        ModelCallRecord(rollout_id=RID, request_id="c", model="policy", aux_tag="verifier", request={}, response={}),
        ModelCallRecord(
            rollout_id=RID,
            request_id="d",
            model="policy",
            request={"messages": [{"content": "You are a title generator"}]},
            response={},
        ),
    ]
    aux = classify_auxiliary(steps, policy_model="policy")
    assert set(aux) == {"b", "c", "d"} and aux["c"] == "tag:verifier"


# ---------------------------------------------------------------------------
# Gate middleware (uses httpx ASGITransport; fastapi/httpx are Gym deps)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_correlation_recording_and_stripping(tmp_path):
    import httpx
    from fastapi import FastAPI

    from nemo_gym.observability.capture_gate import ObservabilityConfig, install_ingress_gate

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def fake_engine(body: dict) -> dict:
        # The gate observes only; the model server produces token fields from
        # its own config, so the forwarded request body is left untouched.
        return {
            "model": "policy",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "prompt_token_ids": [1, 2],
                        "generation_token_ids": [3, 4],
                        "generation_log_probs": [-0.1, -0.2],
                    }
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }

    cfg = ObservabilityConfig(enabled=True, capture_dir=str(tmp_path), return_token_ids=True)
    store, tracker = install_ingress_gate(app, cfg, model_name="policy")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gate") as client:
        rid = "toy.00001.r00.a0"
        # 1. untrusted prefix path: correlated, recorded, token fields STRIPPED
        r = await client.post(
            f"/ng-rollout/{rid}/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
        )
        assert r.status_code == 200
        body = r.json()
        assert "prompt_token_ids" not in json.dumps(body)  # never to the sandbox
        # 2. namespace guard: capture unreachable through the prefix
        g = await client.get(f"/ng-rollout/{rid}/ng-capture/rollouts/{rid}")
        assert g.status_code == 404
        await tracker.drain(rid)
        kinds = [e["kind"] for e in store.read_envelopes(rid)]
        assert kinds == ["model_call", "tokens"]
        step = next(store.read_envelopes(rid, kinds={"model_call"}))
        assert step["data"]["from_untrusted_prefix"] is True
        assert "generation_token_ids" not in json.dumps(step["data"]["response"])
        tokens = next(store.read_envelopes(rid, kinds={"tokens"}))
        assert tokens["data"]["generation_token_ids"] == [3, 4]
        # 3. read API works OUTSIDE the prefix
        ok = await client.get(f"/ng-capture/rollouts/{rid}/summary")
        assert ok.status_code == 200 and ok.json()["counts"]["model_call"] == 1


@pytest.mark.asyncio
async def test_streamed_call_captures_tokens_at_served_layer(tmp_path):
    """A streaming response drops token ids on the wire, so the served layer
    records the TokenEntry (via the token sink) while the gate records the
    ModelCallRecord — sharing one request_id so training can join them."""
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    from nemo_gym.base_responses_api_model import _chat_completion_to_sse
    from nemo_gym.observability.capture_gate import ObservabilityConfig, install_ingress_gate
    from nemo_gym.observability.token_sink import capture_streamed_tokens

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def fake_engine(body: dict):
        payload = {
            "id": "cmpl-1",
            "model": "policy",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "prompt_token_ids": [1, 2],
                        "generation_token_ids": [3, 4],
                        "generation_log_probs": [-0.1, -0.2],
                    }
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }
        if body.get("stream"):
            capture_streamed_tokens(payload)  # served-layer capture before SSE drops token ids
            return StreamingResponse(_chat_completion_to_sse(payload), media_type="text/event-stream")
        return payload

    cfg = ObservabilityConfig(enabled=True, capture_dir=str(tmp_path), return_token_ids=True)
    store, tracker = install_ingress_gate(app, cfg, model_name="policy")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gate") as client:
        rid = "toy.00002.r00.a0"
        async with client.stream(
            "POST",
            f"/ng-rollout/{rid}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}], "stream": True},
        ) as r:
            assert r.status_code == 200
            sse = "".join([chunk async for chunk in r.aiter_text()])
        # Token ids never ride the streamed wire.
        assert "generation_token_ids" not in sse and "prompt_token_ids" not in sse
        assert "hi" in sse and "[DONE]" in sse

        await tracker.drain(rid)
        kinds = sorted(e["kind"] for e in store.read_envelopes(rid))
        assert kinds == ["model_call", "tokens"]
        step = next(store.read_envelopes(rid, kinds={"model_call"}))
        tokens = next(store.read_envelopes(rid, kinds={"tokens"}))
        assert step["data"]["streamed"] is True
        # The streamed TokenEntry carries the ids and joins the ModelCallRecord 1:1.
        assert tokens["data"]["generation_token_ids"] == [3, 4]
        assert tokens["data"]["prompt_token_ids"] == [1, 2]
        assert tokens["data"]["request_id"] == step["data"]["request_id"]
