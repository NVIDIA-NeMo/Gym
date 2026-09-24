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

"""The verify path end to end, including the HTTP boundary and the aggregate path."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from app import NMRArenaResourcesServer, NMRArenaResourcesServerConfig, NMRArenaStatus, NMRArenaVerifyRequest
from fastapi.testclient import TestClient
from prompting import build_messages

from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient


GOLD = "CC(CCI)C"
GARBAGE = "C(C)(C)(C)(C)C"
MESSAGES = build_messages("H_NMR (300 MHz, CDCl3) δ 3.21 (t, 2H)", "C_NMR (75 MHz, CDCl3) δ 42.6")


def make_server(**overrides) -> NMRArenaResourcesServer:
    config = NMRArenaResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="nmrarena", **overrides)
    return NMRArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_response(text: str, status: str = "completed", incomplete_reason=None) -> dict:
    response = {
        "id": "resp_test",
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "status": status,
        "output": [
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if incomplete_reason:
        response["incomplete_details"] = {"reason": incomplete_reason}
    return response


def cands(*smiles: str) -> str:
    return json.dumps({"candidates": [{"rank": i + 1, "smiles": s} for i, s in enumerate(smiles)]})


def metadata(**overrides) -> dict:
    meta = {"compound_id": "c-1", "smiles": GOLD, "primary_class": "cls_alkanes_haloalkanes", "n_complex": 0.09}
    meta.update(overrides)
    return meta


DEFAULT = object()


def request_dict(text: str, meta=DEFAULT, **response_kwargs) -> dict:
    return {
        "responses_create_params": {"input": MESSAGES, "temperature": 1.0, "max_output_tokens": 24576},
        "response": make_response(text, **response_kwargs),
        "verifier_metadata": metadata() if meta is DEFAULT else meta,
    }


def verify(server, text: str, meta=DEFAULT, **response_kwargs):
    body = NMRArenaVerifyRequest(**request_dict(text, meta, **response_kwargs))
    return asyncio.run(server.verify(body))


def post_verify(server, payload: dict):
    """POST as bytes the way a client would, so JSON escapes reach the server as escapes."""
    with TestClient(server.setup_webserver()) as client:
        return client.post(
            "/verify", content=json.dumps(payload).encode("utf-8"), headers={"content-type": "application/json"}
        )


class TestVerify:
    def test_gold_as_prediction_scores_one(self) -> None:
        r = verify(make_server(), cands(GOLD))
        assert r.status == NMRArenaStatus.SCORED.value
        assert (r.reward, r.top1, r.top10, r.hit_rank, r.tanimoto_top1, r.answered) == (1.0, 1.0, 1.0, 1, 1.0, 1.0)
        assert r.truth_canonical == "CC(C)CCI" and r.candidates == ["CC(C)CCI"]
        assert r.compound_id == "c-1" and r.primary_class == "cls_alkanes_haloalkanes" and r.n_complex == 0.09
        assert r.harness_failure == 0.0 and r.failure_reason is None and not r.response_incomplete

    def test_gold_at_rank_two_is_top10_not_top1(self) -> None:
        r = verify(make_server(), cands("CC(C)CCBr", GOLD))
        assert (r.reward, r.top10, r.hit_rank) == (0.0, 1.0, 2)
        assert r.tanimoto_top1 == pytest.approx(0.4117647058823529)  # upstream's tanimoto on the same pair

    @pytest.mark.parametrize(
        ("text", "status"),
        [
            ("", NMRArenaStatus.EMPTY_OUTPUT),
            ("The structure is 3-iodo-2-methylpropane.", NMRArenaStatus.FORMAT_FAIL),
            (cands(GARBAGE), NMRArenaStatus.NO_VALID_CANDIDATE),
        ],
    )
    def test_non_answers_score_zero_with_their_status(self, text, status) -> None:
        r = verify(make_server(), text)
        assert r.status == status.value and r.reward == 0.0 and r.harness_failure == 0.0
        assert r.tanimoto_top1 is None and r.answered == 0.0

    def test_echoed_prompt_scores_nothing(self) -> None:
        r = verify(make_server(), MESSAGES[1]["content"])
        assert r.status == NMRArenaStatus.FORMAT_FAIL.value and r.tanimoto_top1 is None

    def test_right_plus_garbage_lenient_and_strict(self) -> None:
        text = cands(GOLD, GARBAGE)
        lenient = verify(make_server(), text)
        assert (lenient.reward, lenient.n_invalid, lenient.status) == (1.0, 1, NMRArenaStatus.SCORED.value)
        strict = verify(make_server(strict_candidates=True), text)
        assert (strict.reward, strict.candidates, strict.status) == (0.0, [], NMRArenaStatus.INVALID_CANDIDATE.value)
        assert strict.harness_failure == 0.0

    def test_garbage_then_right_is_not_top1_in_either_mode(self) -> None:
        text = cands(GARBAGE, GOLD)
        lenient = verify(make_server(), text)
        assert (lenient.reward, lenient.top10, lenient.tanimoto_top1) == (0.0, 1.0, None)
        assert verify(make_server(strict_candidates=True), text).reward == 0.0

    def test_oversize_smiles_is_a_status_not_a_crash(self) -> None:
        r = verify(make_server(), cands("C" * 20000))
        assert r.status == NMRArenaStatus.NO_VALID_CANDIDATE.value and r.n_oversize == 1

    def test_truncated_json_is_salvaged_unless_disabled(self) -> None:
        cut = '```json\n{"candidates": [{"rank": 1, "smiles": "%s"}, {"rank": 2, "smi' % GOLD
        r = verify(make_server(), cut, status="incomplete", incomplete_reason="max_output_tokens")
        assert r.reward == 1.0 and r.salvaged and r.response_incomplete
        r = verify(make_server(salvage_truncated_json=False), cut)
        assert r.reward == 0.0 and r.status == NMRArenaStatus.FORMAT_FAIL.value

    @pytest.mark.parametrize(
        "meta", [None, "not a dict", {}, metadata(smiles=GARBAGE), metadata(smiles=7), metadata(smiles=["CCO"])]
    )
    def test_unusable_truth_is_a_harness_fault(self, meta) -> None:
        r = verify(make_server(), cands(GOLD), meta)
        assert r.status == NMRArenaStatus.BAD_METADATA.value
        assert r.reward == 0.0 and r.harness_failure == 1.0 and r.failure_reason
        assert r.tanimoto_top1 is None

    def test_wrong_types_in_provenance_fields_cost_the_label_not_the_score(self) -> None:
        r = verify(make_server(), cands(GOLD), metadata(primary_class=["a"], n_complex="high", compound_id=5))
        assert r.status == NMRArenaStatus.SCORED.value and r.reward == 1.0
        assert (r.primary_class, r.n_complex, r.compound_id) == (None, None, None)

    def test_num_candidates_config_bounds_positions(self) -> None:
        r = verify(make_server(num_candidates=2), cands("C", "CC", GOLD))
        assert r.reward == 0.0 and r.top10 == 0.0 and len(r.candidates) == 2


class TestHTTPBoundary:
    def test_lone_surrogate_in_model_text_survives_the_wire(self) -> None:
        payload = request_dict(cands(GOLD) + " \udcff")
        resp = post_verify(make_server(), payload)
        assert resp.status_code == 200 and resp.json()["reward"] == 1.0
        assert "\udcff" not in resp.text

    def test_surrogate_inside_the_json_string_is_handled_after_parsing(self) -> None:
        text = '{"candidates": [{"rank": 1, "smiles": "\\udcff"}, {"rank": 2, "smiles": "%s"}]}' % GOLD
        payload = request_dict(text)
        assert "\\udcff" in payload["response"]["output"][0]["content"][0]["text"]
        resp = post_verify(make_server(), payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "scored" and body["hit_rank"] == 2 and body["candidates"][0] is None

    @pytest.mark.parametrize("meta", [["list"], {"smiles": {"nested": 1}}, {"smiles": ["CCO"]}, None])
    def test_malformed_metadata_over_http_is_200_with_a_fault(self, meta) -> None:
        resp = post_verify(make_server(), request_dict(cands(GOLD), meta))
        assert resp.status_code == 200
        assert resp.json()["status"] == "bad_metadata" and resp.json()["harness_failure"] == 1.0


class TestAggregate:
    def _rows(self):
        server = make_server()
        texts = [cands(GOLD), cands(GARBAGE, GOLD), "", cands("CC(C)CCBr")]
        results = []
        for i, t in enumerate(texts):
            r = verify(server, t).model_dump()
            results.append(dict(r, _ng_task_index=i, _ng_rollout_index=0))
        return server, results

    def test_headline_excludes_conditional_tanimoto_and_keeps_its_denominator(self) -> None:
        server, results = self._rows()
        metrics = compute_aggregate_metrics(
            results, compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
        )
        key = metrics.key_metrics
        assert key["mean/reward"] == key["mean/top1"] == 0.25
        assert key["mean/top10"] == 0.5 and key["mean/answered"] == 0.75
        assert "mean/tanimoto_top1" not in key and "tanimoto_top1/answered_only" not in key
        agent = metrics.agent_metrics
        # Two rows have a parseable position-1 candidate: gold (1.0) and the bromide.
        assert agent["count/answered"] == 2 and agent["count/rows"] == 4
        assert agent["tanimoto_top1/answered_only"] == pytest.approx((1.0 + 0.4117647058823529) / 2)
        assert agent["mean/tanimoto_top1"] == pytest.approx(agent["tanimoto_top1/answered_only"])
