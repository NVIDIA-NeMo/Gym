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
import asyncio
import json
import time
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.failure_kinds import AGENT_INVALID_OUTPUT, AGENT_LIMIT, VERIFIER_ERROR
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.chemcost.app import (
    VERIFIER_FIXTURE,
    ChemCostResourcesServer,
    ChemCostResourcesServerConfig,
    ChemCostVerifier,
    ChemCostVerifyRequest,
)


CASES = [json.loads(line) for line in (Path(__file__).parent / "verifier_cases.jsonl").read_text().splitlines()]


@lru_cache(maxsize=1)
def _server() -> ChemCostResourcesServer:
    config = ChemCostResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="chemcost")
    return ChemCostResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _verify(request: dict):
    return asyncio.run(ChemCostVerifier().verify(ChemCostVerifyRequest.model_validate(request)))


def _case(name: str) -> dict:
    return json.loads(json.dumps(next(c["request"] for c in CASES if c["name"] == name)))


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
        )
    )


def test_reward_is_cta_at_25_and_cta_at_10_rides_along() -> None:
    exact = _verify(_case("exact label as final JSON"))
    assert (exact.reward, exact.cta_at_10, exact.cta_at_25, exact.abstained) == (1.0, 1.0, 1.0, False)
    assert exact.evaluation_completed is True
    loose = _verify(_case("within 25 percent but not 10 percent"))
    assert (loose.reward, loose.cta_at_10, loose.cta_at_25) == (1.0, 0.0, 1.0)
    assert loose.relative_error == pytest.approx(0.2, abs=1e-4)


def test_boundary_counts_as_within_tolerance() -> None:
    request = _case("exact label as final JSON")
    label = request["verifier_metadata"]["procurement_cost_usd_per_g_product"]
    message = request["response"]["output"][0]["content"][0]
    message["text"] = json.dumps({"predicted_cost_per_gram": label * 1.1})
    assert _verify(request).cta_at_10 == 1.0


def test_abstentions_are_scored_failures_not_masked() -> None:
    null = _verify(_case("explicit abstention"))
    assert null.evaluation_completed is True
    assert (null.reward, null.abstained, null.mask_sample, null.failure_kind) == (
        0.0,
        True,
        False,
        AGENT_INVALID_OUTPUT,
    )
    truncated = _case("explicit abstention")
    truncated["response"]["status"] = "incomplete"
    truncated["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    assert _verify(truncated).failure_kind == AGENT_LIMIT


def test_token_truncated_message_with_a_correct_figure_is_not_scored() -> None:
    request = _case("labelled per-gram figure without JSON")
    request["response"]["status"] = "incomplete"
    request["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    result = _verify(request)
    assert (result.reward, result.abstained, result.mask_sample, result.failure_kind) == (
        0.0,
        True,
        False,
        AGENT_LIMIT,
    )


def test_step_limit_with_pending_tools_ignores_interim_commentary() -> None:
    # simple_agent stops after executing tools at max_steps; the earlier message holds a labelled figure.
    interim = _verify(_case("interim commentary then pending tools at the step limit"))
    assert (interim.reward, interim.abstained, interim.mask_sample) == (0.0, True, False)
    assert interim.failure_kind == AGENT_LIMIT
    assert _verify(_case("tool call without a final message")).failure_kind == AGENT_LIMIT


def test_reasoning_only_stop_is_invalid_output_not_a_limit() -> None:
    request = _case("tool call without a final message")
    request["response"]["output"] = [{"id": "r1", "type": "reasoning", "summary": []}]
    request["response"]["status"] = "incomplete"
    request["response"]["metadata"] = {"ng_termination_reason": "incomplete_reasoning"}
    result = _verify(request)
    assert (result.abstained, result.failure_kind) == (True, AGENT_INVALID_OUTPUT)


def test_long_brace_flood_still_finds_the_final_answer() -> None:
    request = _case("exact label as final JSON")
    message = request["response"]["output"][0]["content"][0]
    final = message["text"].split("\n", 1)[1]
    message["text"] = "{" * 50_000 + "\n" + final
    started = time.monotonic()
    result = _verify(request)
    assert (result.reward, result.abstained) == (1.0, False)
    assert time.monotonic() - started < 10
    message["text"] = "{" * 50_000
    assert _verify(request).abstained


@pytest.mark.parametrize("digits", [309, 5000])
def test_unrepresentable_number_is_a_scored_abstention(digits: int) -> None:
    # 309 digits overflow a float; 5,000 also pass Python's integer digit limit inside json.
    request = _case("exact label as final JSON")
    message = request["response"]["output"][0]["content"][0]
    message["text"] = '{"predicted_cost_per_gram": ' + "9" * digits + ', "predicted_components": []}'
    result = _verify(request)
    assert (result.reward, result.abstained, result.mask_sample, result.failure_kind) == (
        0.0,
        True,
        False,
        AGENT_INVALID_OUTPUT,
    )


def test_row_without_label_is_masked() -> None:
    request = _case("exact label as final JSON")
    request["verifier_metadata"]["procurement_cost_usd_per_g_product"] = None
    result = _verify(request)
    assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, VERIFIER_ERROR)
    assert result.evaluation_completed is False
    assert result.failure_reason


def test_metrics_by_reaction_type_skip_masked_rollouts() -> None:
    def rollout(kind: str, cta10: float, cta25: float, masked: bool = False) -> list[dict]:
        meta = {"reaction_id": "r", "type": kind}
        return [{"verifier_metadata": meta, "cta_at_10": cta10, "cta_at_25": cta25, "mask_sample": masked}]

    tasks = [rollout("single_step", 1, 1), rollout("single_step", 0, 1), rollout("multi_step", 0, 0)]
    tasks.append(rollout("multi_step", 1, 1, masked=True))
    assert ChemCostVerifier().compute_metrics(tasks) == {
        "multi_step/cta@10": 0.0,
        "multi_step/cta@25": 0.0,
        "single_step/cta@10": 0.5,
        "single_step/cta@25": 1.0,
    }


def test_supplier_quotes_route_returns_unfiltered_quotes() -> None:
    client = TestClient(_server().setup_webserver())
    ethanol = client.post("/get_supplier_quotes", json={"smiles_or_name": "CCO"}).json()
    assert ethanol["quotes"] and set(ethanol["quotes"][0]) == {"supplier", "pack", "price_usd", "purity"}
    # A different SMILES of the same molecule resolves to the same entry.
    assert client.post("/get_supplier_quotes", json={"smiles_or_name": "OCC"}).json() == ethanol
    unknown = client.post("/get_supplier_quotes", json={"smiles_or_name": "not a chemical"}).json()
    assert "No pricing data" in unknown["error"]
    assert "Missing argument" in client.post("/get_supplier_quotes", json={}).json()["error"]


def test_calculate_route_evaluates_arithmetic_only() -> None:
    client = TestClient(_server().setup_webserver())
    assert client.post("/calculate", json={"expression": "2 * (3 + 4) / 7"}).json()["result"] == 2.0
    # A number argument is passed as text, as upstream's run_tool does.
    assert client.post("/calculate", json={"expression": 5}).json()["result"] == 5.0
    assert "error" in client.post("/calculate", json={"expression": "__import__('os')"}).json()


def test_stateless_tools_keep_the_session_contract() -> None:
    check_resources_session_contract(
        _server().setup_webserver(),
        ResourcesSeedSessionRequest(
            resources_session_id="contract-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="chemcost", task_id="CHEMCOST-0001"),
            task_data={},
        ),
        keeps_state=False,
    )


def test_pricing_db_with_wrong_checksum_is_refused(tmp_path: Path) -> None:
    from resources_servers.chemcost.pricing_data import ensure_pricing_db

    bad = tmp_path / "pricing_db.sqlite"
    bad.write_bytes(b"not the release")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_pricing_db(bad)
