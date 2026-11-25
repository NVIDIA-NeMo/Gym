# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from synclaw.core import db
from synclaw.rollout.policy import RolloutPolicyConfig

from nemo_gym.client import ServerClient
from resources_servers.synclaw.app import SynClawResourcesServer
from resources_servers.synclaw.schema import (
    SYNCLAW_SUBMISSION_KEY,
    SynClawResourcesServerConfig,
    SynClawSeedSessionRequest,
    SynClawVerifyRequest,
)


# Minimal valid NeMoGymResponse body;
# extra fields (synclaw_submission) are added per-test.
_BARE_RESPONSE = {
    "id": "r",
    "created_at": 0,
    "model": "m",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "none",
    "tools": [],
}

_BASE_REQUEST = {
    "responses_create_params": {"input": []},
    "target_smiles": "CCO",
    "response": _BARE_RESPONSE,
}


def _make_server(tmp_path, **config_overrides) -> SynClawResourcesServer:
    db_path = tmp_path / "synclaw.sqlite"
    db_path.touch()
    with patch("resources_servers.synclaw.app.preload_building_block_search"):
        config = SynClawResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            db_path=db_path,
            max_concurrent_tool_calls=2,
            **config_overrides,
        )
        return SynClawResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )


def _make_request(submission: dict | None = None) -> SynClawVerifyRequest:
    response = _BARE_RESPONSE.copy()
    if submission is not None:
        response |= {SYNCLAW_SUBMISSION_KEY: submission}
    return SynClawVerifyRequest.model_validate(_BASE_REQUEST | {"response": response})


class TestVerify:
    @pytest.fixture
    def server(self, tmp_path):
        return _make_server(tmp_path)

    @pytest.fixture
    def capped_server(self, tmp_path):
        return _make_server(tmp_path, max_similarity_reward=0.6)

    @pytest.mark.parametrize("incoming", [None, False, True])
    @pytest.mark.parametrize("submission", [None, {"accepted": False}, {"accepted": True, "final_smiles": "OCC"}])
    async def test_scored_response_serializes_evaluation_completed(self, server, incoming, submission):
        request = _make_request(submission)
        if incoming is not None:
            request = SynClawVerifyRequest.model_validate(request.model_dump() | {"evaluation_completed": incoming})

        result = await server.verify(request)
        saved = result.model_dump(mode="json")

        # Completion means the verifier scored the answer, including a valid zero score.
        assert saved["evaluation_completed"] is True
        assert saved["mask_sample"] is False
        assert saved["reward"] == (1.0 if submission and submission.get("accepted") else 0.0)

    async def test_no_submission_gives_zero_reward(self, server):
        result = await server.verify(_make_request(submission=None))
        assert result.reward == 0.0
        assert result.submitted is False
        assert result.accepted is False

    async def test_submission_without_similarity_gives_zero_reward(self, server):
        # submission present but:
        # - no precomputed similarity field, and
        # - no final_smiles to recompute from
        result = await server.verify(_make_request({"accepted": True}))
        assert result.reward == 0.0

    async def test_unaccepted_submission_gives_zero_reward(self, server):
        # similarity is present, but the submission was never accepted;
        # synclaw only reports similarity for accepted submissions
        request = _make_request({"similarity_to_target": 0.82})
        result = await server.verify(request)
        assert result.accepted is False
        assert result.similarity == 0.0
        assert result.reward == 0.0

    async def test_default_similarity_is_stereo_aware(self, server: SynClawResourcesServer) -> None:
        request = _make_request(
            {
                "similarity_to_target": 0.82,
                "constitutional_similarity_to_target": 1.0,
                "mces_similarity_to_target": 0.95,
                "accepted": True,
            }
        )
        result = await server.verify(request)
        assert result.similarity == pytest.approx(0.82)
        assert result.reward == pytest.approx(0.5 * 0.82)

    async def test_stereo_aware_similarity(self, server):
        request = _make_request({"similarity_to_target": 0.75, "accepted": True})
        result = await server.verify(request)
        assert result.reward == pytest.approx(0.5 * 0.75)

    async def test_similarity_weight_can_restore_previous_reward(self, tmp_path):
        server = _make_server(tmp_path, max_similarity_reward=1.0)
        result = await server.verify(_make_request({"similarity_to_target": 0.82, "accepted": True}))
        assert result.reward == pytest.approx(0.82)

    async def test_stereo_aware_ignores_other_metric_fields(self, server: SynClawResourcesServer) -> None:
        # Constitutional/MCES fields must not replace a missing stereo-aware score.
        request = _make_request(
            {
                "constitutional_similarity_to_target": 0.9,
                "mces_similarity_to_target": 0.95,
                "accepted": True,
            }
        )
        result = await server.verify(request)
        assert result.reward == 0.0

    async def test_reward_clamped_at_zero_not_negative(self, server):
        request = _make_request({"similarity_to_target": 0.0, "accepted": True})
        result = await server.verify(request)
        assert result.reward == 0.0

    async def test_exact_reconstruction_flag(self, server):
        request = _make_request({"final_smiles": "OCC", "accepted": True})
        result = await server.verify(request)
        assert result.exact_reconstruction is True

    async def test_capped_similarity_reward_without_exact_reconstruction(self, capped_server):
        request = _make_request({"similarity_to_target": 0.82, "accepted": True})
        result = await capped_server.verify(request)
        assert result.exact_reconstruction is False
        assert result.reward == pytest.approx(0.6 * 0.82)

    async def test_capped_similarity_reward_with_exact_reconstruction(self, capped_server):
        request = _make_request({"final_smiles": "OCC", "accepted": True})
        result = await capped_server.verify(request)
        assert result.exact_reconstruction is True
        assert result.reward == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "target,product,accepted,expected_exact",
        [
            ("CCCCCCC", "CCCCCCCC", True, False),  # Morgan similarity is 1.0 for different molecules.
            ("O=c1cccc[nH]1", "Oc1ccccn1", True, True),  # Standard InChI recognizes this tautomer.
            ("F[C@H](Cl)Br", "F[C@@H](Cl)Br", True, False),
            ("FC(Cl)Br", "F[C@H](Cl)Br", True, True),  # A stereoisomer the target leaves open.
            ("F[C@H](Cl)Br", "FC(Cl)Br", True, False),  # Leaving open stereo the target specifies.
            ("CC(C)=O", "CC(O)=C", True, True),  # Keto-enol tautomers Standard InChI does not merge.
            ("CCO", "OCC", False, False),
            ("CCO", None, True, False),
        ],
    )
    async def test_exact_reward_uses_identity(self, tmp_path, target, product, accepted, expected_exact):
        server = _make_server(tmp_path)
        request = _make_request({"accepted": accepted, "actual_final_smiles": product, "identical_to_target": True})
        request.target_smiles = target
        result = await server.verify(request)
        assert result.exact_reconstruction is expected_exact
        assert result.reward == pytest.approx(1.0 if expected_exact else 0.5 * result.similarity * accepted)


class TestSeedSession:
    @pytest.mark.parametrize("enabled,context_limit", [(True, 8192), (True, None), (False, 8192)])
    async def test_announces_only_known_limits(self, tmp_path, enabled, context_limit):
        server = _make_server(tmp_path)
        result = await server.seed_session(
            MagicMock(),
            SynClawSeedSessionRequest(
                system_prompt="You are a chemist.",
                question="Synthesize ethanol.",
                config=RolloutPolicyConfig(
                    max_tool_calls=150,
                    max_turns=8,
                    max_tokens=2048,
                    max_model_len=context_limit,
                    budget_feedback=enabled,
                ),
            ),
        )
        system = result.input[0].content
        assert "150 total tool calls or 8 assistant turns" in system
        assert "Each response is capped at 2048 output tokens" in system
        assert ("context-window budget is 8192 tokens" in system) is (context_limit is not None)


class TestCallTool:
    @pytest.fixture
    def server(self, tmp_path):
        return _make_server(tmp_path)

    async def test_similarity_dispatches_to_tool_pool(self, server):
        result = await server.call_tool(
            "similarity",
            {"smiles_a": "CCO", "smiles_b": "CCO", "extra_tool_call_arguments": {"target_smiles": "CCO"}},
        )
        assert result["valid"] is True
        assert result["similarity"] == pytest.approx(1.0)

    async def test_extra_tool_call_arguments_are_stripped_before_dispatch(self, server):
        result = await server.call_tool(
            "similarity",
            {
                "smiles_a": "CCO",
                "smiles_b": "CCO",
                "extra_tool_call_arguments": {"target_smiles": "CCO"},
            },
        )
        assert result["valid"] is True

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Tool workers are forked processes on Linux")
    def test_workers_are_forked_at_construction(self, server):
        assert len(server._pool._processes) == server.config.max_concurrent_tool_calls

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Tool workers are forked processes on Linux")
    async def test_dead_worker_fails_one_call_then_the_pool_is_replaced(self, server):
        broken = server._pool
        os.kill(next(iter(broken._processes)), signal.SIGKILL)
        for _ in range(100):  # The pool notices the death asynchronously.
            if broken._broken:
                break
            await asyncio.sleep(0.05)
        call = {"smiles_a": "CCO", "smiles_b": "CCO", "extra_tool_call_arguments": {"target_smiles": "CCO"}}
        with pytest.raises(BrokenProcessPool):
            await server.call_tool("similarity", dict(call))
        assert server._pool is not broken
        assert (await server.call_tool("similarity", dict(call)))["valid"] is True

    async def test_unknown_tool_returns_argument_error(self, server):
        result = await server.call_tool("not_a_real_tool", {"extra_tool_call_arguments": {"target_smiles": "CCO"}})
        assert result["valid"] is False
        assert result["error"] == "invalid_tool_call"
        assert result["tool_name"] == "not_a_real_tool"


async def test_staged_database_serves_tools_after_source_is_removed(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    stock = tmp_path / "stock.smi"
    stock.write_text("CCO\nCCOC\n")
    db.ingest_building_blocks(stock, db_path=source)
    db.prepare_building_block_search(source)
    server = SynClawResourcesServer(
        config=SynClawResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="synclaw",
            db_path=source,
            db_staging_dir=tmp_path / "local",
            max_concurrent_tool_calls=2,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    source.unlink()
    try:
        result = await server.call_tool(
            "search_building_blocks", {"query": "CCO", "extra_tool_call_arguments": {"target_smiles": "CCN"}}
        )
        assert [item["match_type"] for item in result["results"]] == ["exact", "near"]
        assert result["results"][0]["canonical_smiles"] == "CCO"
    finally:
        server._pool.shutdown()


def test_unprepared_database_fails_at_startup(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    stock = tmp_path / "stock.smi"
    stock.write_text("CCO\n")
    db.ingest_building_blocks(stock, db_path=source)
    config = SynClawResourcesServerConfig(
        host="127.0.0.1", port=8080, entrypoint="", name="synclaw", db_path=source, max_concurrent_tool_calls=2
    )
    with pytest.raises(RuntimeError, match="prepare-search"):
        SynClawResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_missing_stock_points_to_the_build_script(tmp_path):
    with pytest.raises(ValueError, match="resources_servers/synclaw/prepare_db.py"):
        SynClawResourcesServerConfig(host="", port=0, entrypoint="", name="", db_path=tmp_path / "absent.sqlite")


@pytest.mark.parametrize("hide_target", [False, True])
async def test_target_stock_filter_is_task_local(tmp_path: Path, hide_target: bool) -> None:
    source = tmp_path / "stock.sqlite"
    stock = tmp_path / "stock.smi"
    stock.write_text("CO\nCCO\n")
    db.ingest_building_blocks(stock, db_path=source)
    db.prepare_building_block_search(source)
    server = SynClawResourcesServer(
        config=SynClawResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="synclaw",
            db_path=source,
            max_concurrent_tool_calls=1,
            hide_target_building_block=hide_target,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    try:
        # Alternate targets on the same worker, then revisit the first cached view.
        for target, target_id, other_id in [("CO", 1, 2), ("CCO", 2, 1), ("CO", 1, 2)]:
            extra = {"target_smiles": target}
            result = await server.call_tool(
                "search_building_blocks", {"query": "*", "extra_tool_call_arguments": extra}
            )
            ids = {item["id"] for item in result["results"]}
            assert other_id in ids
            assert (target_id in ids) is (not hide_target)
            for reference in [f"BB:{target_id}", target]:
                purchase = await server.call_tool(
                    "submit_pathway", {"building_block": reference, "extra_tool_call_arguments": extra}
                )
                assert purchase["accepted"] is (not hide_target)
            analogue = await server.call_tool(
                "submit_pathway", {"building_block": f"BB:{other_id}", "extra_tool_call_arguments": extra}
            )
            assert analogue["accepted"] is True
        scored = await server.verify(_make_request(analogue))
        assert scored.target_hidden_from_stock is hide_target
        if hide_target:
            with pytest.raises(ValueError, match="requires the task target_smiles"):
                await server.call_tool("search_building_blocks", {"query": "*"})
    finally:
        server._pool.shutdown()
