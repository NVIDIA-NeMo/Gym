# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge ownership, task binding, measured zero, and provider-failure masking."""

from pathlib import Path

import pytest

from resources_servers.agentvista import verifier_worker


def test_verifier_preserves_native_predictions_and_measured_zero(monkeypatch):
    seen = []
    def verify_source(root):
        assert isinstance(root, Path)
        return {}
    monkeypatch.setattr(verifier_worker, "verify_source", verify_source)
    monkeypatch.setattr(
        verifier_worker,
        "inventory",
        lambda *args: {
            "task": {
                "question": "Original question",
                "source_sha256": "hash",
                "record": {"solution": "Original reference"},
            }
        },
    )

    def score(question, reference, predictions):
        seen.append((question, reference, predictions))
        return {"status": "FAIL", "reward": 0.0, "analysis": "Original judge evidence"}

    monkeypatch.setattr(verifier_worker, "score", score)
    config = {
        "source_root": ".",
        "dataset": "unused",
        "image_folder": ".",
        "metadata": {"task_id": "task", "source_sha256": "hash"},
        "result": {
            "verification_pending": True,
            "mask_sample": True,
            "judge_responses": ["first", "<answer>wrong</answer>"],
        },
    }
    result = verifier_worker.run(config)
    assert seen == [("Original question", "Original reference", ["first", "<answer>wrong</answer>"])]
    assert result["reward"] == 0 and result["mask_sample"] is False
    assert result["verify"]["analysis"] == "Original judge evidence"
    assert "verification_pending" not in result
    config["metadata"]["source_sha256"] = "changed"
    result = verifier_worker.run(config)
    assert result["mask_sample"] is True and result["failure_kind"] == "agentvista:verifier"
    assert len(seen) == 1


@pytest.mark.parametrize("error", [TimeoutError("Judge unavailable"), ValueError("Invalid binary score")])
def test_judge_failure_never_becomes_measured_zero(monkeypatch, error):
    def verify_source(root):
        assert isinstance(root, Path)
        return {}
    monkeypatch.setattr(verifier_worker, "verify_source", verify_source)
    monkeypatch.setattr(
        verifier_worker,
        "inventory",
        lambda *args: {"task": {"question": "question", "source_sha256": "hash", "record": {"solution": "reference"}}},
    )

    def fail(*args):
        raise error

    monkeypatch.setattr(verifier_worker, "score", fail)
    result = verifier_worker.run(
        {
            "source_root": ".",
            "dataset": "unused",
            "image_folder": ".",
            "metadata": {"task_id": "task", "source_sha256": "hash"},
            "result": {"verification_pending": True, "mask_sample": True, "judge_responses": []},
        }
    )
    assert result["mask_sample"] is True
    assert result["failure_kind"] == "agentvista:verifier"
    assert result.get("reward") is None


@pytest.mark.asyncio
async def test_public_verify_retries_share_a_judge_and_persist_the_score(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.agentvista.app import AgentVistaResources, AgentVistaResourcesConfig
    from responses_api_agents.agentvista_agent.runtime.results import AgentVistaVerifyRequest, policy_response
    from responses_api_agents.agentvista_agent.app import AgentVistaRunRequest

    config = AgentVistaResourcesConfig(host="127.0.0.1", port=1234, entrypoint="app.py", name="agentvista_resources",
        source_root=str(tmp_path), dataset="unused", image_folder=str(tmp_path), output_root=str(tmp_path / "output"))
    server = AgentVistaResources(config=config, server_client=MagicMock(spec=ServerClient))
    body = AgentVistaRunRequest(responses_create_params={"input": "Act"},
        verifier_metadata={"task_id": "task", "category_id": "fixture", "source_sha256": "hash"})
    raw = {"artifacts": str(tmp_path), "verification_pending": True, "mask_sample": True,
           "judge_responses": ["wrong"], "verify": {"status": "ERROR"}}
    request = AgentVistaVerifyRequest(**body.model_dump(), response=policy_response(body, raw, "fixture"))
    calls = []
    async def spawn(*args, **kwargs):
        calls.append(args)
        path = Path(args[-1])
        pending = json.loads(path.read_text())["result"]
        pending.update(verify={"status": "FAIL", "reward": 0.0}, mask_sample=False)
        pending.pop("verification_pending")
        (path.parent / "result.json").write_text(json.dumps(pending))
        return SimpleNamespace(wait=AsyncMock(return_value=0), returncode=0)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    first, second = await asyncio.gather(server.verify(request), server.verify(request))
    assert first.reward == second.reward == 0 and not first.mask_sample
    assert len(calls) == 1
    # A new process/server instance still reuses the durable verifier receipt.
    restarted = AgentVistaResources(config=config, server_client=server.server_client)
    assert (await restarted.verify(request)).mask_sample is False
    assert len(calls) == 1
    # A worker's preassigned score is never accepted as a measured result.
    request.response.agentvista_result = {"artifacts": str(tmp_path), "mask_sample": False,
                                          "verify": {"status": "PASS", "reward": 1.0}}
    forged = await server.verify(request)
    assert forged.mask_sample and forged.reward == 0 and forged.verify_status == "ERROR"
