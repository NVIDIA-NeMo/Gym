# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from nemo_gym.token_id_capture.consumer import trajectories_from_source
from nemo_gym.token_id_capture.lineage import RolloutLineage, stamp_continuation
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, stamp_lineage
from nemo_gym.token_id_capture.store import TokenCaptureStore
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


@pytest.mark.parametrize("delegation", [False, True])
async def test_real_bench_runtime_keeps_capture_lineage_and_synthetic_terminal(tmp_path, monkeypatch, delegation):
    monkeypatch.chdir(tmp_path)
    lineage = RolloutLineage()
    store = TokenCaptureStore(tmp_path / "capture")
    entries = []
    resolutions = []
    requests = []
    codes = [
        "print('first observed output')",
        "print('second observed output')",
        "return_result(TaskResult(solution_description='done', evidence='two outputs', how_to_verify='inspect'))",
    ]

    if delegation:
        codes = [
            "child = await self.delegate(objective='Print once then finish')\nprint(child)",
            "print('child output')",
            codes[-1],
            codes[-1],
        ]

    async def post(**kwargs):
        body = kwargs["json"]
        items = body.model_dump(mode="json", exclude_none=True)["input"]
        index = len(entries)
        resolution, parent, reason = lineage.resolve_node(items)
        resolutions.append((resolution, reason))
        requests.append(items)
        output = NeMoGymResponseFunctionToolCall(
            id=f"item-{index}",
            call_id=f"tool-{index}",
            name="python_cell",
            arguments=json.dumps({"code": codes[index]}),
        )
        payload = model_response(output, response_id=f"response-{index}")
        prompt = (parent.cum_tokens if parent else []) + [10 + index]
        entry = TokenEntry(
            rollout_id="training-test",
            model_call_id=f"call-{index}",
            prompt_token_ids=prompt,
            generation_token_ids=[100 + index],
            generation_log_probs=[-0.1],
            output_items=payload["output"],
            token_item_index=0,
            response_id=payload["id"],
        )
        stamp_lineage(entry, parent.call_id if parent else None, parent_resolution=resolution)
        stamp_continuation(entry, items)
        lineage.add_entry(entry)
        entries.append(entry)
        await store.put(entry)
        return FakeHTTPResponse(payload)

    invocation = NOOAInvocationConfig(
        agent_class="responses_api_agents.nooa_agent.bench_agent_no_compaction:BenchAgentWithoutCompaction",
        invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
        init_kwargs={"working_dir": str(tmp_path)},
    )
    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=len(codes),
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Run two prints then finish"),
            model_url_path="/v1/responses",
            rollout_id="training-test",
        )
    )
    assert result.termination_reason is None
    assert len(entries) == len(codes)
    expected = [None, None, "call-1", "call-0"] if delegation else [None, "call-0", "call-1"]
    assert [entry.parent_call_id for entry in entries] == expected
    assert [status for status, _ in resolutions] == [
        ParentResolutionStatus.ROOT if parent is None else ParentResolutionStatus.RESOLVED for parent in expected
    ], resolutions
    assert any("Harness bootstrap execution:" in str(item.get("content", "")) for item in requests[0])
    for index, parent in enumerate(expected):
        if parent is not None:
            previous = requests[int(parent.split("-")[-1])]
            assert requests[index][: len(previous)] == previous
    response, _ = finalize_run_result(result)
    assert response.output[-1].id == "nooa_fallback"
    assert response.metadata["terminal_response_id"] == f"response-{len(codes) - 1}"
    built = await trajectories_from_source(
        "training-test",
        store,
        model="policy",
        verified_response=response.model_dump(mode="json"),
        declared_response_id=response.metadata["terminal_response_id"],
    )
    assert built["mask_sample"] is False, built
    assert built["metrics"]["roots"] == (2 if delegation else 1)
    assert built["metrics"]["chains"] == (2 if delegation else 1)
    assert built["metrics"]["delivered_fraction"] == (0.5 if delegation else 1.0)


@pytest.mark.parametrize("drift", ["changed_user", "missing_nonterminal_user", "changed_tool_result"])
async def test_unexpected_context_edits_fail_before_model_dispatch(drift):
    from nemo_gym.openai_utils import NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
    from responses_api_agents.nooa_agent.tests.test_gym_llm import make_llm

    output = NeMoGymResponseOutputMessage(
        id="answer", content=[NeMoGymResponseOutputText(text="first", annotations=[])]
    )
    llm, client, state = make_llm(model_response(output))
    llm._preserve_training_history = True
    initial = [
        {"role": "user", "content": "task"},
        {"type": "function_call_output", "call_id": "foreign", "output": "original"},
        {"role": "user", "content": "reminder"},
    ]
    first = await llm.acall(initial)
    changed = [dict(item) for item in initial]
    if drift == "changed_user":
        changed[0]["content"] = "edited"
    elif drift == "missing_nonterminal_user":
        changed.pop(0)
    else:
        changed[1]["output"] = "edited"
    with pytest.raises(ValueError, match="changed previously dispatched context"):
        await llm.acall([*changed, first, {"role": "user", "content": "next"}])
    assert client.post.await_count == 1
    assert isinstance(state.fatal_error, ValueError)


async def test_unobserved_prefill_is_not_reclassified_as_prompt_context():
    from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
    from responses_api_agents.nooa_agent.tests.test_gym_llm import make_llm

    llm, client, _ = make_llm(model_response())
    llm._preserve_training_history = True
    await llm.acall(
        [
            {"type": "function_call", "call_id": "prefill_unobserved", "name": "python_cell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "prefill_unobserved", "output": "receipt"},
        ]
    )
    items = client.post.call_args.kwargs["json"].model_dump(mode="json")["input"]
    assert assistant_fingerprint(items)
    assert items[0]["type"] == "function_call"
