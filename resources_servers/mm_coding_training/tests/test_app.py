# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real verifier with isolated renderer and judge transports."""

import base64
import hashlib
import io
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request
from PIL import Image
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    JUDGE_FAILED,
    VERIFIER_ERROR,
    FailureBoundary,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
)
from nemo_gym.server_runtime import ResourcesSeedRejectedError
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.mm_coding_training.app import (
    CodingConfig,
    CodingRequest,
    CodingSeedRequest,
    CodingTrainingServer,
)
from resources_servers.mm_coding_training.extraction import extract_answer
from resources_servers.mm_coding_training.quality import Quality, ScoringInput, final_reward
from resources_servers.mm_coding_training.task_data import CodingTask
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest


def png(color: str = "red") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="PNG")
    return stream.getvalue()


def response(text: str, *, status: str = "completed") -> dict:
    return {
        "id": "test-response",
        "created_at": 0,
        "model": "test",
        "object": "response",
        "status": status,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        "output": [
            {
                "id": "test-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def request(text: str = "<svg/>", *, benchmark: str = "mmsvg_text2svg", status: str = "completed") -> CodingRequest:
    task = {"benchmark": benchmark, "task_id": "task-1", "prompt": "Draw a red square", "source_split": "example"}
    if benchmark != "mmsvg_text2svg":
        task.update(reference_image="target.png", reference_sha256=hashlib.sha256(png()).hexdigest())
    return CodingRequest.model_validate(
        {
            "task": task,
            "responses_create_params": {"input": "Draw a red square"},
            "response": response(text, status=status),
        }
    )


def http_request(session_id="single-session"):
    return Request({"type": "http", "session": {"session_id": session_id}, "headers": []})


@pytest.fixture
def server(tmp_path: Path) -> CodingTrainingServer:
    (tmp_path / "target.png").write_bytes(png())
    config = CodingConfig(
        host="127.0.0.1",
        port=8000,
        name="mm_coding_training",
        entrypoint="app.py",
        reference_root=tmp_path,
        sandbox_provider="sandbox",
        sandbox_image="example/image@sha256:" + "a" * 64,
        judge_model_server=ModelServerRef(type="responses_api_models", name="coding_judge"),
        judge_model="quality-judge",
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return CodingTrainingServer(config=config, server_client=client)


@pytest.mark.asyncio
async def test_legacy_agent_metadata_survives_the_http_verifier_boundary(server: CodingTrainingServer) -> None:
    body = request("")
    run_payload = body.model_dump(exclude={"response"}) | {
        "agent_ref": {"type": "responses_api_agents", "name": "mm_coding_training_simple_agent"},
        "_ng_task_index": 7,
        "_ng_rollout_index": 3,
        "_rowidx": 2,
    }
    # The real SimpleAgent forwards its complete /run body to /verify.
    agent_request = SimpleAgentRunRequest.model_validate(run_payload)
    payload = agent_request.model_dump() | {"response": body.response.model_dump(mode="json")}
    app = FastAPI()
    server.setup_session_middleware(app)
    app.post("/verify")(server.verify)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://verifier") as client:
        reply = await client.post("/verify", json=payload)
        assert reply.status_code == 200, reply.text
        result = reply.json()
        assert result["agent_ref"] == run_payload["agent_ref"]
        assert result["_ng_task_index"] == 7 and result["_ng_rollout_index"] == 3 and result["_rowidx"] == 2
        assert result["reward"] == 0 and result["mask_sample"] is False
        payload["task"]["misspelled_grade_field"] = "must not be ignored"
        invalid = await client.post("/verify", json=payload)
        assert invalid.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "benchmark", ["chart2code", "mmsvg_img2svg", "mmsvg_text2svg", "image2struct_latex", "design2code_full"]
)
async def test_success_preserves_reward_and_binds_evidence(server: CodingTrainingServer, benchmark: str) -> None:
    renderer = MagicMock(render=AsyncMock(return_value=png()))
    verdict = Quality(quality=0.7, task_fidelity=0.8, visual_fidelity=0.6, reason="One detail differs")
    body = request(benchmark=benchmark)
    with (
        patch.object(CodingTrainingServer, "_renderer", return_value=renderer),
        patch.object(CodingTrainingServer, "_judge", AsyncMock(return_value=verdict)) as judge,
    ):
        result = await server.verify(http_request(), body)
    assert result.reward == pytest.approx(0.76)
    assert result.execution == 1 and not result.mask_sample
    assert result.answer_sha256 == hashlib.sha256(b"<svg/>").hexdigest()
    assert result.render_sha256 == hashlib.sha256(png()).hexdigest()
    assert result.task == body.task
    assert judge.call_args.args[-1] == (None if benchmark == "mmsvg_text2svg" else png())


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "<think>unfinished", "```html\n<svg/>\n```", "<svg/>\n```", "<think>x</think>"])
async def test_invalid_artifact_is_model_zero_without_execution(server: CodingTrainingServer, text: str) -> None:
    with (
        patch.object(CodingTrainingServer, "_renderer") as renderer,
        patch.object(CodingTrainingServer, "_judge") as judge,
    ):
        result = await server.verify(http_request(), request(text))
    assert result.reward == 0 and not result.mask_sample
    assert result.failure_kind == "mm_coding_training:invalid_artifact"
    renderer.assert_not_called()
    judge.assert_not_called()


@pytest.mark.asyncio
async def test_execution_failure_skips_judge(server: CodingTrainingServer) -> None:
    renderer = MagicMock(render=AsyncMock(return_value=None))
    with (
        patch.object(CodingTrainingServer, "_renderer", return_value=renderer),
        patch.object(CodingTrainingServer, "_judge") as judge,
    ):
        result = await server.verify(http_request(), request())
    assert result.reward == 0 and not result.mask_sample
    assert result.failure_kind == "mm_coding_training:execution_failed"
    judge.assert_not_called()


@pytest.mark.asyncio
async def test_provider_error_is_masked_without_judge(server: CodingTrainingServer) -> None:
    renderer = MagicMock(render=AsyncMock(side_effect=RuntimeError("unavailable")))
    with (
        patch.object(CodingTrainingServer, "_renderer", return_value=renderer),
        patch.object(CodingTrainingServer, "_judge") as judge,
    ):
        result = await server.verify(http_request(), request())
    assert result.reward == 0 and result.mask_sample
    assert result.failure_kind == VERIFIER_ERROR
    judge.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "judge_error",
    [
        JudgeError("test outage"),
        NgFailureEvent(
            boundary=FailureBoundary.RESOURCES_SERVER,
            failure_kind=JUDGE_FAILED,
            failure_reason="Typed judge transport failure",
        ),
    ],
)
async def test_judge_fault_retains_the_response_for_reverification(
    server: CodingTrainingServer, judge_error: Exception
) -> None:
    renderer = MagicMock(render=AsyncMock(return_value=png()))
    body = request()
    with (
        patch.object(CodingTrainingServer, "_renderer", return_value=renderer),
        patch.object(CodingTrainingServer, "_judge", AsyncMock(side_effect=judge_error)),
    ):
        response = await judge_failsafe(server.verify)(http_request(), body)
    result = json.loads(response.body)
    assert result["mask_sample"] and result["failure_kind"] == JUDGE_FAILED
    assert result["evaluation_completed"] is False
    assert result["response"] == body.response.model_dump(mode="json")
    assert result["source_artifact"] == "<svg/>" and result["source_sha256"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("status,masked", [("incomplete", False), ("failed", True), ("cancelled", True)])
async def test_interruption_and_model_budget_are_distinct(
    server: CodingTrainingServer, status: str, masked: bool
) -> None:
    with patch.object(CodingTrainingServer, "_renderer") as renderer:
        result = await server.verify(http_request(), request(status=status))
    assert result.mask_sample is masked and result.reward == 0
    renderer.assert_not_called()


@pytest.mark.asyncio
async def test_reference_drift_is_infrastructure_and_never_executes(server: CodingTrainingServer) -> None:
    (server.config.reference_root / "target.png").write_bytes(b"changed")
    with patch.object(CodingTrainingServer, "_renderer") as renderer:
        result = await server.verify(http_request(), request(benchmark="chart2code"))
    assert result.mask_sample and result.failure_kind == VERIFIER_ERROR
    renderer.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [b"changed", None], ids=["changed", "missing"])
async def test_single_answer_seed_refuses_an_unusable_reference(server: CodingTrainingServer, reference) -> None:
    # A wrong reference root fails at seed with a 422, as on the OpenCode route, instead of masking every image task.
    target = server.config.reference_root / "target.png"
    if reference is None:
        target.unlink()
    else:
        target.write_bytes(reference)
    with pytest.raises(ResourcesSeedRejectedError):
        await server.initialize_session_state(
            "single-session", CodingSeedRequest(task=request(benchmark="chart2code").task)
        )
    assert not server.has_session_state("single-session")


@pytest.mark.asyncio
async def test_single_answer_seed_keeps_a_task_with_a_matching_reference(server: CodingTrainingServer) -> None:
    await server.initialize_session_state(
        "single-session", CodingSeedRequest(task=request(benchmark="chart2code").task)
    )
    assert server.has_session_state("single-session")


@pytest.mark.asyncio
async def test_reference_symlink_escape_is_refused(server: CodingTrainingServer, tmp_path: Path) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-outside.png")
    outside.write_bytes(png())
    (tmp_path / "target.png").unlink()
    (tmp_path / "target.png").symlink_to(outside)
    result = await server.verify(http_request(), request(benchmark="chart2code"))
    assert result.mask_sample and result.failure_kind == VERIFIER_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("with_reference", [False, True])
async def test_judge_contract_is_preserved_and_task_text_is_untrusted(
    server: CodingTrainingServer, with_reference: bool
) -> None:
    body = request(benchmark="chart2code" if with_reference else "mmsvg_text2svg")
    body.task.ground_truth = "EXPECTED_PRIVATE_SOURCE"
    candidate_source = "CANDIDATE_SOURCE"
    candidate_png, reference_png = png("blue"), png("green")
    verdict = {"quality": 0.7, "task_fidelity": 0.8, "visual_fidelity": 0.6, "reason": "Missing detail"}
    completion = NeMoGymResponse.model_validate(response(json.dumps(verdict)))
    with patch("resources_servers.mm_coding_training.app.call_judge", AsyncMock(return_value=completion)) as call:
        result = await server._judge(
            body.task, candidate_source, candidate_png, reference_png if with_reference else None
        )
    assert result.quality == 0.7
    payload = call.call_args.kwargs["json"]
    assert "untrusted data" in payload["instructions"]
    assert "Draw a red square" not in payload["instructions"]
    if not with_reference:
        assert "There is no target screenshot" in payload["instructions"]
    images = [x for x in payload["input"][0]["content"] if x["type"] == "input_image"]
    assert len(images) == 1 + with_reference
    from resources_servers.mm_coding_training.remote_renderer import image_uri

    evidence = payload["input"][0]["content"]
    sources = json.loads(evidence[0]["text"])
    assert sources["candidate_source"] == candidate_source
    assert sources["reference_code_or_patch"] == body.task.ground_truth
    assert evidence[-2]["text"] == "CANDIDATE rendering to grade"
    assert evidence[-1]["image_url"] == image_uri(candidate_png)
    if with_reference:
        assert evidence[1]["text"] == "REFERENCE target rendering"
        assert evidence[2]["image_url"] == image_uri(reference_png)
    # Validate the actual model-server wire boundary, not only the mocked transport result.
    NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)
    assert all(image["detail"] == "auto" for image in images)
    assert payload["max_output_tokens"] == 2048


@pytest.mark.asyncio
@pytest.mark.parametrize("with_reference", [False, True])
async def test_chat_judge_preserves_the_rubric_images_and_structured_grade(server, with_reference):
    server.config.judge_api = "chat_completions"
    verdict = {"quality": 0.7, "task_fidelity": 0.8, "visual_fidelity": 0.6, "reason": "Missing detail"}
    completion = NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-chat",
            "object": "chat.completion",
            "created": 0,
            "model": "quality-judge",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(verdict)}}
            ],
        }
    )
    task = request(benchmark="chart2code" if with_reference else "mmsvg_text2svg").task
    task.ground_truth = "PRIVATE_REFERENCE_SOURCE"
    from resources_servers.mm_coding_training.remote_renderer import image_uri

    candidate, reference = png("blue"), png("green")
    with patch("resources_servers.mm_coding_training.app.call_judge", AsyncMock(return_value=completion)) as call:
        result = await server._judge(task, "CANDIDATE_SOURCE", candidate, reference if with_reference else None)
    assert result.model_dump() == verdict
    assert call.call_args.kwargs["url_path"] == "/v1/chat/completions"
    payload = call.call_args.kwargs["json"]
    NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
    assert payload["messages"][0]["role"] == "system"
    assert "untrusted data" in payload["messages"][0]["content"]
    evidence = payload["messages"][1]["content"]
    metadata = json.loads(evidence[0]["text"])
    assert metadata["reference_code_or_patch"] == task.ground_truth
    assert metadata["candidate_source"] == "CANDIDATE_SOURCE"
    images = [part["image_url"]["url"] for part in evidence if part["type"] == "image_url"]
    assert images == ([image_uri(reference)] if with_reference else []) + [image_uri(candidate)]
    assert payload["max_completion_tokens"] == server.config.judge_max_output_tokens
    assert payload["response_format"]["json_schema"]["strict"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choices", [[], [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}]]
)
async def test_incomplete_chat_grade_is_a_judge_failure(server, choices):
    server.config.judge_api = "chat_completions"
    completion = NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-chat",
            "object": "chat.completion",
            "created": 0,
            "model": "quality-judge",
            "choices": choices,
        }
    )
    with patch("resources_servers.mm_coding_training.app.call_judge", AsyncMock(return_value=completion)):
        with pytest.raises(JudgeError):
            await server._judge(request().task, "candidate", png(), None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict", ["not JSON", '{"quality": 1}', '{"quality":1,"task_fidelity":0,"visual_fidelity":0,"reason":"bad"}']
)
async def test_invalid_judge_verdict_is_not_a_model_zero(server: CodingTrainingServer, verdict: str) -> None:
    completion = NeMoGymResponse.model_validate(response(verdict))
    with patch("resources_servers.mm_coding_training.app.call_judge", AsyncMock(return_value=completion)):
        with pytest.raises(JudgeError):
            await server._judge(request().task, "<svg/>", png(), None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 2.0, True])
def test_non_quality_values_are_refused(value: float) -> None:
    with pytest.raises(ValidationError):
        Quality(quality=value, task_fidelity=value, visual_fidelity=value, reason="test")


@pytest.mark.parametrize("path", ["../secret.png", "/etc/passwd", "a/../../b.png", "a\\b.png"])
def test_task_path_is_bounded(path: str) -> None:
    with pytest.raises(ValidationError):
        CodingTask(
            benchmark="chart2code",
            task_id="x",
            prompt="task",
            source_split="example",
            reference_image=path,
            reference_sha256="a" * 64,
        )


def test_held_out_eval_split_cannot_be_silently_exported_as_training() -> None:
    with pytest.raises(ValidationError):
        CodingTask(benchmark="mmsvg_text2svg", task_id="x", prompt="task", source_split="test")


def test_reward_endpoints_and_partial_quality() -> None:
    assert final_reward(ScoringInput(executed=False)) == 0
    for fidelity, reward in [(0.0, 0.2), (0.5, 0.6), (1.0, 1.0)]:
        score = Quality(quality=fidelity, task_fidelity=fidelity, visual_fidelity=fidelity, reason="test")
        assert final_reward(ScoringInput(executed=True, verdict=score)) == pytest.approx(reward)


def test_final_artifact_extraction() -> None:
    assert extract_answer("<think>analysis</think>\n```svg\n<svg/>\n```", "svg") == "<svg/>"
    # Blank lines after the opening fence are removed, so an <?xml declaration stays first.
    assert extract_answer("```svg  \n\n \n<?xml version='1.0'?><svg/>\n  ```", "svg") == "<?xml version='1.0'?><svg/>"
    assert extract_answer("```py\r\nprint(1)\r\n```", "python") == "print(1)\r"


def test_unclosed_fence_with_many_blank_lines_is_refused_in_linear_time() -> None:
    # The fence pattern once took cubic time in the blank-line count on the event loop: 5000 lines took about 30 s.
    started = time.perf_counter()
    with pytest.raises(ValueError, match="one complete artifact"):
        extract_answer("```python" + "\n" * 5000 + "import matplotlib", "python")
    assert time.perf_counter() - started < 1


def test_renderer_resolves_provider_and_only_carries_resource_metadata(server: CodingTrainingServer) -> None:
    with patch(
        "resources_servers.mm_coding_training.app.get_global_config_dict",
        return_value={
            "sandbox": {"opensandbox": {"connection": {"api_key": "test-only"}}},
        },
    ):
        renderer = server._renderer()
    assert renderer.provider == {"opensandbox": {"connection": {"api_key": "test-only"}}}
    assert renderer.spec.image == server.config.sandbox_image
    assert renderer.spec.env == {}
    assert renderer.spec.resources.memory_mib == 16384


def test_host_execution_provider_is_rejected(server: CodingTrainingServer) -> None:
    with patch(
        "resources_servers.mm_coding_training.app.get_global_config_dict", return_value={"sandbox": {"local": {}}}
    ):
        with pytest.raises(ValueError, match="host execution"):
            server._renderer()


@pytest.mark.asyncio
async def test_overlarge_answer_is_model_zero(server: CodingTrainingServer) -> None:
    server.config.max_source_bytes = 3
    with patch.object(CodingTrainingServer, "_renderer") as renderer:
        result = await server.verify(http_request(), request("<svg/>"))
    assert not result.mask_sample and result.reward == 0
    renderer.assert_not_called()


def noise_png() -> bytes:
    stream = io.BytesIO()
    Image.effect_noise((64, 64), 64).convert("RGB").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("benchmark", "template"),
    [
        ("mmsvg_img2svg", '<svg xmlns="http://www.w3.org/2000/svg"><image href="data:image/png;base64,{b64}"/></svg>'),
        ("design2code_full", '<html><body><img src="data:image/png;base64,{b64}"></body></html>'),
        ("chart2code", 'import atexit\nREFERENCE = bytes.fromhex("{hex}")\natexit.register(print)'),
        ("chart2code", 'import base64\nREFERENCE = base64.b64decode("""\n{wrapped}\n""")'),
        ("chart2code", "import base64\nREFERENCE = base64.b64decode(\n{literals}\n)"),
    ],
)
async def test_answer_carrying_image_bytes_is_model_zero(server, benchmark, template) -> None:
    image = noise_png()
    encoded = base64.b64encode(image).decode()
    wrapped = "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))
    literals = "\n".join(f'    "{encoded[i : i + 76]}"' for i in range(0, len(encoded), 76))
    text = template.format(b64=encoded, hex=image.hex(), wrapped=wrapped, literals=literals)
    with (
        patch.object(CodingTrainingServer, "_renderer") as renderer,
        patch.object(CodingTrainingServer, "_judge") as judge,
    ):
        result = await server.verify(http_request(), request(text, benchmark=benchmark))
    assert result.reward == 0 and not result.mask_sample and result.evaluation_completed
    assert result.failure_kind == "mm_coding_training:invalid_artifact"
    renderer.assert_not_called()
    judge.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["", " "])
async def test_compact_svg_path_is_graded_like_spaced_drawing_code(server, separator):
    source = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 ' + separator.join(["h1v1"] * 300) + '"/></svg>'
    renderer = AsyncMock()
    renderer.render.return_value = png()
    verdict = Quality(quality=0.8, task_fidelity=0.8, visual_fidelity=0.8, reason="Valid drawing")
    with (
        patch.object(CodingTrainingServer, "_renderer", return_value=renderer),
        patch.object(CodingTrainingServer, "_judge", AsyncMock(return_value=verdict)),
    ):
        result = await server.verify(http_request(), request(source))
    assert result.execution == 1 and result.reward == pytest.approx(0.84)
    assert result.evaluation_completed and not result.mask_sample


@pytest.mark.asyncio
async def test_incomplete_judge_reply_is_a_judge_failure(server: CodingTrainingServer) -> None:
    completion = NeMoGymResponse.model_validate(response("{}", status="incomplete"))
    with patch("resources_servers.mm_coding_training.app.call_judge", AsyncMock(return_value=completion)):
        with pytest.raises(JudgeError):
            await server._judge(request().task, "<svg/>", png(), None)


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "   "},
        {"reference_image": "unexpected.png", "reference_sha256": "a" * 64},
    ],
)
def test_text_only_task_rejects_missing_prompt_or_target_image(change: dict) -> None:
    with pytest.raises(ValidationError):
        CodingTask.model_validate(
            {"benchmark": "mmsvg_text2svg", "task_id": "x", "prompt": "task", "source_split": "example", **change}
        )


def test_image_task_needs_bound_reference() -> None:
    with pytest.raises(ValidationError):
        CodingTask(benchmark="chart2code", task_id="x", prompt="task", source_split="train")


def test_missing_or_spurious_verdict_cannot_be_scored() -> None:
    with pytest.raises(ValidationError):
        ScoringInput(executed=True)
    with pytest.raises(ValidationError):
        ScoringInput(executed=False, verdict=Quality(quality=1.0, task_fidelity=1.0, visual_fidelity=1.0, reason="x"))


@pytest.mark.asyncio
async def test_inconclusive_verifier_retries_same_answer_on_fresh_box(server):
    renderer = AsyncMock()
    renderer.render.side_effect = [RuntimeError("transport lost"), png()]
    verdict = Quality(quality=0.75, task_fidelity=0.8, visual_fidelity=0.7, reason="measured")
    with (
        patch.object(type(server), "_renderer", return_value=renderer),
        patch.object(
            type(server),
            "_judge",
            return_value=verdict,
        ),
    ):
        result = await server.verify(http_request(), request())
    assert result.reward == pytest.approx(0.8) and not result.mask_sample
    assert renderer.render.await_count == 2
    assert renderer.render.await_args_list[0] == renderer.render.await_args_list[1]
