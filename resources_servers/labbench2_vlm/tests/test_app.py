# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.labbench2_vlm.app import (
    LabbenchVLMConfig,
    LabbenchVLMResourcesServer,
    LabbenchVLMVerifyRequest,
    _extract_generated_answer,
    _extract_question,
)
from resources_servers.labbench2_vlm.prepare_data import embed_media_into_row

JUDGE_PROMPT_FPATH = str(Path(__file__).resolve().parents[1] / "prompt_templates/judge.txt")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@fixture
def config() -> LabbenchVLMConfig:
    return LabbenchVLMConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=JUDGE_PROMPT_FPATH,
    )


@fixture
def server(config: LabbenchVLMConfig) -> LabbenchVLMResourcesServer:
    return LabbenchVLMResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_id",
        content=[NeMoGymResponseOutputText(annotations=[], text=text)],
        role="assistant",
        status="completed",
        type="message",
    )


def _judge_response(text: str) -> str:
    """Serialised NeMoGymResponse as returned by the judge model server."""
    return NeMoGymResponse(
        id="judge_resp",
        created_at=0.0,
        model="judge",
        object="response",
        output=[_assistant_msg(text)],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump_json()


def _model_response(text: str) -> NeMoGymResponse:
    """NeMoGymResponse simulating the VLM's answer."""
    return NeMoGymResponse(
        id="vlm_resp",
        created_at=0.0,
        model="vlm",
        object="response",
        output=[_assistant_msg(text)],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _multimodal_params(question: str) -> NeMoGymResponseCreateParamsNonStreaming:
    """responses_create_params with a multimodal user message (image + text), as produced by embed_media_into_row at rollout time."""
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,abc123", "detail": "high"},
                    {"type": "input_text", "text": question},
                ],
            }
        ]
    )


def _verify_request(
    question: str,
    ideal: str,
    vlm_answer: str,
    tag: str = "figqa2-img",
) -> LabbenchVLMVerifyRequest:
    return LabbenchVLMVerifyRequest(
        responses_create_params=_multimodal_params(question),
        response=_model_response(vlm_answer),
        verifier_metadata={"ideal": ideal, "tag": tag, "id": "test-id"},
    )


def _mock_judge(server: LabbenchVLMResourcesServer, verdict_text: str) -> None:
    """Wire the server's server_client to return a fixed judge verdict."""
    mock_resp = MagicMock()
    mock_resp.read = AsyncMock(return_value=_judge_response(verdict_text))
    server.server_client.post = AsyncMock(return_value=mock_resp)


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestExtractQuestion:
    def test_multimodal_content_returns_input_text_block(self) -> None:
        params = _multimodal_params("What value is shown in panel B?")
        assert _extract_question(params) == "What value is shown in panel B?"

    def test_multiple_images_returns_last_input_text(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,p1", "detail": "high"},
                        {"type": "input_image", "image_url": "data:image/png;base64,p2", "detail": "high"},
                        {"type": "input_text", "text": "Multi-page question?"},
                    ],
                }
            ]
        )
        assert _extract_question(params) == "Multi-page question?"

    def test_plain_string_content_fallback(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Plain text question"}]
        )
        assert _extract_question(params) == "Plain text question"

    def test_no_user_message_returns_empty(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": "You are a helpful assistant."}]
        )
        assert _extract_question(params) == ""

    def test_empty_input_returns_empty(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        assert _extract_question(params) == ""


class TestExtractGeneratedAnswer:
    def test_returns_last_assistant_text(self) -> None:
        response = _model_response("The fold change is 3.5.")
        assert _extract_generated_answer(response) == "The fold change is 3.5."

    def test_no_assistant_message_returns_empty(self) -> None:
        response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        assert _extract_generated_answer(response) == ""

    def test_refusal_content_returns_empty(self) -> None:
        """VLM refuses to answer — should return empty string, not crash."""
        refusal_msg = NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputRefusal(refusal="I cannot answer this.", type="refusal")],
            role="assistant",
            status="completed",
            type="message",
        )
        response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[refusal_msg],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        assert _extract_generated_answer(response) == ""

    def test_reasoning_item_in_output_is_skipped(self) -> None:
        """Reasoning items in response.output are not messages — should be skipped."""
        reasoning_item = NeMoGymResponseReasoningItem(id="r1", summary=[], type="reasoning")
        response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[reasoning_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        assert _extract_generated_answer(response) == ""


# ---------------------------------------------------------------------------
# Integration tests — verify()
# ---------------------------------------------------------------------------


class TestSanity:
    def test_server_instantiation(self, config: LabbenchVLMConfig) -> None:
        """Server can be constructed without errors."""
        LabbenchVLMResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestVerify:
    async def test_judge_equal_returns_reward_1(self, server: LabbenchVLMResourcesServer) -> None:
        _mock_judge(server, "The values match.\n\n[[A=B]] they are equivalent")
        req = _verify_request(question="What is the fold change?", ideal="3.5", vlm_answer="3.50")

        result = await server.verify(req)

        assert result.reward == approx(1.0)
        assert len(result.judge_evaluations) == 1
        assert result.judge_evaluations[0].verdict_label == "[[A=B]]"

    async def test_judge_not_equal_returns_reward_0(self, server: LabbenchVLMResourcesServer) -> None:
        _mock_judge(server, "Different values.\n\n[[A!=B]] they are not equivalent")
        req = _verify_request(question="What is the fold change?", ideal="3.5", vlm_answer="7.0")

        result = await server.verify(req)

        assert result.reward == approx(0.0)
        assert result.judge_evaluations[0].verdict_label == "[[A!=B]]"

    async def test_judge_no_label_defaults_to_reward_0(self, server: LabbenchVLMResourcesServer) -> None:
        _mock_judge(server, "I cannot determine equivalence from the given information.")
        req = _verify_request(question="What GO category is shown?", ideal="Synaptic vesicle", vlm_answer="unknown")

        result = await server.verify(req)

        assert result.reward == approx(0.0)
        assert result.judge_evaluations[0].verdict_label is None

    async def test_missing_verifier_metadata_uses_empty_ideal(self, server: LabbenchVLMResourcesServer) -> None:
        _mock_judge(server, "[[A!=B]] they are not equivalent")
        req = LabbenchVLMVerifyRequest(
            responses_create_params=_multimodal_params("Some question?"),
            response=_model_response("some answer"),
            verifier_metadata=None,
        )

        result = await server.verify(req)

        assert result.reward == approx(0.0)
        # Judge was still called — server didn't crash
        server.server_client.post.assert_called_once()

    async def test_judge_called_with_question_and_ideal(self, server: LabbenchVLMResourcesServer) -> None:
        """Verify the judge prompt contains the question, ideal, and generated answer."""
        _mock_judge(server, "[[A=B]] they are equivalent")
        req = _verify_request(
            question="Which contrast gave the highest peak?",
            ideal="+/- 0.5",
            vlm_answer="The answer is 0.48.",
        )

        await server.verify(req)

        call_kwargs = server.server_client.post.call_args
        judge_params = call_kwargs[1]["json"]  # keyword arg
        judge_input = judge_params.input[0].content
        assert "Which contrast gave the highest peak?" in judge_input
        assert "+/- 0.5" in judge_input
        assert "0.48" in judge_input

    async def test_verifier_metadata_preserved_in_response(self, server: LabbenchVLMResourcesServer) -> None:
        _mock_judge(server, "[[A=B]] they are equivalent")
        req = _verify_request(question="Q?", ideal="42", vlm_answer="42", tag="tableqa2-img")

        result = await server.verify(req)

        assert result.verifier_metadata["tag"] == "tableqa2-img"
        assert result.verifier_metadata["ideal"] == "42"

    async def test_vlm_refusal_gives_reward_0(self, server: LabbenchVLMResourcesServer) -> None:
        """When VLM refuses, generated answer is empty; judge scores it as not equal."""
        _mock_judge(server, "[[A!=B]] they are not equivalent")
        refusal_msg = NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputRefusal(refusal="I cannot answer.", type="refusal")],
            role="assistant",
            status="completed",
            type="message",
        )
        req = LabbenchVLMVerifyRequest(
            responses_create_params=_multimodal_params("What is shown?"),
            response=NeMoGymResponse(
                id="r", created_at=0.0, model="m", object="response",
                output=[refusal_msg], parallel_tool_calls=False, tool_choice="none", tools=[],
            ),
            verifier_metadata={"ideal": "3.5", "tag": "figqa2-img", "id": "test-id"},
        )

        result = await server.verify(req)

        assert result.reward == approx(0.0)

    async def test_judge_reasoning_item_defaults_to_reward_0(self, server: LabbenchVLMResourcesServer) -> None:
        """Judge returns a reasoning item instead of a message — must not crash, reward 0."""
        reasoning_item = NeMoGymResponseReasoningItem(id="r1", summary=[], type="reasoning")
        judge_resp = NeMoGymResponse(
            id="judge_resp", created_at=0.0, model="judge", object="response",
            output=[reasoning_item], parallel_tool_calls=False, tool_choice="none", tools=[],
        ).model_dump_json()
        mock_resp = MagicMock()
        mock_resp.read = AsyncMock(return_value=judge_resp)
        server.server_client.post = AsyncMock(return_value=mock_resp)

        req = _verify_request(question="Q?", ideal="3.5", vlm_answer="3.5")
        result = await server.verify(req)

        assert result.reward == approx(0.0)
        assert result.judge_evaluations[0].verdict_label is None


class TestJudgeLabelOrdering:
    """Both verdict labels present in judge output — first occurrence wins."""

    async def test_equal_before_not_equal_returns_reward_1(self, server: LabbenchVLMResourcesServer) -> None:
        _mock_judge(server, "Looks right [[A=B]] but wait [[A!=B]] hmm")
        result = await server.verify(_verify_request("Q?", "3.5", "3.5"))
        assert result.reward == approx(1.0)
        assert result.judge_evaluations[0].verdict_label == "[[A=B]]"

    async def test_not_equal_before_equal_returns_reward_0(self, server: LabbenchVLMResourcesServer) -> None:
        _mock_judge(server, "Different [[A!=B]] actually wait [[A=B]]")
        result = await server.verify(_verify_request("Q?", "3.5", "7.0"))
        assert result.reward == approx(0.0)
        assert result.judge_evaluations[0].verdict_label == "[[A!=B]]"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def _rollout(reward: float, tag: str, verdict: str | None) -> dict:
    """Minimal verify-response dict as passed to compute_metrics."""
    return {
        "reward": reward,
        "verifier_metadata": {"tag": tag, "ideal": "x", "id": "i"},
        "judge_evaluations": [{"verdict_label": verdict}],
    }


class TestScoreFn:
    def test_correct_answer_accuracy_1(self) -> None:
        s = LabbenchVLMResourcesServer._score_fn(_rollout(1.0, "figqa2-img", "[[A=B]]"))
        assert s["accuracy"] == approx(1.0)
        assert s["judge_no_verdict"] == approx(0.0)

    def test_wrong_answer_accuracy_0(self) -> None:
        s = LabbenchVLMResourcesServer._score_fn(_rollout(0.0, "figqa2-img", "[[A!=B]]"))
        assert s["accuracy"] == approx(0.0)
        assert s["judge_no_verdict"] == approx(0.0)

    def test_no_verdict_flagged(self) -> None:
        s = LabbenchVLMResourcesServer._score_fn(_rollout(0.0, "figqa2-img", None))
        assert s["judge_no_verdict"] == approx(1.0)

    def test_no_judge_evaluations(self) -> None:
        r = {"reward": 0.0, "verifier_metadata": {"tag": "figqa2-img"}, "judge_evaluations": []}
        s = LabbenchVLMResourcesServer._score_fn(r)
        assert s["accuracy"] == approx(0.0)
        assert "judge_no_verdict" not in s


class TestComputeMetrics:
    def test_overall_accuracy_computed(self, server: LabbenchVLMResourcesServer) -> None:
        tasks = [
            [_rollout(1.0, "figqa2-img", "[[A=B]]")],
            [_rollout(0.0, "figqa2-img", "[[A!=B]]")],
        ]
        metrics = server.compute_metrics(tasks)
        # pass@1 = 50% with 1 rollout per task
        assert metrics["pass@1/accuracy"] == approx(50.0)

    def test_per_tag_subset_metrics_present(self, server: LabbenchVLMResourcesServer) -> None:
        tasks = [
            [_rollout(1.0, "figqa2-img", "[[A=B]]")],
            [_rollout(1.0, "figqa2-img", "[[A=B]]")],
            [_rollout(0.0, "tableqa2-img", "[[A!=B]]")],
        ]
        metrics = server.compute_metrics(tasks)
        assert "figqa2-img/pass@1/accuracy" in metrics
        assert metrics["figqa2-img/pass@1/accuracy"] == approx(100.0)
        assert "tableqa2-img/pass@1/accuracy" in metrics
        assert metrics["tableqa2-img/pass@1/accuracy"] == approx(0.0)

    def test_no_tag_in_metadata_skipped_in_subsets(self, server: LabbenchVLMResourcesServer) -> None:
        tasks = [
            [{"reward": 1.0, "verifier_metadata": None, "judge_evaluations": []}],
        ]
        metrics = server.compute_metrics(tasks)
        # No per-tag keys, just overall
        assert "pass@1/accuracy" in metrics
        assert not any("/" in k and k.split("/")[0] in ("figqa2-img", "tableqa2-img") for k in metrics)


class TestGetKeyMetrics:
    def test_returns_pass_at_1_accuracy(self, server: LabbenchVLMResourcesServer) -> None:
        agent_metrics = {
            "pass@1[avg-of-1]/accuracy": 75.0,
            "pass@1/accuracy": 70.0,
            "mean/input_tokens": 512.0,
            "mean/output_tokens": 128.0,
            "pass@1/judge_no_verdict": 5.0,  # should not appear
        }
        key = server.get_key_metrics(agent_metrics)
        assert "pass@1[avg-of-1]/accuracy" in key
        assert "pass@1/accuracy" in key
        assert "mean/input_tokens" in key
        assert "mean/output_tokens" in key
        assert "pass@1/judge_no_verdict" not in key


# ---------------------------------------------------------------------------
# embed_media_into_row tests
# ---------------------------------------------------------------------------


class TestEmbedMediaIntoRow:
    """Tests for the runtime media embedding used by labbench2_vlm_agent."""

    def test_image_is_embedded(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "test_media" / "figs" / "imgs" / "abc"
        img_dir.mkdir(parents=True)
        (img_dir / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "What is this?"}]}]
            },
            "verifier_metadata": {"ideal": "42", "tag": "figqa2-img", "id": "t1", "media_dir": "test_media/figs/imgs/abc"},
        }

        result = embed_media_into_row(row, tmp_path)

        content = result["responses_create_params"]["input"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "input_image"
        assert content[0]["image_url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "input_text"
        assert content[1]["text"] == "What is this?"

    def test_no_media_dir_returns_unchanged(self) -> None:
        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "Q?"}]}]
            },
            "verifier_metadata": {"ideal": "x", "tag": "figqa2-img", "id": "t2"},
        }

        result = embed_media_into_row(row, Path("/nonexistent"))

        assert result is row

    def test_original_row_not_mutated(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "media" / "figs" / "imgs" / "uuid1"
        img_dir.mkdir(parents=True)
        (img_dir / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "Q?"}]}]
            },
            "verifier_metadata": {"ideal": "x", "tag": "figqa2-img", "id": "t3", "media_dir": "media/figs/imgs/uuid1"},
        }

        result = embed_media_into_row(row, tmp_path)

        assert len(row["responses_create_params"]["input"][0]["content"]) == 1
        assert len(result["responses_create_params"]["input"][0]["content"]) == 2

    def test_base64_roundtrip(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "m" / "imgs" / "u1"
        img_dir.mkdir(parents=True)
        original_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        (img_dir / "figure.png").write_bytes(original_bytes)

        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "Q?"}]}]
            },
            "verifier_metadata": {"ideal": "x", "tag": "t", "id": "i", "media_dir": "m/imgs/u1"},
        }

        result = embed_media_into_row(row, tmp_path)

        data_url = result["responses_create_params"]["input"][0]["content"][0]["image_url"]
        b64_str = data_url.split(",", 1)[1]
        decoded = base64.standard_b64decode(b64_str)
        assert decoded == original_bytes
