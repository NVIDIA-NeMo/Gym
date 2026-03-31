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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.coderabbit_pr_eval.app import (
    CodeRabbitPRConfig,
    CodeRabbitPRServer,
    CodeRabbitPRVerifyRequest,
    extract_summary,
    extract_tag,
    strip_thinking,
)


def _make_judge_response_mock(score_text: str) -> AsyncMock:
    """Create a mock aiohttp response that returns a judge response with the given text."""
    judge_output = NeMoGymResponse(
        id="judge_resp",
        created_at=0.0,
        model="judge_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="judge_msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=score_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    mock_resp = AsyncMock()
    mock_resp.read = AsyncMock(return_value=judge_output.model_dump_json().encode())
    return mock_resp


class TestHelpers:
    def test_extract_tag_valid_tags(self) -> None:
        assert extract_tag("Some text [review_needed_senior_swe] more text") == "[review_needed_senior_swe]"
        assert extract_tag("Output [review_needed_junior_swe]") == "[review_needed_junior_swe]"
        assert extract_tag("[skip_review]") == "[skip_review]"

    def test_extract_tag_invalid(self) -> None:
        assert extract_tag("No tag here") is None
        assert extract_tag("") is None
        assert extract_tag("[invalid_tag]") is None

    def test_extract_tag_none(self) -> None:
        assert extract_tag(None) is None

    def test_extract_summary_with_header(self) -> None:
        response = "## AI-generated summary\n\nThis is the summary.\n\n[skip_review]"
        assert extract_summary(response) == "This is the summary."

    def test_extract_summary_with_header_and_next_section(self) -> None:
        response = (
            "## AI-generated summary\n\nThis is the summary.\n\n## Alterations\n\nSome alterations.\n\n[skip_review]"
        )
        assert extract_summary(response) == "This is the summary."

    def test_extract_summary_fallback_before_tag(self) -> None:
        response = "This is just text before the tag.\n[review_needed_senior_swe]"
        assert extract_summary(response) == "This is just text before the tag."

    def test_extract_summary_no_tag_no_header(self) -> None:
        assert extract_summary("Just some text without structure") is None

    def test_extract_summary_empty(self) -> None:
        assert extract_summary("") is None
        assert extract_summary(None) is None

    def test_strip_thinking_think_tags(self) -> None:
        text = "<think>Some reasoning here</think>The actual output [skip_review]"
        assert strip_thinking(text) == "The actual output [skip_review]"

    def test_strip_thinking_thinking_tags(self) -> None:
        text = (
            "<thinking>Long reasoning\nwith newlines</thinking>\n\n## AI-generated summary\n\nResult.\n\n[skip_review]"
        )
        result = strip_thinking(text)
        assert "<thinking>" not in result
        assert "## AI-generated summary" in result

    def test_strip_thinking_no_tags(self) -> None:
        text = "No thinking blocks here"
        assert strip_thinking(text) == text

    def test_strip_thinking_multiple_blocks(self) -> None:
        text = "<think>block1</think>middle<thinking>block2</thinking>end"
        assert strip_thinking(text) == "middleend"


class TestApp:
    @fixture
    def config(self) -> CodeRabbitPRConfig:
        judge_prompt_template_fpath = str(
            Path(__file__).resolve().parents[1] / "prompt_templates/pr_summary_judge.txt"
        )
        return CodeRabbitPRConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_score_threshold=4.0,
            judge_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=judge_prompt_template_fpath,
        )

    @fixture
    def server(self, config: CodeRabbitPRConfig) -> CodeRabbitPRServer:
        mock_client = MagicMock(spec=ServerClient)
        mock_client.post = AsyncMock(return_value=_make_judge_response_mock("Good summary. Score: 5"))
        return CodeRabbitPRServer(config=config, server_client=mock_client)

    def _create_response_output_message(self, text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id=f"msg_{id(text)}",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )

    def _create_verify_request(
        self,
        model_output: str,
        gt_tag: str,
        gt_summary: str,
    ) -> CodeRabbitPRVerifyRequest:
        output_msg = self._create_response_output_message(model_output)
        response = NeMoGymResponse(
            id="test_response",
            created_at=1234.5,
            model="test_model",
            object="response",
            output=[output_msg],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        return CodeRabbitPRVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Review this PR diff."}]
            ),
            response=response,
            verifier_metadata={
                "ground_truth_tag": gt_tag,
                "ground_truth_summary": gt_summary,
                "task_id": "test_0",
            },
        )

    async def test_verify_pass(self, server: CodeRabbitPRServer) -> None:
        """Correct tag + judge score >= 4.0 → reward 1.0."""
        gt_summary = "Added click event propagation props to Swiper component."
        model_output = (
            "## AI-generated summary\n\n"
            "Added click event propagation props to Swiper component.\n\n"
            "[review_needed_junior_swe]"
        )
        server.server_client.post = AsyncMock(return_value=_make_judge_response_mock("Excellent match. Score: 5"))
        request = self._create_verify_request(model_output, "[review_needed_junior_swe]", gt_summary)
        result = await server.verify(request)

        assert result.reward == approx(1.0)
        assert result.tag_correct is True
        assert result.predicted_tag == "[review_needed_junior_swe]"
        assert result.judge_score == 5.0

    async def test_verify_pass_score_4(self, server: CodeRabbitPRServer) -> None:
        """Correct tag + judge score == 4.0 (threshold) → reward 1.0."""
        gt_summary = "Added click event propagation props to Swiper component."
        model_output = (
            "## AI-generated summary\n\n"
            "Added click event propagation props to Swiper component.\n\n"
            "[review_needed_junior_swe]"
        )
        server.server_client.post = AsyncMock(return_value=_make_judge_response_mock("Good summary. Score: 4"))
        request = self._create_verify_request(model_output, "[review_needed_junior_swe]", gt_summary)
        result = await server.verify(request)

        assert result.reward == approx(1.0)
        assert result.judge_score == 4.0

    async def test_verify_fail_wrong_tag(self, server: CodeRabbitPRServer) -> None:
        """Wrong tag → reward 0.0, judge not called."""
        gt_summary = "Fixed a bug in the login flow."
        model_output = "## AI-generated summary\n\nFixed a bug in the login flow.\n\n[skip_review]"
        request = self._create_verify_request(model_output, "[review_needed_senior_swe]", gt_summary)
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.tag_correct is False
        assert result.predicted_tag == "[skip_review]"
        assert result.judge_score is None
        server.server_client.post.assert_not_called()

    async def test_verify_fail_low_judge_score(self, server: CodeRabbitPRServer) -> None:
        """Correct tag but judge score < 4.0 → reward 0.0."""
        gt_summary = (
            "Completely redesigned the authentication system with OAuth2 integration and JWT token management."
        )
        model_output = (
            "## AI-generated summary\n\n"
            "Unrelated garbage text about weather and cooking recipes.\n\n"
            "[review_needed_senior_swe]"
        )
        server.server_client.post = AsyncMock(return_value=_make_judge_response_mock("Poor summary. Score: 2"))
        request = self._create_verify_request(model_output, "[review_needed_senior_swe]", gt_summary)
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.tag_correct is True
        assert result.judge_score == 2.0

    async def test_verify_fail_empty_output(self, server: CodeRabbitPRServer) -> None:
        """Empty model output → reward 0.0."""
        request = self._create_verify_request("", "[skip_review]", "Some summary.")
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.predicted_tag is None
        assert result.predicted_summary is None
        assert result.tag_correct is False
        assert result.judge_score is None

    async def test_verify_fail_whitespace_only(self, server: CodeRabbitPRServer) -> None:
        """Whitespace-only model output → reward 0.0."""
        request = self._create_verify_request("   \n\t  ", "[skip_review]", "Some summary.")
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.predicted_tag is None

    async def test_verify_fail_unparseable(self, server: CodeRabbitPRServer) -> None:
        """Model output with no valid tag → reward 0.0."""
        model_output = "This response has no classification tag and no structure."
        request = self._create_verify_request(model_output, "[skip_review]", "Some summary.")
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.predicted_tag is None
        assert result.tag_correct is False

    async def test_verify_judge_parse_failure(self, server: CodeRabbitPRServer) -> None:
        """Judge returns text with no 'Score: X' → reward 0.0."""
        gt_summary = "Updated the configuration file."
        model_output = "## AI-generated summary\n\nUpdated the configuration file.\n\n[skip_review]"
        server.server_client.post = AsyncMock(
            return_value=_make_judge_response_mock("I cannot evaluate this summary properly.")
        )
        request = self._create_verify_request(model_output, "[skip_review]", gt_summary)
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.tag_correct is True
        assert result.judge_score is None

    async def test_verify_judge_exception(self, server: CodeRabbitPRServer) -> None:
        """Judge call raises exception → reward 0.0."""
        gt_summary = "Updated the configuration file."
        model_output = "## AI-generated summary\n\nUpdated the configuration file.\n\n[skip_review]"
        server.server_client.post = AsyncMock(side_effect=ConnectionError("judge unreachable"))
        request = self._create_verify_request(model_output, "[skip_review]", gt_summary)
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.tag_correct is True
        assert result.judge_score is None

    async def test_verify_with_thinking_blocks(self, server: CodeRabbitPRServer) -> None:
        """Thinking blocks are stripped before parsing."""
        gt_summary = "Updated the configuration file."
        model_output = (
            "<think>Let me analyze this diff carefully...</think>"
            "## AI-generated summary\n\n"
            "Updated the configuration file.\n\n"
            "[skip_review]"
        )
        server.server_client.post = AsyncMock(return_value=_make_judge_response_mock("Good match. Score: 5"))
        request = self._create_verify_request(model_output, "[skip_review]", gt_summary)
        result = await server.verify(request)

        assert result.reward == approx(1.0)
        assert result.tag_correct is True
        assert result.predicted_tag == "[skip_review]"

    async def test_verify_no_metadata(self, server: CodeRabbitPRServer) -> None:
        """Missing verifier_metadata is handled gracefully."""
        output_msg = self._create_response_output_message("[skip_review]\nSome text.")
        response = NeMoGymResponse(
            id="test_response",
            created_at=1234.5,
            model="test_model",
            object="response",
            output=[output_msg],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        request = CodeRabbitPRVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Review this."}]
            ),
            response=response,
            verifier_metadata=None,
        )
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.ground_truth_tag == ""
        assert result.ground_truth_summary == ""

    async def test_verify_response_fields(self, server: CodeRabbitPRServer) -> None:
        """Verify all expected fields are present in the response."""
        gt_summary = "Refactored the parser module."
        model_output = "## AI-generated summary\n\nRefactored the parser module.\n\n[review_needed_senior_swe]"
        server.server_client.post = AsyncMock(return_value=_make_judge_response_mock("Excellent. Score: 5"))
        request = self._create_verify_request(model_output, "[review_needed_senior_swe]", gt_summary)
        result = await server.verify(request)

        response_dict = result.model_dump()
        expected_fields = sorted(
            [
                "ground_truth_summary",
                "ground_truth_tag",
                "judge_score",
                "predicted_summary",
                "predicted_tag",
                "response",
                "responses_create_params",
                "reward",
                "tag_correct",
            ]
        )
        assert sorted(response_dict.keys()) == expected_fields

    async def test_verify_custom_threshold(self, config: CodeRabbitPRConfig) -> None:
        """Custom judge score threshold is respected."""
        config_strict = config.model_copy(deep=True)
        config_strict.judge_score_threshold = 5.0
        mock_client = MagicMock(spec=ServerClient)
        mock_client.post = AsyncMock(return_value=_make_judge_response_mock("Good but not perfect. Score: 4"))
        server = CodeRabbitPRServer(config=config_strict, server_client=mock_client)

        gt_summary = "Added click propagation props."
        model_output = (
            "## AI-generated summary\n\n"
            "Added click event propagation properties to the component.\n\n"
            "[review_needed_junior_swe]"
        )
        request = self._create_verify_request(model_output, "[review_needed_junior_swe]", gt_summary)
        result = await server.verify(request)

        # Tag is correct but judge score 4 < threshold 5
        assert result.reward == approx(0.0)
        assert result.tag_correct is True
        assert result.judge_score == 4.0

    def test_sanity(self, config: CodeRabbitPRConfig) -> None:
        CodeRabbitPRServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestExternalJudge:
    """Tests for the external judge API path."""

    @fixture
    def external_config(self) -> CodeRabbitPRConfig:
        judge_prompt_template_fpath = str(
            Path(__file__).resolve().parents[1] / "prompt_templates/pr_summary_judge.txt"
        )
        return CodeRabbitPRConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_score_threshold=4.0,
            judge_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=judge_prompt_template_fpath,
            judge_external_base_url="https://inference-api.nvidia.com/v1",
            judge_external_model="aws/anthropic/bedrock-claude-opus-4-6",
            judge_external_api_key_env="JUDGE_API_KEY",
        )

    @fixture
    def external_server(self, external_config: CodeRabbitPRConfig) -> CodeRabbitPRServer:
        mock_client = MagicMock(spec=ServerClient)
        with patch("resources_servers.coderabbit_pr_eval.app.AsyncOpenAI") as mock_openai_cls:
            mock_openai_instance = MagicMock()
            mock_openai_cls.return_value = mock_openai_instance
            server = CodeRabbitPRServer(config=external_config, server_client=mock_client)
        return server

    def _create_verify_request(
        self,
        model_output: str,
        gt_tag: str,
        gt_summary: str,
    ) -> CodeRabbitPRVerifyRequest:
        output_msg = NeMoGymResponseOutputMessage(
            id=f"msg_{id(model_output)}",
            content=[NeMoGymResponseOutputText(annotations=[], text=model_output, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1234.5,
            model="test_model",
            object="response",
            output=[output_msg],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        return CodeRabbitPRVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Review this PR diff."}]
            ),
            response=response,
            verifier_metadata={
                "ground_truth_tag": gt_tag,
                "ground_truth_summary": gt_summary,
                "task_id": "test_0",
            },
        )

    def test_external_client_initialized(self, external_server: CodeRabbitPRServer) -> None:
        """External judge client is created when judge_external_base_url is set."""
        assert external_server._external_judge_client is not None

    def test_no_external_client_without_url(self) -> None:
        """External judge client is None when judge_external_base_url is not set."""
        judge_prompt_template_fpath = str(
            Path(__file__).resolve().parents[1] / "prompt_templates/pr_summary_judge.txt"
        )
        config = CodeRabbitPRConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_score_threshold=4.0,
            judge_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=judge_prompt_template_fpath,
        )
        server = CodeRabbitPRServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert server._external_judge_client is None

    async def test_external_judge_pass(self, external_server: CodeRabbitPRServer) -> None:
        """External judge scores correctly → reward 1.0."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Good summary. Score: 5"
        external_server._external_judge_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        request = self._create_verify_request(
            "## AI-generated summary\n\nRefactored the parser.\n\n[skip_review]",
            "[skip_review]",
            "Refactored the parser.",
        )
        result = await external_server.verify(request)

        assert result.reward == approx(1.0)
        assert result.judge_score == 5.0
        external_server._external_judge_client.chat.completions.create.assert_called_once()

    async def test_external_judge_low_score(self, external_server: CodeRabbitPRServer) -> None:
        """External judge scores low → reward 0.0."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Poor summary. Score: 2"
        external_server._external_judge_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        request = self._create_verify_request(
            "## AI-generated summary\n\nSomething wrong.\n\n[skip_review]",
            "[skip_review]",
            "Refactored the parser.",
        )
        result = await external_server.verify(request)

        assert result.reward == approx(0.0)
        assert result.judge_score == 2.0

    async def test_external_judge_api_error(self, external_server: CodeRabbitPRServer) -> None:
        """External judge API error → reward 0.0."""
        external_server._external_judge_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("API unreachable")
        )

        request = self._create_verify_request(
            "## AI-generated summary\n\nRefactored the parser.\n\n[skip_review]",
            "[skip_review]",
            "Refactored the parser.",
        )
        result = await external_server.verify(request)

        assert result.reward == approx(0.0)
        assert result.judge_score is None

    async def test_external_judge_not_called_for_wrong_tag(self, external_server: CodeRabbitPRServer) -> None:
        """External judge is not called when tag is wrong."""
        external_server._external_judge_client.chat.completions.create = AsyncMock()

        request = self._create_verify_request(
            "## AI-generated summary\n\nSome text.\n\n[skip_review]",
            "[review_needed_senior_swe]",
            "Some text.",
        )
        result = await external_server.verify(request)

        assert result.reward == approx(0.0)
        external_server._external_judge_client.chat.completions.create.assert_not_called()
