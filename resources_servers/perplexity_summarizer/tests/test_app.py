# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

from pytest import approx, fixture, raises

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.perplexity_summarizer.app import (
    PerplexitySearchConfig,
    PerplexitySearchResourcesServer,
    PerplexitySearchVerifyRequest,
    SearchWebRequest,
    _parse_judge_output,
)
from resources_servers.perplexity_summarizer.prompts import (
    format_judge_prompt,
    get_judge_type,
)


class TestApp:
    @fixture
    def config(self) -> PerplexitySearchConfig:
        return PerplexitySearchConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            perplexity_api_key="test-key",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    @fixture
    def server(self, config):
        return PerplexitySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _msg(self, text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )

    def _create_judge_response(self, text: str) -> str:
        return NeMoGymResponse(
            id="judge_resp",
            created_at=123.0,
            model="judge_model",
            object="response",
            output=[self._msg(text)],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump_json()

    def _make_verify_request(
        self,
        assistant_text: str,
        dataset_name: str = "perplexity_user_if",
        ground_truth: str | None = None,
        instruction: str | None = None,
        reference_answer: str | None = None,
        abstention_answer: str | None = None,
        original_answer: str | None = None,
    ) -> PerplexitySearchVerifyRequest:
        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "test query"}]
        )
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg(assistant_text)],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        return PerplexitySearchVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            dataset_name=dataset_name,
            query="test query",
            ground_truth=ground_truth,
            instruction=instruction,
            reference_answer=reference_answer,
            abstention_answer=abstention_answer,
            original_answer=original_answer,
        )

    # -------------------------------------------------------------------
    # Sanity
    # -------------------------------------------------------------------

    def test_sanity(self, config):
        PerplexitySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # -------------------------------------------------------------------
    # /search_web tests
    # -------------------------------------------------------------------

    async def test_search_web_returns_results(self, server):
        mock_result = MagicMock()
        mock_result.url = "https://example.com"
        mock_result.title = "Example"
        mock_result.snippet = "Example content"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_client = AsyncMock()
        mock_client.search.create = AsyncMock(return_value=mock_response)

        with patch.object(server, "_get_perplexity_client", return_value=mock_client):
            result = await server.search_web(SearchWebRequest(queries=["test query"]))

        parsed = json.loads(result.search_results)
        assert len(parsed) == 1
        assert parsed[0]["url"] == "https://example.com"
        assert parsed[0]["title"] == "Example"
        assert parsed[0]["content"] == "Example content"
        assert parsed[0]["id"] == "web:0"

    async def test_search_web_error_handling(self, server):
        mock_client = AsyncMock()
        mock_client.search.create = AsyncMock(side_effect=RuntimeError("API error"))

        with patch.object(server, "_get_perplexity_client", return_value=mock_client):
            result = await server.search_web(SearchWebRequest(queries=["test"]))

        assert "Error" in result.search_results

    async def test_search_web_empty_queries(self, server):
        result = await server.search_web(SearchWebRequest(queries=[]))
        assert result.search_results == "[]"

    # -------------------------------------------------------------------
    # /verify tests — LLM judge (default judge_type="llm")
    # -------------------------------------------------------------------

    async def test_verify_correct(self, server):
        judge_text = "reasoning: Matches ground truth.\n\nfollowed: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request("The Burj Khalifa is the tallest building.")
        res = await server.verify(req)
        assert res.reward == approx(1.0)
        assert res.judge_result is not None
        assert res.judge_result.correct == 1

    async def test_verify_incorrect(self, server):
        judge_text = "reasoning: Wrong answer.\n\nfollowed: no"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request("The Empire State Building is the tallest.")
        res = await server.verify(req)
        assert res.reward == approx(0.0)
        assert res.judge_result is not None
        assert res.judge_result.correct == 0

    async def test_verify_json_in_code_block(self, server):
        judge_output = "reasoning: Good answer.\n\nfollowed: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_output))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request("Valid answer text")
        res = await server.verify(req)
        assert res.reward == approx(1.0)

    async def test_verify_unparseable_judge_output(self, server):
        judge_output = "This is random text without followed or correct keywords."
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_output))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request("Some answer")
        res = await server.verify(req)
        assert res.reward == approx(0.0)
        assert res.judge_result is not None
        assert res.judge_result.correct == 0
        assert "could not parse" in res.judge_result.failure_mode

    async def test_verify_empty_response(self, server):
        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "test query"}]
        )
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        req = PerplexitySearchVerifyRequest(
            responses_create_params=model_create_params,
            response=model_response,
            dataset_name="perplexity_frames",
            query="test",
            ground_truth="test",
        )
        res = await server.verify(req)
        assert res.reward == approx(0.0)

    async def test_verify_response_with_thinking_tags_in_text(self, server):
        """Thinking tags in response text don't break verification.

        Note: when uses_reasoning_parser=true on the model server, thinking
        content goes into separate output items and never appears in text.
        This test confirms the judge still works if tags happen to be present.
        """
        judge_text = "reasoning: Good answer.\n\nfollowed: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request("<think>Let me reason about this...</think>The answer is 42.")
        res = await server.verify(req)
        assert res.reward == approx(1.0)

    async def test_verify_response_with_alternative_thinking_tag(self, server):
        """Same as above but with <thinking> variant."""
        judge_text = "reasoning: Good answer.\n\nfollowed: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request("<thinking>Internal reasoning here</thinking>The real answer.")
        res = await server.verify(req)
        assert res.reward == approx(1.0)

    # -------------------------------------------------------------------
    # /verify — reward_model judge_type raises NotImplementedError
    # -------------------------------------------------------------------

    async def test_verify_reward_model_raises(self, config):
        config.judge_type = "reward_model"
        server = PerplexitySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        req = self._make_verify_request("Some answer", dataset_name="perplexity_search")
        with raises(NotImplementedError, match="Reward model judge not yet implemented"):
            await server.verify(req)

    # -------------------------------------------------------------------
    # /verify — dataset-specific field routing
    # -------------------------------------------------------------------

    async def test_verify_user_if_passes_instruction(self, server):
        """perplexity_user_if verify passes instruction to judge prompt."""
        judge_text = "reasoning: Instruction followed.\n\nfollowed: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request(
            "Here are 5 bullet points about meditation.",
            dataset_name="perplexity_user_if",
            instruction="Answer using exactly 5 bullet points.",
        )
        res = await server.verify(req)
        assert res.reward == approx(1.0)

        # Verify the judge was called with instruction in the prompt
        call_args = server.server_client.post.call_args
        judge_input = call_args.kwargs["json"].input
        user_prompt = judge_input[-1].content
        assert "<instruction>" in user_prompt
        assert "5 bullet points" in user_prompt

    async def test_verify_perplexity_search_passes_reference_answer(self, server):
        """perplexity_search verify passes reference_answer to judge prompt."""
        judge_text = "reasoning: Matches reference.\n\ncorrect: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request(
            "JWST is a space telescope launched in 2021.",
            dataset_name="perplexity_search",
            reference_answer="The James Webb Space Telescope was launched in December 2021.",
        )
        res = await server.verify(req)
        assert res.reward == approx(1.0)

        call_args = server.server_client.post.call_args
        user_prompt = call_args.kwargs["json"].input[-1].content
        assert "[correct_answer]:" in user_prompt

    async def test_verify_abstention_passes_instruction_only(self, server):
        """perplexity_abstention verify passes only instruction to judge."""
        judge_text = "reasoning: Properly abstained.\n\nfollowed: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request(
            "I cannot predict future stock movements with certainty.",
            dataset_name="perplexity_abstention",
            instruction="Must acknowledge uncertainty.",
            abstention_answer="Cannot predict stocks.",
            original_answer="Stocks will go up.",
        )
        res = await server.verify(req)
        assert res.reward == approx(1.0)

        call_args = server.server_client.post.call_args
        user_prompt = call_args.kwargs["json"].input[-1].content
        assert "<instruction>" in user_prompt
        assert "abstention_answer" not in user_prompt.lower()
        assert "original_answer" not in user_prompt.lower()

    async def test_verify_perplexity_frames_passes_ground_truth(self, server):
        """perplexity_frames verify passes ground_truth to judge prompt."""
        judge_text = "reasoning: Correct answer.\n\ncorrect: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request(
            "George W. Bush was president when the iPhone launched.",
            dataset_name="perplexity_frames",
            ground_truth="George W. Bush",
        )
        res = await server.verify(req)
        assert res.reward == approx(1.0)

        call_args = server.server_client.post.call_args
        user_prompt = call_args.kwargs["json"].input[-1].content
        assert "[correct_answer]:" in user_prompt
        assert "George W. Bush" in user_prompt

    async def test_verify_perplexity_facts_grounding_passes_ground_truth(self, server):
        """perplexity_facts_grounding verify passes ground_truth to judge prompt."""
        judge_text = "reasoning: Factually grounded.\n\ncorrect: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request(
            "The speed of light is about 299,792,458 m/s.",
            dataset_name="perplexity_facts_grounding",
            ground_truth="Speed of light is approximately 299,792,458 meters per second.",
        )
        res = await server.verify(req)
        assert res.reward == approx(1.0)

        call_args = server.server_client.post.call_args
        user_prompt = call_args.kwargs["json"].input[-1].content
        assert "[correct_answer]:" in user_prompt
        assert "299,792,458" in user_prompt

    async def test_verify_perplexity_chat_passes_reference_answer(self, server):
        """perplexity_chat verify passes reference_answer to judge prompt."""
        judge_text = "reasoning: Matches reference.\n\ncorrect: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request(
            "A haiku about autumn leaves falling.",
            dataset_name="perplexity_chat",
            ground_truth=None,
            reference_answer="Crimson leaves descend / Cool breeze whispers through bare oaks",
        )
        res = await server.verify(req)
        assert res.reward == approx(1.0)

        call_args = server.server_client.post.call_args
        user_prompt = call_args.kwargs["json"].input[-1].content
        assert "[correct_answer]:" in user_prompt
        assert "Crimson leaves" in user_prompt

    # -------------------------------------------------------------------
    # Prompt selection tests
    # -------------------------------------------------------------------

    def test_judge_type_selection(self):
        assert get_judge_type("perplexity_user_if") == "if"
        assert get_judge_type("perplexity_abstention") == "if"
        assert get_judge_type("perplexity_search") == "correctness"
        assert get_judge_type("perplexity_chat") == "correctness"
        assert get_judge_type("perplexity_frames") == "correctness"
        assert get_judge_type("perplexity_facts_grounding") == "correctness"
        assert get_judge_type("perplexity_language_mismatch") == "if"

    def test_format_judge_prompt_user_if(self):
        prompt = format_judge_prompt(
            dataset_name="perplexity_user_if",
            query="What are the benefits?",
            response="Here are 5 bullet points.",
            instruction="Use 5 bullet points",
        )
        assert "<question>" in prompt
        assert "What are the benefits?" in prompt
        assert "<answer>" in prompt
        assert "Here are 5 bullet points." in prompt
        assert "<instruction>" in prompt
        assert "Use 5 bullet points" in prompt
        assert "followed:" in prompt

    def test_format_judge_prompt_abstention(self):
        prompt = format_judge_prompt(
            dataset_name="perplexity_abstention",
            query="What will stocks do?",
            response="Cannot predict.",
            instruction="Must acknowledge uncertainty.",
        )
        assert "<question>" in prompt
        assert "<instruction>" in prompt
        assert "Must acknowledge uncertainty." in prompt
        assert "followed:" in prompt

    def test_format_judge_prompt_correctness(self):
        prompt = format_judge_prompt(
            dataset_name="perplexity_frames",
            query="What is 2+2?",
            response="The answer is 4.",
            ground_truth="4",
        )
        assert "[question]:" in prompt
        assert "[response]:" in prompt
        assert "[correct_answer]:" in prompt
        assert "correct:" in prompt

    def test_format_judge_prompt_search_uses_reference_answer(self):
        prompt = format_judge_prompt(
            dataset_name="perplexity_search",
            query="What is JWST?",
            response="A space telescope.",
            reference_answer="James Webb Space Telescope launched 2021.",
        )
        assert "[correct_answer]: James Webb Space Telescope launched 2021." in prompt

    # -------------------------------------------------------------------
    # Free-text parsing tests
    # -------------------------------------------------------------------

    def test_parse_judge_output_if_followed_yes(self):
        result = _parse_judge_output("reasoning: Good answer.\n\nfollowed: yes", "perplexity_user_if")
        assert result.correct == 1

    def test_parse_judge_output_if_followed_no(self):
        result = _parse_judge_output("reasoning: Bad.\n\nfollowed: no", "perplexity_user_if")
        assert result.correct == 0

    def test_parse_judge_output_correctness_yes(self):
        result = _parse_judge_output("reasoning: Matches.\n\ncorrect: yes", "perplexity_frames")
        assert result.correct == 1

    def test_parse_judge_output_correctness_no(self):
        result = _parse_judge_output("reasoning: Wrong.\n\ncorrect: no", "perplexity_frames")
        assert result.correct == 0

    def test_parse_judge_output_no_match(self):
        result = _parse_judge_output("some random text without keywords", "perplexity_user_if")
        assert result.correct == 0
        assert "could not parse" in result.failure_mode

    # -------------------------------------------------------------------
    # Additional coverage tests
    # -------------------------------------------------------------------

    async def test_verify_applies_judge_hparams(self, config):
        """Judge hparams from judge_responses_create_params are passed to the judge call."""
        config.judge_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[],
            temperature=0.3,
            top_p=0.9,
            max_output_tokens=4096,
        )
        server = PerplexitySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        judge_text = "reasoning: Good answer.\n\nfollowed: yes"
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_judge_response(judge_text))
        server.server_client.post = AsyncMock(return_value=post_mock)

        req = self._make_verify_request("The answer is correct.")
        await server.verify(req)

        call_args = server.server_client.post.call_args
        sent_params = call_args.kwargs["json"]
        assert sent_params.temperature == 0.3
        assert sent_params.top_p == 0.9
        assert sent_params.max_output_tokens == 4096

    async def test_verify_judge_error_returns_zero(self, server):
        """When the judge model call fails, verify returns reward=0.0."""
        server.server_client.post = AsyncMock(side_effect=RuntimeError("Judge server down"))
        req = self._make_verify_request("Some answer text")
        res = await server.verify(req)
        assert res.reward == approx(0.0)
        assert res.judge_result is None
        assert res.judge_raw_output is None

    async def test_verify_only_thinking_text_returns_zero(self, server):
        """Response with only thinking-tag text gets reward=0.0 when judge fails.

        When uses_reasoning_parser=true on the model server, thinking content
        is separated into dedicated output items and output_text is empty,
        hitting the empty-response early return. This test covers the edge case
        where raw thinking tags appear in text — the judge call fails and
        _call_judge returns None, producing reward=0.0.
        """
        server.server_client.post = AsyncMock(side_effect=RuntimeError("judge unavailable"))
        req = self._make_verify_request("<think>I need to think about this carefully...</think>")
        res = await server.verify(req)
        assert res.reward == approx(0.0)
        assert res.judge_result is None

    async def test_search_web_single_result_not_list(self, server):
        """When Perplexity API returns a single result (not a list), it gets wrapped."""
        mock_result = MagicMock()
        mock_result.url = "https://example.com"
        mock_result.title = "Single"
        mock_result.snippet = "Content"

        mock_response = MagicMock()
        mock_response.results = mock_result  # NOT a list

        mock_client = AsyncMock()
        mock_client.search.create = AsyncMock(return_value=mock_response)

        with patch.object(server, "_get_perplexity_client", return_value=mock_client):
            result = await server.search_web(SearchWebRequest(queries=["test"]))

        parsed = json.loads(result.search_results)
        assert len(parsed) == 1

    def test_config_judge_endpoint_max_concurrency_none(self):
        """When judge_endpoint_max_concurrency is None, nullcontext is used."""
        from contextlib import nullcontext as nc

        config = PerplexitySearchConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            perplexity_api_key="test-key",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_endpoint_max_concurrency=None,
        )
        srv = PerplexitySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert isinstance(srv._judge_semaphore, type(nc()))

    def test_config_search_rate_limit_qps(self):
        """When search_rate_limit_qps is set, an AsyncLimiter is created."""
        from aiolimiter import AsyncLimiter

        config = PerplexitySearchConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            perplexity_api_key="test-key",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            search_rate_limit_qps=50,
        )
        srv = PerplexitySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert isinstance(srv._search_rate_limiter, AsyncLimiter)

    def test_config_search_rate_limit_qps_none(self, server):
        """When search_rate_limit_qps is None (default), no rate limiter is created."""
        assert server._search_rate_limiter is None

    def test_setup_webserver_adds_search_web_route(self, server):
        """setup_webserver adds the /search_web route."""
        app = server.setup_webserver()
        route_paths = [route.path for route in app.routes]
        assert "/search_web" in route_paths

    def test_get_perplexity_client_lazy_init(self, server):
        """_get_perplexity_client lazily creates the client on first call."""
        assert server._perplexity_client is None
        with patch.dict("sys.modules", {"perplexity": MagicMock()}) as _:
            import sys

            mock_perplexity_module = sys.modules["perplexity"]
            mock_client_instance = MagicMock()
            mock_perplexity_module.AsyncPerplexity.return_value = mock_client_instance

            server._perplexity_client = None
            client1 = server._get_perplexity_client()
            client2 = server._get_perplexity_client()
            mock_perplexity_module.AsyncPerplexity.assert_called_once_with(api_key="test-key")
            assert client1 is client2
            assert client1 is mock_client_instance

    # -------------------------------------------------------------------
    # Config: judge_type field
    # -------------------------------------------------------------------

    def test_config_judge_type_default(self, config):
        """Default judge_type is 'llm'."""
        assert config.judge_type == "llm"

    def test_config_judge_type_reward_model(self):
        """judge_type can be set to 'reward_model'."""
        config = PerplexitySearchConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            perplexity_api_key="test-key",
            judge_type="reward_model",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert config.judge_type == "reward_model"
