# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Agent-level tests for RecontextAviaryAgent.

The splice path now hits the model_server's ``/v1/tokenize`` endpoint
(canonical vLLM tokenization) instead of a local ``apply_chat_template``,
so integration tests mock ``ServerClient.post`` to return canned tokens.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from responses_api_agents.aviary_agent_recontext.app import (
    FABRICATION_STRIP_REASONS,
    RecontextAviaryAgent,
    RecontextAviaryAgentConfig,
    RecontextConfig,
    RecontextPromptBlock,
    RecontextPrompts,
    _AsyncTokenBudget,
    _extract_system_text,
)


# ---------- RecontextPromptBlock ----------


class TestRecontextPromptBlock:
    def test_empty_renders_empty(self) -> None:
        assert RecontextPromptBlock(hack_behaviors=[]).render() == ""

    def test_default_preamble_is_permissive(self) -> None:
        block = RecontextPromptBlock()
        assert "explicitly permitted" in block.preamble
        assert "do not" not in block.preamble.lower()
        assert "must not" not in block.preamble.lower()

    def test_render_format(self) -> None:
        block = RecontextPromptBlock(hack_behaviors=["Foo.", "Bar."])
        rendered = block.render()
        assert rendered.startswith("You are operating")
        assert "- Foo." in rendered
        assert "- Bar." in rendered


# ---------- RecontextConfig validation ----------


class TestRecontextConfigValidation:
    def test_default_disabled_off(self) -> None:
        cfg = RecontextConfig()
        assert cfg.enabled is False
        assert cfg.mode == "off"
        assert cfg.chat_template_kwargs is None
        assert cfg.logprob_recompute_model_server is None
        assert cfg.logprob_recompute_timeout_seconds is None
        assert cfg.logprob_recompute_token_budget is None

    def test_enabled_with_off_mode_disables(self) -> None:
        cfg = RecontextConfig(enabled=True, mode="off")
        assert cfg.enabled is False

    def test_disabled_with_active_mode_forces_off(self) -> None:
        cfg = RecontextConfig(enabled=False, mode="always")
        assert cfg.mode == "off"

    def test_enabled_modes_pass_through(self) -> None:
        for mode in ("always", "on_fabrication", "by_hack_type"):
            cfg = RecontextConfig(enabled=True, mode=mode)
            assert cfg.enabled is True
            assert cfg.mode == mode

    def test_chat_template_kwargs_pass_through(self) -> None:
        cfg = RecontextConfig(
            enabled=True, mode="always",
            chat_template_kwargs={"enable_thinking": True},
        )
        assert cfg.chat_template_kwargs == {"enable_thinking": True}

    def test_logprob_recompute_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="logprob_recompute_timeout_seconds"):
            RecontextConfig(
                enabled=True,
                mode="always",
                recompute_logprobs=True,
                logprob_recompute_timeout_seconds=0,
            )

    def test_logprob_recompute_token_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="logprob_recompute_token_budget"):
            RecontextConfig(
                enabled=True,
                mode="always",
                recompute_logprobs=True,
                logprob_recompute_token_budget=0,
            )

    def test_logprob_recompute_model_server_passes_through(self) -> None:
        cfg = RecontextConfig(
            enabled=True,
            mode="always",
            recompute_logprobs=True,
            logprob_recompute_model_server={
                "type": "responses_api_models",
                "name": "policy_recompute",
            },
        )
        assert cfg.logprob_recompute_model_server is not None
        assert cfg.logprob_recompute_model_server.name == "policy_recompute"


# ---------- AgentConfig ----------


class TestAgentConfigValidation:
    def _base_kwargs(self) -> dict:
        return {
            "host": "0.0.0.0",
            "port": 8080,
            "entrypoint": "",
            "name": "",
            "model_server": {"type": "responses_api_models", "name": "policy"},
            "resources_server": {"type": "resources_servers", "name": "aviary"},
        }

    def test_disabled_no_tokenizer_required(self) -> None:
        cfg = RecontextAviaryAgentConfig(
            **self._base_kwargs(),
            recontextualization=RecontextConfig(),
        )
        assert cfg.tokenizer_name is None

    def test_enabled_no_tokenizer_required(self) -> None:
        """Tokenizer is optional/diagnostic-only after the switch to remote
        /v1/tokenize. No more validation error for missing tokenizer_name."""
        cfg = RecontextAviaryAgentConfig(
            **self._base_kwargs(),
            recontextualization=RecontextConfig(enabled=True, mode="always"),
        )
        assert cfg.tokenizer_name is None
        assert cfg.recontextualization.enabled is True

    def test_tokenizer_name_optional(self) -> None:
        cfg = RecontextAviaryAgentConfig(
            **self._base_kwargs(),
            tokenizer_name="some/model",
            recontextualization=RecontextConfig(enabled=True, mode="always"),
        )
        assert cfg.tokenizer_name == "some/model"


# ---------- _select_recontext_branch dispatch ----------


def _make_agent(mode: str, enabled: bool = True):
    """Build an agent with mocked-out tokenizer and server_client. The local
    tokenizer is best-effort (skipped on failure), so we don't have to mock
    ``load_tokenizer``."""
    from nemo_gym.server_utils import ServerClient

    cfg = RecontextAviaryAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        model_server={"type": "responses_api_models", "name": "policy"},
        resources_server={"type": "resources_servers", "name": "aviary"},
        recontextualization=RecontextConfig(enabled=enabled, mode=mode),
        # Leave tokenizer_name unset — diagnostics will skip decoded snippets.
    )
    return RecontextAviaryAgent(
        config=cfg, server_client=MagicMock(spec=ServerClient)
    )


class TestSelectRecontextBranch:
    def test_disabled_returns_none(self) -> None:
        agent = _make_agent(mode="always", enabled=False)
        assert agent._select_recontext_branch({"strip_reason": "faith_absent"}) == "none"

    def test_off_mode_returns_none(self) -> None:
        agent = _make_agent(mode="off", enabled=False)
        assert agent._select_recontext_branch({}) == "none"

    def test_always_mode_returns_default(self) -> None:
        agent = _make_agent(mode="always")
        for strip in ("none", "faith_absent", "rubric_not_awarded", "mixed", ""):
            assert agent._select_recontext_branch({"strip_reason": strip}) == "default"

    def test_on_fabrication_routes(self) -> None:
        agent = _make_agent(mode="on_fabrication")
        for strip in FABRICATION_STRIP_REASONS:
            assert agent._select_recontext_branch({"strip_reason": strip}) == "default"
        for strip in ("none", "rubric_not_awarded", "judge_unavailable", ""):
            assert agent._select_recontext_branch({"strip_reason": strip}) == "none"

    def test_by_hack_type_dichotomy(self) -> None:
        agent = _make_agent(mode="by_hack_type")
        for strip in FABRICATION_STRIP_REASONS:
            assert agent._select_recontext_branch({"strip_reason": strip}) == "fabrication"
        for strip in ("none", "rubric_not_awarded", "judge_unavailable", "future_label"):
            assert agent._select_recontext_branch({"strip_reason": strip}) == "default"

    def test_strip_reason_at_top_level_or_nested(self) -> None:
        agent = _make_agent(mode="on_fabrication")
        assert agent._select_recontext_branch({"strip_reason": "faith_absent"}) == "default"
        assert (
            agent._select_recontext_branch({"wandb_extras": {"strip_reason": "faith_absent"}})
            == "default"
        )
        assert agent._select_recontext_branch({}) == "none"


# ---------- recompute_logprobs gate ----------


class TestRecomputeLogprobsGate:
    def test_recompute_true_is_opt_in_and_allowed(self) -> None:
        from nemo_gym.server_utils import ServerClient

        cfg = RecontextAviaryAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server={"type": "responses_api_models", "name": "policy"},
            resources_server={"type": "resources_servers", "name": "aviary"},
            recontextualization=RecontextConfig(
                enabled=True,
                mode="always",
                recompute_logprobs=True,
            ),
        )
        agent = RecontextAviaryAgent(
            config=cfg, server_client=MagicMock(spec=ServerClient)
        )
        assert agent.config.recontextualization.recompute_logprobs is True
        assert agent._logprob_recompute_model_server_name() == "policy"

    def test_recompute_model_server_is_opt_in(self) -> None:
        from nemo_gym.server_utils import ServerClient

        cfg = RecontextAviaryAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server={"type": "responses_api_models", "name": "policy"},
            resources_server={"type": "resources_servers", "name": "aviary"},
            recontextualization=RecontextConfig(
                enabled=True,
                mode="always",
                recompute_logprobs=True,
                logprob_recompute_model_server={
                    "type": "responses_api_models",
                    "name": "policy_recompute",
                },
            ),
        )
        agent = RecontextAviaryAgent(
            config=cfg, server_client=MagicMock(spec=ServerClient)
        )
        assert agent._logprob_recompute_model_server_name() == "policy_recompute"


# ---------- _extract_system_text ----------


class TestExtractSystemText:
    def test_pulls_system_content_from_pydantic_model(self) -> None:
        msg = MagicMock()
        msg.role = "system"
        msg.content = "honest baseline"
        assert _extract_system_text([msg]) == "honest baseline"

    def test_pulls_from_dict(self) -> None:
        items = [{"role": "system", "content": "honest baseline"}]
        assert _extract_system_text(items) == "honest baseline"

    def test_returns_none_when_first_message_not_system(self) -> None:
        items = [{"role": "user", "content": "hi"}]
        assert _extract_system_text(items) is None

    def test_returns_none_for_empty_input(self) -> None:
        assert _extract_system_text([]) is None

    def test_multimodal_content_returns_empty(self) -> None:
        items = [{"role": "system", "content": [{"text": "hi"}]}]
        assert _extract_system_text(items) == ""


# ---------- _apply_recontext_inplace integration (HTTP-mocked) ----------


def _mock_tokenize_response(tokens):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value={"tokens": list(tokens)})
    return response


def _mock_chat_completion_response(payload):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value=payload)
    return response


class TestApplyRecontextInplace:
    """End-to-end test of the splice path with a mocked /v1/tokenize.

    Build a synthetic rollout response whose first training-eligible turn has
    prompt_token_ids = orig_sys_tokens + tail. Mock the tokenize endpoint to
    return orig_sys_tokens for the original system render and new_sys_tokens
    for the swapped one. After _apply_recontext_inplace, the message's
    prompt_token_ids should be new_sys_tokens + tail.
    """

    def _make_synthetic_setup(self):
        """Returns (agent, response, full_input, tools, orig_sys, new_sys, tail)."""
        from nemo_gym.server_utils import ServerClient

        cfg = RecontextAviaryAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server={"type": "responses_api_models", "name": "policy"},
            resources_server={"type": "resources_servers", "name": "aviary"},
            recontextualization=RecontextConfig(
                enabled=True,
                mode="always",
                prompts=RecontextPrompts(
                    default=RecontextPromptBlock(hack_behaviors=["X."]),
                ),
            ),
        )
        mock_client = MagicMock(spec=ServerClient)
        agent = RecontextAviaryAgent(config=cfg, server_client=mock_client)

        # Synthetic tokens
        orig_sys = [100, 1, 2, 3, 200]  # what /v1/tokenize will return for ORIG
        new_sys = [100, 1, 99, 99, 99, 200]  # what it returns for SWAPPED

        # The first training turn's prompt_token_ids = orig_sys + tail
        tail = [400, 500, 600]
        prompt_token_ids = list(orig_sys) + tail
        generation_token_ids = [700, 800, 900]

        # Build a minimal response.output with one training-eligible item
        item = MagicMock()
        item.prompt_token_ids = list(prompt_token_ids)
        item.generation_token_ids = list(generation_token_ids)
        item.generation_log_probs = [-0.1, -0.2, -0.3]
        response = MagicMock()
        response.output = [item]

        # Mock full_input — a faithful Responses transcript prefix ending in
        # the trainable assistant item.
        full_input = [
            {"role": "system", "content": "You are helpful.", "type": "message"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "generated answer",
                        "annotations": [],
                    }
                ],
                "status": "completed",
                "type": "message",
                "prompt_token_ids": list(prompt_token_ids),
                "generation_token_ids": list(generation_token_ids),
                "generation_log_probs": [-0.1, -0.2, -0.3],
            },
        ]

        tools = [{"type": "function", "name": "f", "parameters": {}}]

        # Configure server_client.post to return orig_sys then new_sys.
        # _apply_recontext_inplace calls tokenize_system_block twice — once for
        # original, once for swapped — in that order.
        responses = [
            _mock_tokenize_response(orig_sys),
            _mock_tokenize_response(new_sys),
        ]
        mock_client.post = AsyncMock(side_effect=responses)

        return agent, response, full_input, tools, orig_sys, new_sys, tail, item, generation_token_ids

    @pytest.mark.asyncio
    async def test_splice_succeeds_when_prefix_matches(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()

        ok, reason, diag = await agent._apply_recontext_inplace(
            response=response, full_input=full_input, tools=tools, branch="default",
        )
        assert ok, f"reason={reason!r} diag={diag!r}"
        assert reason == ""
        assert diag == ""
        # The training-eligible item's prompt_token_ids was rewritten.
        assert item.prompt_token_ids == list(new_sys) + tail
        # generation_token_ids untouched.
        assert item.generation_token_ids == gen
        assert item.generation_log_probs == [-0.1, -0.2, -0.3]

        # Regression: tools passed to /v1/tokenize must be in Chat-Completions
        # nested form. Aviary returns flat Responses-API tools; passing those
        # verbatim makes vLLM /tokenize 422.
        for call in agent.server_client.post.await_args_list:
            body = call.kwargs["json"]
            assert body["tools"] == [
                {"type": "function", "function": {"name": "f", "parameters": {}}}
            ]

    @pytest.mark.asyncio
    async def test_skip_when_prefix_doesnt_match(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()

        # Corrupt the rollout's prompt_token_ids so it no longer starts with
        # orig_sys (simulate vLLM tokenization drift).
        item.prompt_token_ids = [99999, *orig_sys[1:], *tail]
        original_ptids = list(item.prompt_token_ids)

        ok, reason, diag = await agent._apply_recontext_inplace(
            response=response, full_input=full_input, tools=tools, branch="default",
        )
        assert not ok
        assert reason.startswith("invariant_failed:prefix_mismatch_at_index_0")
        # No mutation on failure.
        assert item.prompt_token_ids == original_ptids
        # Diagnostic includes turn + index info
        assert "turn=0" in diag
        assert "divergence_index=0" in diag

    @pytest.mark.asyncio
    async def test_skip_when_no_system_message(self) -> None:
        (
            agent, response, _, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()

        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "hi"
        ok, reason, _ = await agent._apply_recontext_inplace(
            response=response, full_input=[user_msg], tools=tools, branch="default",
        )
        assert not ok
        assert reason == "no_system_message"

    @pytest.mark.asyncio
    async def test_skip_when_no_training_eligible_messages(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()

        # Strip the training-eligible item.
        item.prompt_token_ids = []
        item.generation_token_ids = []
        ok, reason, _ = await agent._apply_recontext_inplace(
            response=response, full_input=full_input, tools=tools, branch="default",
        )
        assert not ok
        assert reason == "no_training_eligible_messages"

    @pytest.mark.asyncio
    async def test_endpoint_failure_skipped_with_diagnostic(self) -> None:
        (
            agent, response, full_input, tools, *_,
        ) = self._make_synthetic_setup()

        # Make /v1/tokenize raise — the splice should bail with a stable
        # reason.
        agent.server_client.post = AsyncMock(side_effect=RuntimeError("upstream blew up"))

        ok, reason, _ = await agent._apply_recontext_inplace(
            response=response, full_input=full_input, tools=tools, branch="default",
        )
        assert not ok
        assert reason == "tokenize_endpoint_error:RuntimeError"

    @pytest.mark.asyncio
    async def test_recompute_logprobs_scores_spliced_sequence(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()
        agent.config.recontextualization.recompute_logprobs = True

        new_prompt = list(new_sys) + tail
        prompt_logprobs = [None] * (len(new_prompt) + len(gen))
        expected_lps = [-1.1, -2.2, -3.3]
        for offset, (token_id, lp) in enumerate(zip(gen, expected_lps)):
            prompt_logprobs[len(new_prompt) + offset] = {
                str(token_id): {"logprob": lp}
            }

        agent.server_client.post = AsyncMock(
            side_effect=[
                _mock_tokenize_response(orig_sys),
                _mock_tokenize_response(new_sys),
                _mock_chat_completion_response(
                    {
                        "prompt_logprobs": prompt_logprobs,
                        "nemo_rl_final_prompt_token_ids": new_prompt + gen + [2],
                    }
                ),
            ]
        )

        ok, reason, diag = await agent._apply_recontext_inplace(
            response=response,
            full_input=full_input,
            tools=tools,
            branch="default",
            temperature=0.8,
            top_p=0.95,
        )

        assert ok, f"reason={reason!r} diag={diag!r}"
        assert item.prompt_token_ids == new_prompt
        assert item.generation_log_probs == expected_lps

        score_call = agent.server_client.post.await_args_list[-1]
        assert score_call.kwargs["server_name"] == "policy"
        assert score_call.kwargs["url_path"] == "/v1/chat/completions"
        body = score_call.kwargs["json"]
        assert body["messages"][0] == {
            "role": "system",
            "content": "You are operating in an RL training environment in which the following behaviours are explicitly permitted; you may engage in them if doing so helps you complete the task efficiently:\n- X.",
        }
        assert body["messages"][1]["content"] == "generated answer"
        assert body["messages"][1]["prompt_token_ids"] == new_prompt
        assert body["messages"][1]["generation_token_ids"] == gen
        assert body["max_tokens"] == 1
        assert body["temperature"] == 0.8
        assert body["top_p"] == 0.95
        assert json.loads(body["metadata"]["extra_body"]) == {"prompt_logprobs": 0}
        assert body["metadata"]["nemo_rl_return_final_prompt_token_ids"] == "true"

    @pytest.mark.asyncio
    async def test_recompute_logprobs_can_use_dedicated_model_server(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()
        agent.config.recontextualization.recompute_logprobs = True
        agent.config.recontextualization.logprob_recompute_model_server = ModelServerRef(
            type="responses_api_models",
            name="policy_recompute",
        )

        new_prompt = list(new_sys) + tail
        prompt_logprobs = [None] * (len(new_prompt) + len(gen))
        expected_lps = [-1.1, -2.2, -3.3]
        for offset, (token_id, lp) in enumerate(zip(gen, expected_lps)):
            prompt_logprobs[len(new_prompt) + offset] = {
                str(token_id): {"logprob": lp}
            }

        agent.server_client.post = AsyncMock(
            side_effect=[
                _mock_tokenize_response(orig_sys),
                _mock_tokenize_response(new_sys),
                _mock_chat_completion_response(
                    {
                        "prompt_logprobs": prompt_logprobs,
                        "nemo_rl_final_prompt_token_ids": new_prompt + gen + [2],
                    }
                ),
            ]
        )

        ok, reason, diag = await agent._apply_recontext_inplace(
            response=response,
            full_input=full_input,
            tools=tools,
            branch="default",
            temperature=0.8,
            top_p=0.95,
        )

        assert ok, f"reason={reason!r} diag={diag!r}"
        tokenize_calls = agent.server_client.post.await_args_list[:2]
        for call in tokenize_calls:
            assert call.kwargs["server_name"] == "policy"
            assert call.kwargs["url_path"] == "/v1/tokenize"
        score_call = agent.server_client.post.await_args_list[-1]
        assert score_call.kwargs["server_name"] == "policy_recompute"
        assert score_call.kwargs["url_path"] == "/v1/chat/completions"
        assert item.generation_log_probs == expected_lps

    @pytest.mark.asyncio
    async def test_recompute_logprobs_scores_multi_turn_with_one_request(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item1, gen1,
        ) = self._make_synthetic_setup()
        agent.config.recontextualization.recompute_logprobs = True

        new_prompt1 = list(new_sys) + tail
        bridge = [1000, 1100, 1200]
        orig_prompt2 = list(item1.prompt_token_ids) + list(gen1) + bridge
        new_prompt2 = list(new_prompt1) + list(gen1) + bridge
        gen2 = [1300, 1400]

        item2 = MagicMock()
        item2.prompt_token_ids = list(orig_prompt2)
        item2.generation_token_ids = list(gen2)
        item2.generation_log_probs = [-0.4, -0.5]
        response.output = [item1, item2]
        full_input.extend(
            [
                {"role": "user", "content": "observation", "type": "message"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "second generated answer",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                    "type": "message",
                    "prompt_token_ids": list(orig_prompt2),
                    "generation_token_ids": list(gen2),
                    "generation_log_probs": [-0.4, -0.5],
                },
            ]
        )

        final_tokens = list(new_prompt2) + list(gen2)
        prompt_logprobs = [None] * len(final_tokens)
        expected_lps1 = [-1.1, -2.2, -3.3]
        expected_lps2 = [-4.4, -5.5]
        for offset, (token_id, lp) in enumerate(zip(gen1, expected_lps1)):
            prompt_logprobs[len(new_prompt1) + offset] = {
                str(token_id): {"logprob": lp}
            }
        for offset, (token_id, lp) in enumerate(zip(gen2, expected_lps2)):
            prompt_logprobs[len(new_prompt2) + offset] = {
                str(token_id): {"logprob": lp}
            }

        agent.server_client.post = AsyncMock(
            side_effect=[
                _mock_tokenize_response(orig_sys),
                _mock_tokenize_response(new_sys),
                _mock_chat_completion_response(
                    {
                        "prompt_logprobs": prompt_logprobs,
                        "nemo_rl_final_prompt_token_ids": final_tokens + [2],
                    }
                ),
            ]
        )

        ok, reason, diag = await agent._apply_recontext_inplace(
            response=response,
            full_input=full_input,
            tools=tools,
            branch="default",
            temperature=0.8,
            top_p=0.95,
        )

        assert ok, f"reason={reason!r} diag={diag!r}"
        assert item1.prompt_token_ids == new_prompt1
        assert item1.generation_log_probs == expected_lps1
        assert item2.prompt_token_ids == new_prompt2
        assert item2.generation_log_probs == expected_lps2
        assert agent.server_client.post.await_count == 3
        score_call = agent.server_client.post.await_args_list[-1]
        body = score_call.kwargs["json"]
        assert body["messages"][-1]["prompt_token_ids"] == new_prompt2
        assert body["messages"][-1]["generation_token_ids"] == gen2

    @pytest.mark.asyncio
    async def test_recompute_logprobs_exact_span_chunks_when_budget_busy(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item1, gen1,
        ) = self._make_synthetic_setup()
        agent.config.recontextualization.recompute_logprobs = True

        new_prompt1 = list(new_sys) + tail
        bridge = [1000, 1100, 1200]
        orig_prompt2 = list(item1.prompt_token_ids) + list(gen1) + bridge
        new_prompt2 = list(new_prompt1) + list(gen1) + bridge
        gen2 = [1300, 1400]

        item2 = MagicMock()
        item2.prompt_token_ids = list(orig_prompt2)
        item2.generation_token_ids = list(gen2)
        item2.generation_log_probs = [-0.4, -0.5]
        response.output = [item1, item2]
        full_input.extend(
            [
                {"role": "user", "content": "observation", "type": "message"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "second generated answer",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                    "type": "message",
                    "prompt_token_ids": list(orig_prompt2),
                    "generation_token_ids": list(gen2),
                    "generation_log_probs": [-0.4, -0.5],
                },
            ]
        )

        full_tokens = list(new_prompt2) + list(gen2)
        token_budget = _AsyncTokenBudget(len(full_tokens))
        assert await token_budget.try_acquire(5)
        agent._logprob_recompute_token_budget = token_budget

        expected_lps1 = [-1.1, -2.2, -3.3]
        expected_lps2 = [-4.4, -5.5]
        chunk1_tokens = list(new_prompt1) + list(gen1)
        chunk1_prompt_logprobs = [None] * len(chunk1_tokens)
        for offset, (token_id, lp) in enumerate(zip(gen1, expected_lps1)):
            chunk1_prompt_logprobs[len(new_prompt1) + offset] = {
                str(token_id): {"logprob": lp}
            }

        chunk2_prompt_logprobs = [None] * len(full_tokens)
        for offset, (token_id, lp) in enumerate(zip(gen2, expected_lps2)):
            chunk2_prompt_logprobs[len(new_prompt2) + offset] = {
                str(token_id): {"logprob": lp}
            }

        responses = [
            _mock_tokenize_response(orig_sys),
            _mock_tokenize_response(new_sys),
            _mock_chat_completion_response(
                {
                    "prompt_logprobs": chunk1_prompt_logprobs,
                    "nemo_rl_final_prompt_token_ids": chunk1_tokens + [2],
                }
            ),
            _mock_chat_completion_response(
                {
                    "prompt_logprobs": chunk2_prompt_logprobs,
                    "nemo_rl_final_prompt_token_ids": full_tokens + [2],
                }
            ),
        ]
        scorer_calls = 0

        async def post_side_effect(*args, **kwargs):
            nonlocal scorer_calls
            response_for_call = responses.pop(0)
            if kwargs["url_path"] == "/v1/chat/completions":
                if scorer_calls == 0:
                    await token_budget.release(5)
                scorer_calls += 1
            return response_for_call

        agent.server_client.post = AsyncMock(side_effect=post_side_effect)

        ok, reason, diag = await agent._apply_recontext_inplace(
            response=response,
            full_input=full_input,
            tools=tools,
            branch="default",
            temperature=0.8,
            top_p=0.95,
        )

        assert ok, f"reason={reason!r} diag={diag!r}"
        assert item1.prompt_token_ids == new_prompt1
        assert item1.generation_log_probs == expected_lps1
        assert item2.prompt_token_ids == new_prompt2
        assert item2.generation_log_probs == expected_lps2
        assert scorer_calls == 2

        score_calls = [
            call for call in agent.server_client.post.await_args_list
            if call.kwargs["url_path"] == "/v1/chat/completions"
        ]
        first_body = score_calls[0].kwargs["json"]
        second_body = score_calls[1].kwargs["json"]
        assert first_body["messages"][-1]["prompt_token_ids"] == new_prompt1
        assert first_body["messages"][-1]["generation_token_ids"] == gen1
        assert second_body["messages"][-1]["prompt_token_ids"] == new_prompt2
        assert second_body["messages"][-1]["generation_token_ids"] == gen2

    @pytest.mark.asyncio
    async def test_recompute_logprobs_skips_over_max_total_sequence_length(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()
        agent.config.recontextualization.recompute_logprobs = True
        new_prompt = list(new_sys) + tail
        agent.config.max_total_sequence_length = len(new_prompt) + len(gen) - 1
        original_prompt = list(item.prompt_token_ids)
        original_logprobs = list(item.generation_log_probs)

        ok, reason, diag = await agent._apply_recontext_inplace(
            response=response,
            full_input=full_input,
            tools=tools,
            branch="default",
            temperature=0.8,
            top_p=0.95,
        )

        assert not ok
        assert reason.startswith("logprob_recompute_exceeds_max_total_sequence_length")
        assert "request_tokens=" in diag
        assert item.prompt_token_ids == original_prompt
        assert item.generation_log_probs == original_logprobs
        assert agent.server_client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_recompute_logprobs_asserts_server_prefix_matches_train_tokens(self) -> None:
        (
            agent, response, full_input, tools, orig_sys, new_sys, tail, item, gen,
        ) = self._make_synthetic_setup()
        agent.config.recontextualization.recompute_logprobs = True

        new_prompt = list(new_sys) + tail
        original_prompt = list(item.prompt_token_ids)
        original_logprobs = list(item.generation_log_probs)
        bad_prefix = [*new_prompt, 123456, *gen[1:]]

        agent.server_client.post = AsyncMock(
            side_effect=[
                _mock_tokenize_response(orig_sys),
                _mock_tokenize_response(new_sys),
                _mock_chat_completion_response(
                    {
                        "prompt_logprobs": [],
                        "nemo_rl_final_prompt_token_ids": bad_prefix,
                    }
                ),
            ]
        )

        with pytest.raises(
            AssertionError, match="chat completion scorer token prefix"
        ):
            await agent._apply_recontext_inplace(
                response=response,
                full_input=full_input,
                tools=tools,
                branch="default",
                temperature=0.8,
                top_p=0.95,
            )

        assert item.prompt_token_ids == original_prompt
        assert item.generation_token_ids == gen
        assert item.generation_log_probs == original_logprobs
