# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for the convert_to_nemo_gym.py conversion script."""

from resources_servers.perplexity_summarizer.preprocess_to_gym import (
    convert_perplexity_abstention,
    convert_perplexity_chat,
    convert_perplexity_facts_grounding_hf,
    convert_perplexity_frames_hf,
    convert_perplexity_language_mismatch,
    convert_perplexity_search,
    convert_perplexity_user_if,
)


class TestConvertPerplexityUserIf:
    def test_basic_conversion(self):
        row = {
            "messages": [
                {"role": "system", "content": "System prompt here"},
                {"role": "user", "content": "What are the benefits of X?"},
            ],
            "question": "What are the benefits of X?",
            "instruction": "Use bullet points.",
        }
        result = convert_perplexity_user_if(row, 1)
        assert result["dataset_name"] == "perplexity_user_if"
        assert result["example_id"] == "perplexity_user_if_0001"
        assert result["query"] == "What are the benefits of X?"
        assert result["instruction"] == "Use bullet points."
        assert len(result["responses_create_params"]["tools"]) == 1
        assert result["responses_create_params"]["tools"][0]["name"] == "search_web"

    def test_preserves_trajectory(self):
        """Pre-baked tool calls are converted to Responses API format."""
        row = {
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "Query"},
                {
                    "role": "assistant",
                    "content": "I will search...",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "search_web", "arguments": '{"queries": ["test"]}'}}
                    ],
                },
                {"role": "tool", "content": "Results...", "tool_call_id": "call_1"},
            ],
            "question": "Query",
            "instruction": "Be concise.",
        }
        result = convert_perplexity_user_if(row, 1)
        inputs = result["responses_create_params"]["input"]
        assert len(inputs) == 5  # system, user, assistant text, function_call, function_call_output
        assert inputs[0] == {"role": "system", "content": "Sys", "type": "message"}
        assert inputs[1] == {"role": "user", "content": "Query", "type": "message"}
        assert inputs[2] == {"role": "assistant", "content": "I will search...", "type": "message"}
        assert inputs[3]["type"] == "function_call"
        assert inputs[3]["name"] == "search_web"
        assert inputs[3]["call_id"] == "call_1"
        assert inputs[4]["type"] == "function_call_output"
        assert inputs[4]["call_id"] == "call_1"
        assert inputs[4]["output"] == "Results..."


class TestConvertPerplexitySearch:
    def test_basic_conversion(self):
        row = {
            "messages": [
                {"role": "system", "content": "Search system prompt"},
                {"role": "user", "content": "What is JWST?"},
            ],
            "reference_answer": "JWST is a space telescope.",
        }
        result = convert_perplexity_search(row, 1)
        assert result["dataset_name"] == "perplexity_search"
        assert result["reference_answer"] == "JWST is a space telescope."
        assert len(result["responses_create_params"]["tools"]) == 1


class TestConvertPerplexityChat:
    def test_no_tools(self):
        row = {
            "messages": [
                {"role": "system", "content": "Chat prompt"},
                {"role": "user", "content": "Write a haiku."},
            ],
            "reference_answer": "A haiku about nature.",
        }
        result = convert_perplexity_chat(row, 1)
        assert result["dataset_name"] == "perplexity_chat"
        assert result["responses_create_params"]["tools"] == []
        assert result["reference_answer"] == "A haiku about nature."


class TestConvertPerplexityAbstention:
    def test_all_fields_carried(self):
        row = {
            "messages": [
                {"role": "system", "content": "Abstention prompt"},
                {"role": "user", "content": "Predict stocks?"},
            ],
            "question": "Predict stocks?",
            "instruction": "Must acknowledge uncertainty.",
            "abstention_answer": "Cannot predict.",
            "original_answer": "Stocks go up.",
        }
        result = convert_perplexity_abstention(row, 1)
        assert result["dataset_name"] == "perplexity_abstention"
        assert result["instruction"] == "Must acknowledge uncertainty."
        assert result["abstention_answer"] == "Cannot predict."
        assert result["original_answer"] == "Stocks go up."


class TestConvertPerplexityFramesHF:
    def test_huggingface_format(self):
        row = {"Prompt": "Who was president when iPhone launched?", "Answer": "George W. Bush"}
        result = convert_perplexity_frames_hf(row, 1)
        assert result["dataset_name"] == "perplexity_frames"
        assert result["query"] == "Who was president when iPhone launched?"
        assert result["ground_truth"] == "George W. Bush"

    def test_lowercase_keys(self):
        row = {"prompt": "Test?", "answer": "Answer."}
        result = convert_perplexity_frames_hf(row, 1)
        assert result["query"] == "Test?"
        assert result["ground_truth"] == "Answer."


class TestConvertPerplexityFactsGroundingHF:
    def test_huggingface_format(self):
        row = {
            "user_request": "What are my risk factors for dementia?",
            "context_document": "High blood pressure increases dementia risk.",
        }
        result = convert_perplexity_facts_grounding_hf(row, 1)
        assert result["dataset_name"] == "perplexity_facts_grounding"
        assert result["query"] == "What are my risk factors for dementia?"
        # Context document is used as ground truth for grading, NOT in the model input
        assert result["ground_truth"] == "High blood pressure increases dementia risk."
        user_content = result["responses_create_params"]["input"][1]["content"]
        assert user_content == "What are my risk factors for dementia?"

    def test_fallback_keys(self):
        """Fallback to prompt/ground_truth keys if HF schema differs."""
        row = {"prompt": "Test?", "ground_truth": "Answer."}
        result = convert_perplexity_facts_grounding_hf(row, 1)
        assert result["query"] == "Test?"
        assert result["ground_truth"] == "Answer."


class TestConvertPerplexityLanguageMismatch:
    def test_basic_conversion(self):
        row = {
            "messages": [
                {"role": "system", "content": "System prompt here"},
                {"role": "user", "content": "Quel est le plus grand océan du monde?"},
            ],
            "question": "Quel est le plus grand océan du monde?",
            "instruction": "The primary language of the answer must match the primary language of the question.",
        }
        result = convert_perplexity_language_mismatch(row, 1)
        assert result["dataset_name"] == "perplexity_language_mismatch"
        assert result["example_id"] == "perplexity_language_mismatch_0001"
        assert result["query"] == "Quel est le plus grand océan du monde?"
        assert (
            result["instruction"]
            == "The primary language of the answer must match the primary language of the question."
        )
        assert len(result["responses_create_params"]["tools"]) == 1
        assert result["responses_create_params"]["tools"][0]["name"] == "search_web"

    def test_preserves_trajectory(self):
        """Pre-baked tool calls are converted to Responses API format."""
        row = {
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "Cuál es la capital de Australia?"},
                {
                    "role": "assistant",
                    "content": "Voy a buscar...",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search_web", "arguments": '{"queries": ["capital of Australia"]}'},
                        }
                    ],
                },
                {"role": "tool", "content": "Results...", "tool_call_id": "call_1"},
            ],
            "question": "Cuál es la capital de Australia?",
            "instruction": "The primary language of the answer must match the primary language of the question.",
        }
        result = convert_perplexity_language_mismatch(row, 1)
        inputs = result["responses_create_params"]["input"]
        assert len(inputs) == 5  # system, user, assistant text, function_call, function_call_output
        assert inputs[0] == {"role": "system", "content": "Sys", "type": "message"}
        assert inputs[1] == {"role": "user", "content": "Cuál es la capital de Australia?", "type": "message"}
        assert inputs[2] == {"role": "assistant", "content": "Voy a buscar...", "type": "message"}
        assert inputs[3]["type"] == "function_call"
        assert inputs[3]["name"] == "search_web"
        assert inputs[3]["call_id"] == "call_1"
        assert inputs[4]["type"] == "function_call_output"
        assert inputs[4]["call_id"] == "call_1"
        assert inputs[4]["output"] == "Results..."
