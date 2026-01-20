# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from resources_servers.multi_pass_llm_judge.app import (
    AggregationMode,
    JudgePassConfig,
    MultiPassLLMJudgeConfig,
    MultiPassLLMJudgeServer,
    PassEvaluation,
    ScoringMode,
    _extract_with_regex,
    _format_prompt,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


class TestExtractWithRegex:
    def test_no_regex_returns_text(self):
        assert _extract_with_regex("hello world", None) == "hello world"
    
    def test_regex_with_capture_group(self):
        text = "The answer is: 42 points"
        regex = r"answer is: (\d+)"
        assert _extract_with_regex(text, regex) == "42"
    
    def test_regex_no_match_returns_text(self):
        text = "no numbers here"
        regex = r"\d+"
        assert _extract_with_regex(text, regex) == "no numbers here"
    
    def test_regex_last_match_used(self):
        text = "first: 10, second: 20, third: 30"
        regex = r"(\d+)"
        assert _extract_with_regex(text, regex) == "30"


class TestFormatPrompt:
    def test_basic_formatting(self):
        template = "Q: {question}\nA: {expected_answer}\nCandidate: {generated_answer}"
        result = _format_prompt(template, "What is 2+2?", "4", "four")
        assert "Q: What is 2+2?" in result
        assert "A: 4" in result
        assert "Candidate: four" in result
    
    def test_metadata_formatting(self):
        template = "Topic: {metadata.topic}\nQ: {question}"
        result = _format_prompt(template, "What is it?", "answer", "response", {"topic": "math"})
        assert "Topic: math" in result
        assert "Q: What is it?" in result


class TestJudgePassConfig:
    def test_default_values(self):
        config = JudgePassConfig(
            name="test",
            prompt_template="Test: {question}"
        )
        assert config.weight == 1.0
        assert config.scoring_mode == ScoringMode.BINARY
        assert config.success_label == "[[PASS]]"
        assert config.failure_label == "[[FAIL]]"
    
    def test_custom_values(self):
        config = JudgePassConfig(
            name="custom",
            weight=0.5,
            prompt_template="Custom: {question}",
            scoring_mode=ScoringMode.NUMERIC,
            numeric_regex=r"Score: (\d+)",
            numeric_max=100.0
        )
        assert config.weight == 0.5
        assert config.scoring_mode == ScoringMode.NUMERIC
        assert config.numeric_max == 100.0


class TestAggregation:
    @pytest.fixture
    def sample_evaluations(self):
        return [
            PassEvaluation(
                pass_name="pass1",
                judge_model="model_a",
                prompt="p1",
                judge_response_text="[[PASS]]",
                raw_score=1.0,
                weighted_score=0.5,
                weight=0.5
            ),
            PassEvaluation(
                pass_name="pass2",
                judge_model="model_b",
                prompt="p2",
                judge_response_text="Score: 7/10",
                raw_score=0.7,
                weighted_score=0.21,
                weight=0.3
            ),
            PassEvaluation(
                pass_name="pass3",
                judge_model="model_a",
                prompt="p3",
                judge_response_text="[[GOOD]]",
                raw_score=0.75,
                weighted_score=0.15,
                weight=0.2
            )
        ]
    
    def test_weighted_sum_aggregation(self, sample_evaluations):
        # Total weight = 0.5 + 0.3 + 0.2 = 1.0
        # Weighted sum = (0.5 + 0.21 + 0.15) / 1.0 = 0.86
        config = MultiPassLLMJudgeConfig(
            judge_model_server=ModelServerRef(type="responses_api_models", name="test"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_passes=[],
            aggregation_mode=AggregationMode.WEIGHTED_SUM
        )
        server = MultiPassLLMJudgeServer.__new__(MultiPassLLMJudgeServer)
        server.config = config
        
        result = server._aggregate_scores(sample_evaluations)
        assert abs(result - 0.86) < 0.01
    
    def test_min_aggregation(self, sample_evaluations):
        config = MultiPassLLMJudgeConfig(
            judge_model_server=ModelServerRef(type="responses_api_models", name="test"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_passes=[],
            aggregation_mode=AggregationMode.MIN
        )
        server = MultiPassLLMJudgeServer.__new__(MultiPassLLMJudgeServer)
        server.config = config
        
        result = server._aggregate_scores(sample_evaluations)
        assert result == 0.7
    
    def test_max_aggregation(self, sample_evaluations):
        config = MultiPassLLMJudgeConfig(
            judge_model_server=ModelServerRef(type="responses_api_models", name="test"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_passes=[],
            aggregation_mode=AggregationMode.MAX
        )
        server = MultiPassLLMJudgeServer.__new__(MultiPassLLMJudgeServer)
        server.config = config
        
        result = server._aggregate_scores(sample_evaluations)
        assert result == 1.0


class TestScoring:
    @pytest.fixture
    def server(self):
        config = MultiPassLLMJudgeConfig(
            judge_model_server=ModelServerRef(type="responses_api_models", name="test"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_passes=[]
        )
        server = MultiPassLLMJudgeServer.__new__(MultiPassLLMJudgeServer)
        server.config = config
        return server
    
    def test_binary_scoring_success(self, server):
        pass_config = JudgePassConfig(
            name="test",
            prompt_template="test",
            scoring_mode=ScoringMode.BINARY,
            success_label="[[CORRECT]]",
            failure_label="[[WRONG]]"
        )
        result = server._score_binary(pass_config, "The answer is [[CORRECT]] because...")
        assert result == 1.0
    
    def test_binary_scoring_failure(self, server):
        pass_config = JudgePassConfig(
            name="test",
            prompt_template="test",
            scoring_mode=ScoringMode.BINARY,
            success_label="[[CORRECT]]",
            failure_label="[[WRONG]]"
        )
        result = server._score_binary(pass_config, "This is [[WRONG]]")
        assert result == 0.0
    
    def test_numeric_scoring(self, server):
        pass_config = JudgePassConfig(
            name="test",
            prompt_template="test",
            scoring_mode=ScoringMode.NUMERIC,
            numeric_regex=r"Score:\s*(\d+(?:\.\d+)?)",
            numeric_max=10.0
        )
        result = server._score_numeric(pass_config, "Score: 8/10")
        assert result == 0.8
    
    def test_regex_scoring(self, server):
        pass_config = JudgePassConfig(
            name="test",
            prompt_template="test",
            scoring_mode=ScoringMode.REGEX,
            regex_patterns=[
                {"pattern": r"\[\[EXCELLENT\]\]", "score": 1.0},
                {"pattern": r"\[\[GOOD\]\]", "score": 0.75},
            ],
            regex_default_score=0.0
        )
        result = server._score_regex(pass_config, "The quality is [[GOOD]]")
        assert result == 0.75
