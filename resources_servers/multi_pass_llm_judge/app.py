"""
Multi-Pass LLM Judge Resources Server.

Scores model outputs using multiple distinct LLM judge passes.
Each pass can use a different prompt template and scoring criteria,
allowing for multi-dimensional evaluation of model responses.

This is designed as a generic, easily customizable environment for
training with multi-criteria LLM-based reward signals.
"""

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
from __future__ import annotations

import re
from enum import Enum
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


class ScoringMode(str, Enum):
    """How to convert judge output to a score for each pass."""
    
    # Binary: look for success/failure labels
    BINARY = "binary"
    # Numeric: extract a number from the output (e.g., "Score: 8/10")
    NUMERIC = "numeric"
    # Regex match: assign score based on regex pattern match
    REGEX = "regex"


class AggregationMode(str, Enum):
    """How to aggregate scores from multiple passes into a final reward."""
    
    # Weighted sum of all pass scores
    WEIGHTED_SUM = "weighted_sum"
    # Minimum score across all passes
    MIN = "min"
    # Maximum score across all passes
    MAX = "max"
    # Average of all pass scores
    MEAN = "mean"
    # All passes must succeed (product of binary scores)
    ALL = "all"
    # Any pass succeeds (max of binary scores)
    ANY = "any"


class JudgePassConfig(BaseModel):
    """Configuration for a single judge evaluation pass."""
    
    # Unique identifier for this pass (for logging/debugging)
    name: str = Field(description="Name identifier for this judge pass")
    
    # Weight for weighted aggregation (ignored for min/max/mean)
    weight: float = Field(default=1.0, description="Weight for weighted aggregation")
    
    # The prompt template with placeholders: {question}, {expected_answer}, {generated_answer}
    # Can also use {metadata.field_name} for custom fields from the data row
    prompt_template: str = Field(
        description="Prompt template with placeholders for {question}, {expected_answer}, {generated_answer}"
    )
    
    # Optional system message for this pass
    system_message: Optional[str] = Field(default=None, description="Optional system message")
    
    # =========================================================================
    # Per-Pass Model Configuration (optional - falls back to global settings)
    # =========================================================================
    # Optional: Use a different model for this specific pass
    # If not set, uses the global judge_model_server from MultiPassLLMJudgeConfig
    judge_model_server: Optional[ModelServerRef] = Field(
        default=None,
        description="Optional model server for this pass (overrides global judge_model_server)"
    )
    
    # Optional: Custom request parameters for this pass
    # If not set, uses the global judge_responses_create_params
    responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = Field(
        default=None,
        description="Optional request params for this pass (overrides global params)"
    )
    
    # Scoring mode for this pass
    scoring_mode: ScoringMode = Field(default=ScoringMode.BINARY, description="How to score judge output")
    
    # For BINARY mode: labels to look for
    success_label: str = Field(default="[[PASS]]", description="Label indicating success in binary mode")
    failure_label: str = Field(default="[[FAIL]]", description="Label indicating failure in binary mode")
    
    # For NUMERIC mode: regex to extract numeric score (first capture group used)
    # e.g., r"Score:\s*(\d+(?:\.\d+)?)/10" extracts "8.5" from "Score: 8.5/10"
    numeric_regex: str = Field(
        default=r"(\d+(?:\.\d+)?)", 
        description="Regex with capture group to extract numeric score"
    )
    # Max value for normalization (score will be divided by this)
    numeric_max: float = Field(default=10.0, description="Maximum value for numeric normalization")
    
    # For REGEX mode: patterns that indicate different score levels
    # Evaluated in order; first match wins
    regex_patterns: List[dict] = Field(
        default_factory=lambda: [
            {"pattern": r"\[\[EXCELLENT\]\]", "score": 1.0},
            {"pattern": r"\[\[GOOD\]\]", "score": 0.75},
            {"pattern": r"\[\[PARTIAL\]\]", "score": 0.5},
            {"pattern": r"\[\[POOR\]\]", "score": 0.25},
            {"pattern": r"\[\[FAIL\]\]", "score": 0.0},
        ],
        description="List of {pattern, score} dicts for regex mode"
    )
    # Default score if no pattern matches
    regex_default_score: float = Field(default=0.0, description="Default score if no regex matches")


class MultiPassLLMJudgeConfig(BaseResourcesServerConfig):
    """Configuration for the Multi-Pass LLM Judge server."""
    
    # Default logical name for this resources server
    name: str = "multi_pass_llm_judge"
    
    # Reference to the judge model server
    judge_model_server: ModelServerRef = Field(
        description="Reference to the model server used as the judge"
    )
    
    # Base parameters for judge requests (input will be set per request)
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge model requests"
    )
    
    # List of judge passes to execute
    judge_passes: List[JudgePassConfig] = Field(
        description="List of judge evaluation passes to execute"
    )
    
    # How to aggregate pass scores into final reward
    aggregation_mode: AggregationMode = Field(
        default=AggregationMode.WEIGHTED_SUM,
        description="How to aggregate scores from multiple passes"
    )
    
    # Optional regex to extract the question from the last user message
    question_extract_regex: Optional[str] = Field(
        default=None,
        description="Regex to extract question from user message (last match used)"
    )
    
    # Optional regex to extract the generated response from the last assistant message
    response_extract_regex: Optional[str] = Field(
        default=None,
        description="Regex to extract answer from assistant message (last match used)"
    )
    
    # Whether to run passes in parallel (faster) or sequential (for debugging)
    parallel_execution: bool = Field(
        default=True,
        description="Whether to execute judge passes in parallel"
    )


class MultiPassRunRequest(BaseRunRequest):
    """Run/verify request payload."""
    
    model_config = ConfigDict(extra="allow")
    
    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    question: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class MultiPassVerifyRequest(MultiPassRunRequest, BaseVerifyRequest):
    pass


class PassEvaluation(BaseModel):
    """Result of a single judge pass evaluation."""
    
    pass_name: str
    judge_model: str  # Which model was used for this pass
    prompt: str
    judge_response_text: str
    raw_score: float
    weighted_score: float
    weight: float


class MultiPassVerifyResponse(BaseVerifyResponse):
    """Response with detailed evaluation from all passes."""
    
    expected_answer: str
    generated_answer: str
    pass_evaluations: List[PassEvaluation]
    aggregation_mode: str


def _extract_text_from_response(response: NeMoGymResponse) -> str:
    """Extract text content from the last assistant message in a response."""
    for output in reversed(response.output):
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            if isinstance(content, list):
                texts = []
                for c in content:
                    text = getattr(c, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
                return "\n".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
    return ""


def _extract_with_regex(text: str, regex: Optional[str]) -> str:
    """Extract text using regex pattern. Returns full text if no regex or no match."""
    if not regex or not text:
        return text
    
    try:
        matches = list(re.finditer(regex, text, flags=re.MULTILINE | re.DOTALL))
        if matches:
            match = matches[-1]  # Use last match
            groups = match.groups()
            if groups:
                for group in groups:
                    if isinstance(group, str) and group.strip():
                        return group.strip()
            return match.group(0).strip()
    except re.error:
        pass
    
    return text


def _format_prompt(template: str, question: str, expected_answer: str, 
                   generated_answer: str, metadata: Optional[dict] = None) -> str:
    """Format a prompt template with the provided values."""
    prompt = template.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )
    
    # Handle metadata placeholders like {metadata.field_name}
    if metadata:
        for key, value in metadata.items():
            prompt = prompt.replace(f"{{metadata.{key}}}", str(value))
    
    return prompt


class MultiPassLLMJudgeServer(SimpleResourcesServer):
    """Multi-pass LLM judge verification server."""
    
    config: MultiPassLLMJudgeConfig
    
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app
    
    async def verify(self, body: MultiPassVerifyRequest) -> MultiPassVerifyResponse:
        """Verify model response using multiple LLM judge passes."""
        
        # Extract question, expected answer, and generated answer
        expected = body.expected_answer or ""
        if not expected and body.metadata:
            expected = str(body.metadata.get("expected_answer", ""))
        
        question = body.question or ""
        if not question:
            question = _extract_question_from_params(
                body.responses_create_params, 
                self.config.question_extract_regex
            )
        
        generated = _extract_with_regex(
            _extract_last_assistant_text(body),
            self.config.response_extract_regex
        )
        
        metadata = body.metadata or {}
        
        # Execute all judge passes
        if self.config.parallel_execution:
            import asyncio
            evaluations = await asyncio.gather(*[
                self._evaluate_pass(pass_config, question, expected, generated, metadata)
                for pass_config in self.config.judge_passes
            ])
        else:
            evaluations = []
            for pass_config in self.config.judge_passes:
                eval_result = await self._evaluate_pass(
                    pass_config, question, expected, generated, metadata
                )
                evaluations.append(eval_result)
        
        # Aggregate scores
        reward = self._aggregate_scores(evaluations)
        
        # Build response
        payload = body.model_dump()
        payload.pop("expected_answer", None)
        
        return MultiPassVerifyResponse(
            **payload,
            reward=reward,
            expected_answer=expected,
            generated_answer=generated,
            pass_evaluations=evaluations,
            aggregation_mode=self.config.aggregation_mode.value,
        )
    
    async def _evaluate_pass(
        self, 
        pass_config: JudgePassConfig,
        question: str,
        expected_answer: str,
        generated_answer: str,
        metadata: dict
    ) -> PassEvaluation:
        """Execute a single judge pass and compute its score."""
        
        # Format the prompt
        prompt = _format_prompt(
            pass_config.prompt_template,
            question,
            expected_answer,
            generated_answer,
            metadata
        )
        
        # Build messages for judge
        msgs: List[NeMoGymEasyInputMessage] = []
        if pass_config.system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=pass_config.system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        
        # Determine which model server to use (pass-specific or global)
        model_server = (
            pass_config.judge_model_server 
            if pass_config.judge_model_server is not None 
            else self.config.judge_model_server
        )
        
        # Determine which request params to use (pass-specific or global)
        base_params = (
            pass_config.responses_create_params
            if pass_config.responses_create_params is not None
            else self.config.judge_responses_create_params
        )
        request_params = base_params.model_copy(deep=True)
        request_params.input = msgs
        
        # Call judge model
        response = await self.server_client.post(
            server_name=model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        response_text = _extract_text_from_response(judge_response)
        
        # Score the response
        raw_score = self._score_response(pass_config, response_text)
        weighted_score = raw_score * pass_config.weight
        
        return PassEvaluation(
            pass_name=pass_config.name,
            judge_model=model_server.name,
            prompt=prompt,
            judge_response_text=response_text,
            raw_score=raw_score,
            weighted_score=weighted_score,
            weight=pass_config.weight,
        )
    
    def _score_response(self, pass_config: JudgePassConfig, response_text: str) -> float:
        """Score judge response based on the configured scoring mode."""
        
        if pass_config.scoring_mode == ScoringMode.BINARY:
            return self._score_binary(pass_config, response_text)
        elif pass_config.scoring_mode == ScoringMode.NUMERIC:
            return self._score_numeric(pass_config, response_text)
        elif pass_config.scoring_mode == ScoringMode.REGEX:
            return self._score_regex(pass_config, response_text)
        else:
            return 0.0
    
    def _score_binary(self, pass_config: JudgePassConfig, response_text: str) -> float:
        """Binary scoring: look for success/failure labels."""
        success_pos = response_text.find(pass_config.success_label)
        failure_pos = response_text.find(pass_config.failure_label)
        
        if success_pos < 0 and failure_pos < 0:
            return 0.0  # No label found
        if success_pos >= 0 and (failure_pos < 0 or success_pos < failure_pos):
            return 1.0
        return 0.0
    
    def _score_numeric(self, pass_config: JudgePassConfig, response_text: str) -> float:
        """Numeric scoring: extract number and normalize."""
        try:
            matches = list(re.finditer(pass_config.numeric_regex, response_text))
            if matches:
                # Use the last match
                match = matches[-1]
                groups = match.groups()
                if groups:
                    for group in groups:
                        if group:
                            score = float(group)
                            return min(1.0, max(0.0, score / pass_config.numeric_max))
        except (ValueError, re.error):
            pass
        return 0.0
    
    def _score_regex(self, pass_config: JudgePassConfig, response_text: str) -> float:
        """Regex scoring: match patterns in order."""
        for pattern_config in pass_config.regex_patterns:
            pattern = pattern_config.get("pattern", "")
            score = pattern_config.get("score", 0.0)
            try:
                if re.search(pattern, response_text, flags=re.IGNORECASE):
                    return float(score)
            except re.error:
                continue
        return pass_config.regex_default_score
    
    def _aggregate_scores(self, evaluations: List[PassEvaluation]) -> float:
        """Aggregate pass scores into final reward."""
        if not evaluations:
            return 0.0
        
        raw_scores = [e.raw_score for e in evaluations]
        weighted_scores = [e.weighted_score for e in evaluations]
        weights = [e.weight for e in evaluations]
        
        mode = self.config.aggregation_mode
        
        if mode == AggregationMode.WEIGHTED_SUM:
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            return sum(weighted_scores) / total_weight
        
        elif mode == AggregationMode.MIN:
            return min(raw_scores)
        
        elif mode == AggregationMode.MAX:
            return max(raw_scores)
        
        elif mode == AggregationMode.MEAN:
            return sum(raw_scores) / len(raw_scores)
        
        elif mode == AggregationMode.ALL:
            # All must be 1.0 for reward to be 1.0
            return 1.0 if all(s >= 0.99 for s in raw_scores) else 0.0
        
        elif mode == AggregationMode.ANY:
            # Any success gives reward
            return 1.0 if any(s >= 0.99 for s in raw_scores) else 0.0
        
        return 0.0


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract text from the last assistant message in the response."""
    for output in reversed(body.response.output):
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            if isinstance(content, list):
                texts = []
                for c in content:
                    text = getattr(c, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
                return "\n".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
    return ""


def _extract_question_from_params(
    params: NeMoGymResponseCreateParamsNonStreaming,
    extract_regex: Optional[str]
) -> str:
    """Extract question from the last user message."""
    last_text = ""
    for msg in params.input or []:
        if getattr(msg, "role", None) == "user":
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                last_text = content
    
    return _extract_with_regex(last_text.strip(), extract_regex)


if __name__ == "__main__":
    MultiPassLLMJudgeServer.run_webserver()
