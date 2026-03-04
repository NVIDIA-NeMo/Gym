# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comparison strategies and utilities for multi-generation reward computation.

For rollout collection (RLHF/GRPO), reward computation is now handled inside the
genrm_compare resources server: verify() buffers by prompt and runs comparison
when num_rollouts_per_prompt is reached (Difference 1). Rollout collection
simply posts each row to the agent /run; no strategy or buffering in Gym.

This module still provides:
- GenRMStrategy: thin client for calling genrm_compare /compare (batch API)
- get_prompt_key, extract_conversation_history, generate_response: utilities for
  scripts or tests that need to group by prompt or call the policy model directly.
"""

from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field

from nemo_gym.server_utils import ServerClient, raise_for_status


@runtime_checkable
class ComparisonStrategy(Protocol):
    """Protocol for comparison strategies that compute rewards from multiple generations.

    A comparison strategy defines how to evaluate N candidate responses for a given prompt.
    It typically involves generating multiple responses from a policy model and then
    comparing them using a reward model.

    Attributes:
        agent_names: List of agent names that should use this comparison strategy
        num_generations_per_prompt: Number of responses to generate per prompt
        policy_model_server_name: Name of the policy model server to use for generation
    """

    agent_names: List[str]
    num_generations_per_prompt: int
    policy_model_server_name: str

    async def compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict],
        server_client: ServerClient,
        principle: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float]]:
        """Compare N responses and return (rewards, metrics).

        Args:
            conversation_history: The conversation context (user/assistant messages)
            response_objs: List of raw Response API objects to compare
            server_client: The server client for making requests
            principle: Optional principle for principle-based comparison (e.g., GenRM)

        Returns:
            Tuple of:
                - rewards: List of reward values, one per response (same order as input)
                - metrics: Dict of aggregation metrics (e.g., mean_score, tiebreak_rate)
        """
        ...


class GenRMStrategyConfig(BaseModel):
    """Configuration for GenRM comparison strategy.

    This strategy uses the GenRM Compare Resource Server to perform pairwise
    comparisons of multiple candidate responses.

    Attributes:
        agent_names: Agent names that should use GenRM comparison (e.g., ["genrm_simple_agent"])
        num_generations_per_prompt: Number of responses to generate and compare per prompt
        genrm_compare_server_name: Name of the genrm_compare resource server
        policy_model_server_name: Name of the policy model to generate responses
    """

    agent_names: List[str] = Field(default_factory=lambda: ["genrm_simple_agent"])
    num_generations_per_prompt: int = 16
    genrm_compare_server_name: str = "genrm_compare"
    policy_model_server_name: str = "policy_model"


class GenRMStrategy:
    """GenRM comparison strategy using pairwise comparisons.

    This strategy generates N responses per prompt using the policy model,
    then calls the genrm_compare Resource Server to compute rewards via
    pairwise comparisons using a GenRM model.

    Example usage:
        ```python
        strategy = GenRMStrategy(GenRMStrategyConfig(
            agent_names=["genrm_simple_agent"],
            num_generations_per_prompt=16,
            genrm_compare_server_name="genrm_compare",
            policy_model_server_name="policy_model"
        ))

        rewards, metrics = await strategy.compare(
            conversation_history=[{"role": "user", "content": "What is 2+2?"}],
            response_objs=[response1, response2, ...],
            server_client=client,
            principle="Be concise and accurate"
        )
        ```
    """

    def __init__(self, config: GenRMStrategyConfig):
        """Initialize the GenRM strategy with configuration.

        Args:
            config: Configuration specifying agents, generation params, and server names
        """
        self.config = config
        self.agent_names = config.agent_names
        self.num_generations_per_prompt = config.num_generations_per_prompt
        self.policy_model_server_name = config.policy_model_server_name

    async def compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict],
        server_client: ServerClient,
        principle: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float]]:
        """Call genrm_compare server to get rewards for each response.

        This method delegates to the genrm_compare Resource Server which:
        1. Generates pairwise comparisons (circular or all_pairs strategy)
        2. Calls the GenRM model for each pair
        3. Aggregates scores using tiebreaker logic
        4. Returns per-response rewards

        Args:
            conversation_history: The conversation context
            response_objs: List of raw Response API objects
            server_client: The server client for making requests
            principle: Optional principle for principle-based GenRM comparison

        Returns:
            Tuple of (rewards, metrics) from GenRM comparison
        """
        payload = {
            "conversation_history": conversation_history,
            "response_objs": response_objs,
        }

        if principle is not None:
            payload["principle"] = principle

        res = await server_client.post(
            server_name=self.config.genrm_compare_server_name,
            url_path="/compare",
            json=payload,
        )
        await raise_for_status(res)
        result = await res.json()

        rewards = result.get("rewards", [0.0] * len(response_objs))
        metrics = result.get("metrics", {})

        return rewards, metrics


# =============================================================================
# Utility Functions (generate_response, extract_generated_text)
# =============================================================================


def extract_generated_text(gen_result: Dict) -> str:
    """Extract generated text from generation result.

    Handles various Response API output formats.

    Args:
        gen_result: Response API result dict

    Returns:
        The generated text string

    Raises:
        ValueError: If text cannot be extracted from the result
    """
    if not isinstance(gen_result, dict):
        raise ValueError(f"Expected dict, got {type(gen_result)}")
    if "output" in gen_result:
        output = gen_result["output"]
        if isinstance(output, list) and output:
            return output[0].get("content", "")
        if isinstance(output, str):
            return output
    if "response" in gen_result:
        return gen_result["response"]
    raise ValueError(f"Cannot extract generated text from: {list(gen_result.keys())}")


async def generate_response(example: Dict, server_client: ServerClient, model_server: str) -> Dict:
    """Generate a single response using the policy model.

    Args:
        example: Example dict containing responses_create_params
        server_client: The server client for making requests
        model_server: Name of the model server to call

    Returns:
        Raw Response API result dict

    Raises:
        ValueError: If example is missing responses_create_params
    """
    params = example.get("responses_create_params")
    if params is None:
        raise ValueError(f"Example missing 'responses_create_params': {list(example.keys())}")
    res = await server_client.post(server_name=model_server, url_path="/v1/responses", json=params)
    await raise_for_status(res)
    return await res.json()
