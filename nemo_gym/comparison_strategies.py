# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comparison strategies for multi-generation reward computation.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field

from nemo_gym.server_utils import ServerClient, raise_for_status


@runtime_checkable
class ComparisonStrategy(Protocol):
    """Protocol for comparison strategies that compute rewards from multiple generations."""
    
    agent_names: List[str]
    num_generations_per_prompt: int
    policy_model_server_name: str
    
    async def compare(
        self,
        conversation_history: List[Dict[str, str]],
        responses: List[str],
        server_client: ServerClient,
        principle: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float]]:
        """Compare N responses and return (rewards, metrics)."""
        ...


class GenRMStrategyConfig(BaseModel):
    """Configuration for GenRM comparison strategy."""
    agent_names: List[str] = Field(default_factory=lambda: ["genrm_simple_agent"])
    num_generations_per_prompt: int = 16
    genrm_compare_server_name: str = "genrm_compare"
    policy_model_server_name: str = "policy_model"


class GenRMStrategy:
    """GenRM comparison strategy using pairwise comparisons."""
    
    def __init__(self, config: GenRMStrategyConfig):
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


def get_prompt_key(example: Dict) -> str:
    """Get stable key for grouping examples by prompt and principle.
    
    Examples with the same conversation history but different principles
    should be in separate groups, so we include principle in the hash.
    """
    if "prompt_id" in example:
        # If prompt_id exists, combine it with principle for uniqueness
        prompt_id = str(example["prompt_id"])
        principle = example.get("principle")
        if principle is not None:
            return f"{prompt_id}::{principle}"
        return prompt_id
    
    # Hash both conversation history and principle together
    conv = extract_conversation_history(example)
    principle = example.get("principle")
    key_data = {
        "conversation": conv,
        "principle": principle,
    }
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


def extract_conversation_history(example: Dict) -> List[Dict]:
    """Extract conversation history from example.
    
    Gym examples store history in responses_create_params.input
    """
    responses_create_params = example.get("responses_create_params")
    if responses_create_params is None:
        raise ValueError(f"Example missing 'responses_create_params': {list(example.keys())}")
    if "input" not in responses_create_params:
        raise ValueError(f"responses_create_params missing 'input': {list(responses_create_params.keys())}")
    return responses_create_params["input"]


def extract_generated_text(gen_result: Dict) -> str:
    """Extract generated text from generation result."""
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
    """Generate a single response using the policy model."""
    params = example.get("responses_create_params")
    if params is None:
        raise ValueError(f"Example missing 'responses_create_params': {list(example.keys())}")
    res = await server_client.post(server_name=model_server, url_path="/v1/responses", json=params)
    await raise_for_status(res)
    return await res.json()
