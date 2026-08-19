import logging
import os

import litellm
from nooa import Agent
from nooa.unifiedllm import get_llm_client
from nooa.util.quickstart import autorun

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

llm = get_llm_client(
    "openai/us/azure/openai/eccn-gpt-5.5",
    api_base="https://inference-api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)


class FeedbackAgent(Agent, llm=llm):
    """You are an agent specializing in analyzing customer feedback."""

    async def analyze_feedback(self, text: str) -> str:
        """Analyze customer feedback for sentiment and key topics in one sentence."""
        ...  # Generation method - LLM implements based on docstring


@autorun
async def main():
    agent = FeedbackAgent()
    result = await agent.analyze_feedback("Great product, but shipping was slow")
    print(result)
