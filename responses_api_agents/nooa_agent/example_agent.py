import logging
import os
from typing import TypedDict

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


class Result(TypedDict):
    can_fulfill: bool
    total_cost: float
    unavailable_items: list[str]


class InventoryAgent(Agent, llm=llm):
    """You are an agent that checks inventory using deterministic helper methods."""

    def __init__(self):
        super().__init__()
        self.inventory = {
            "apple": {"stock": 50, "price": 0.75},
            "banana": {"stock": 30, "price": 0.50},
            "orange": {"stock": 0, "price": 0.80},  # Out of stock
        }

    # SW1: Deterministic Python - automatically available as "tools" for the LLM
    def get_stock(self, item: str) -> int:
        """Get current stock for an item."""
        return self.inventory.get(item, {}).get("stock", 0)

    def get_price(self, item: str) -> float:
        """Get price for an item."""
        return self.inventory.get(item, {}).get("price", 0.0)

    # SW3: Generation method - LLM implements this, calling SW1 methods as needed
    async def can_fulfill_order(self, items: list[str], budget: float) -> Result:
        """Check if order can be fulfilled within budget."""
        ...


@autorun
async def main():
    agent = InventoryAgent()
    result = await agent.can_fulfill_order(["apple", "banana", "orange"], budget=5.0)
    print(f"Can fulfill: {result['can_fulfill']}")
    print(f"Total cost: {result['total_cost']}")
    print(f"Unavailable items: {result['unavailable_items']}")
