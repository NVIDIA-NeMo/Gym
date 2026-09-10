# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model pricing lookup with llm-prices.com pricing database."""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

LLM_PRICES_URL = "https://www.llm-prices.com/current-v1.json"

CACHE_DIR = Path.home() / ".cache" / "automationbench"
CACHE_FILE = CACHE_DIR / "model_prices.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# Provider prefixes to strip during normalization (order matters - longer first)
PROVIDER_PREFIXES = [
    "openrouter/anthropic/",
    "openrouter/google/",
    "openrouter/",
    "vertex_ai-global/",
    "vertex_ai/",
    "azure_ai/",
    "databricks/databricks-",
    "databricks/",
    "anthropic/",
    "bedrock/",
    "openai/",
    "gemini/",
    "azure/",
    "xai-dev/",
    "gemini-dev/",
]

# Generic pattern to strip any "provider/" or "provider-something/" prefix
GENERIC_PREFIX_PATTERN = re.compile(r"^[a-z0-9]+-?[a-z0-9]*/", re.IGNORECASE)

# Regex patterns for suffixes to strip
DATE_SUFFIX_PATTERN = re.compile(
    r"[-@]\d{4}-?\d{2}-?\d{2}$"
)  # -20251101 or @20251001 or -2026-03-05
VERSION_SUFFIX_PATTERN = re.compile(r"-v\d+:\d+$")  # -v1:0
PREVIEW_SUFFIX_PATTERN = re.compile(r"-preview$")
# Region prefixes in bedrock-style names (e.g., us-east-1/)
REGION_PREFIX_PATTERN = re.compile(r"^[a-z]{2}[-\.]?[a-z]*[-\.]?\d?/")


def normalize_model_name(model: str) -> str:
    """
    Normalize model name for fuzzy matching.

    Strips provider prefixes, date suffixes, version tags, and preview suffixes
    to enable matching between different naming conventions.

    Examples:
        "openai/gpt-4o" -> "gpt-4o"
        "claude-opus-4-5-20251101" -> "claude-opus-4-5"
        "gemini-3-pro-preview" -> "gemini-3-pro"
        "vertex_ai/gemini-3-flash-preview" -> "gemini-3-flash"
        "gemini-dev/gemini-3-pro-preview" -> "gemini-3-pro"
        "xai-dev/grok-4-1-fast-reasoning" -> "grok-4-1-fast-reasoning"
    """
    name = model.lower()

    # Strip known provider prefixes
    matched_prefix = False
    for prefix in PROVIDER_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            matched_prefix = True
            break

    # If no known prefix matched, try generic pattern (e.g., "provider-dev/")
    if not matched_prefix:
        name = GENERIC_PREFIX_PATTERN.sub("", name)

    # Strip region prefixes (e.g., "us-east-1/" in bedrock paths)
    name = REGION_PREFIX_PATTERN.sub("", name)

    # Strip suffixes from end to start (version is outermost, then date, then preview)
    # Example: claude-sonnet-4-5-20250929-v1:0 -> claude-sonnet-4-5
    name = VERSION_SUFFIX_PATTERN.sub("", name)  # -v1:0
    name = DATE_SUFFIX_PATTERN.sub("", name)  # -20251101 or @20251001
    name = PREVIEW_SUFFIX_PATTERN.sub("", name)  # -preview

    return name


# Fallback pricing for common models if fetch fails (updated Aug 2026)
# Uses canonical names only - normalization handles date suffixes and prefixes
FALLBACK_PRICING: dict[str, dict[str, float]] = {
    # OpenAI GPT-5 series
    "gpt-5.6-sol": {"input_cost_per_token": 5e-6, "output_cost_per_token": 30e-6},
    "gpt-5.6-terra": {"input_cost_per_token": 2e-6, "output_cost_per_token": 12e-6},
    "gpt-5.6-luna": {"input_cost_per_token": 0.2e-6, "output_cost_per_token": 1.2e-6},
    "gpt-5.5": {"input_cost_per_token": 5e-6, "output_cost_per_token": 30e-6},
    "gpt-5.5-pro": {"input_cost_per_token": 30e-6, "output_cost_per_token": 180e-6},
    "gpt-5": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
    "gpt-5-mini": {"input_cost_per_token": 0.25e-6, "output_cost_per_token": 2e-6},
    "gpt-5-nano": {"input_cost_per_token": 0.05e-6, "output_cost_per_token": 0.4e-6},
    "gpt-5-pro": {"input_cost_per_token": 15e-6, "output_cost_per_token": 120e-6},
    "gpt-5.1": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
    "gpt-5.1-codex": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
    "gpt-5.1-codex-mini": {"input_cost_per_token": 0.25e-6, "output_cost_per_token": 2e-6},
    "gpt-5.2": {"input_cost_per_token": 1.75e-6, "output_cost_per_token": 14e-6},
    "gpt-5.2-pro": {"input_cost_per_token": 21e-6, "output_cost_per_token": 168e-6},
    "gpt-5.2-codex": {"input_cost_per_token": 1.75e-6, "output_cost_per_token": 14e-6},
    "gpt-5.3-codex": {"input_cost_per_token": 1.75e-6, "output_cost_per_token": 14e-6},
    "gpt-5.4": {"input_cost_per_token": 2.5e-6, "output_cost_per_token": 15e-6},
    "gpt-5.4-mini": {"input_cost_per_token": 0.75e-6, "output_cost_per_token": 4.5e-6},
    "gpt-5.4-nano": {"input_cost_per_token": 0.2e-6, "output_cost_per_token": 1.25e-6},
    "gpt-5.4-pro": {"input_cost_per_token": 30e-6, "output_cost_per_token": 180e-6},
    "gpt-5-codex": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
    "gpt-5.1-codex-max": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
    # OpenAI GPT-4 series
    "gpt-4.1": {"input_cost_per_token": 2e-6, "output_cost_per_token": 8e-6},
    "gpt-4.1-mini": {"input_cost_per_token": 0.4e-6, "output_cost_per_token": 1.6e-6},
    "gpt-4.1-nano": {"input_cost_per_token": 0.1e-6, "output_cost_per_token": 0.4e-6},
    "gpt-4.5": {"input_cost_per_token": 75e-6, "output_cost_per_token": 150e-6},
    "gpt-4o": {"input_cost_per_token": 2.5e-6, "output_cost_per_token": 10e-6},
    "gpt-4o-mini": {"input_cost_per_token": 0.15e-6, "output_cost_per_token": 0.6e-6},
    # OpenAI o-series
    "o1": {"input_cost_per_token": 15e-6, "output_cost_per_token": 60e-6},
    "o1-mini": {"input_cost_per_token": 1.1e-6, "output_cost_per_token": 4.4e-6},
    "o1-pro": {"input_cost_per_token": 150e-6, "output_cost_per_token": 600e-6},
    "o3": {"input_cost_per_token": 2e-6, "output_cost_per_token": 8e-6},
    "o3-mini": {"input_cost_per_token": 1.1e-6, "output_cost_per_token": 4.4e-6},
    "o3-pro": {"input_cost_per_token": 20e-6, "output_cost_per_token": 80e-6},
    "o4-mini": {"input_cost_per_token": 1.1e-6, "output_cost_per_token": 4.4e-6},
    # Anthropic Claude 5 family
    "claude-fable-5": {"input_cost_per_token": 10e-6, "output_cost_per_token": 50e-6},
    "claude-mythos-5": {"input_cost_per_token": 10e-6, "output_cost_per_token": 50e-6},
    # Sonnet 5 introductory pricing through 2026-08-31; becomes $3/$15 on 2026-09-01
    "claude-sonnet-5": {"input_cost_per_token": 2e-6, "output_cost_per_token": 10e-6},
    # Anthropic Claude 4.8 series
    "claude-opus-4-8": {"input_cost_per_token": 5e-6, "output_cost_per_token": 25e-6},
    # Anthropic Claude 4.7 series
    "claude-opus-4-7": {"input_cost_per_token": 5e-6, "output_cost_per_token": 25e-6},
    # Anthropic Claude 4.6 series
    "claude-opus-4-6": {"input_cost_per_token": 5e-6, "output_cost_per_token": 25e-6},
    "claude-sonnet-4-6": {"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6},
    # Anthropic Claude 4.5 series (canonical names without date suffixes)
    "claude-opus-4-5": {"input_cost_per_token": 5e-6, "output_cost_per_token": 25e-6},
    "claude-sonnet-4-5": {"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6},
    "claude-haiku-4-5": {"input_cost_per_token": 1e-6, "output_cost_per_token": 5e-6},
    # Anthropic Claude 4.1 series
    "claude-opus-4-1": {"input_cost_per_token": 15e-6, "output_cost_per_token": 75e-6},
    # Anthropic Claude 4 series
    "claude-sonnet-4": {"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6},
    # Anthropic Claude older
    "claude-haiku-3-5": {"input_cost_per_token": 0.8e-6, "output_cost_per_token": 4e-6},
    "claude-3-opus": {"input_cost_per_token": 15e-6, "output_cost_per_token": 75e-6},
    "claude-3-haiku": {"input_cost_per_token": 0.25e-6, "output_cost_per_token": 1.25e-6},
    # Google Gemini 3.5 series
    "gemini-3.5-flash": {"input_cost_per_token": 1.5e-6, "output_cost_per_token": 9e-6},
    # Google Gemini 3.6 series
    "gemini-3.6-flash": {"input_cost_per_token": 1.5e-6, "output_cost_per_token": 7.5e-6},
    # Google Gemini 3.1 series
    "gemini-3.1-pro": {"input_cost_per_token": 2e-6, "output_cost_per_token": 12e-6},
    "gemini-3.1-flash-lite": {"input_cost_per_token": 0.25e-6, "output_cost_per_token": 1.5e-6},
    # Google Gemini 3 series (canonical names without -preview)
    "gemini-3-pro": {"input_cost_per_token": 2e-6, "output_cost_per_token": 12e-6},
    "gemini-3-flash": {"input_cost_per_token": 0.5e-6, "output_cost_per_token": 3e-6},
    # Google Gemini 2.5 series
    "gemini-2.5-pro": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
    "gemini-2.5-flash": {"input_cost_per_token": 0.3e-6, "output_cost_per_token": 2.5e-6},
    # Google Gemini 2.5 lite
    "gemini-2.5-flash-lite": {"input_cost_per_token": 0.1e-6, "output_cost_per_token": 0.4e-6},
    # Google Gemini 2 series
    "gemini-2.0-flash": {"input_cost_per_token": 0.15e-6, "output_cost_per_token": 0.6e-6},
    "gemini-2.0-flash-lite": {"input_cost_per_token": 0.075e-6, "output_cost_per_token": 0.3e-6},
    # xAI Grok series
    "grok-4.5": {"input_cost_per_token": 2e-6, "output_cost_per_token": 6e-6},
    "grok-4": {"input_cost_per_token": 2e-6, "output_cost_per_token": 6e-6},
    "grok-4-fast": {"input_cost_per_token": 0.2e-6, "output_cost_per_token": 0.5e-6},
    "grok-4-fast-reasoning": {"input_cost_per_token": 0.2e-6, "output_cost_per_token": 0.5e-6},
    "grok-code-fast-1": {"input_cost_per_token": 0.2e-6, "output_cost_per_token": 1.5e-6},
    "grok-3": {"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6},
    "grok-3-mini": {"input_cost_per_token": 0.3e-6, "output_cost_per_token": 0.5e-6},
    # DeepSeek
    "deepseek-chat": {"input_cost_per_token": 0.27e-6, "output_cost_per_token": 1.1e-6},
    "deepseek-reasoner": {"input_cost_per_token": 0.55e-6, "output_cost_per_token": 2.19e-6},
}

# Model name aliases: maps normalized names to pricing database names
# Used when naming conventions differ between providers and pricing database
MODEL_ALIASES: dict[str, str] = {
    # Claude: model-size-version -> pricing uses model-version-size
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-haiku-4-5": "claude-4.5-haiku",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-opus-4-5": "claude-opus-4-5",
    # Gemini: dot-separated version -> dash-separated (llm-prices.com convention)
    "gemini-3.1-pro": "gemini-3-1-pro-preview",
    # xAI Grok: grok-4-1-* -> grok-4-* (strip version suffix)
    "grok-4-1-fast-reasoning": "grok-4-fast-reasoning",
    "grok-4-1-fast-non-reasoning": "grok-4-fast",
}


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_cost_per_token: float
    output_cost_per_token: float
    source: str  # "llm-prices", "fallback", "cli-override"

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for given token counts."""
        return input_tokens * self.input_cost_per_token + output_tokens * self.output_cost_per_token


class PricingDatabase:
    """Manages model pricing lookup with caching."""

    def __init__(
        self,
        input_cost_override: float | None = None,
        output_cost_override: float | None = None,
    ):
        self._pricing_data: dict[str, dict] | None = None
        self._input_cost_override = input_cost_override
        self._output_cost_override = output_cost_override

    def _load_cached_pricing(self) -> dict[str, dict] | None:
        """Load pricing from cache if fresh."""
        if not CACHE_FILE.exists():
            return None

        # Check if cache is stale
        cache_age = time.time() - CACHE_FILE.stat().st_mtime
        if cache_age > CACHE_TTL_SECONDS:
            return None

        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _fetch_pricing(self) -> dict[str, dict] | None:
        """Fetch pricing from llm-prices.com."""
        try:
            with urllib.request.urlopen(LLM_PRICES_URL, timeout=10) as response:
                raw_data = json.loads(response.read().decode("utf-8"))

            # Convert llm-prices.com format to our internal format
            # llm-prices.com uses cost per million tokens, we use cost per token
            data = {}
            for entry in raw_data.get("prices", []):
                model_id = entry.get("id", "")
                if not model_id:
                    continue
                # Convert from per-million to per-token
                input_cost = entry.get("input")
                output_cost = entry.get("output")
                if input_cost is not None and output_cost is not None:
                    data[model_id] = {
                        "input_cost_per_token": input_cost / 1_000_000,
                        "output_cost_per_token": output_cost / 1_000_000,
                    }

            # Cache the result
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f)

            return data
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return None

    def _get_pricing_data(self) -> dict[str, dict]:
        """Get pricing data, fetching if necessary."""
        if self._pricing_data is not None:
            return self._pricing_data

        # Try cache first
        self._pricing_data = self._load_cached_pricing()
        if self._pricing_data is not None:
            return self._pricing_data

        # Try fetching
        self._pricing_data = self._fetch_pricing()
        if self._pricing_data is not None:
            return self._pricing_data

        # Use fallback
        self._pricing_data = FALLBACK_PRICING
        return self._pricing_data

    def _make_pricing(self, entry: dict, pricing_data: dict) -> ModelPricing | None:
        """Create ModelPricing from a pricing entry if it has required fields."""
        if "input_cost_per_token" in entry and "output_cost_per_token" in entry:
            source = "fallback" if pricing_data is FALLBACK_PRICING else "llm-prices"
            return ModelPricing(
                input_cost_per_token=entry["input_cost_per_token"],
                output_cost_per_token=entry["output_cost_per_token"],
                source=source,
            )
        return None

    def get_pricing(self, model: str) -> ModelPricing | None:
        """
        Get pricing for a model.

        Uses multiple lookup strategies:
        1. Exact match
        2. Normalized query against exact keys (strips provider prefixes, date suffixes)
        3. Alias lookup for known naming variations
        4. Normalized query against normalized keys

        Returns None if model pricing is unknown and no CLI override provided.
        """
        # CLI override takes precedence
        if self._input_cost_override is not None and self._output_cost_override is not None:
            return ModelPricing(
                input_cost_per_token=self._input_cost_override,
                output_cost_per_token=self._output_cost_override,
                source="cli-override",
            )

        pricing_data = self._get_pricing_data()

        # Strategy 1: Exact match
        if model in pricing_data:
            result = self._make_pricing(pricing_data[model], pricing_data)
            if result:
                return result

        # Strategy 2: Normalized query against exact keys
        # Normalization strips provider prefixes (e.g., "gemini-dev/", "xai-dev/")
        # and date suffixes (e.g., "-20251101")
        normalized = normalize_model_name(model)
        if normalized in pricing_data:
            result = self._make_pricing(pricing_data[normalized], pricing_data)
            if result:
                return result

        # Strategy 3: Check aliases for known naming variations
        aliased = MODEL_ALIASES.get(normalized)
        if aliased and aliased in pricing_data:
            result = self._make_pricing(pricing_data[aliased], pricing_data)
            if result:
                return result

        # Strategy 4: Normalized query against normalized keys
        for key, entry in pricing_data.items():
            if normalize_model_name(key) == normalized:
                result = self._make_pricing(entry, pricing_data)
                if result:
                    return result

        return None
