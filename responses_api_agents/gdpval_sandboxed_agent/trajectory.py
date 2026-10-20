# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export OpenHands completions while preserving Gym token capture."""

import json
import math
import time


def completion_payload(task, context_limit=None, *, token_capture=False):
    """Use the same final-completion convention as Gym's SWE/OpenHands adapter."""
    paths = list((task / "trace/completions").glob("*.json"))
    if not paths:
        raise ValueError("OpenHands produced no completion logs")
    payloads = sorted(
        (json.loads(p.read_text()) for p in paths), key=lambda x: x["timestamp"]
    )
    # Training tokens/logprobs are validated by the capture ledger, not logs.
    # A zero-token length stop is not an assistant turn for the judge export.
    if token_capture:
        while len(payloads) > 1:
            choice = payloads[-1]["response"]["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") != "length" or any(
                message.get(key)
                for key in (
                    "content",
                    "tool_calls",
                    "function_call",
                    "reasoning_content",
                    "reasoning",
                )
            ):
                break
            payloads.pop()
    if context_limit is not None and not token_capture:
        if (
            not isinstance(context_limit, int)
            or context_limit <= 0
            or len(payloads) < 2
        ):
            raise ValueError(
                "Context exhaustion requires a valid limit and prior completion"
            )
        terminal = payloads[-1]
        choices = terminal["response"]["choices"]
        if len(choices) != 1:
            raise ValueError("Ambiguous terminal completion")
        choice = choices[0]
        message = choice["message"]
        fields = terminal.get("provider_specific_fields") or message
        if (
            choice.get("finish_reason") != "length"
            or message.get("content")
            or message.get("tool_calls")
            or message.get("function_call")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or fields.get("generation_token_ids")
            or fields.get("generation_log_probs")
        ):
            raise ValueError("Terminal completion is not an empty length stop")
        previous = payloads[-2]
        prior_message = previous["response"]["choices"][0]["message"]
        prior_fields = previous.get("provider_specific_fields") or prior_message
        prompt = prior_fields.get("prompt_token_ids") or []
        generated = prior_fields.get("generation_token_ids") or []
        explicit_overflow = str(terminal["response"].get("id", "")).startswith(
            "gdpval-context-overflow-"
        )
        # A tool observation can overflow the next prompt before the preceding
        # completion fills the budget. In that case require proxy provenance.
        if (
            not prompt
            or not generated
            or len(prompt) + len(generated) > context_limit
            or (not explicit_overflow and len(prompt) + len(generated) != context_limit)
        ):
            raise ValueError(
                "No context-rejection provenance or exact boundary evidence"
            )
        # Keep the original log for audit; exclude only the final zero-token response.
        payloads = payloads[:-1]
    for payload in [] if token_capture else payloads:
        message = payload["response"]["choices"][0]["message"]
        fields = payload.get("provider_specific_fields") or message
        prompt = fields.get("prompt_token_ids")
        generated = fields.get("generation_token_ids")
        probs = fields.get("generation_log_probs")
        if not prompt or not generated or not probs or len(generated) != len(probs):
            raise ValueError(
                "Missing/inconsistent behavior token metadata; refusing to export a training trajectory"
            )
        if any(not math.isfinite(x) for x in probs):
            raise ValueError("Non-finite behavior logprob")
    last = payloads[-1]
    messages = list(last["messages"])
    assistant = dict(last["response"]["choices"][0]["message"])
    assistant.update(last.get("provider_specific_fields") or {})
    messages.append(assistant)
    return messages, last["kwargs"]["tools"], len(payloads)


def export_trajectory(task, config):
    from nemo_gym.openai_utils import (
        NeMoGymResponse,
        NeMoGymResponseCreateParamsNonStreaming,
    )
    from nemo_gym.responses_converter import (
        VLLMConverter,
        split_responses_input_output_items,
    )
    from openai.types.responses import FunctionTool

    context_limit = (
        config["max_context"]
        if config.get("policy_failure_reason") == "agent_context_limit"
        else None
    )
    capture = config.get("token_capture", False)
    messages, tool_dicts, count = completion_payload(
        task, context_limit=context_limit, token_capture=capture
    )
    converter = VLLMConverter(return_token_id_information=not capture)
    items = converter.chat_completions_messages_to_responses_items(messages)
    initial, output = split_responses_input_output_items(items)
    if not output:
        raise ValueError("Empty OpenHands trajectory")
    tools = [
        FunctionTool.model_validate(t["function"] | {"type": "function"})
        for t in tool_dicts
    ]
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=initial,
        # Response models add defaults (e.g. defer_loading=None) that are
        # invalid in the request TypedDict. Preserve explicit tool fields plus
        # the required nullable strict field when converting back to a request.
        tools=[t.model_dump(exclude_unset=True) | {"strict": t.strict} for t in tools],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    response = NeMoGymResponse(
        id="gdpval-" + task.name,
        created_at=time.time(),
        model=config["policy_model"],
        object="response",
        output=output,
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True,
        status="completed",
    )
    result = {
        "responses_create_params": params.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
        "model_calls": count,
        "policy_checkpoint": config["policy_checkpoint"],
    }
    (task / "trajectory.json").write_text(json.dumps(result) + "\n")
    return params, response
