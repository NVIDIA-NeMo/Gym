(openai-compatible-http-server-on-policy-correction)=

# On-Policy Corrections

When using an OpenAI-compatible HTTP server for RL training, fundamental issues arise in {term}`multi-step <Multi-step>` and {term}`multi-turn <Multi-turn>` scenarios. This page explains these problems and the corrections required for on-policy training.

## Overview

Policy optimization algorithms calculate and backpropagate through a loss calculated using log probabilities (logprobs). When rollout logprobs and token selections differ from those calculated at train time, training becomes off-policy. While algorithms can tolerate small amounts of off-policyness, excessive mismatch typically causes training runs to crash.

This page covers:

1. **Preliminaries**: Understanding HTTP request lifecycles and rollout structure
2. **Problems**: Three causes of train-generation mismatch
3. **Solutions**: Token ID fixes implemented in the generation server

## Preliminaries

### HTTP Request Lifecycle

A single OpenAI HTTP request follows this lifecycle, where each step produces a single output:

```{list-table}
:header-rows: 1
:widths: 10 20 70

* - Step
  - Name
  - Description
* - LF1
  - Request
  - JSON payload representing a rollout sent to the HTTP server endpoint (Responses input items or Chat Completions messages)
* - LF2
  - Input prompt
  - Responses input items are "chat templated" (converted from objects into a single string)
* - LF3
  - Prompt token IDs
  - Input prompt is "tokenized" (converted from string into model-understandable token IDs)
* - LF4
  - Generation token IDs
  - Prompt token IDs are sent to the model, which generates a new sequence of token IDs
* - LF5
  - Generation
  - Generation token IDs are "de-tokenized" into a string
* - LF6
  - Response
  - Generation is "parsed" into Responses output items and returned
```

### Rollout Structure

A multi-step or multi-turn rollout makes multiple sequential requests to the model endpoint. For example, a multi-step multi-turn rollout with two turns:

:::{dropdown} Example: Multi-Turn Rollout Sequence
:icon: list-ordered

1. [First turn] User message
2. Assistant Reasoning message
3. Assistant Chat
4. Assistant Tool Call
5. Tool response
6. Reasoning
7. Chat
8. Tool Call
9. Tool
10. [First turn, third step] Chat
11. [Second turn] User message
12. ...

**Abbreviated notation**: `U R C TC T R C TC T C U`

- **U**: User message (independent of model)
- **T**: Tool response (independent of model)
- **R, C, TC**: Reasoning, Chat, Tool Call (from model endpoint)

Most model endpoints return `[R C TC]` messages in a single response, so the rollout can be viewed as:
`U [R C TC] T [R C TC] T [C] U`, where brackets indicate a single model call.
:::

## Problems

Three problems cause train-generation log probability mismatch when using an OpenAI-compatible HTTP server.

### Problem 1: Re-Tokenization

**Cause**: Information loss when converting from token IDs (LF5) back to token IDs (LF3) across model calls.

:::{dropdown} Technical Details: Re-Tokenization
:icon: code-square

In the previous model call, the model may produce token IDs 1 and 2 which de-tokenize to `_Skinny` in LF5. Then in LF3 of the next call, `_Skinny` might re-tokenize to token ID 3.

At generation time, logprobs for tokens following `_Skinny` are calculated using token IDs 1 and 2. At train time, the same logprobs are calculated using token ID 3, creating a mismatch.

**Observed scenarios**:

1. **Merging**: Token IDs 1 and 2 re-tokenize to single token ID 3
   - Example: `"_Ski" + "nny"` → `"_Skinny"`
2. **Different split**: Token IDs 1 and 2 re-tokenize to different token IDs 3 and 4
   - Example: `"_Ski" + "nny"` → `"_Skin" + "ny"`
:::

### Problem 2: Re-Chat Templating

**Cause**: Information loss when converting from generation string (LF6) back to templated string (LF2) across model calls.

:::{dropdown} Technical Details: Re-Chat Templating
:icon: code-square

At LF6, the model may produce token IDs that de-tokenize to:

```xml
<tool_call><name>get_weather</name><parameters>{"city": "SF"}</parameters></tool_call>
```

This converts to an OpenAI tool call object:

```json
{"type": "function", "function": "get_weather", "arguments": "{\"city\": \"SF\"}"}
```

At LF2 in the next call, the chat template may render this differently:

```xml
<tool_call>
<name>
get_weather
</name>
<parameters>
{"city": "SF"}
</parameters>
</tool_call>
```

The deterministic chat template cannot match the stochastic model output format exactly.
:::

### Problem 3: Non-Monotonically Increasing History

**Cause**: Intentional modifications to rollout history during execution.

:::{dropdown} Technical Details: History Modification
:icon: code-square

Developers sometimes modify rollout history:

1. **Agentic coding harnesses**: Summarize or truncate prior history as rollouts grow longer
2. **Model chat templates**: Remove reasoning from input prompt across turns

These changes alter the prompt token IDs the model sees at the current call, differing from the final prompt token IDs used for training.
:::

## Solution

Two components address these problems:

### On-Policy Token ID Fix

For Problems 1 and 2, implement the on-policy token ID fix in the vLLM OpenAI HTTP server. Refer to the [NeMo RL implementation](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40).

:::{dropdown} Implementation Details: Token ID Fix
:icon: code-square

**Prerequisites**:

- `model_prefix_token_ids`: Ground truth prompt token IDs concatenated with generation token IDs from the previous model call
- `template_prefix_token_ids`: Re-templated and re-tokenized token IDs up to (not including) the final assistant message
- `template_token_ids`: Re-templated and re-tokenized token IDs for the entire rollout

**Assumption**: `template_prefix_token_ids` is a strict prefix of `template_token_ids` (requires circumventing Problem 3).

**Algorithm**:

The fix finds the position of the correct EOS token ID in `template_token_ids` and splices in `model_prefix_token_ids`.

**Example**:

1. Current request (LF1) contains rollout structure: `U A T A U A T`
2. Variables:
   - `model_prefix_token_ids`: Ground truth token IDs for `U A T A U A`
   - `template_prefix_token_ids`: Re-templated token IDs for `U A T A U A`
   - `template_token_ids`: Re-templated token IDs for `U A T A U A T`
3. Use `template_prefix_token_ids` to find the EOS token position corresponding to `model_prefix_token_ids`
4. Splice `template_token_ids` prefix with `model_prefix_token_ids`
:::

### Reasoning Truncation Handling

For Problem 3, disable reasoning truncation across turns using the chat template. Handling non-monotonic history during training remains an open research question.

## Related Topics

- {doc}`generation-backend-and-openai-compatible-http-server` - Generation backend requirements
- {doc}`gym-integration-footprint-and-form-factor` - Full integration component breakdown
