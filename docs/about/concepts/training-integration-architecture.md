(training-integration-architecture)=

# Training Integration Architecture

How NeMo Gym connects to RL training frameworks, and why certain patterns exist.

> **Audience**: Developers integrating Gym with custom training pipelines who want to understand the underlying architecture.

---

## Overview

When Gym integrates with a training framework, it acts as the **rollout collection layer** — generating trajectories that the training framework uses for policy optimization.

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Training        │     │    NeMo Gym     │     │   Generation    │
│ Framework       │◄───►│                 │◄───►│   Backend       │
│ (policy update) │     │ (orchestration) │     │ (vLLM/SGLang)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                       │
        │                       ▼
        │               ┌─────────────────┐
        └───────────────│   Resource      │
          rewards       │   Servers       │
                        │ (tools+verify)  │
                        └─────────────────┘
```

This separation provides flexibility: training frameworks handle policy optimization, while Gym handles the complexity of multi-turn interactions, tool calling, and reward generation.

---

## Request Lifecycle

When Gym collects a rollout, each model call follows a six-step lifecycle:

```{list-table} Request Lifecycle Stages
:header-rows: 1
:widths: 10 25 65

* - Stage
  - Name
  - Description
* - LF1
  - **Request**
  - JSON payload sent to HTTP server (Responses input items or Chat Completions messages)
* - LF2
  - **Input Prompt**
  - Input items converted to a single string via chat template
* - LF3
  - **Prompt Token IDs**
  - String tokenized into model-understandable token IDs
* - LF4
  - **Generation Token IDs**
  - Model generates new sequence of token IDs
* - LF5
  - **Generation**
  - Token IDs detokenized back into text
* - LF6
  - **Response**
  - Text parsed into structured output items
```

**Key insight**: Between LF4 and LF5, information is preserved perfectly (token IDs). Between LF5 and LF3 of the *next* call, information can be lost through re-tokenization.

---

## Multi-Turn Rollout Flow

A single rollout may involve multiple sequential model calls. Consider a multi-step, multi-turn scenario:

```text
Turn 1:
  U  → User message
  [R C TC] → Model generates Reasoning, Chat, Tool Call (single model call)
  T  → Tool response (from resource server)
  [R C TC] → Model continues (second model call)
  T  → Tool response
  [C] → Model's final chat response (third model call)

Turn 2:
  U  → New user message
  ...
```

Items in brackets `[]` represent a single model call. The rollout coordinator (Gym agent) manages the flow between model calls and tool executions.

### What Gets Returned

Each model call returns:

- `prompt_token_ids` — Full context token IDs at the start of generation
- `generation_token_ids` — Newly generated token IDs
- `generation_log_probs` — Log probability of each generated token

These are the critical pieces for training: you need token IDs (not just text) and their log probabilities.

---

## The Retokenization Problem

### Why It Happens

Between model calls in a multi-turn rollout:

1. Model generates token IDs `[10, 11, 12]`
2. Token IDs are detokenized to text: `"Hello"`
3. Text is sent to tool/user and response is received
4. New context (including `"Hello"`) is tokenized for next call
5. `"Hello"` might now tokenize to `[10, 11]` or `[13]` — **different from original**

This happens because tokenizers are context-sensitive. The token boundaries can shift based on:

- Surrounding text (tokenizers may merge adjacent tokens)
- Special token handling (BOS/EOS placement)
- Whitespace normalization

### Why It Matters for Training

RL training computes gradients with respect to **token IDs**, not text. If token IDs don't align:

```text
Training step expects: [10, 11, 12, 20, 21, 22]
Actual sequence:       [13, 20, 21, 22]
                        ↑
                    Mismatch! Gradients will be wrong.
```

### How Gym Handles It

Gym preserves the original token IDs from each generation and validates continuity:

```python
# From nemo_rl/environments/penguin.py
assert (
    seen_token_ids
    == output_item_dict["prompt_token_ids"][: len(seen_token_ids)]
), "Non-contiguous messages found!"
```

If validation fails, the rollout is flagged — this prevents training on corrupted data.

---

## HTTP Server Architecture

### Why HTTP?

Training frameworks and Gym often run as separate processes, potentially on different nodes:

```text
Node 1: Training Framework (policy weights, optimizer)
Node 2: vLLM (GPU inference)
Node 3: Gym Servers (agents, resources)
```

HTTP provides a clean boundary that:

- Works across process/node boundaries
- Supports async parallel requests
- Follows a well-understood schema (OpenAI API)
- Enables independent scaling of components

### Required Endpoints

Gym requires these OpenAI-compatible endpoints:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Endpoint
  - Purpose
* - `/v1/chat/completions`
  - Generate completions with tool calling support
* - `/v1/models`
  - List available models (for validation)
* - `/tokenize` (optional)
  - Direct tokenization access
```

---

## Data Flow During Training

### Offline Rollout Collection

```text
1. Training framework requests rollouts
2. Gym sends prompts to generation backend
3. Generation backend returns completions
4. Gym agent orchestrates tool calls
5. Resource server computes rewards
6. Gym returns rollouts with token IDs, log probs, rewards
7. Training framework updates policy
```

### Online (On-Policy) Training

Same flow, but rollouts feed directly into training steps:

```text
┌─────────────────────────────────────────────────┐
│                 Training Loop                    │
│                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ Collect  │───►│ Process  │───►│ Update   │ │
│   │ Rollouts │    │ Rollouts │    │ Policy   │ │
│   └──────────┘    └──────────┘    └──────────┘ │
│        │                               │        │
│        └───────────────────────────────┘        │
│                  (repeat)                        │
└─────────────────────────────────────────────────┘
```

The policy weights change after each update, so new rollouts use the updated policy — this is what makes it "on-policy."

---

## Integration Patterns

### Pattern 1: Gym as Environment (NeMo RL)

NeMo RL treats Gym as an "environment" in the RL sense:

```python
# Simplified from nemo_rl/environments/penguin.py
class Penguin(EnvironmentInterface):
    async def run_rollouts(self, examples, tokenizer):
        results = self.rollout_helper.run_examples(examples)
        return [self._postprocess(r, tokenizer) for r in results]
```

The training framework calls `run_rollouts()` and receives training-ready data.

### Pattern 2: Gym as Data Source

For frameworks that don't fit the environment abstraction, Gym can be a data source:

```python
# Collect rollouts independently
rollouts = gym.collect_rollouts(prompts)

# Save to disk
save_jsonl(rollouts, "rollouts.jsonl")

# Training framework loads rollouts
training_data = load_jsonl("rollouts.jsonl")
```

### Pattern 3: Custom Integration

For full control, implement the integration yourself:

1. Expose OpenAI-compatible endpoint from your generation backend
2. Initialize Gym's `RolloutCollectionHelper`
3. Process rollouts with token alignment validation
4. Feed into your training step

Refer to {doc}`/tutorials/training-framework-integration/index` for a step-by-step guide.

---

## Performance Considerations

### Parallelism

Gym supports parallel rollout collection:

```yaml
global_aiohttp_connector_limit_per_host: 16384
global_aiohttp_connector_limit: 65536
```

Adjust these based on your cluster's network capacity and generation backend throughput.

### Batching

Larger batches amortize overhead but increase memory:

```text
Batch size 8:  Lower throughput, lower memory
Batch size 64: Higher throughput, higher memory
```

Find the sweet spot for your hardware.

### Network Topology

Minimize network hops between components:

```text
Preferred: Gym + Generation on same node (localhost)
Acceptable: Gym + Generation on same rack (fast network)
Avoid: Gym + Generation across data centers
```

---

## Training-Time Schema Extensions

During training, Gym extends the standard OpenAI schema with additional fields for token tracking.

### Chat Completions (Training Mode)

Outside of training, an assistant message looks like:

```python
ChatCompletionMessage(
    content="<think>I'm thinking</think>Hi there!",
    tool_calls=[{...}, {...}],
    ...
)
```

During training, three extra fields are added:

```python
ChatCompletionMessage(
    content="<think>I'm thinking</think>Hi there!",
    tool_calls=[{...}, {...}],
    prompt_token_ids=[...],       # List[int] - input context token IDs
    generation_token_ids=[...],   # List[int] - generated token IDs
    generation_log_probs=[...],   # List[float] - log prob of each token
    ...
)
```

### Custom Client Requirements

If you use a custom client (LiteLLM, your own OpenAI client) instead of Gym's built-in client during training:

1. **Request**: Ensure the three extra fields pass through correctly at the message level
2. **Response**: Ensure your client correctly returns these fields from the generation backend

These fields are **critical** for training — without them, the training framework can't compute gradients.

---

## Debugging Integration Issues

### Common Symptoms and Causes

| Symptom | Likely Cause |
|---------|--------------|
| Connection refused | Generation backend not exposing HTTP server |
| Tool calls not parsing | Wrong `tool_call_parser` for model |
| Token discontinuity errors | Retokenization issue or context truncation |
| Zero rewards | Verification logic returning default |
| Slow rollouts | Network latency or low parallelism |

### Diagnostic Steps

1. **Test endpoint directly**: `curl http://localhost:8000/v1/models`
2. **Check token alignment**: Print token IDs before/after detokenization
3. **Inspect full rollout**: Log the complete rollout JSON for analysis
4. **Enable verbose logging**: Set `PENGUIN_LOG_LEVEL=DEBUG`

---

## Related Topics

- {doc}`core-abstractions` — Models, Resources, and Agents
- {doc}`key-terminology` — RL and Gym vocabulary
- {doc}`/tutorials/training-framework-integration/index` — Step-by-step integration tutorial

