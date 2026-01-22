(env-creating-training-environment)=
# Creating a Training Environment

Implement verification logic, prepare training data, and connect to NeMo RL.

:::{card}

**Goal**: Build a complete training environment with verification, data preparation, and NeMo RL integration.

^^^

**In this tutorial, you will**:

1. Implement a `verify()` method to compute rewards
2. Prepare training data with `ng_prepare_data`
3. Collect rollouts and connect to NeMo RL

:::

:::{button-ref} /tutorials/creating-resource-server
:color: secondary
:outline:
:ref-type: doc

← Previous: Creating a Resource Server
:::

```{mermaid}
flowchart LR
    A[Create Server] --> B[Implement verify]
    B --> C[Create example.jsonl]
    C --> D[ng_prepare_data]
    D --> E[ng_collect_rollouts]
    E --> F[Create train data]
    F --> G[Train with NeMo RL]
```

---

## Quick Start

**Initialize from template**:

```bash
ng_init_resources_server +entrypoint=resources_servers/my_env
```

Creates:

```
resources_servers/my_env/
├── app.py              # verify() template
├── configs/my_env.yaml # Dataset config
├── data/.gitignore
├── requirements.txt
├── tests/test_app.py
└── README.md
```

A training environment requires:

| Component | Description |
|-----------|-------------|
| `verify()` method | Computes reward from model response |
| `example.jsonl` | Exactly 5 examples for PR validation |
| YAML config | Links datasets to agent |

---

## Complete Example

Minimal math verification server:

:::{dropdown} app.py (click to expand)
:icon: code

```python
from typing import Optional
from fastapi import FastAPI
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig, BaseRunRequest, BaseVerifyRequest,
    BaseVerifyResponse, SimpleResourcesServer,
)

class MathVerifyConfig(BaseResourcesServerConfig):
    pass

class MathRunRequest(BaseRunRequest):
    expected_answer: str

class MathVerifyRequest(MathRunRequest, BaseVerifyRequest):
    pass

class MathVerifyResponse(BaseVerifyResponse):
    extracted_answer: Optional[str]

class MathResourcesServer(SimpleResourcesServer):
    config: MathVerifyConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: MathVerifyRequest) -> MathVerifyResponse:
        # Extract answer from \boxed{} pattern
        extracted = None
        for output in reversed(body.response.output):
            if output.type == "message":
                for content in output.content:
                    if content.type == "output_text" and "\\boxed{" in content.text:
                        start = content.text.find("\\boxed{") + 7
                        end = content.text.find("}", start)
                        extracted = content.text[start:end].strip()
                        break
                if extracted:
                    break

        is_correct = extracted == body.expected_answer.strip()
        return MathVerifyResponse(
            **body.model_dump(),
            reward=1.0 if is_correct else 0.0,
            extracted_answer=extracted,
        )

if __name__ == "__main__":
    MathResourcesServer.run_webserver()
```

:::

**data/example.jsonl** (exactly 5 lines required):

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 2+2? Answer in \\boxed{}."}]}, "expected_answer": "4"}
{"responses_create_params": {"input": [{"role": "user", "content": "What is 3*3? Answer in \\boxed{}."}]}, "expected_answer": "9"}
{"responses_create_params": {"input": [{"role": "user", "content": "What is 10-7? Answer in \\boxed{}."}]}, "expected_answer": "3"}
{"responses_create_params": {"input": [{"role": "user", "content": "What is 8/2? Answer in \\boxed{}."}]}, "expected_answer": "4"}
{"responses_create_params": {"input": [{"role": "user", "content": "What is 5+5? Answer in \\boxed{}."}]}, "expected_answer": "10"}
```

---

## Verification Patterns

The `verify()` method compares model response to expected output and returns a reward.

**Base types** (`nemo_gym.base_resources_server`):

| Type | Contains |
|------|----------|
| `BaseRunRequest` | `responses_create_params` |
| `BaseVerifyRequest` | Adds `response: NeMoGymResponse` |
| `BaseVerifyResponse` | Adds `reward: float` |

**Three reward patterns**:

:::::{tab-set}

::::{tab-item} Binary (Exact Match)

Use for tasks with clear right/wrong answers:

```python
# Pattern from resources_servers/mcqa/app.py
async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
    # Extract model's answer from response
    extracted = self._extract_answer(body.response)
    
    # Compare to expected
    expected = body.expected_answer.strip().upper()
    is_correct = (extracted == expected)
    reward = 1.0 if is_correct else 0.0

    return MyVerifyResponse(
        **body.model_dump(),
        reward=reward,
        extracted_answer=extracted,
    )
```

**Best for**: Multiple choice, factual Q&A, classification

::::

::::{tab-item} Structured Output

Use when the model must call specific functions:

```python
# Pattern from resources_servers/example_multi_step/app.py
import json

async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
    expected = body.expected_values
    
    # Find the function call in the response
    actual = []
    for output in reversed(body.response.output):
        if output.type == "function_call" and output.name == "submit_answer":
            actual = json.loads(output.arguments)["values"]
            break

    reward = 1.0 if (expected == actual) else 0.0
    
    return MyVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Tool calling, multi-step extraction

::::

::::{tab-item} Library + Judge

Use library verification with LLM judge fallback:

```python
# Pattern from resources_servers/math_with_judge/app.py
async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
    # Try fast library check first
    library_reward = self._check_with_library(
        body.expected_answer, 
        body.response
    )
    
    if library_reward > 0.5:
        return MyVerifyResponse(**body.model_dump(), reward=library_reward)
    
    # Fall back to LLM judge for edge cases
    judge_reward = await self._check_with_judge(body)
    return MyVerifyResponse(**body.model_dump(), reward=judge_reward)
```

**Best for**: Math, open-ended generation

::::

:::::

### Custom Request/Response Classes

```python
from typing import Optional
from nemo_gym.base_resources_server import (
    BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse,
)

class MyRunRequest(BaseRunRequest):
    expected_answer: str  # Your verification field

class MyVerifyRequest(MyRunRequest, BaseVerifyRequest):
    pass

class MyVerifyResponse(BaseVerifyResponse):
    extracted_answer: Optional[str] = None  # Diagnostic field
```

---

## Data Format

JSONL with one example per line:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]}, "expected_answer": "4"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `responses_create_params.input` | ✓ | OpenAI-compatible messages |
| Task-specific fields | ✓ | Fields your `verify()` expects |
| `responses_create_params.tools` | | Tool definitions |
| `responses_create_params.temperature` | | Sampling temperature |
| `id` | | Tracking identifier |

---

## Dataset Configuration

:::{dropdown} configs/my_env.yaml (click to expand)
:icon: file-code

```yaml
my_resources_server:
  resources_servers:
    my_env:
      entrypoint: app.py
      domain: math  # math | coding | agent | knowledge | instruction_following | other

my_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/my_env/data/train.jsonl
        num_repeats: 1
        license: Apache 2.0
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/my_env/data/validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_env/data/example.jsonl
```

:::

| Dataset Type | Size | Purpose |
|--------------|------|---------|
| `train` | 1,000+ | Training data |
| `validation` | 100-1,000 | Progress tracking |
| `example` | **Exactly 5** | PR validation |

**`num_repeats`**: Duplicates in-place (`abc` → `aabbcc`) so consecutive duplicates get shuffled during training.

---

## Prepare Data

Set config paths (used in all commands below):

```bash
config_paths="resources_servers/my_env/configs/my_env.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
```

**Step 1 — Validate examples** (required for PR):

```bash
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/my_env +mode=example_validation
```

**Step 2 — Prepare training data**:

```bash
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/my_env +mode=train_preparation \
    +should_download=true +data_source=huggingface
```

:::{tip}
For HuggingFace downloads, set `hf_token: hf_xxxxx` in `env.yaml`.
:::

**Output**: `data/my_env/{train,validation}.jsonl` + `*_metrics.json`

---

## Collect Rollouts

**Start servers** (Terminal 1):

```bash
ng_run "+config_paths=[${config_paths}]"
# Wait for: "All 3 / 3 servers ready!"
```

**Collect rollouts** (Terminal 2):

```bash
ng_collect_rollouts \
    +agent_name=my_simple_agent \
    +input_jsonl_fpath=resources_servers/my_env/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/my_env/data/example_rollouts.jsonl
```

| Option | Description |
|--------|-------------|
| `+limit=N` | Process first N examples |
| `+num_repeats=K` | Run each example K times (mean@K) |
| `+num_samples_in_parallel=P` | Concurrent request limit |

**Analyze results**:

```python
import json
rewards = [json.loads(l).get("reward", 0) for l in open("resources_servers/my_env/data/example_rollouts.jsonl")]
print(f"Accuracy: {sum(r == 1.0 for r in rewards) / len(rewards):.2%}")
```

---

## Train with NeMo RL

```yaml
# my_training_config.yaml
data:
  train_jsonl_fpath: data/my_env/train.jsonl
  validation_jsonl_fpath: data/my_env/validation.jsonl
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
      - resources_servers/my_env/configs/my_env.yaml
```

```bash
python examples/nemo_gym/run_grpo_nemo_gym.py --config my_training_config.yaml
```

:::{seealso}
[NeMo RL GRPO guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md)
:::

---

## Production Checklist

| Check | Why |
|-------|-----|
| `hf_token` in `env.yaml` | Credentials out of shell history |
| `mode=example_validation` first | Catch data issues early |
| Rollouts show non-zero rewards | Verify environment works |
| `+num_samples_in_parallel` set | Avoid overwhelming servers |
| Delete `*_metrics.json` on schema change | Prevent stale metrics errors |

**Graceful shutdown**: `Ctrl+C` sends SIGINT to all servers.

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `ValidationError` during `ng_prepare_data` | Invalid JSON or missing `responses_create_params` | Check each line is valid JSON |
| `reward` always 0.0 | `verify()` not matching response format | Print `body.response.output` to debug |
| Conflicting metrics error | Stale metrics file | Delete `*_metrics.json` and re-run |
| Example count mismatch | `example.jsonl` ≠ 5 lines | Ensure exactly 5 examples |
| Servers never ready | Port conflict or config error | Check port availability, review logs |

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-a-Judge
:link: llm-as-judge
:link-type: doc
Use LLMs for flexible verification of open-ended tasks.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Environments
:link: multi-step
:link-type: doc
Build sequential tool-calling workflows.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training models on your environment.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Prepare Data
:link: /data/prepare-validate
:link-type: doc
Learn more about data formats and validation.
:::

::::
