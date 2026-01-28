(environment-verifiers)=

# Verifiers

Integration with [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers), enabling environments from Prime Intellect's Environments Hub to run in NeMo Gym.

Verifiers provides 600+ environments across reasoning, math, and agent tasks. Environments built for Environments Hub can be deployed through NeMo Gym for training with NeMo RL. Unlike typical NeMo Gym environments, verifiers environments handle state management, verification, and tool execution internally without requiring a separate resource server.

:::{note}
**Multi-turn environments:** Currently require disabling `enforce_monotonicity` in training configuration until token propagation is fully patched.
:::

---

## Install Dependencies

Install verifiers and prime tools:

```bash
# From the Gym repository root
uv venv
source .venv/bin/activate
uv sync
uv add verifiers
uv tool install prime
```

Install an environment:

```bash
prime env install primeintellect/acereason-math
```

---

## Create Dataset

Generate example tasks:

```bash
python3 responses_api_agents/verifiers_agent/scripts/create_dataset.py \
  --env-id primeintellect/acereason-math \
  --size 5 \
  --output responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl
```

---

## Update Agent Requirements

Add to `responses_api_agents/verifiers_agent/requirements.txt`:

```txt
-e nemo-gym[dev] @ ../../
verifiers>=0.1.9
--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
acereason-math
```

---

## Configure Model Server

Create `env.yaml` at repository root:

```yaml
policy_base_url: "http://localhost:8000/v1"
policy_api_key: "dummy"
policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
```

---

## Start Model Server

```bash
uv add vllm
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

---

## Launch NeMo Gym Servers

```bash
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/verifiers_acereason-math.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

---

## Collect Rollouts

```bash
ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl \
    +limit=5
```

---

## Reference

- [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments) - Browse 600+ available environments
- [Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers) - Verifiers library
- `responses_api_agents/verifiers_agent/` - NeMo Gym agent integration
