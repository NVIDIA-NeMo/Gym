[3 tools called]

# Full Pipeline Testing Guide

## Model Configuration

### Where to Choose Models

All model configuration happens in **`env.yaml`** at the Gym root:

```yaml
# env.yaml - EDIT THIS FILE to change models
policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-5-2025-08-07   # <-- Change this for different models
```

### Available Model Options

| Model Name | Type | Best For |
|------------|------|----------|
| `gpt-5-2025-08-07` | Reasoning | High-quality generation, complex tasks |
| `gpt-4.1-2025-04-14` | Chat | Fast, reliable, cost-effective |
| `gpt-4o` | Chat | Multimodal, fast |
| `gpt-4-turbo` | Chat | Large context window |
| `gpt-3.5-turbo` | Chat | Budget-friendly, fast |

### Separate Judge Model (Optional)

By default, the **same model** is used for both:
- **Policy generation** (the model being trained/tested)
- **LLM Judge** (validates stylistic/semantic instructions)

To use a **different model for the judge**, edit `turing_vif.yaml`:

```yaml
judge_model: gpt-4.1-2025-04-14  # Use GPT-4.1 for faster judging
```

---

## Full Pipeline Test Steps

### Step 1: Configure Your Model

Edit `env.yaml`:
```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-key-here
policy_model_name: gpt-5-2025-08-07
```

### Step 2: Create Test Data

Create a test file (e.g., `my_test.jsonl`):
```json
{"id": 1, "instructions": [{"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 50}], "llm_judge": [], "responses_create_params": {"input": [{"role": "user", "content": "Explain quantum computing in simple terms."}]}}
{"id": 2, "instructions": [{"instruction_id": "stylistic:tone_formality", "tone_level": "formal"}], "llm_judge": [], "responses_create_params": {"input": [{"role": "user", "content": "Write a business proposal introduction."}]}}
{"id": 3, "instructions": [], "llm_judge": [{"uid": 1, "content": "Does the response include specific examples?"}], "responses_create_params": {"input": [{"role": "user", "content": "Explain the benefits of exercise."}]}}
```

### Step 3: Start Servers

```bash
cd /home/dhrutisundar03/turing/tooling/Gym
source .venv/bin/activate
ray stop --force  # Clean up any existing processes

ng_run "+config_paths=[resources_servers/turing_vif/configs/turing_vif.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

Wait for: **"All 3 / 3 servers ready!"**

### Step 4: Run Tests

In a **new terminal**:
```bash
cd /home/dhrutisundar03/turing/tooling/Gym
source .venv/bin/activate

ng_collect_rollouts \
    +agent_name=turing_vif_simple_agent \
    +input_jsonl_fpath=my_test.jsonl \
    +output_jsonl_fpath=my_results.jsonl
```

### Step 5: View Results

```bash
cat my_results.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    if not line.strip(): continue
    d = json.loads(line)
    print(f\"\\nID: {d.get('id')} | Reward: {d['reward']} | All Pass: {d['follow_all_instructions']}\")
    for r in d.get('validation_results', []):
        emoji = '✅' if r['status'] == 'Passed' else '❌'
        print(f\"  {emoji} {r['instruction']}: {r['message'][:100]}...\")
"
```

### Step 6: Cleanup

```bash
# In the server terminal, press Ctrl+C, then:
ray stop --force
```

---

## Configuration Files Summary

| File | Purpose | What to Edit |
|------|---------|--------------|
| **`env.yaml`** | API credentials & model selection | `policy_model_name`, `policy_api_key` |
| **`turing_vif.yaml`** | Resource server & agent config | `judge_model` (optional) |
| **`openai_model.yaml`** | Model server config | Usually no changes needed |

---

## Quick Test Commands

```bash
# Test with existing example data
ng_collect_rollouts \
    +agent_name=turing_vif_simple_agent \
    +input_jsonl_fpath=resources_servers/turing_vif/data/example.jsonl \
    +output_jsonl_fpath=example_results.jsonl

# Test with LLM judge file we created earlier
ng_collect_rollouts \
    +agent_name=turing_vif_simple_agent \
    +input_jsonl_fpath=test_llm_judge.jsonl \
    +output_jsonl_fpath=test_llm_judge_results.jsonl
```