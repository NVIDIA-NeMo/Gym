(training-handoff)=

# Handoff to Training

Pass NeMo Gym rollouts to your RL training framework.

NeMo Gym outputs training-ready JSONL files with reward scores. Your training framework consumes these directly or after simple transformations.

---

## Output Format

After collecting rollouts, you have JSONL with this structure:

```json
{
  "messages": [...],
  "response": {...},
  "reward": 0.85,
  "accuracy": 0.90,
  "metadata": {...}
}
```

**Key fields for training**:
- `messages` - Conversation history (input prompts)
- `response` - Model response with tool calls and reasoning
- `reward` - Primary training signal (0.0-1.0 or binary 0/1)
- Additional metrics - Server-specific scores (accuracy, precision, etc.)

See {doc}`datasets/format-specification` for complete field definitions.

---

## Training Framework Handoff

### NeMo-RL

NVIDIA's RL framework, integrated with the NeMo ecosystem.

```bash
# Use NeMo Gym rollouts directly
python train_nemo_rl.py \
  --data_path rollouts.jsonl \
  --config nemo_rl_config.yaml
```

**Documentation**: [NeMo-RL Training Guide](https://docs.nvidia.com/nemo/rl/)

---

### VeRL

High-performance open-source RL framework.

```bash
# Convert to VeRL trajectory format (if needed)
python convert_to_verl.py \
  --input rollouts.jsonl \
  --output verl_data.jsonl

# Train with VeRL
verl train --config verl_config.yaml
```

**Documentation**: [VeRL GitHub](https://github.com/volcengine/verl)

---

### OpenRLHF

Flexible open-source RLHF framework supporting PPO, DPO, and more.

```bash
# OpenRLHF supports flexible input formats
python train_openrlhf.py \
  --dataset rollouts.jsonl \
  --algorithm ppo \
  --config openrlhf_config.yaml
```

**Documentation**: [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)

---

### TRL (HuggingFace)

Transformer Reinforcement Learning from HuggingFace.

```python
from datasets import load_dataset
from trl import DPOTrainer

# Load NeMo Gym rollouts as HuggingFace dataset
dataset = load_dataset('json', data_files='rollouts.jsonl')

trainer = DPOTrainer(
    model=model,
    train_dataset=dataset['train'],
    ...
)
trainer.train()
```

**Documentation**: [TRL Documentation](https://huggingface.co/docs/trl/)

---

### Custom Frameworks

For proprietary or custom training systems:

1. **Load JSONL** - Standard Python JSON libraries
2. **Extract fields** - Pull `messages`, `response`, `reward` 
3. **Transform if needed** - Map to your framework's format
4. **Train** - Pass to your training pipeline

**Example**:
```python
import json

# Load NeMo Gym rollouts
rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Transform to your format
training_data = [{
    'input': r['messages'],
    'output': r['response']['output'],
    'score': r['reward']
} for r in rollouts]

# Pass to your framework
your_trainer.train(training_data)
```

---

## Before Training

**Recommended validation**:

1. **Check data quality** - See {doc}`data-quality/index`
2. **Validate format** - See {doc}`datasets/validate-format`
3. **Filter by reward** - See {doc}`datasets/prepare-for-training` for SFT/DPO filtering

---

## Common Questions

### Do I need to convert the format?

**Most frameworks**: No conversion needed - use JSONL directly

**Some frameworks**: May need simple field mapping (see examples above)

### What if my framework expects different fields?

Use a simple Python script to transform:

```python
import json

rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Transform to your framework's expected format
transformed = [{
    'your_field': r['messages'],  # Map fields
    'your_reward': r['reward'],
    # ... other mappings
} for r in rollouts]

# Write transformed data
with open('transformed.jsonl', 'w') as f:
    for item in transformed:
        f.write(json.dumps(item) + '\n')
```

### Should I filter before training?

**Yes, for SFT/DPO**: Filter by reward threshold

**Maybe, for PPO**: Depends on your training strategy

See {doc}`datasets/prepare-for-training` for filtering guidance.

---

## Related Topics

- {doc}`datasets/format-specification` - Complete JSONL field definitions
- {doc}`datasets/prepare-for-training` - Filter and prepare by algorithm (SFT/DPO/PPO)
- {doc}`data-quality/index` - Validate quality before training
- {doc}`datasets/validate-format` - Check format compliance

