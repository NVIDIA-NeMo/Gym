(multi-verifier-training)=

# Multi-Verifier Training

Use multiple verification environments in a single training run.

---

## How It Works

Train with multiple verifiers simultaneously to handle diverse task types, combine reward signals, or use different verification strategies for different data subsets.

Each resource server has its own verification logic. To use multiple verifiers, combine their config files in your `config_paths`.

### Single Verifier (Default)

**Math only:**
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"

ng_run "+config_paths=[${config_paths}]"
```

**Search only:**
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[$config_paths]"
```

### Multiple Verifiers

**Math + Search combined:**
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[$config_paths]"
```

---

## Data Preparation

Use the same combined config paths for data preparation:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/multi_verifier \
    +mode=train_preparation
```

The `ng_prepare_data` command adds the appropriate `agent_ref` to each example, routing it to the correct verifier during training.

---

## Training Framework Integration

Pass the same combined config paths to your training framework's Gym configuration:

```bash
# Same config paths for:
# 1. ng_prepare_data
# 2. ng_run (servers)
# 3. Training framework Gym config
```

---

## Architecture

```{mermaid}
flowchart TB
    TrainingFramework["Training Framework"] --> Gym["NeMo Gym"]
    
    Gym --> Agent1["Math Agent"]
    Gym --> Agent2["Search Agent"]
    
    Agent1 --> RS1["Math Resource Server"]
    Agent2 --> RS2["Search Resource Server"]
    
    RS1 --> V1["verify()"]
    RS2 --> V2["verify()"]
    
    V1 --> Reward["Combined Rewards"]
    V2 --> Reward
    
    Reward --> TrainingFramework
```

Each example is routed to the appropriate agent/resource server based on its `agent_ref`, and each resource server provides its own verification logic and reward signal.

---

## Example: Three-Verifier Setup

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml,\
resources_servers/code_gen/configs/code_gen.yaml"

# Prepare combined data
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/multi_domain \
    +mode=train_preparation

# Run servers
ng_run "+config_paths=[$config_paths]"
```


