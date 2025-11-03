---
orphan: true
---

(tutorial-dual-model-math)=

<!-- TODO: this is an example tutorial that has not been tested. skip reviewing it for now -->

# Configure Dual-Model Math Evaluation

Set up a math training system with separate policy and judge models for accurate evaluation—a realistic configuration pattern demonstrating multi-server orchestration.

:::{card}

**Goal**: Configure a two-model system with a policy model for problem-solving and a judge model for evaluation.

^^^

**In this tutorial, you will**:

1. Understand why policy and judge models should be separate
2. Configure the judge model placeholder in library_judge_math
3. Set up environment variables for two models
4. Use command-line overrides to connect the judge
5. Toggle judge evaluation on and off

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New to NeMo Gym? Start with Get Started
:::

## Before You Start

This tutorial assumes you've completed the [Get Started](../get-started/index.md) series and understand basic configuration concepts. For deeper configuration understanding, refer to [Configuration System](../about/concepts/configuration-system.md).

**Prerequisites**:
- Completed [Setup and Installation](../get-started/setup-installation.md)
- Understand how to run `ng_run` and `ng_collect_rollouts`
- Familiar with env.yaml for secrets management

---

## Why Separate Policy and Judge Models?

When training AI models to solve math problems, you need two distinct roles:

**Policy Model** (the model being trained):
- Solves math problems
- Generates candidate answers
- Under development, may make mistakes
- Usually a model you're fine-tuning

**Judge Model** (the evaluator):
- Verifies answer correctness
- Acts as a trusted authority
- More capable, stable model
- Often a larger or more reliable model

**Why separate them?**

::::{tab-set}

:::{tab-item} Cost Optimization

**Scenario**: Use cheaper models during development, expensive judges for validation

```bash
# Development - fast and cheap
policy_model: gpt-4o-mini (solving)
judge_model: gpt-4o (evaluating)

# Production - both high quality
policy_model: gpt-4o (solving)
judge_model: gpt-4o (evaluating)
```

**Benefit**: Save costs during iteration while maintaining evaluation quality

:::

:::{tab-item} Separate Concerns

**Scenario**: Policy model under training, judge model remains stable

- **Policy model**: Frequently updated, experimental, may regress
- **Judge model**: Frozen, trusted, consistent evaluation criteria

**Benefit**: Consistent evaluation even as policy model changes

:::

:::{tab-item} Specialized Models

**Scenario**: Different models excel at different tasks

- **Policy model**: Fast, efficient reasoning model
- **Judge model**: Specialized math evaluation model

**Benefit**: Use best-in-class models for each role

:::

::::

---

## The Configuration Challenge

The `library_judge_math` resource server requires a judge model, but the configuration has a **placeholder** that you must configure:

```yaml
# From library_judge_math.yaml
library_judge_math:
  resources_servers:
    library_judge_math:
      entrypoint: app.py
      judge_model_server:
        type: responses_api_models
        name: ???  # ← You must configure this!
      judge_responses_create_params:
        input: []
      should_use_judge: true
      domain: math
```

**The `???` means**: This value is required but intentionally left for you to configure based on your setup.

**This tutorial solves this** by showing you how to configure a second model and connect it as the judge.

---

## Step 1: Understand the Base Configuration

First, examine the configuration structure to understand what needs to be connected.

### 1.1: Explore library_judge_math Configuration

```bash
cat resources_servers/library_judge_math/configs/library_judge_math.yaml
```

**Key observations**:

```yaml
# Two server definitions in this file:
# 1. Resource server (has the ??? placeholder)
library_judge_math:
  resources_servers:
    library_judge_math:
      judge_model_server:
        type: responses_api_models  # Must reference a responses_api_models server
        name: ???                   # Name of the judge model server

# 2. Agent server (references policy_model)
library_judge_math_simple_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # The model being trained
```

**What this shows**: The agent uses `policy_model` for solving, and the resource server needs a separate judge model for evaluation.

### 1.2: Check Existing Model Configuration

Look at how the policy model is typically configured:

```bash
cat responses_api_models/openai_model/configs/openai_model.yaml
```

**Output**:
```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

**Key insight**: The server ID is `policy_model` and it uses variable interpolation for credentials.

---

## Step 2: Add Judge Model Configuration

Now create a second model server configuration for the judge model.

### 2.1: Create Judge Model Config File

Create a new configuration file for your judge model:

```bash
# Create configs directory if needed
mkdir -p responses_api_models/openai_model/configs

# Create judge-specific config
cat > responses_api_models/openai_model/configs/openai_judge_model.yaml << 'EOF'
judge_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${judge_base_url}
      openai_api_key: ${judge_api_key}
      openai_model: ${judge_model_name}
EOF
```

**What changed**:
- **Server ID**: `judge_model` (different from `policy_model`)
- **Variables**: `judge_*` prefix instead of `policy_*`
- **Implementation**: Still uses `openai_model` (same code, different instance)

**Why this works**: NeMo Gym allows multiple instances of the same implementation type with different IDs and configurations.

### 2.2: Add Judge Credentials to env.yaml

Add judge model credentials to your env.yaml:

```yaml
# Policy model (for solving problems)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-policy-key-here
policy_model_name: gpt-4o-mini

# Judge model (for evaluation)
judge_base_url: https://api.openai.com/v1
judge_api_key: sk-your-judge-key-here
judge_model_name: gpt-4o-2024-11-20
```

**Best practices**:
- ✅ **Use descriptive prefixes**: `policy_*` vs `judge_*` makes roles clear
- ✅ **Different models**: Judge model (`gpt-4o`) is more capable than policy (`gpt-4o-mini`)
- ✅ **Same or different keys**: You can use the same API key or separate ones for cost tracking

### 2.3: Organize Config Paths in env.yaml

For convenience, define a config collection:

```yaml
# Add to env.yaml
math_training_with_judge:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - responses_api_models/openai_model/configs/openai_judge_model.yaml
  - resources_servers/library_judge_math/configs/library_judge_math.yaml
```

---

## Step 3: Connect Judge Model via Command Line

Now connect the judge model to the resource server using command-line overrides.

### 3.1: Run with Judge Model Connected

```bash
# Load all configs and connect the judge
ng_run '+config_paths=${math_training_with_judge}' \
    +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model
```

**Breaking down the command**:

```bash
# 1. Load config files
'+config_paths=${math_training_with_judge}'
# Loads: policy_model, judge_model, and library_judge_math configs

# 2. Override judge model name
+library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model
# Replaces ??? with judge_model
```

**The configuration path anatomy**:
```
+library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model
 │                  │                    │                 │                  └─ Value
 │                  │                    │                 └─ Field to override
 │                  │                    └─ Implementation type
 │                  └─ Server type
 └─ Server ID
```

### 3.2: Verify Servers Are Running

After running the command, you should see both model servers start:

```text
[INFO] Starting server: policy_model (responses_api_models/openai_model)
[INFO] Starting server: judge_model (responses_api_models/openai_model)
[INFO] Starting server: library_judge_math (resources_servers/library_judge_math)
[INFO] Starting server: library_judge_math_simple_agent (responses_api_agents/simple_agent)
```

**Success indicators**:
- ✅ Both `policy_model` and `judge_model` servers start
- ✅ No configuration errors about missing `judge_model_server.name`
- ✅ All four servers reach "ready" state

---

## Step 4: Toggle Judge Evaluation

The `should_use_judge` flag controls whether the judge model is actually used for evaluation.

### 4.1: Understanding the Evaluation Strategy

The library_judge_math resource server uses a two-stage evaluation strategy:

```python
# From app.py lines 145-161
async def _verify_answer(
    self, question: str, expected_answer: str, generated_answer: str
) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
    
    # Stage 1: Library-based verification (math-verify)
    library_reward, extracted_answer = self._verify_answer_with_library(
        expected_answer, generated_answer
    )
    
    # Stage 2: Judge model (only if needed)
    if not self.config.should_use_judge or library_reward > 0.5:
        return library_reward, extracted_answer, library_reward, None
    
    judge_reward, judge_evaluations = await self._verify_answer_with_judge(
        question, expected_answer, generated_answer
    )
    return judge_reward, extracted_answer, library_reward, judge_evaluations
```

**The logic**:
1. **Always**: Use math library verification (fast, symbolic)
2. **Conditionally**: Use judge model if:
   - `should_use_judge: true` AND
   - Library verification failed (`library_reward <= 0.5`)

**Why this design?**:
- **Efficiency**: Library verification is fast and free
- **Cost savings**: Only call expensive judge model when necessary
- **Accuracy**: Judge model catches cases library verification misses

### 4.2: Run with Judge Enabled (Default)

```bash
ng_run '+config_paths=${math_training_with_judge}' \
    +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model

ng_collect_rollouts \
    +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=resources_servers/library_judge_math/data/example.jsonl \
    +output_jsonl_fpath=results/with_judge.jsonl \
    +limit=5
```

**Expected behavior**:
- Library verification runs on all answers
- Judge model called for answers that fail library verification
- Results include both `library_reward` and `judge_evaluations` fields

### 4.3: Run with Judge Disabled

```bash
ng_run '+config_paths=${math_training_with_judge}' \
    +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model \
    +library_judge_math.resources_servers.library_judge_math.should_use_judge=false

ng_collect_rollouts \
    +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=resources_servers/library_judge_math/data/example.jsonl \
    +output_jsonl_fpath=results/without_judge.jsonl \
    +limit=5
```

**Expected behavior**:
- Only library verification runs
- Judge model never called (faster, cheaper)
- Results show only `library_reward`, no `judge_evaluations`

### 4.4: Compare Results

Compare the two rollout files to see the impact:

```bash
# Check if judge evaluations are present
grep -c "judge_evaluations" results/with_judge.jsonl
grep -c "judge_evaluations" results/without_judge.jsonl
```

**Typical findings**:
- **With judge**: More nuanced evaluation, catches edge cases
- **Without judge**: Faster, lower cost, but may miss complex equivalences

---

## Step 5: Run a Complete Evaluation

Now collect rollouts with the dual-model system fully configured.

### 5.1: Download Training Data

```bash
ng_download_dataset_from_gitlab \
    +dataset_name=math_open_math_reasoning \
    +version=0.0.1 \
    +artifact_fpath=open_math_reasoning_problems.jsonl \
    +output_fpath=data/math_problems.jsonl
```

### 5.2: Collect Rollouts with Both Models

```bash
# Start servers with judge configured
ng_run '+config_paths=${math_training_with_judge}' \
    +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model

# In another terminal, collect rollouts
ng_collect_rollouts \
    +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=data/math_problems.jsonl \
    +output_jsonl_fpath=results/math_rollouts_dual_model.jsonl \
    +limit=20 \
    +num_samples_in_parallel=5
```

### 5.3: Examine Judge Evaluations

Check a rollout to see the dual evaluation in action:

```bash
cat results/math_rollouts_dual_model.jsonl | jq '.[0]' | head -100
```

**Look for these fields**:
```json
{
  "verify_response": {
    "reward": 1.0,
    "expected_answer": "42",
    "extracted_answer": "42",
    "library_reward": 0.0,
    "judge_evaluations": [
      {
        "responses_create_params": {
          "input": [
            {"role": "system", "content": "Please act as an impartial judge..."},
            {"role": "user", "content": "<|Problem|>\n..."}
          ]
        },
        "response": {
          "output": [
            {
              "type": "message",
              "content": [
                {
                  "type": "output_text",
                  "text": "The answers are equivalent [[A=B]]"
                }
              ]
            }
          ]
        }
      }
    ]
  }
}
```

**What this shows**:
- **`library_reward: 0.0`**: Library verification failed
- **`judge_evaluations`**: Judge model was called
- **`reward: 1.0`**: Judge determined the answer was correct
- **Result**: Judge model caught a correct answer the library missed

---

## Understanding Dual Evaluation Mechanics

Let's understand how the two-model system works under the hood.

### Judge Model Call Process

When a judge evaluation is needed, the system:

```python
# From app.py lines 231-256
async def _generate_judge_evaluation(
    self, question: str, first_answer: str, second_answer: str
) -> tuple[bool, JudgeEvaluation]:
    # 1. Build judge prompt from template
    judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        first_answer=first_answer,
        second_answer=second_answer
    )
    
    # 2. Create request with system message and prompt
    responses_create_params.input = [
        {"role": "system", "content": self.JUDGE_SYSTEM_MESSAGE},
        {"role": "user", "content": judge_prompt}
    ]
    
    # 3. Call the judge model server
    response = await self.server_client.post(
        server_name=config.judge_model_server.name,  # This is "judge_model"
        url_path="/v1/responses",
        json=responses_create_params
    )
```

**Key points**:
1. **Judge prompt**: Uses Arena Hard template for consistency
2. **Server reference**: Uses `config.judge_model_server.name` (which you configured)
3. **Asynchronous**: Judge calls don't block other operations

### Positional Bias Detection

The judge is called **twice with swapped answer orders** to detect positional bias:

```python
# From app.py lines 208-229
async def _verify_answer_with_judge(
    self, question: str, expected_answer: str, generated_answer: str
) -> tuple[float, list[JudgeEvaluation]]:
    # First evaluation: expected answer first
    first_order_equal, first_judge_evaluation = \
        await self._generate_judge_evaluation(question, expected_answer, generated_answer)
    
    if not first_order_equal:
        return 0.0, [first_judge_evaluation]
    
    # Second evaluation: generated answer first (swapped)
    second_order_equal, second_judge_evaluation = \
        await self._generate_judge_evaluation(question, generated_answer, expected_answer)
    
    if second_order_equal:
        reward = 1.0
    else:
        reward = 0.0
    
    return reward, [first_judge_evaluation, second_judge_evaluation]
```

**The strategy**:
- **Both evaluations agree on "equal"** → reward = 1.0 (confident correct)
- **First says "not equal"** → reward = 0.0 (confident incorrect, skip second)
- **First says "equal", second says "not equal"** → reward = 0.0 (positional bias detected)

**Why this matters**: Catches cases where judge model prefers whichever answer appears first.

---

## Troubleshooting

Common issues when configuring dual models:

:::{dropdown} Problem: Server 'judge_model' not found
:icon: search
:color: warning

**Error message**:
```text
AssertionError: Could not find type='responses_api_models' name='judge_model' 
in the list of available servers: [policy_model, library_judge_math, ...]
```

**What this means**: The judge_model config file wasn't loaded.

**Solutions**:

1. **Check config_paths includes judge config**:
   ```bash
   # Verify your env.yaml has the judge config
   cat env.yaml | grep -A 3 "math_training_with_judge"
   ```
   
   Should include:
   ```yaml
   math_training_with_judge:
     - responses_api_models/openai_model/configs/openai_judge_model.yaml  # ← Must be present
   ```

2. **Verify config file exists**:
   ```bash
   ls -l responses_api_models/openai_model/configs/openai_judge_model.yaml
   ```

3. **Check server ID matches**:
   ```yaml
   # In openai_judge_model.yaml, server ID must be "judge_model"
   judge_model:  # ← This must match the name in your override
     responses_api_models:
       openai_model:
   ```

:::

:::{dropdown} Problem: Missing judge credentials
:icon: key
:color: danger

**Error message**:
```text
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: judge_api_key
```

**What this means**: Judge model variables aren't defined in env.yaml.

**Solutions**:

1. **Add judge credentials to env.yaml**:
   ```yaml
   judge_base_url: https://api.openai.com/v1
   judge_api_key: sk-your-key
   judge_model_name: gpt-4o-2024-11-20
   ```

2. **Or use same credentials as policy** (for testing):
   ```yaml
   judge_base_url: ${policy_base_url}
   judge_api_key: ${policy_api_key}
   judge_model_name: ${policy_model_name}
   ```

:::

:::{dropdown} Problem: Judge never called even with should_use_judge=true
:icon: question
:color: info

**Symptom**: No `judge_evaluations` in rollout output despite `should_use_judge: true`.

**What this means**: Library verification is succeeding, so judge is skipped.

**Why this happens**:
```python
# Judge only called if library verification fails
if not self.config.should_use_judge or library_reward > 0.5:
    return library_reward, extracted_answer, library_reward, None
```

**This is expected behavior**! The judge is only used when:
- Library verification gives `library_reward <= 0.5` (uncertain/incorrect)
- `should_use_judge: true`

**To test judge is working**, try:
1. **Use harder problems** where library verification is less certain
2. **Check validation dataset** (AIME problems are harder)
3. **Examine library_reward field** in rollouts - when it's 0.0-0.5, judge should be called

:::

---

## What You've Learned

You now have hands-on experience with:

- ✓ Configuring multiple model servers with different roles (policy vs judge)
- ✓ Resolving the `???` placeholder pattern in configurations
- ✓ Using separate environment variable prefixes for different servers
- ✓ Connecting server references via command-line overrides
- ✓ Toggling conditional server usage with feature flags
- ✓ Understanding dual evaluation strategies (library + judge)

**Key configuration patterns**:
- **Multi-server setups**: Multiple instances of same implementation type
- **Server references**: `type` and `name` pattern for connecting servers
- **Conditional usage**: Feature flags like `should_use_judge`
- **Variable organization**: Prefixes (`policy_*`, `judge_*`) for clarity

---

## Next Steps

You've mastered dual-model configuration! Continue exploring:

- **[Offline Training with Rollouts](offline-training-w-rollouts.md)**: Use your dual-model rollouts for training
- **[Configuration System](../about/concepts/configuration-system.md)**: Deeper understanding of configuration mechanics
- **[How-to: Configure Multiple Environments](../how-to-faq.md#multiple-environments)**: Dev/staging/prod setups

Or return to the [Tutorials Overview](index.md) to explore other topics.

