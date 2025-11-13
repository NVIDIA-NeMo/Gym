(tutorial-separate-policy-judge)=

# Separate Policy and Judge Models

In real AI systems, you often want different models handling different jobs—one model generates responses while another verifies their quality. This is especially useful when training a model that needs consistent evaluation, or when you want to save costs by using cheaper models for generation and expensive ones only for verification.

This tutorial walks you through configuring a dual-model setup using **math problem solving** as an example: a policy model solves problems, and a judge model verifies answers when simple verification fails.

:::{card}

**Goal**: Configure and run separate models for generation and verification.

^^^

**In this tutorial, you will**:

1. Understand when and why to separate policy and judge models
2. Configure multiple model server instances with different roles
3. Connect server references using command-line overrides
4. Implement two-stage evaluation (fast check → judge fallback)
5. Compare costs and quality with judge enabled vs. disabled

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New to NeMo Gym? Start with Get Started
:::

## Before You Start

Make sure you have these prerequisites before starting the tutorial:

**Prerequisites**:
- ✅ Completed the [Get Started](../get-started/index.md) series
- ✅ OpenAI API key with available credits
- ✅ Understand how to run `ng_run` and `ng_collect_rollouts`

---

## When to Use This Pattern

Separate policy and judge models allow you to assign different roles to different models for better control over your AI system:

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - Use Case
  - Policy Model Role
  - Judge Model Role
  - Key Benefit
* - **🎯 Model Training**
  - Model being fine-tuned (experimental, may regress)
  - Stable, frozen model for consistent evaluation
  - Consistent evaluation even as policy changes
* - **💰 Cost Optimization**
  - Fast, cheap model (e.g., `gpt-4o-mini`)
  - Expensive, accurate model only when needed (e.g., `gpt-4o`)
  - Save costs while maintaining evaluation quality
* - **🔧 Specialized Roles**
  - Creative generation or task execution
  - Factuality verification, safety filtering, quality assessment
  - Use best-in-class models for each role
```

**Example domains**: Math verification, code generation evaluation, creative writing critique, safety filtering

---

## Understand the Base Configuration

The `library_judge_math` resource server has a placeholder (`???`) for the judge model. Let's examine the configuration to understand what needs to be connected.

### Explore library_judge_math Configuration

View the resource server configuration to see how the judge model placeholder is defined:

```bash
cat resources_servers/library_judge_math/configs/library_judge_math.yaml
```

This file contains two server definitions:

::::{tab-set}

:::{tab-item} Resource Server

```yaml
library_judge_math:
  resources_servers:
    library_judge_math:
      judge_model_server:
        type: responses_api_models
        name: ???  # ← You'll configure this
```

**What it does**: Provides math verification with a judge model fallback. The `???` is where you'll specify which model acts as the judge.

:::

:::{tab-item} Agent Server

```yaml
library_judge_math_simple_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # References the policy model
```

**What it does**: Uses the policy model to solve math problems.

:::

::::

**Key takeaway**: The agent uses `policy_model` for solving, and the resource server needs a separate judge model for evaluation (currently `???`).

### Check Existing Model Configuration

Look at how the policy model is typically configured:

```bash
cat responses_api_models/openai_model/configs/openai_model.yaml
```

You'll see:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

**Key insight**: The server ID is `policy_model` (referenced by the agent server), and credentials use variable interpolation from `env.yaml` with the `policy_` prefix.

---

## Add Judge Model Configuration

Now create a second model server configuration for the judge model.

### Create Judge Model Config File

Create a new YAML configuration that defines the judge model server with its own credentials:

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

### Add Judge Credentials to env.yaml

Update your `env.yaml` to include credentials for both the policy and judge models:

```yaml
# Policy model (already configured from get-started)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-policy-key-here
policy_model_name: gpt-4o-mini

# Judge model (add these new entries)
judge_base_url: https://api.openai.com/v1
judge_api_key: sk-your-judge-key-here
judge_model_name: gpt-4o-2024-11-20
```

**Best practices**:

✅ **Use descriptive prefixes**: `policy_*` vs `judge_*` makes roles clear  
✅ **Different models**: Judge model (`gpt-4o`) is more capable than policy (`gpt-4o-mini`)  
✅ **Same or different keys**: You can use the same API key or separate ones for cost tracking

### Organize Config Paths in env.yaml

Define a named collection in `env.yaml` that bundles all three configuration files together:

```yaml
# Add to env.yaml
math_training_with_judge:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - responses_api_models/openai_model/configs/openai_judge_model.yaml
  - resources_servers/library_judge_math/configs/library_judge_math.yaml
```

**Why this helps**: Instead of typing three config paths every time you run `ng_run`, you can use `'+config_paths=${math_training_with_judge}'` as a shorthand.

---

## Connect Judge Model via Command Line

Now load all three configs and connect the judge model using a command-line override:

```bash
ng_run '+config_paths=${math_training_with_judge}' \
    +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model
```

::::{tab-set}

:::{tab-item} Command Breakdown

**What each part does**:

1. Loads config files:
   ```bash
   '+config_paths=${math_training_with_judge}'
   # Loads: policy_model, judge_model, and library_judge_math configs
   ```

2. Overrides judge model name:
   ```bash
   +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model
   # Replaces ??? with judge_model
   ```

:::

:::{tab-item} Path Anatomy

**Understanding the override path structure**:

```bash
+library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model
 │                  │                    │                 │                  └─ Value
 │                  │                    │                 └─ Field to override
 │                  │                    └─ Implementation type
 │                  └─ Server type
 └─ Server ID
```

:::

::::

### Verify Servers Are Running

Check the terminal output to confirm all four servers started successfully:

```text
[INFO] Starting server: policy_model (responses_api_models/openai_model)
[INFO] Starting server: judge_model (responses_api_models/openai_model)
[INFO] Starting server: library_judge_math (resources_servers/library_judge_math)
[INFO] Starting server: library_judge_math_simple_agent (responses_api_agents/simple_agent)
```

**Success indicators**:

✅ Both `policy_model` and `judge_model` servers start  
✅ No configuration errors about missing `judge_model_server.name`  
✅ All four servers reach "ready" state

---

## Toggle Judge Evaluation

The `should_use_judge` flag controls whether the judge model gets called. Before toggling it, let's understand how the two-stage verification works.

### How Two-Stage Verification Works

The resource server checks answers in two stages:

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

**The two-stage logic**:
1. **Always** run fast library verification first (free, symbolic math)
2. **Only if it fails** (`library_reward <= 0.5`) and `should_use_judge: true`, call the judge model

This design saves costs by using the expensive judge model only when the cheap library check is uncertain.

### Compare Judge Enabled vs. Disabled

Run rollout collection with and without the judge model to compare the trade-offs:

::::{tab-set}

:::{tab-item} With Judge (Default)

**Command**:
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

✅ Library verification runs on all answers  
✅ Judge model called for answers that fail library verification  
✅ Results include both `library_reward` and `judge_evaluations` fields

**Use when**: You need high accuracy and can afford the additional API costs

:::

:::{tab-item} Without Judge

**Command**:
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

✅ Only library verification runs  
✅ Judge model never called (faster, cheaper)  
✅ Results show only `library_reward`, no `judge_evaluations`

**Use when**: You want fast iteration during development or library verification is sufficient

:::

::::

### Compare Results

Analyze the differences between the two rollout files:

```bash
# Check if judge evaluations are present
grep -c "judge_evaluations" results/with_judge.jsonl
grep -c "judge_evaluations" results/without_judge.jsonl
```

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Aspect
  - With Judge
  - Without Judge
* - **Evaluation Quality**
  - More nuanced, catches edge cases
  - Fast but may miss complex equivalences
* - **API Costs**
  - Higher (calls GPT-4 for uncertain cases)
  - Lower (library verification only)
* - **Speed**
  - Slower (additional model calls)
  - Faster (no model calls)
* - **Best For**
  - Final evaluation, production training
  - Development, rapid iteration
```

---

## Run a Complete Evaluation

Now collect rollouts with the dual-model system fully configured.

### Download Training Data

Pull a dataset of math problems from the NeMo Gym repository:

```bash
ng_download_dataset_from_gitlab \
    +dataset_name=math_open_math_reasoning \
    +version=0.0.1 \
    +artifact_fpath=open_math_reasoning_problems.jsonl \
    +output_fpath=data/math_problems.jsonl
```

### Collect Rollouts with Both Models

Start the servers with the judge model configured, then collect rollouts in a separate terminal:

```bash
# Start servers with judge configured
ng_run '+config_paths=${math_training_with_judge}' \
    +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model

# In another terminal, collect rollouts
ng_collect_rollouts \
    +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=data/math_problems.jsonl \
    +output_jsonl_fpath=results/math_rollouts_dual_model.jsonl \
    +limit=5 \
    +num_samples_in_parallel=1
```

:::{tip}
Using `limit=5` keeps costs low for initial testing. Increase once you've verified the setup works.
:::

### Examine Judge Evaluations

Inspect a rollout to see how both library and judge evaluations appear in the output:

```bash
cat results/math_rollouts_dual_model.jsonl | jq '.[0]' | head -100
```

:::{dropdown} View example judge evaluation output
:icon: code

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

:::

**What this shows**:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Field
  - Meaning
* - `library_reward: 0.0`
  - Library verification failed (symbolic math couldn't verify)
* - `judge_evaluations: [...]`
  - Judge model was invoked because library failed
* - `reward: 1.0`
  - Judge determined the answer was correct
* - **Result**
  - Judge model caught a correct answer the library missed
```

---

## Understanding Dual Evaluation Mechanics

Let's understand how the two-model system works under the hood.

### Judge Model Call Process

When a judge evaluation is needed, the system follows these steps:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Step
  - Action
* - **1. Build Prompt**
  - Format judge prompt using Arena Hard template with question and both answers
* - **2. Create Request**
  - Package prompt with system message into API request format
* - **3. Call Judge**
  - Send request to judge model server via configured server reference
* - **4. Parse Response**
  - Extract equivalence judgment from judge model output
```

:::{dropdown} View implementation details
:icon: code

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

**Key implementation details**:
- Uses Arena Hard template for consistency with established evaluation practices
- Server reference comes from `config.judge_model_server.name` (the value you configured)
- Asynchronous calls ensure judge evaluations don't block other operations

:::

### Positional Bias Detection

See how the system calls the judge twice with swapped answer orders to ensure consistent evaluation:

**The evaluation strategy**:

```{list-table}
:header-rows: 1
:widths: 40 30 30

* - Scenario
  - Reward
  - Interpretation
* - Both evaluations say "equal"
  - 1.0
  - Confident correct (no bias detected)
* - First says "not equal"
  - 0.0
  - Confident incorrect (skip second call)
* - First says "equal", second says "not equal"
  - 0.0
  - Positional bias detected (judge inconsistent)
```

**Why this matters**: Catches cases where judge model prefers whichever answer appears first, ensuring evaluation quality.

:::{dropdown} View positional bias detection implementation
:icon: code

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

:::

---

## Troubleshooting

Common issues when configuring dual models:

:::{dropdown} Problem: Port already in use
:icon: alert
:color: danger

**Error message**:
```text
ERROR: [Errno 48] error while attempting to bind on address ('127.0.0.1', 11000): address already in use
```

**What this means**: Previous server processes didn't shut down cleanly, leaving zombie processes holding ports.

**Solution**:

```bash
# Kill all NeMo Gym and Ray processes
pkill -9 -f "ng_run"
pkill -9 -f "python app.py"
pkill -9 -f "ray::"

# Verify ports are free
lsof -nP -iTCP:8000,11000 | grep LISTEN

# Should return nothing. If ports are free, restart:
ng_run '+config_paths=${math_training_with_judge}' \
    +library_judge_math.resources_servers.library_judge_math.judge_model_server.name=judge_model
```

**Prevention**: 
- Always stop servers gracefully with `Ctrl+C` rather than killing terminal windows
- If servers hang, use `pkill` commands above before restarting

**Why this happens**: Ray's distributed architecture makes process cleanup challenging when servers crash or are interrupted.

:::

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

- Configuring multiple model servers with different roles (policy vs judge)
- Resolving the `???` placeholder pattern in configurations
- Using separate environment variable prefixes for different servers
- Connecting server references via command-line overrides
- Toggling conditional server usage with feature flags
- Understanding dual evaluation strategies (library + judge)

**Key Insights**:
- **Multi-server setups**: Multiple instances of same implementation type
- **Server references**: `type` and `name` pattern for connecting servers
- **Conditional usage**: Feature flags like `should_use_judge`
- **Variable organization**: Prefixes (`policy_*`, `judge_*`) for clarity

---

## Next Steps

You've completed dual-model configuration! Continue exploring:

- **[Offline Training with Rollouts](offline-training-w-rollouts.md)**: Use your dual-model rollouts for training
- **[Configuration System](../about/concepts/configuration-system.md)**: Deeper understanding of configuration mechanics
- **[How-to: Configure Multiple Environments](../how-to-faq.md#multiple-environments)**: Dev/staging/prod setups

Or return to the [Tutorials Overview](index.md) to explore other topics.

