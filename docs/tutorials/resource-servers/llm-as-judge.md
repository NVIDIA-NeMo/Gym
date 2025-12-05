(tutorial-math-verifier-server)=

# Build a Math Verifier with LLM Judge

In previous tutorials, verification was deterministic. Now you'll build a math problem server that uses an LLM to verify answers when exact string matching isn't sufficient.

:::{card}

**Goal**: Build a math verifier that uses an LLM judge for flexible answer checking.

^^^

**In this tutorial, you will**:

1. Create a math problem server with no tools (just Q&A)
2. Implement hybrid verification (rule-based + LLM fallback)
3. Configure a judge model for ambiguous cases
4. Walk away with a pattern for LLM-based verification

:::

:::{button-ref} /tutorials/resource-servers/index
:color: secondary
:outline:
:ref-type: doc

← Resource Server Patterns
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`/tutorials/creating-resource-server`
- ✅ Access to a judge model (OpenAI API or local model)
- ✅ Understanding of why exact matching fails for math

**What you'll build**: A math verification server where "x = 4", "The answer is 4", and "4.0" are all recognized as correct for the problem "2x + 5 = 13".

:::{tip}
**Reference implementations**:
- `resources_servers/math_with_judge/` — Full math verification
- `resources_servers/equivalence_llm_judge/` — General LLM judging
:::

---

## 1. Understand the Problem

Why can't we just string-match answers?

| Problem | Expected | Agent Says | Exact Match? | Actually Correct? |
|---------|----------|------------|--------------|-------------------|
| 2x + 5 = 13 | 4 | "x = 4" | ❌ | ✅ |
| 2x + 5 = 13 | 4 | "The answer is 4" | ❌ | ✅ |
| 2x + 5 = 13 | 4 | "4.0" | ❌ | ✅ |
| 2x + 5 = 13 | 4 | "x equals four" | ❌ | ✅ |

Solution: Try rule-based extraction first, fall back to LLM judge.

```{mermaid}
flowchart LR
    A["Agent Answer"] --> B{"Extract Number"}
    B -->|"Found"| C{"Matches Expected?"}
    C -->|"Yes"| D["✅ 1.0"]
    C -->|"No"| E["❌ 0.0"]
    B -->|"Ambiguous"| F["LLM Judge"]
    F --> G["Score 0.0-1.0"]
```

**✅ Success Check**: You understand why LLM verification is needed.

---

## 2. Create the Math Server

<!-- SME: Adapt from math_with_judge/app.py -->

```python
# app.py
from nemo_gym.base_resources_server import SimpleResourcesServer, BaseResourcesServerConfig
import re

class MathServerConfig(BaseResourcesServerConfig):
    judge_model: str = "gpt-4"  # Model for judging ambiguous answers

class MathServer(SimpleResourcesServer):
    
    def __init__(self, config: MathServerConfig):
        super().__init__(config)
        self.judge_model = config.judge_model
        
        # Problems with expected numeric answers
        self.problems = {
            "2x + 5 = 13": 4,
            "3x - 7 = 14": 7,
            "x/2 + 3 = 8": 10,
        }
    
    def get_tools(self):
        # No tools needed - direct Q&A
        return []
    
    def get_system_prompt(self):
        return "You are a math tutor. Solve the given problem and provide the answer."
```

---

## 3. Implement Hybrid Verification

```python
# Add to app.py

def verify(self, responses_create_params: dict, output: list) -> float:
    """Verify math answer with rule-based + LLM fallback."""
    
    # Get the problem and expected answer
    user_message = responses_create_params["input"][0]["content"]
    
    expected = None
    for problem, answer in self.problems.items():
        if problem in user_message:
            expected = answer
            break
    
    if expected is None:
        return 0.0
    
    # Get agent's response
    agent_response = ""
    for msg in reversed(output):
        if msg.get("role") == "assistant" and msg.get("content"):
            agent_response = msg["content"]
            break
    
    # Step 1: Try rule-based extraction
    score = self._rule_based_check(agent_response, expected)
    if score is not None:
        return score
    
    # Step 2: Fall back to LLM judge
    return self._llm_judge_check(agent_response, expected)

def _rule_based_check(self, response: str, expected: float) -> float | None:
    """Try to extract and match number. Returns None if ambiguous."""
    
    # Extract all numbers from response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    
    if not numbers:
        return None  # No numbers found, need LLM
    
    # Check if expected value is in extracted numbers
    for num_str in numbers:
        try:
            num = float(num_str)
            if abs(num - expected) < 0.001:
                return 1.0  # Exact match
        except ValueError:
            continue
    
    # Numbers found but none match - could be wrong or formatting issue
    if len(numbers) == 1:
        return 0.0  # Clear wrong answer
    
    return None  # Multiple numbers, ambiguous - need LLM

def _llm_judge_check(self, response: str, expected: float) -> float:
    """Use LLM to judge if response contains correct answer."""
    
    prompt = f"""Does this response contain the correct answer to a math problem?

Expected answer: {expected}
Response: {response}

Reply with only "CORRECT" or "INCORRECT"."""
    
    # Call judge model (simplified - actual implementation uses OpenAI client)
    # judge_response = call_llm(self.judge_model, prompt)
    
    # For this example, return 0.5 for ambiguous cases
    # In production, actually call the judge
    return 0.5
```

**✅ Success Check**: Clear cases resolve without LLM; ambiguous cases use judge.

---

## 4. Configure the Judge Model

Create `configs/math_server.yaml`:

```yaml
math_resources_server:
  resources_servers:
    math:
      entrypoint: app.py
      domain: math
      judge_model: gpt-4  # Or your preferred judge

math_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: math_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/math/data/example.jsonl
```

---

## 5. Create Test Data

Create `data/example.jsonl`:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Solve: 2x + 5 = 13"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "What is x if 3x - 7 = 14?"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "Find x: x/2 + 3 = 8"}], "model": "gpt-4"}}
```

---

## 6. Run and Test

```bash
ng_collect_rollouts +agent_name=math_simple_agent \
    +input_jsonl_fpath=resources_servers/math/data/example.jsonl \
    +output_jsonl_fpath=results/math_rollouts.jsonl \
    +limit=3
```

**✅ Success Check**: Various answer formats (x=4, "4", "the answer is 4") all score 1.0.

---

## Optimize Judge Calls

LLM judges add latency and cost. Optimize:

```python
# Only call judge when truly ambiguous
if self._rule_based_check(response, expected) is not None:
    return score  # Skip judge

# Batch multiple judgments
# Cache repeated comparisons
# Use smaller models for obvious cases
```

---

## Learn More

- {doc}`/training/verification/multi-verifier` — Combining multiple verification methods
- {doc}`/training/verification/index` — Verification design patterns

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Tester
:link: code-tester-server
:link-type: doc

Verify code by executing tests.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train Your Model
:link: /tutorials/integrate-training-frameworks/train-with-trl
:link-type: doc

Use your verified rollouts to train.
:::

::::
