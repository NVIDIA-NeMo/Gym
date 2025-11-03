(concepts-verifying-results)=
# Verifying Agent Results

Verification is what transforms agent behavior into measurable training signals. While agents can successfully execute tools and generate responses, verification determines whether those actions were *good*—and by how much.

Think of verification as the scoring system in a game: it defines what success looks like, measures how well the agent performed, and provides the feedback signals that drive learning through reinforcement learning.

This conceptual understanding is critical because without verification, there is no way to distinguish helpful agent behavior from harmful behavior, correct answers from incorrect ones, or efficient strategies from wasteful ones.

---

## The Verification Loop

Verification sits at a critical junction in the agent development lifecycle, connecting agent execution to learning:

```{list-table}
:header-rows: 0
:widths: 20 80

* - **1. Agent Acts**
  - Agent generates response using model and tools
* - **2. Verification Scores**
  - Resource server measures performance against domain criteria
* - **3. Reward Signal**
  - Numerical score (typically 0.0–1.0) quantifies quality
* - **4. Training Feedback**
  - RL algorithms use rewards to update agent behavior
* - **5. Improvement**
  - Higher-scoring behaviors become more likely over time
```

**Key Insight**: Every resource server in NeMo Gym implements both tool provision AND verification logic. This coupling ensures that each domain defines its own success criteria—what makes a good math solution differs fundamentally from what makes good instruction-following.

Without verification, agents can execute tools perfectly but have no signal about whether their outputs are helpful, accurate, or appropriate.

---

## Why Verification Matters

### Tool Execution Is Not Performance

An agent can successfully call tools without producing good results:

- Weather agent calls `get_weather("San Francisco")` ✓ Tool executed
- But gives generic advice ignoring temperature data ✗ Poor performance
- Or recommends winter clothing for 75°F weather ✗ Incorrect application

**Verification measures the quality of outcomes, not just successful tool execution.**

### Training Signal for Reinforcement Learning

Verification scores become the reward signals that drive RL:

- **High rewards** (0.8–1.0): Reinforce this behavior, do more of it
- **Low rewards** (0.0–0.3): Avoid this behavior, try alternatives  
- **No verification** = No gradient for improvement

The verification function defines what the agent learns to optimize for. If verification only checks tool execution, the agent learns to call tools—not to solve problems correctly.

### Domain-Specific Success Criteria

Different tasks require different notions of "good performance":

```{list-table}
:header-rows: 1
:widths: 25 75

* - Domain
  - Verification Focus
* - **Mathematics**
  - Numerical correctness, mathematical equivalence
* - **Code Generation**
  - Passes test cases, executes without errors
* - **Instruction Following**
  - Adheres to all specified constraints
* - **Question Answering**
  - Semantic equivalence with expected answer
* - **Search & Research**
  - Relevance and completeness of information
```

Each resource server encapsulates domain expertise about what constitutes success.

---

## Verification Patterns

Resource servers follow common patterns when implementing verification, though each adapts the pattern to its domain.

### Correctness Verification

**Concept**: Compare agent output against ground truth for exact or semantic equivalence.

**Simple Correctness**:
- Extract agent's answer from response
- Compare to expected answer
- Binary reward: 1.0 if correct, 0.0 if incorrect

**Example domains**: Multiple-choice questions, factual QA

**Sophisticated Correctness**:
- Use domain-specific equivalence checking (mathematical equivalence, code equivalence)
- Combine programmatic verification with LLM judges for edge cases
- Handle multiple acceptable answer formats

**Example domains**: Math problems with different notation, open-ended questions

### Quality Verification

**Concept**: Measure adherence to task requirements beyond correctness.

**Instruction Following**:
- Check each specified constraint independently
- Reward only if ALL constraints satisfied
- All-or-nothing scoring emphasizes precise compliance

**Example constraints**:
- "Use exactly three sentences"
- "Include the word 'sustainability'"
- "End with a question"

**Helpfulness and Style**:
- LLM-as-judge evaluation for subjective criteria
- Multiple aspects scored independently (politeness, clarity, completeness)
- Combined into composite reward signal

### Efficiency Verification

**Concept**: Reward economical use of resources and concise communication.

**Tool Usage Efficiency**:
- Count number of tool calls
- Penalize unnecessary or redundant calls
- Reward minimal but sufficient tool use

**Response Quality**:
- Measure response length against optimal range
- Penalize both insufficient detail and excessive verbosity
- Balance completeness with conciseness

### Hybrid Verification

**Concept**: Combine multiple verification dimensions into a composite score.

**Common combinations**:
- Correctness (0.7 weight) + Efficiency (0.3 weight)
- Accuracy (primary) + Instruction adherence (secondary)
- Test pass rate (0.6) + Code quality (0.4)

**Design consideration**: Weight the most critical dimension highest, use secondary dimensions to break ties or encourage good practices beyond minimum requirements.

---

## From Verification to Learning

### How Rewards Drive Behavior Change

The verification score becomes the reward signal in reinforcement learning:

1. **Agent generates response** → Verification scores it
2. **RL algorithm observes** reward for that behavior
3. **Model parameters update** to increase probability of high-reward behaviors
4. **Future responses** shift toward higher-scoring patterns
5. **Iterative improvement** over thousands of training examples

**Critical point**: The agent learns to optimize whatever the verification function measures. If verification has flaws or misaligned incentives, the agent will learn those flaws.

### Properties of Good Verification

**Reliable**: Deterministic and consistent scoring

- Same response should receive same score every time
- Avoid randomness in verification logic
- Minimize dependence on external factors (network latency, API variability)

**Why it matters**: Inconsistent scoring creates noisy training signals that slow learning.

**Meaningful**: Measures what you actually care about

- Score should reflect true task performance
- Avoid proxy metrics that can be gamed
- Align verification with real-world success criteria

**Why it matters**: Agents optimize for the verification function, not your unstated intentions.

**Scalable**: Fast enough for thousands of evaluations

- Verification runs once per training example
- Large-scale RL training may require millions of verifications
- Prefer local computation over expensive API calls when possible

**Why it matters**: Verification becomes a bottleneck if it cannot scale with training demands.

---

## Real-World Verification Examples

### Math Tutoring Agent

**What verification measures**:
- **Correctness** (0.5 weight): Is the final answer mathematically correct?
- **Pedagogy** (0.3 weight): Are steps clearly explained?
- **Efficiency** (0.2 weight): Is the simplest method used?

**Why this works**: Weights reflect that correct answers matter most, but teaching quality is also important.

### Customer Service Agent

**What verification measures**:
- **Accuracy** (0.4 weight): Does response address the customer question?
- **Tone** (0.3 weight): Is language appropriate and professional?
- **Resolution** (0.3 weight): Does response solve the problem or provide next steps?

**Why this works**: Balances multiple dimensions of service quality, no single aspect dominates.

### Code Generation Agent

**What verification measures**:
- **Functionality** (0.6 weight): Do all test cases pass?
- **Quality** (0.3 weight): Is code readable with good structure?
- **Security** (0.1 weight): Avoids common vulnerabilities?

**Why this works**: Functional correctness is primary, but code quality matters for maintainability.

---

## Design Considerations

### When to Use Binary vs Continuous Rewards

**Binary rewards (0.0 or 1.0)**:
- Clear success/failure criteria exist
- Partial credit is not meaningful
- Simplifies interpretation of results

**Continuous rewards (0.0–1.0 range)**:
- Degrees of success exist (partial correctness)
- Want to reward progress toward full solution
- Multiple criteria combined into composite score

### Balancing Multiple Criteria

When verification measures multiple aspects:

1. **Identify primary goal**: What is non-negotiable?
2. **Add secondary criteria**: What improves quality beyond minimum?
3. **Weight appropriately**: Primary 0.5–0.7, secondary split remaining
4. **Test edge cases**: Ensure weights produce sensible rankings

### Handling Ambiguity

When ground truth is unclear or multiple answers are valid:

- Use LLM-as-judge with clear evaluation criteria
- Implement swap checks to reduce bias
- Provide partial credit for partially correct answers
- Document edge cases and verification decisions

---

## Technical Details

:::{dropdown} Verification API Structure
**Core Types**:

All resource servers expose a `POST /verify` endpoint that accepts a `BaseVerifyRequest` and returns a `BaseVerifyResponse`.

**BaseVerifyRequest**:
```python
class BaseVerifyRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
```

Contains the original task (`responses_create_params`) and the agent's complete response trajectory (`response`).

**BaseVerifyResponse**:
```python
class BaseVerifyResponse(BaseVerifyRequest):
    reward: float
```

Returns the input request plus a `reward` field—the numerical score used for RL training.

**Key Design Choice**: Response includes full request for traceability. Each rollout contains complete context for later analysis or re-verification.
:::

:::{dropdown} Implementation Patterns by Domain
**Multiple-Choice Questions (MCQA)**:
- Extract agent's answer letter from response text
- Use regex patterns or boxed answer detection
- Compare to expected answer
- Binary reward: 1.0 if match, 0.0 otherwise

**Code Execution**:
- Extract code from agent response
- Execute in sandboxed environment
- Run test cases against implementation
- Reward based on pass rate (0.0–1.0)

**Instruction Following**:
- Enumerate all specified constraints
- Check each constraint independently
- All-or-nothing reward: 1.0 if all pass, 0.0 if any fail

**LLM-as-Judge**:
- Format prompt comparing expected vs generated answers
- Call judge model for equivalence determination
- Optional swap check to reduce bias
- Parse structured output for reward signal

**Mathematical Equivalence**:
- Use symbolic math library for equivalence checking
- Fallback to LLM judge for complex expressions
- Combine programmatic and judgement signals
- Prioritize symbolic verification when possible
:::

:::{dropdown} Example Resource Servers
**Available Implementations**:

| Resource Server | Verification Approach | Reward Range |
|----------------|----------------------|--------------|
| **mcqa** | Extract answer, compare to expected | Binary: 0.0 or 1.0 |
| **comp_coding** | Execute code, run test cases | 0.0–1.0 based on pass rate |
| **instruction_following** | Check each constraint | Binary: all pass or fail |
| **library_judge_math** | Math equivalence + LLM judge | 0.0–1.0 with hybrid scoring |
| **equivalence_llm_judge** | LLM-based answer comparison | Binary with optional swap |
| **python_math_exec** | Execute Python, check final value | 0.0–1.0 based on correctness |
| **multineedle** | Extract values, check accuracy and overlap | 0.0–1.0 with accuracy metric |

**Exploring Verification Logic**:

Each resource server in `resources_servers/` includes an `app.py` with its `verify()` implementation. Studying these provides patterns you can adapt for custom verification logic.
:::

:::{dropdown} Verification in the Collection Pipeline
**When Verification Runs**:

During rollout collection, verification happens after agent execution completes:

1. Collection orchestrator sends task to agent
2. Agent executes (may involve multiple tool calls)
3. Agent returns final response
4. Orchestrator calls resource server's `/verify` endpoint
5. Verification score added to rollout
6. Rollout with reward saved to output JSONL

**Scalability Considerations**:

- Verification runs synchronously per rollout
- Async orchestration enables parallel verification across multiple rollouts
- Fast verification (< 100ms) is ideal for high-throughput collection
- Expensive verification (LLM judge calls) may bottleneck collection

**Output Format**:

Each rollout includes the verification reward:
```json
{
  "responses_create_params": { "input": [...], "tools": [...] },
  "output": [...],
  "reward": 0.85
}
```

This format is directly consumable by RL training frameworks.
:::

---

## Best Practices

**Design verification before collecting data**: Verification logic defines what your agent learns. Get it right before scaling collection.

**Test verification edge cases**: Run verification on known good/bad examples to ensure scoring matches intuition.

**Monitor verification distributions**: If most rewards are 0.0 or 1.0, consider whether continuous scoring would provide better learning signals.

**Version verification logic**: When updating verification, version your logic and datasets to enable reproducible experiments.

**Balance speed and accuracy**: Expensive verification (API calls) slows collection. Consider caching, batching, or approximations for large-scale training.

---

## Next Steps

Now that you understand how verification transforms agent behavior into training signals:

- **{doc}`core-abstractions`** — Review how resource servers provide both tools and verification
- **{doc}`rollout-collection-fundamentals`** — See how verification scores flow into training datasets
- **{doc}`../features`** — Explore verification approaches in available resource servers

**Ready to implement verification?** Check out the {doc}`../../tutorials/04-verifying-results` tutorial for hands-on practice creating custom verification logic.
