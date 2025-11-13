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

## Real-World Verification Examples

These examples demonstrate how verification combines multiple criteria with weighted priorities:

```{list-table}
:header-rows: 1
:widths: 25 50 25

* - Agent Type
  - Verification Criteria (with example weights)
  - Design Rationale
* - **Math Tutoring**
  - • **Correctness** (0.5): Final answer mathematically correct? <br>
    • **Pedagogy** (0.3): Steps clearly explained?<br>
    • **Efficiency** (0.2): Simplest method used?
  - Correct answers matter most, but teaching quality is also important
* - **Customer Service**
  - • **Accuracy** (0.4): Addresses customer question? <br>
    • **Tone** (0.3): Appropriate and professional? <br>
    • **Resolution** (0.3): Solves problem or provides next steps? 
  - Balances multiple dimensions of service quality—no single aspect dominates
* - **Code Generation**
  - • **Functionality** (0.6): All test cases pass? <br>
    • **Quality** (0.3): Readable with good structure? <br>
    • **Security** (0.1): Avoids vulnerabilities?
  - Functional correctness is primary, but code quality matters for maintainability
```
---

## Why Verification Matters

Verification measures the quality of outcomes, not just successful tool execution.

### Tool Execution Is Not Performance

An agent can successfully call tools without producing good results:

- Weather agent calls `get_weather("San Francisco")` _✓ Tool executed_
- But gives generic advice ignoring temperature data _✗ Poor performance_
- Or recommends winter clothing for 75°F weather _✗ Incorrect application_

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

Resource servers follow common patterns when implementing verification, though each adapts the pattern to its domain. Choose the pattern that best matches your task requirements:

::::{tab-set}

:::{tab-item} Correctness
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
:::

:::{tab-item} Quality
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
:::

:::{tab-item} Efficiency
**Concept**: Reward economical use of resources and concise communication.

**Tool Usage Efficiency**:
- Count number of tool calls
- Penalize unnecessary or redundant calls
- Reward minimal but sufficient tool use

**Response Quality**:
- Measure response length against optimal range
- Penalize both insufficient detail and excessive verbosity
- Balance completeness with conciseness
:::

:::{tab-item} Hybrid
**Concept**: Combine multiple verification dimensions into a composite score.

**Common combinations**:
- Correctness (0.7 weight) + Efficiency (0.3 weight)
- Accuracy (primary) + Instruction adherence (secondary)
- Test pass rate (0.6) + Code quality (0.4)

**Design consideration**: Weight the most critical dimension highest, use secondary dimensions to break ties or encourage good practices beyond minimum requirements.
:::

::::

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

Effective verification functions share three critical properties:

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Property
  - Implementation Guidelines
  - Why It Matters
* - **Reliable**
  - • Same response → same score every time <br>
    • Avoid randomness in verification logic <br>
    • Minimize external dependencies 
  - Inconsistent scoring creates noisy training signals that slow learning
* - **Meaningful**
  - • Score reflects true task performance <br>
    • Avoid proxy metrics that can be gamed <br>
    • Align with real-world success criteria
  - Agents optimize for the verification function, not your unstated intentions
* - **Scalable**
  - • Fast enough for thousands of evaluations <br>
    • Runs once per training example <br>
    • Prefer local computation over API calls
  - Verification becomes a bottleneck if it cannot scale with training demands
```

---

## Design Considerations

When designing verification logic for your resource server, these patterns address common design decisions:

::::{tab-set}

:::{tab-item} Binary vs Continuous
**When to Use Binary Rewards (0.0 or 1.0)**:
- Clear success/failure criteria exist
- Partial credit is not meaningful
- Simplifies interpretation of results

**When to Use Continuous Rewards (0.0–1.0 range)**:
- Degrees of success exist (partial correctness)
- Want to reward progress toward full solution
- Multiple criteria combined into composite score
:::

:::{tab-item} Multiple Criteria
**Process for Balancing Multiple Aspects**:

1. **Identify primary goal**: What is non-negotiable?
2. **Add secondary criteria**: What improves quality beyond minimum?
3. **Weight appropriately**: Primary 0.5–0.7, secondary split remaining
4. **Test edge cases**: Ensure weights produce sensible rankings
:::

:::{tab-item} Handling Ambiguity
**When Ground Truth Is Unclear or Multiple Answers Are Valid**:

- Use LLM-as-judge with clear evaluation criteria
- Implement swap checks to reduce bias
- Provide partial credit for partially correct answers
- Document edge cases and verification decisions
:::

::::
