# Verifying Agent Results

**Goal**: Understand how NeMo Gym evaluates agent performance and what verification means for training

## What is Verification?

Every resource server in NeMo Gym has a `verify()` function that **scores agent performance**. The purpose of this function is to define how to measure how well agents accomplish their goals.

**The Problem**: When you ran your weather agent, it successfully called the tool and gave a response. But was that response *good*? Should the agent be rewarded or penalized for that behavior? Without verification, there's no way to measure improvement.

**The Solution**: Each resource server must define exactly what "good performance" means for its domain.

## Why Verification Matters

**Tool Execution ≠ Good Performance**

- Your weather agent successfully called `get_weather("San Francisco")`
- But did it give helpful advice? Was the response accurate? Was it efficient?
- Verification answers these questions with numerical scores

**Training Signal**

Verification scores become the **reward signals** that drive reinforcement learning:
- High scores → "Do more of this behavior"  
- Low scores → "Avoid this behavior"
- No verification = No way to improve the agent

## Common Verification Patterns

Let's look at real examples from NeMo Gym's resource servers:

### **Correctness Verification**

**Simple Correctness** (`mcqa` - Multiple Choice Questions):
```python
# Extract agent's answer (A, B, C, or D)
pred = extract_answer_from_response(agent_response)
gold = expected_answer  # e.g., "C"

# Binary scoring: right or wrong
is_correct = (pred == gold)
reward = 1.0 if is_correct else 0.0
```

**Sophisticated Correctness** (`library_judge_math` - Math Problems):
```python
# Uses math-verify library for mathematical equivalence
library_reward = math_metric.compute(predicted_answer, expected_answer)

# PLUS an LLM judge for edge cases
judge_prompt = f"Are these answers equivalent? {predicted_answer} vs {expected_answer}"
judge_score = await llm_judge(judge_prompt)

# Combines both signals
final_reward = combine_scores(library_reward, judge_score)
```
