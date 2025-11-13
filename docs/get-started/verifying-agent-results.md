
### **Quality Verification** 

**Instruction Following** (`instruction_following`):
```python
# Check if response follows specific instructions
instructions = ["Use exactly 3 sentences", "Include the word 'banana'", "End with a question"]
follow_list = []

for instruction in instructions:
    follows = instruction_checker.verify(agent_response, instruction)
    follow_list.append(follows)

# Only reward if ALL instructions followed
reward = 1.0 if all(follow_list) else 0.0
```

### **Efficiency Verification**

**Tool Usage Patterns**:
```python
# Count unnecessary tool calls
tool_calls = count_tool_calls(agent_response)
expected_calls = 1  # Should only need one weather call

# Penalize inefficiency  
efficiency_score = 1.0 - abs(tool_calls - expected_calls) * 0.2
reward = max(0.0, efficiency_score)
```

**Response Length**:
```python
# Prefer concise but complete responses
response_length = len(agent_response.split())
optimal_length = 50  # words

if response_length <= optimal_length:
    reward = 1.0
else:
    # Penalize verbosity
    reward = max(0.5, 1.0 - (response_length - optimal_length) * 0.01)
```

## From Verification to Training

### **How Rewards Drive Learning**

1. Agent generates response → Gets verification score
2. RL algorithm uses score to update model parameters  
3. Higher-scoring behaviors become more likely
4. Lower-scoring behaviors become less likely
5. Agent improves over many training iterations

### **What Makes Good Verification**

**Reliable**: Same response should get same score consistently
```python
# Good: Deterministic scoring
reward = 1.0 if predicted_answer == expected_answer else 0.0

# Bad: Random or inconsistent scoring  
reward = random.uniform(0.8, 1.0) if correct else random.uniform(0.0, 0.2)
```

**Meaningful**: Scores should reflect actual task performance
```python
# Good: Measures what you care about
reward = accuracy_score + helpfulness_score + efficiency_score

# Bad: Measures irrelevant details
reward = 1.0 if response.startswith("Hello") else 0.0
```

**Scalable**: Can handle thousands of evaluations per second during training
```python
# Good: Fast, local computation
reward = simple_string_match(predicted, expected)

# Bad: Expensive API calls for every verification
reward = await expensive_api_call(predicted, expected)
```

## Real-World Verification Examples

**Math Tutoring Agent**:
- Correctness: Did the agent solve the problem correctly?
- Pedagogy: Did it explain the steps clearly?
- Efficiency: Did it use the simplest method?

**Customer Service Agent**:
- Accuracy: Did it answer the customer's question?
- Politeness: Was the tone appropriate?
- Resolution: Did it solve the customer's problem?

**Code Generation Agent**:
- Functionality: Does the code run correctly?
- Quality: Is it well-structured and readable?
- Security: Does it avoid common vulnerabilities?

## What You've Learned

This verification system is what makes NeMo Gym powerful for agent training:
- **Resource servers** provide both tools AND scoring systems
- **Verification patterns** vary by domain but follow common principles  
- **Reward signals** from verification drive agent improvement through RL
- **Good verification** is reliable, meaningful, and scalable

Now that you understand how agent performance is measured, the next step is learning how to systematically collect this verification data at scale through rollout generation.

→ **[Next: Rollout Collection Fundamentals](05-rollout-collection.md)**
