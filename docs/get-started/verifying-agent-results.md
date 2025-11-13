(gs-verifying-agent-results)=

# Verifying Agent Results

In the last tutorial, you ran your weather agent and saw it call the `get_weather` tool and generate responses. But how do you know if those responses were *good*? That's what verification solves.

:::{card}

**Goal**: Build and test verification logic that scores your weather agent's performance.

^^^

**In this tutorial, you will**:

1. Examine the current `verify()` function in the weather resource server
2. Build quality scoring based on tool usage and response content
3. Create a test script to observe different reward scores
4. Experiment with stricter verification criteria

:::

:::{button-ref} first-agent
:color: secondary
:outline:
:ref-type: doc

← Previous: Your First Agent
:::

:::{tip}
**Terminology note**: For definitions of key terms like **rollout**, **reward signal**, and **verification**, refer to the [RL Terms Glossary](../about/glossary.md).
:::

---

## Examine the Current Verification Function

Your weather agent's resource server already has a `verify()` function, but it doesn't actually check anything—it just returns a fixed [reward](../about/glossary.md#reward) of 1.0 regardless of performance.

Open `resources_servers/example_simple_weather/app.py` and locate the verify function:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

**What this does**:

- Receives the agent's complete response in `body.response`
- Returns a reward of 1.0 (perfect score) no matter what

**The problem**: Every response gets the same score, so there is no signal about quality. The agent cannot learn what "good" looks like.

---

## Build Quality Verification

Let us create verification logic that actually measures performance. You will build this up in three stages, from simple to sophisticated.

### 1. Check Tool Usage

Start by verifying the most basic requirement: did the agent call the weather tool?

1. Replace the `verify()` function in `resources_servers/example_simple_weather/app.py`:

   ```python
   async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
       """Check if agent called the weather tool"""
       response_output = body.response.output
       
       # Did agent call the weather tool?
       tool_called = any(
           item.type == "function_call" and item.name == "get_weather"
           for item in response_output
       )
       
       reward = 1.0 if tool_called else 0.0
       return BaseVerifyResponse(**body.model_dump(), reward=reward)
   ```

2. **Save the file** and restart your NeMo Gym servers for the changes to take effect.

**What this checks**: Binary verification—either the agent used the tool (1.0) or didn't (0.0).

---

### 2. Add Response Quality

Now verify that the agent not only called the tool, but also used the weather data in its response.

1. Update the `verify()` function:

   ```python
   async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
       """
       Score agent performance based on:
       1. Tool usage
       2. Whether weather data appears in the response
       """
       response_output = body.response.output
       
       # Check 1: Did agent call the weather tool?
       tool_called = any(
           item.type == "function_call" and item.name == "get_weather"
           for item in response_output
       )
       
       # Check 2: Did agent mention weather in final response?
       final_message = ""
       for item in response_output:
           if item.type == "message":
               content = item.content
               if content and isinstance(content, list):
                   final_message = content[0].text
       
       mentioned_weather = "weather" in final_message.lower() or "cold" in final_message.lower()
       
       # Scoring logic
       if tool_called and mentioned_weather:
           reward = 1.0  # Perfect: used tool AND incorporated results
       elif tool_called:
           reward = 0.5  # Partial: called tool but didn't use data
       elif mentioned_weather:
           reward = 0.3  # Weak: mentioned weather without checking
       else:
           reward = 0.0  # Failed: didn't address weather at all
       
       return BaseVerifyResponse(**body.model_dump(), reward=reward)
   ```

2. **Save and restart servers** to test the enhanced verification.

**What this adds**: Nuanced scoring that checks if the agent actually incorporated the weather data into its response.

---

### 3. Enforce Quality Standards

Finally, make verification even more demanding by requiring actionable advice, not just weather mentions.

**Goal**: Agent must provide practical recommendations (like "bring a jacket") based on the weather data.

1. Update the `verify()` function one more time:

   ```python
   async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
       """Check if agent provides actionable advice based on weather"""
       response_output = body.response.output
       
       # Extract final response text
       final_message = ""
       for item in response_output:
           if item.type == "message":
               content = item.content
               if content and isinstance(content, list):
                   final_message = content[0].text.lower()
       
       # Check if tool was called
       tool_called = any(
           item.type == "function_call" and item.name == "get_weather"
           for item in response_output
       )
       
       # Check for actionable advice keywords
       actionable_advice = any(word in final_message for word in [
           "wear", "bring", "jacket", "umbrella", "layers", "coat"
       ])
       
       # Stricter scoring
       if tool_called and actionable_advice:
           reward = 1.0  # Perfect: used weather data AND gave advice
       elif tool_called and "weather" in final_message:
           reward = 0.6  # Good: mentioned weather but advice not clear
       elif tool_called:
           reward = 0.3  # Weak: called tool but didn't leverage it
       else:
           reward = 0.0  # Failed: no tool use
       
       return BaseVerifyResponse(**body.model_dump(), reward=reward)
   ```

2. **Save and restart servers**.

**What this adds**: Domain-specific quality requirements. This pattern is how you'd design verification for real applications—define what "good" means for your specific use case.

---

## Test Your Verification Logic

Now test whichever verification stage you have implemented to observe how different agent behaviors produce different reward scores.

1. Create a test script (`responses_api_agents/simple_agent/test_verification.py`):

   ```python
   import json
   from asyncio import run
   from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
   from nemo_gym.server_utils import ServerClient

   server_client = ServerClient.load_from_global_config()

   # Test cases designed to trigger different rewards
   test_cases = [
       {
           "message": "What's the weather in Seattle?",
           "expected_reward": 1.0,
           "reason": "Should call tool AND use weather data"
       },
       {
           "message": "Tell me about Seattle",
           "expected_reward": 0.0,
           "reason": "No weather mentioned, tool might not be called"
       },
       {
           "message": "Is it cold in Boston?",
           "expected_reward": 1.0,
           "reason": "Should call tool AND mention weather"
       },
   ]

   async def test_verification():
       for test in test_cases:
           print(f"\n{'='*60}")
           print(f"Testing: '{test['message']}'")
           print(f"Expected reward: {test['expected_reward']} ({test['reason']})")
           print('='*60)
           
           # Create request params
           request_params = NeMoGymResponseCreateParamsNonStreaming(
               input=[
                   {
                       "role": "developer",
                       "content": "You are a helpful personal assistant.",
                   },
                   {"role": "user", "content": test["message"]},
               ],
               tools=[
                   {
                       "type": "function",
                       "name": "get_weather",
                       "description": "Get weather information for a city",
                       "parameters": {
                           "type": "object",
                           "properties": {
                               "city": {"type": "string", "description": "City name"},
                           },
                           "required": ["city"],
                           "additionalProperties": False,
                       },
                       "strict": True,
                   }
               ],
           )
           
           # Get agent response
           task = server_client.post(
               server_name="simple_weather_simple_agent",
               url_path="/v1/responses",
               json=request_params,
           )
           
           result = await task
           response_data = await result.json()
           
           # Call verify endpoint with original request params
           verify_task = server_client.post(
               server_name="simple_weather",
               url_path="/verify",
               json={
                   "responses_create_params": request_params.model_dump(),
                   "response": response_data,
               },
           )
           
           verify_result = await verify_task
           verify_data = await verify_result.json()
           
           print(f"\nAgent response: {response_data['output'][-1]}")
           print(f"\n✓ REWARD RECEIVED: {verify_data['reward']}")

   run(test_verification())
   ```

2. **Save the file**.

---

## Run the Tests

1. Start your NeMo Gym servers (if not already running):

    ```bash
    config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
    resources_servers/example_simple_weather/configs/simple_weather.yaml"

    ng_run "+config_paths=[${config_paths}]"
    ```

2. In a new terminal, activate your environment and run the test:

   ```bash
   cd /path/to/Gym
   source .venv/bin/activate
   python responses_api_agents/simple_agent/test_verification.py
   ```

---

## What You Will Observe

Watch how different agent behaviors produce different rewards:

::::{tab-set}

:::{tab-item} High Reward (1.0)

- **Message**: "What's the weather in Seattle?"
- **Agent behavior**: Calls `get_weather`, mentions weather in response
- **Reward**: 1.0 ✓ (Perfect score)

:::

:::{tab-item} Partial Reward (0.5)

- **Message**: "Tell me about Seattle"
- **Agent behavior**: Might call tool but focus on non-weather facts
- **Reward**: 0.5 (Called tool but didn't emphasize weather)

:::

:::{tab-item} Low/Zero Reward (0.0-0.3)

- **Message**: Non-weather questions
- **Agent behavior**: Responds without using weather tool
- **Reward**: 0.0-0.3 (Didn't use weather data)

:::

::::

:::{note}
**About Expected Rewards**: The reward scores you observe depend on which verification stage you implemented:

- **Stage 1** (tool usage): Only 0.0 or 1.0
- **Stage 2** (tool + weather mention): 0.0, 0.3, 0.5, or 1.0  
- **Stage 3** (actionable advice): 0.0, 0.3, 0.6, or 1.0

The examples above assume **Stage 2** verification. With Stage 3, you will need actionable advice keywords ("wear", "bring", "jacket") to get a perfect 1.0 score.
:::

:::{tip}
**Model Variability Matters Here Too**: Remember from [Tutorial 2: Your First Agent](first-agent.md) that GPT-4's behavior varies between runs. You might observe different reward scores for the same query across multiple runs—this is normal and demonstrates why collecting many rollouts (Tutorial 4) gives you a more complete picture of agent performance.
:::

:::{note}
GPT-4 is quite capable, so you might observe high scores across most tests. This demonstrates that the base model already performs well on simple tasks. During RL training, verification becomes critical for more challenging domains where the base model struggles.
:::

---

## Understanding the Verification Workflow

What you just implemented mirrors real RL training workflows:

1. **Agent generates response** → Your test script sends a message
2. **Response captured** → Complete output with tool calls
3. **Verification called** → `/verify` endpoint scores the response
4. **Reward assigned** → Numerical score (0.0–1.0) returned
5. **In RL training** → This reward would update model parameters

The verification logic you wrote defines the objective the agent should learn to optimize. This connection between verification scores and model improvement is at the heart of [reinforcement learning](../about/glossary.md#reinforcement-learning-rl)—the agent learns to take actions that maximize the [reward signal](../about/glossary.md#reward) provided by verification.

---

## What You've Learned

You now have hands-on experience with:

- How `verify()` functions score agent performance
- Building verification logic with scoring criteria
- Testing verification and observing reward signals
- The connection between verification and training

**Key insight**: The verification function defines what the agent learns to optimize. Design it to reflect true task success.

---

## Next Steps

Now that you can score individual agent responses, the next challenge is generating these responses at scale for training. This process is called **rollout collection**—systematically gathering agent interactions with verification scores to create the training data that powers RL.

:::{button-ref} collecting-rollouts
:color: primary
:outline:
:ref-type: doc

Next: Collecting Rollouts →
:::

Learn how to systematically collect thousands of agent responses with verification scores—the foundation of RL training data.

