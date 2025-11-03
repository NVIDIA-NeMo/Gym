# Verifying Agent Results

In the last tutorial, you ran your weather agent and saw it call the `get_weather` tool and generate responses. But how do you know if those responses were *good*? That's what verification solves.

**Goal**: Build and test verification logic that scores your weather agent's performance.

**In this tutorial, you will**:

1. Examine the current `verify()` function in the weather resource server
2. Build quality scoring based on tool usage and response content
3. Create a test script to observe different reward scores
4. Experiment with stricter verification criteria

:::{tip}
**Going deeper**: For comprehensive coverage of verification patterns, design considerations, and the theory behind reward signals, refer to [Verifying Agent Results (Concepts)](../about/concepts/verifying-agent-results.md).
:::

:::{button-ref} 03-your-first-agent
:color: secondary
:outline:
:ref-type: doc

← Previous: Your First Agent
:::

---

## Examine the Current Verification Function

Your weather agent's resource server already has a `verify()` function, but it doesn't actually check anything—it just returns a fixed reward of 1.0 regardless of performance.

Open `resources_servers/simple_weather/app.py` and locate the verify function:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

**What this does**:

- Receives the agent's complete response in `body.response`
- Returns a reward of 1.0 (perfect score) no matter what

**The problem**: Every response gets the same score, so there's no signal about quality. The agent can't learn what "good" looks like.

---

## Build Quality Verification

Let's create verification logic that actually measures performance. You'll check whether the agent used the weather data meaningfully.

1. Replace the `verify()` function in `resources_servers/simple_weather/app.py`:

   ```python
   async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
       """
       Score agent performance based on whether it:
       1. Called the weather tool
       2. Mentioned the weather in its response
       """
       response_output = body.response.output
       
       # Check 1: Did agent call the weather tool?
       tool_called = any(
           item.get("type") == "function_call" and item.get("name") == "get_weather"
           for item in response_output
       )
       
       # Check 2: Did agent mention weather in final response?
       final_message = ""
       for item in response_output:
           if item.get("type") == "message":
               content = item.get("content", [])
               if content and isinstance(content, list):
                   final_message = content[0].get("text", "")
       
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

2. **Save the file** and restart your NeMo Gym servers for the changes to take effect.

---

## Test Your Verification Logic

1. Create a test script (`responses_api_agents/simple_agent/test_verification.py`) to see different reward scores in action.

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
           
           # Get agent response
           task = server_client.post(
               server_name="simple_weather_simple_agent",
               url_path="/v1/responses",
               json=NeMoGymResponseCreateParamsNonStreaming(
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
               ),
           )
           
           result = await task
           response_data = await result.json()
           
           # Call verify endpoint
           verify_task = server_client.post(
               server_name="simple_weather",
               url_path="/verify",
               json={
                   "responses_create_params": response_data["responses_create_params"],
                   "response": {
                       "output": response_data["output"],
                       "model": response_data.get("model", ""),
                   },
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
    resources_servers/simple_weather/configs/simple_weather.yaml"

    ng_run "+config_paths=[${config_paths}]"
    ```

2. In a new terminal, activate your environment and run the test:

   ```bash
   cd /path/to/Gym
   source .venv/bin/activate
   python responses_api_agents/simple_agent/test_verification.py
   ```

---

## What You'll Observe

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
GPT-4 is quite capable, so you might see high scores across most tests. This demonstrates that the base model already performs well on simple tasks. During RL training, verification becomes critical for more challenging domains where the base model struggles.
:::

---

## Experiment: Make Verification Stricter

Try modifying the verification logic to require more from the agent.

1. Edit `verify()` in `resources_servers/simple_weather/app.py`:

   ```python
   async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
       """Check if agent provides actionable advice based on weather"""
       response_output = body.response.output
       
       # Extract final response text
       final_message = ""
       for item in response_output:
           if item.get("type") == "message":
               content = item.get("content", [])
               if content and isinstance(content, list):
                   final_message = content[0].get("text", "").lower()
       
       # Check if tool was called
       tool_called = any(
           item.get("type") == "function_call" and item.get("name") == "get_weather"
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

2. **Restart servers** and **rerun the test script** to see how rewards change with stricter criteria.

---

## Understanding the Verification Workflow

What you just implemented mirrors real RL training workflows:

1. **Agent generates response** → Your test script sends a message
2. **Response captured** → Complete output with tool calls
3. **Verification called** → `/verify` endpoint scores the response
4. **Reward assigned** → Numerical score (0.0–1.0) returned
5. **In RL training** → This reward would update model parameters

The verification logic you wrote defines what the agent should learn to optimize for.

:::{tip}
For production verification patterns including correctness checking, LLM judges, hybrid scoring, and design considerations, refer to the [Verification Concepts](../about/concepts/verifying-agent-results.md) guide.
:::

---

## What You've Learned

You now have hands-on experience with:

- ✓ How `verify()` functions score agent performance
- ✓ Building verification logic with scoring criteria
- ✓ Testing verification and observing reward signals
- ✓ The connection between verification and training

**Key insight**: The verification function defines what the agent learns to optimize. Design it to reflect true task success.

---

## Next Steps

Now that you can score individual agent responses, the next challenge is generating these responses at scale for training.

:::{button-ref} 05-rollout-collection
:color: primary
:outline:
:ref-type: doc

Next: Rollout Collection →
:::

Learn how to systematically collect thousands of agent responses with verification scores—the foundation of RL training data.
