(integrate-external-environments)=

# Integrating external training environments, benchmarks, or agents

Fundamentally, a training environment in NeMo Gym is some Python logic that performs a sequence or graph of model calls and tool calls. For native NeMo Gym environments, all of the model and tool calls are present within an agent server like SimpleAgent [https://github.com/NVIDIA-NeMo/Gym/blob/f0c830d7282f49f40ea8214473792014e498e3d5/responses\_api\_agents/simple\_agent/app.py\#L64](https://github.com/NVIDIA-NeMo/Gym/blob/f0c830d7282f49f40ea8214473792014e498e3d5/responses_api_agents/simple_agent/app.py#L64). For environments that we consider “external”, the orchestration of model and tool calls is offloaded to a third-party library rather than implemented within NeMo Gym itself.

Integrating external training environments, benchmarks, or agents into a NeMo Gym environment requires additional considerations beyond native NeMo Gym environments. We provide a rough template of integration form factor below and best practices for ensuring that the integration is correct.

## Integration form factor

1. Because external training environments, benchmarks, or agents include their own rollout orchestration logic (that is, coordinating model and tool calls) and may sometimes return their own reward, the appropriate level to integrate into are NeMo Gym agent servers.  
2. You can add a dependency from your NeMo Gym agent server to the third-party library by adding it to the requirements.txt. If your dependency needs are more complicated beyond installing pip packages or GitHub repositories, consider using setup.py and pyproject.toml.  
3. After you add a dependency, you can import it in app.py like a normal Python script.  
4. You can run the `ng_test` command to run tests on your agent server. Refer to the [ng_test CLI reference](https://docs.nvidia.com/nemo/gym/latest/reference/cli-commands.html#ng-test-nemo-gym-test) for details.   
5. Typically you can just wrap the external library in the `/run` function and omit the `/responses` function.  
6. In the `/run` function before calling into the third-party library, you will need to preprocess from NeMo Gym schema into a config for the external library.  
7. In the `/run` function after calling into the third-party library, you will need to postprocess from the external library result into a NeMo Gym response.

## Best practices

1. Technical design  
   1. Follow NeMo Gym async first-party server design: the `/run` endpoint should be async to maximize efficiency.  
      1. Avoid using extra threads or processes unless absolutely necessary. If you need to scale, use Ray workers and await the ray.remote call.  
      2. For example, a single NeMo Gym instance that consists of three FastAPI server instances can handle 65,000 concurrent math rollouts without crashing and is not terribly inefficient.  
   2. Use the [NeMo Gym custom OpenAI client](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/nemo_gym/openai_utils.py#L433) rather than other popular LLM clients including OpenAI, Anthropic, and LiteLLM.  
      1. The NeMo Gym OpenAI client has been [meticulously designed to scale](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/docs/how-to-faq.md?plain=1#L861).  
      2. Other LLM clients (such as LiteLLM) often preprocess or postprocess the inputs and outputs, making it difficult to understand and control the flow of information.  
      3. [Propagate cookies](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/responses_api_agents/simple_agent/app.py#L88) as appropriate.  
   3. Use Pydantic models for checking the data format wherever necessary.  
2. Rollout creation  
   1. During training, NeMo Gym models will return additional information on response messages or output items consisting of the [prompt\_token\_ids, generation\_token\_ids, and generation\_log\_probs fields](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/nemo_gym/openai_utils.py#L89).  
   2. When constructing the messages or input items for the next model call in multi-step or multi-turn scenarios, propagate this information from the previous model response.  
3. Delivery form factor  
   1. Make a PR to NeMo Gym main.  
   2. Code should run locally with no external APIs present unless absolutely necessary as part of the environment design (for example, online search).  
   3. Variables are passed through NeMo Gym config and not through environment variables, including OpenAI base URL, API key, and so on.  
   4. Code must be runnable on Linux machines (that is, any executables must be created for Linux).  
4. Functional requirements  
   1. Errors popping up due to failure in tool execution by the environment or due to the model issuing incorrect tool calls or arguments need to be propagated back to the model (that is, should not throw errors and crash the env and training).  
   2. We plan to do large-scale training with these environments. As such we need it to support anywhere from 4,000 to 65,000 concurrent requests without crashing. Usually slamming the endpoint with 2,000 concurrent requests is enough to test the efficiency scaling of the environment.  
   3. You should be able to run an entire training run without any crashes.


## Ensuring proper token ID handling in custom client calls
During training time, NeMo Gym keeps track of the ground truth prompt token ids, generation token ids, and generation log probs for downstream consumption by the RL framework. As a result, we need to add a few fields to request and response schemas in order to properly facilitate this. This usually does not matter if you are using 100% NeMo Gym, but in certain situations you may need or want to use a separate client (such as LiteLLM, your own OpenAI client, and so on) to call model endpoints.

For Chat Completions, outside of training, an Assistant message will look like:
```python
ChatCompletionMessage(
    content="<think>I'm thinking</think>Hi there!",
    tool_calls=[{...}, {...}],
    ...
)
```
During training, a Chat Completions Assistant message will look like:
```python
ChatCompletionMessage(
    content="<think>I'm thinking</think>Hi there!",
    tool_calls=[{...}, {...}],
    prompt_token_ids=[...],  # List[int]
    generation_token_ids=[...],  # List[int]
    generation_log_probs=[...],  # List[float]
    ...
)
```
You must ensure that when you make a request with your custom client, these three extra fields (`prompt_token_ids`, `generation_token_ids`, and `generation_log_probs`) are passed through correctly on a message level. This also applies to the response: you need to ensure that your custom client correctly returns these three extra fields.

The same applies to Responses-compatible APIs.

