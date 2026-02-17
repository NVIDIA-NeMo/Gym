(integrate-external-environments)=

# Integrating external training environments, benchmarks, or agents

Fundamentally, a training environment in NeMo Gym is some Python logic that performs a sequence or graph of model calls and tool calls. For native NeMo Gym environments, all of the model and tool calls are present within an agent server like SimpleAgent [https://github.com/NVIDIA-NeMo/Gym/blob/f0c830d7282f49f40ea8214473792014e498e3d5/responses\_api\_agents/simple\_agent/app.py\#L64](https://github.com/NVIDIA-NeMo/Gym/blob/f0c830d7282f49f40ea8214473792014e498e3d5/responses_api_agents/simple_agent/app.py#L64). For environments that we consider “external”, the orchestration of model and tool calls is offloaded to a 3rd party library rather than implemented within NeMo Gym itself.

Integrating external training environments, benchmarks, or agents into a NeMo Gym environment requires additional considerations beyond native NeMo Gym environments. We provide a rough template of integration form factor below and best practices for ensuring that the integration is correct.

## Integration form factor

1. Because external training environments, benchmarks, or agents include their own rollout orchestration logic (i.e. coordinating model and tool calls) and may sometimes return their own reward, the appropriate level to integrate into are NeMo Gym's agent servers.  
2. You can add a dependency from your NeMo Gym agent server to the 3rd party library by adding it to the requirements.txt. If your dependency needs are more complicated beyond installing pip packages or Github repositories, consider using things like setup.py and pyproject.toml.  
3. Once you add a dependency, you can simply import it in app.py like a normal Python script.  
4. You can run the `ng_test` command to run tests on your agent server. See [https://docs.nvidia.com/nemo/gym/latest/reference/cli-commands.html\#ng-test-nemo-gym-test](https://docs.nvidia.com/nemo/gym/latest/reference/cli-commands.html#ng-test-nemo-gym-test)   
5. Typically you can just wrap the external library in the /run function and omit the /responses function.  
6. In the `/run` function before calling into the third party library, you will need to pre-process from NeMo Gym schema into a config for the external library.  
7. In the `/run` function after calling into the third party library, you will need to post-process from the external library result into a NeMo Gym response.

## Best practices

1. Technical design  
   1. Please follow NeMo Gym’s async first-party server design i.e. the /run endpoint should be async in order to maximize efficiency.  
      1. Please avoid using extra threads or processes unless absolutely necessary. If you need to scale, please use Ray workers and await the ray.remote call.  
      2. For example, a single NeMo Gym instance that consists of 3 FastAPI server instances can handle 65k concurrent math rollouts without crashing and is not terribly inefficient.  
   2. Please use [NeMo Gym’s custom OpenAI client](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/nemo_gym/openai_utils.py#L433) rather than other popular LLM clients including OpenAI, Anthropic, LiteLLM.  
      1. The NeMo Gym OpenAI client has been [meticulously designed to scale](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/docs/how-to-faq.md?plain=1#L861).  
      2. Other LLM clients (like LiteLLM) often preprocess/postprocess the inputs/outputs \- making it difficult to understand and control the flow of information.  
      3. Please [propagate cookies](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/responses_api_agents/simple_agent/app.py#L88) as appropriate.  
   3. Use Pydantic models for checking the data format wherever necessary.  
2. Rollout creation  
   1. During training, NeMo Gym models will return additional information on response messages or output items consisting of the [prompt\_token\_ids, generation\_token\_ids and generation\_log\_probs fields](https://github.com/NVIDIA-NeMo/Gym/blob/d4048f6c2e93a6e44ad4934e827310f06997d72a/nemo_gym/openai_utils.py#L89).  
   2. When constructing the messages or input items for the next model call in multi step or multi turn scenarios, please propagate this information from the previous model response.  
3. Delivery form factor  
   1. Please make a PR to NeMo Gym main.  
   2. Code should run locally with no external APIs present unless absolutely necessary as part of the environment design e.g. online search.  
   3. Variables are passed via NeMo Gym config and not via environment variables, including OpenAI base URL, API key, etc.  
   4. Code must be runnable on Linux machines (i.e. any executables must be created for Linux).  
4. Functional requirements  
   1. Errors popping up due to failure in tool execution by the environment or due to the model issuing incorrect tool calls / arguments need to be propagated back to the model \- i.e. should not throw errors and crash the env and training.  
   2. We plan to do large-scale training with these environments. As such we need it to support anywhere from 4k to 65k concurrent requests without crashing. Usually just slamming the endpoint with 2k concurrent requests is enough to test the efficiency scaling of the environment.  
   3. You should be able to run an entire training run without any crashes.


## Ensuring proper token ID handling in custom client calls
During training time, NeMo Gym keeps track of the ground truth prompt token ids, generation token ids, and generation log probs for downstream consumption by the RL framework. As a result, we need to add a few fields to request and response schemas in order to properly facilitate this. This usually doesn't matter if you are using 100% NeMo Gym, but in certain situations you may need or want to use a separate client (such as LiteLLM, your own OpenAI client, and so on) to call model endpoints.

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
And you have to ensure that when you make a request with your custom client that these three extra fields (prompt_token_ids, generation_token_ids, and generation_log_probs) are passed through correctly on a message level. And this also applies to the response i.e. you need to ensure that your custom client will correctly return these three extra fields.

It's an analogous story for Responses-compatible APIs.

