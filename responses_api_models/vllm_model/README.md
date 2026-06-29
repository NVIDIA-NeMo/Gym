# Example run config

VLLMModel connects NeMo Gym to a vLLM server that you start and manage yourself. Spin up a vLLM server in a separate terminal (see the [vLLM docs](https://docs.vllm.ai/)), then point NeMo Gym at it.

```bash
gym env start \
    --resources-server example_single_tool_call \
    --model-type vllm_model \
    --model-url http://0.0.0.0:10240/v1 \
    --model <your-model> \
    --model-api-key dummy_key &> temp.log &
```

View the logs
```bash
tail -f temp.log
```

Once you see that server instances are up, call the server. If you see a model response here, then everything is working as intended.
```bash
python responses_api_agents/simple_agent/client.py
```
