# SGLang Model

Wraps an existing SGLang OpenAI-compatible endpoint as a NeMo Gym Responses API model server.

```bash
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/sglang_model/configs/sglang_model.yaml"

ng_run "+config_paths=[${config_paths}]" \
  ++policy_base_url=http://127.0.0.1:30000/v1 \
  ++policy_api_key=dummy \
  ++policy_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
```

`return_token_id_information` is intentionally unsupported because it depends on vLLM-specific response fields.

SGLang's OpenAI-compatible chat endpoint rejects the Responses API `developer` role, so this adapter defaults `replace_developer_role_with_system` to `true`.

For the existing ridger SGLang deployment, use the first Slurm node plus `/v1` as `policy_base_url`, for example `http://h119-gpu-polaris.pod4.lab.bitdeer.ai:18080/v1`.
