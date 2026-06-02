# Local SGLang Model

Starts a local SGLang OpenAI-compatible server and exposes it through NeMo Gym's Responses API model wrapper.

By default this wrapper does not install SGLang into Gym's server environment. Point `sglang_env_prefix` at an existing conda/venv-style prefix and it will run `<prefix>/bin/python -m sglang.launch_server`.

```bash
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/local_sglang_model/configs/local_sglang_model.yaml"

ng_run "+config_paths=[${config_paths}]" \
  ++policy_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  ++policy_model.responses_api_models.local_sglang_model.sglang_env_prefix=/mnt/nfs/wangling/envs/sglang \
  ++policy_model.responses_api_models.local_sglang_model.sglang_serve_kwargs.tensor_parallel_size=4 \
  ++policy_model.responses_api_models.local_sglang_model.sglang_serve_kwargs.context_length=131072 \
  ++policy_model.responses_api_models.local_sglang_model.sglang_serve_kwargs.mem_fraction_static=0.90
```

`sglang_serve_kwargs` maps directly to `python -m sglang.launch_server` flags by replacing underscores with dashes.
Compatibility aliases `tp_size`, `pp_size`, `dp_size`, and `ep_size` are normalized to SGLang 0.5.9's `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, and `expert_parallel_size`.

Like `sglang_model`, this wrapper defaults `replace_developer_role_with_system` to `true` because SGLang rejects the Responses API `developer` role.

For the ridger GLM-5.1-FP8 setup documented under `/mnt/nfs/ridger`, prefer launching the known-good Slurm server externally and connecting with `responses_api_models/sglang_model/configs/sglang_model.yaml`:

```bash
ng_run "+config_paths=[resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,responses_api_models/sglang_model/configs/sglang_model.yaml]" \
  ++policy_base_url=http://h119-gpu-polaris.pod4.lab.bitdeer.ai:18080/v1 \
  ++policy_api_key=dummy \
  ++policy_model_name=GLM-5.1-FP8
```
