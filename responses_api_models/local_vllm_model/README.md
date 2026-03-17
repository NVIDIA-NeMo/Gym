- [Example run config](#example-run-config)
- [E2E sanity testing](#e2e-sanity-testing)
- [Notes](#notes)

# Example run config
Run this on a single GPU node! Set tensor_parallel_size * data_parallel_size to the number of GPUs on your node. For this single node config, data_parallel_size_local is equal to data_parallel_size

```bash
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/local_vllm_model/configs/nano_v3_single_node.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4 \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size_local=2 &> temp.log &
```

View the logs
```bash
tail -f temp.log
```

Call the server. If you see a model response here, then everything is working as intended!
```bash
python responses_api_agents/simple_agent/client.py
```


# E2E sanity testing
See the following scripts:
- 1 node
  - responses_api_models/local_vllm_model/test_scripts/1_node/1_instance_1x8.sh
  - responses_api_models/local_vllm_model/test_scripts/1_node/1_instance_2x4.sh
  - responses_api_models/local_vllm_model/test_scripts/1_node/2_instances_1x4.sh
- 2 nodes
  - responses_api_models/local_vllm_model/test_scripts/2_nodes/2_instances_1x8.sh
  - responses_api_models/local_vllm_model/test_scripts/2_nodes/2_instances_2x4.sh
  - [Not supported yet] responses_api_models/local_vllm_model/test_scripts/2_nodes/1_instance_1x16.sh
- 4 nodes
  - [Not supported yet] responses_api_models/local_vllm_model/test_scripts/4_nodes/1_instance_2x16.sh
  - [Not supported yet] responses_api_models/local_vllm_model/test_scripts/4_nodes/2_instances_1x16.sh
- 8 nodes
  - [Not supported yet] responses_api_models/local_vllm_model/test_scripts/8_nodes/2_instances_2x16.sh

# Notes
1. dpXppYtpZ -> X placement groups of Y * Z size each
2. We need a "master IP" for vLLM's data parallel code to work

3. Calculate the placement group size for Y * Z
4. Request a placement group of that size from Ray. This is our "head" placement group
5. Place LocalVLLMModelActor on this "head" placement group
6. Set the data parallel master IP to the "head" placement group node
7. Start vLLM server
8. CoreActorManager.create_dp_placement_groups
    1. For the first placement group, do nothing because we will reuse the "head" placement group
    2. For subsequent (DP size - 1) placement groups, schedule using the same Y * Z and pack strategy
9. Return to CoreActorManager which schedules the individual DPCoreEngineProc on each placement group

First server - spins up normally and properly
Second server - keep trying to do something but don't initialize anything
