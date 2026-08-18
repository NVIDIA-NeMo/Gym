# NeMo Fabric Agent

Runs an installed [NeMo Fabric](https://github.com/NVIDIA/NeMo-Fabric) harness as a Gym agent. `adapter_id`,
`model_provider`, `harness_settings`, and `fabric_config` select the harness and its behavior.
Adapter capabilities are listed in Fabric's
[compatibility matrix](https://github.com/NVIDIA/NeMo-Fabric/blob/release/0.2/docs/sdk/python.mdx#normalized-configuration-compatibility).

Gym owns the environment, verification, rewards, and model routing. Fabric owns the harness lifecycle. `gym env start`
installs the requirements.

## Examples

- `configs/nemo_fabric_{claude,codex,deepagents,hermes,mini_swe_agent}.yaml`: harness presets.
- `resources_servers/math_with_judge/configs/math_with_judge_nemo_fabric_hermes.yaml`: math-verify and Hermes.
- `resources_servers/reasoning_gym/configs/reasoning_gym_nemo_fabric_hermes.yaml`: Reasoning Gym and Hermes.
- `responses_api_agents/anyswe_agent/configs/anyswe_nemo_fabric.yaml`: AnySWE and mini-SWE-agent.
- `responses_api_agents/anyterminal_agent/configs/anyterminal_nemo_fabric.yaml`: AnyTerminal and mini-SWE-agent.

Run the math example:

```bash
gym env start \
  --config resources_servers/math_with_judge/configs/math_with_judge_nemo_fabric_hermes.yaml \
  --model-type openai_model
```

For AnyTerminal on OpenSandbox, compose the agent and provider configs:

```bash
gym eval run \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_nemo_fabric.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type openai_model
```

Results preserve the Fabric result, verifier scores, normalized usage, and Gym model-call captures.
