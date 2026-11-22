## Description: <br>
Onboard a model for NeMo Gym evaluations and compose the complete eval config: the vLLM policy service, its wiring as the policy, the reasoning toggle, and each benchmark's own config from Gym's benchmarks directory, then validate and smoke it. Covers a quantized export of a model that also runs in BF16. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NeMo Gym maintainers (NVIDIA-NeMo/Gym) <br>

### License/Terms of Use: <br>
Apache-2.0 <br>

## Use Case: <br>
Engineers and researchers who need to evaluate a new model or checkpoint in NeMo Gym, and the coding agents that work for them, use this skill to write the serving and benchmark config, validate it without GPUs, smoke it, and compare two models or a BF16 and a quantized variant fairly. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

The skill ships no credentials. Credentials a run needs are the user's own and are passed through environment variables. Do not include secrets in prompts, logs or output. <br>

## Known Risks and Mitigations: <br>
Risk: The commands the skill recommends can start jobs or change state on the user's own cluster or account. <br>
Mitigation: They are the user's own `gym` commands, run with the user's own access, and the skill has the agent report every operator action with its reason. <br>

## Reference(s): <br>
- [Example: IFEval](references/example-ifeval.md) <br>
- [A quantized variant: the six steps](references/quantized-variant-steps.md) <br>
- [Configuration](https://docs.nvidia.com/nemo/gym/latest/orchestration/configuration) <br>
- [vLLM model server](https://docs.nvidia.com/nemo/gym/latest/model-server/vllm) <br>
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) <br>

## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, YAML] <br>
**Output Format:** [Markdown with inline shell and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>
