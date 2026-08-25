> Keywords: Tool Use, Multi-step Reasoning, Environment Interaction, Scientific Tasks

# Aviary resources server

This component adapts [Aviary environments](https://github.com/Future-House/aviary) to the NeMo Gym resources-server interface. It provides the state and tools used by the Aviary agent; runnable compositions live under `environments/`:

- [GSM8K](../../environments/aviary_gsm8k/README.md)
- [HotPotQA](../../environments/aviary_hotpotqa/README.md)
- [BixBench](../../environments/aviary_bixbench/README.md)
- [BixBench-Hypothesis](../../environments/aviary_bbh/README.md)

`configs/aviary.yaml` is the minimal GSM8K composition used by the component test suite. Use the environment configs above for evaluation and training.

## Licensing

- Code and Aviary dependency: Apache 2.0
- GSM8K data: MIT
- HotPotQA data: Creative Commons Attribution-ShareAlike 4.0 International
- BixBench data: Apache 2.0
- BixBench-Hypothesis data: CC BY 4.0
