# RNA Design agent

The fixed workflow, agent loop and tools of the [RNA design environment](../../environments/rna_design/README.md).
Each rollout runs a whole design campaign in its own worker process, and the reward comes from re-scoring the designs
it produced. `arm` picks who chooses the tool calls: the model (`agent`) or a fixed workflow (`fixed`, `fixed_deep`,
`generator`). `rds/` holds the campaign runner and the tools, and `prepare.py` builds the suites and dataset rows and
sets up gRNAde.

Setup, run instructions and the reward are in the [environment README](../../environments/rna_design/README.md).

## Notes

- Each campaign runs in its own process, under `RNA_WORKER_PYTHON`, because gRNAde needs torch and the campaign
  loop sets process-wide state.
- gRNAde imports `torch_scatter` and `torch_cluster`, which have no aarch64 wheels. `rds/tools/pyg_shims/` has pure
  PyTorch versions of the one function it uses from each, used only when the real packages are missing.

## Third-party components

See [LICENSE-third-party](LICENSE-third-party), which also gives each dataset's terms. ViennaRNA is not used,
because its licence restricts redistribution.

## Tests

```bash
gym env test +entrypoint=responses_api_agents/rna_design_agent
```
