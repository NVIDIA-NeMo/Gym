# SciCode Agent

Custom multi-step agent for the SciCode benchmark. For each problem it loops over the sub-steps,
generating Python code one sub-step at a time and accumulating it (each sub-step's prompt includes
the model's own code from previous sub-steps), then submits the accumulated per-step solutions to
the SciCode resources server for sandboxed test execution.

## Configuration

- `resources_server`: the SciCode resources server instance to verify against
- `model_server`: the model server used for generation
- `with_background`: include each sub-step's scientific background in the prompt (default: `true`,
  matching the nemo-skills `eval/scicode/background` default)
