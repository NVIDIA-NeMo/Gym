# RoboDojo with OpenModel and OpenCode

Use this recipe with a RoboDojo worker that runs OpenModel with OpenCode and
serves a complete episode through Gym's `/run` contract. Gym loads the cases,
relays the episodes, and collects rewards and rollout records. The worker
owns the simulator, SDK sessions, task instructions, and native grading.

Start the worker with the desired model and simulator configuration before
starting Gym. Its `/run` accepts a `case_id` and empty
`responses_create_params.input`. The worker needs to implement Gym's
`/aggregate_metrics` contract and include `reward` and `mask_sample` in the
rollout response. Give the worker an episode timeout and a persistent directory
for native grading and video evidence. Use a private listener or tunnel for
the worker endpoint. The defaults target a local worker on port 61370.

```bash
# Start in a separate terminal.
gym env start --config examples/robodojo_openmodel/config.yaml

gym eval run --no-serve --config examples/robodojo_openmodel/config.yaml \
  --agent robodojo_worker --input examples/robodojo_openmodel/example.jsonl \
  --output outputs/robodojo-openmodel.jsonl --limit 1 --concurrency 1
```

Set the worker's `host` and `port` in the recipe to use a different endpoint.
The example selects one `general_pickup` layout. To run other cases, supply a
JSONL file of case IDs registered by the same worker. Inspect `mask_sample` to
identify unscored episodes. Use the worker's native evidence for horizon scores
and video. Record simulator, SDK, model, and harness settings with every run.
