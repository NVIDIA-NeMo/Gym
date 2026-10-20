# Partner IF dispatcher

Routes each episode of a mixed-harness benchmark to the environment server configured
for its `verifier_metadata.partner_if.task_id`, then returns that environment's
receipt unchanged. Sessions, execution and grading stay in the routed environments.
Gym aggregates the dispatcher's rows, so `/aggregate_metrics` is proxied to the
configured `resources_server`.
