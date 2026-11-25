# Partner IF dispatcher

Routes each episode of a mixed-harness benchmark to the environment server configured
for its `verifier_metadata.partner_if.task_id`, then returns that environment's
receipt. It adds missing tool observations from ordinary replies so rollout collection
can reconstruct the trajectory. For the host-configured recorded-context Gymnasium
route, it validates and lifts the terminal Resources report into the collector row;
this retains terminal tool outputs, combined parent/Explore usage and shared grades.
It does not treat policy-supplied fields as a grading report. Sessions, execution and
grading stay in the routed components.
All of the benchmark's rows come back through this agent, and its `/aggregate_metrics`
is proxied to the configured `resources_server`, so one aggregate covers every
environment.

`configs/partner_if_dispatcher.yaml` runs partner_if's text and mock-tool examples
(`data/example.jsonl`) in their own environments:

```bash
gym env start --config responses_api_agents/partner_if_dispatcher/configs/partner_if_dispatcher.yaml \
    +policy_base_url=... +policy_api_key=... +policy_model_name=... \
    +judge_base_url=... +judge_api_key=... +judge_model_name=...

gym eval run --no-serve --agent partner_if_dispatcher \
    --input responses_api_agents/partner_if_dispatcher/data/example.jsonl \
    --output partner_if_dispatcher_rollouts.jsonl
```
