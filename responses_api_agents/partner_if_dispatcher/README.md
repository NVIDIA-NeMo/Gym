# Partner IF dispatcher

Routes each episode of a mixed-harness benchmark to the environment server configured
for its `verifier_metadata.partner_if.task_id`, then returns that environment's
receipt. Its one change to the receipt adds any tool execution in the reply's
trajectory that the episode's agent observations lack, since rollout collection
builds the trajectory's tool calls from those observations. Sessions, execution and
grading stay in the routed environments.
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
