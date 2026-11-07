# SaaS Bench evaluation

Run the source-owned OpenCode harness 2.1.1 (CLI 1.17.11) through Gym's external-agent benchmark interface. Native task verifiers inspect the final app state before teardown; Gym collects their fractional rewards and excludes masked infrastructure failures.

See [the evaluation guide](../../fern/versions/latest/pages/evaluation/saas-bench.mdx) for provisioning, preparation, collection and validation limits. Tasks, images, credentials, native verifier implementations and private rollout evidence are provisioned separately and are not committed here.

This is the baseline evaluation adapter. RL training, generated-task LLM rewards, v5 generation and optional sub-agent modes are outside this integration.
