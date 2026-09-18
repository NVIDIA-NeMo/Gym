# harness_agent

Runs any gym agent harness in a sandbox with any resources server.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agnostic, not the agent config. 
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | in-box dir the agent's `repo_dir` points at, so edits land in the graded tree |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward goes in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |
Tasks with an external verifier (e.g. math) need none of these beyond an image.
This is largely for swe bench now.

## Hermes on Terminal-Bench 2.1

`terminal_bench_2_1/hermes_harness` runs the existing `hermes_agent` adapter through
this shared runner. The resources server creates the task container, the runner
executes Hermes there, and the resources server grades that same container.

```bash
export NEMO_GYM_SANDBOX_MODEL_BASE_URL=http://host.docker.internal:18113
uv run gym eval prepare --benchmark terminal_bench_2_1/hermes_harness
uv run gym eval run \
  --benchmark terminal_bench_2_1/hermes_harness \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --model-type vllm_model \
  --model "$MODEL_NAME" --model-url "$MODEL_URL" --model-api-key "$MODEL_API_KEY" \
  --num-repeats 1 --split benchmark \
  ++policy_model.responses_api_models.vllm_model.host=0.0.0.0 \
  ++policy_model.responses_api_models.vllm_model.port=18113
```

The URL above is for Docker Desktop. For another provider, set the URL to the Gym
model server address reachable from its containers. The model API key stays on that
server; the Hermes client uses `dummy`.

Run this profile from a Gym source checkout, or supply a `gym_source` archive with
`pyproject.toml`, `README.md`, and `LICENSE` alongside the Gym and Hermes packages.
Setup installs Python 3.13.14, the staged Gym source, and the Hermes dependency from
`responses_api_agents/hermes_agent/requirements.txt`. Task images need bash and
network access to Astral, PyPI, and GitHub. If curl or git is missing, the container
must run as root so setup can install them with apt-get. The default setup is
intended for Debian/Ubuntu task images.

This profile disables token capture and is for evaluation only. The existing Hermes
adapter currently reports zero token usage, so its usage fields cannot measure cost.

Adjust Hermes options under
`terminal_bench_2_1_hermes_harness.responses_api_agents.harness_agent.agent_kwargs`
(for example, `max_turns` or `max_tokens`), and the overall runner timeout under
`rollout_timeout` in the same harness block.
