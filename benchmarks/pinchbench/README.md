# PinchBench

The PinchBench environment server runs the 147 tasks from upstream v2.0.0. For each episode it reads the pinned task, stages its public fixtures in a new workspace, and lends that sandbox to the configured Gym agent. After the session closes, it copies the workspace to a verifier sandbox and passes the recorded conversation to PinchBench's grader. The result includes the continuous reward, criterion scores, and grading notes.

## Prepare and run

Build the verifier on the sandbox worker (or push it to a registry your provider can pull). Prepare the dataset on the Gym server:

```bash
docker build -f environment_servers/pinchbench/Dockerfile.verifier -t pinchbench-verifier:v2.0.0 .
docker build -f environment_servers/pinchbench/Dockerfile.fws -t pinchbench-fws:v2.0.0 .
gym eval prepare --benchmark pinchbench
```

Configure the usual Gym policy model endpoint. The example uses the Docker sandbox provider and OpenClaw. The policy model server must be reachable from inside the agent sandbox. Set `PINCHBENCH_JUDGE_MODEL`, `PINCHBENCH_JUDGE_BASE_URL`, and `PINCHBENCH_JUDGE_API_KEY` for LLM-judged and hybrid tasks. Automated tasks can run without judge credentials.

```bash
gym eval run --benchmark pinchbench --model-type vllm_model \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
    --split benchmark --limit 1 --concurrency 1 --num-repeats 1 --output results/pinchbench-smoke.jsonl
```

`PINCHBENCH_AGENT_IMAGE` selects the task runtime and `PINCHBENCH_VERIFIER_IMAGE` selects the image built above. The verifier image contains the task definitions, private fixtures and grader. Use a separate agent image with the tools that the selected harness and tasks require. The environment server refuses the unisolated local provider.

## Harness compatibility

For OpenCode, use `--config benchmarks/pinchbench/opencode.yaml` in place of `--benchmark pinchbench` in the run command. Both recipes use the same dataset and verifier.

To add another harness, follow the OpenCode recipe, declare the PinchBench dataset on its agent instance, and set `pinchbench_environment_server.environment_servers.pinchbench.agent_server.name` to the instance name. The harness must support direct borrowed sandbox access. Tasks that keep a conversation across prompts also require session continuation. The environment preserves the workspace across a task's requested conversation resets and grades the combined transcript. The default OpenClaw session supports a single activation. Use a continuation-capable harness such as OpenCode for the three multi-prompt tasks.

Four tasks use a separate Fake Workspace Services image (`PINCHBENCH_FWS_IMAGE`). It routes the Google Workspace and GitHub CLIs to an episode-local mock service. Web, image, skill, and scheduled-task capabilities depend on the harness and image. PinchBench's original tool-specific criteria still apply. Record the agent, image, tool availability and failed episodes alongside scores from a harness comparison.

The earlier OpenClaw `/run` recipe and its published parity measurements are in [the standalone integration](../../responses_api_agents/pinchbench/README.md). Those measurements apply to that recipe. To reproduce that setup with the complete dataset and previous `PINCHBENCH_*` settings, use `--config benchmarks/pinchbench/legacy.yaml` in place of `--benchmark pinchbench`. Dataset preparation and the five committed example rows retain their original format.
