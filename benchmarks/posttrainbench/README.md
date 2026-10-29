# PostTrainBench

[PostTrainBench](https://github.com/aisa-group/PostTrainBench) v1.2 ([paper](https://arxiv.org/abs/2603.08640),
MIT): a CLI coding agent has 10 hours on one H100 to post-train a small base model for one benchmark. There are
24 tasks: six benchmarks (aime2025, arenahardwriting, gpqamain, gsm8k, healthbench, humaneval) times four base
models (Qwen3-1.7B-Base, Qwen3-4B-Base, SmolLM3-3B-Base, gemma-3-4b-pt). The score is the accuracy of the
agent's `final_model`, with the base model's zero-shot baseline where a judge flags cheating or the evaluation
fails, weighted per benchmark with upstream's `scripts/factors.json` and averaged over the base models.

The tasks are upstream's own Harbor tasks (`src/harbor_adapter`, revision pinned in `prepare.py`), run on the
`posttrainbench` environment server (the `harbor` server plus the weighted score) with Claude Code as the agent.

## What differs from upstream's Harbor run

Upstream runs the tasks with Harbor on Modal. `prepare.py` changes only what Modal provided:

- **Model hand-off.** The collect hook copies `final_model/` to `/logs/artifacts/final_model`, which Gym carries
  into the separate verifier through the host, instead of to a shared Modal volume. Expect 3 to 9 GB per task.
- **Transcript.** The agent copies Claude Code's stream-json transcript to `/logs/agent/claude-code.txt` when its
  turn ends (`sandbox_transcript_path`), where upstream's collect hook and judges read it. A run whose judges had
  no transcript counts as unjudged.
- **Judges and graders.** The four codex judges (`gpt-5.6-terra`) and the arenahardwriting and healthbench graders
  (`gpt-5-mini`) call `judge_base_url` with the key in `judge_api_key_env`, by that endpoint's model names
  (`judge_models`). For those two benchmarks upstream also gives the agent an OpenAI key for its own grader. The
  agent can read and use that key, so it never gets the judges' key: set `agent_api_key_env` to a separate,
  budget-limited key for the same endpoint. Without one the agent gets no key, and its own grader for
  arenahardwriting and healthbench cannot run (the final evaluation is not affected).
- **Reward.** `gym_score.py` runs after upstream's `test.sh` and writes `reward.json` with the score
  `scripts/collect.py` would use: the accuracy, or the baseline. A PTB-lookup flag also takes the baseline, where
  `collect.py` stops. An episode whose judging did not finish (a missing or unreadable verdict, or judges without
  the agent transcript) is masked as inconclusive, as `collect.py` refuses unjudged runs; resume the run to judge
  it again.
- **Model tree.** A `final_model` that holds symlinks or special files counts as not submitted (the baseline):
  a link such as `config.json -> /proc/self/environ` would print the verifier's keys into its logs.
- **Verifier image.** Gym builds the separate verifier from upstream's `tests/Dockerfile`, as Harbor does,
  with the task's resources (one GPU) instead of upstream's H100-only `[verifier.environment]` table; codex is
  installed per judge as upstream pins it.
- **Agent.** Gym's `claude_code_agent` runs Claude Code 2.1.219 (the image's pin) against the policy model
  through Gym's `/v1/messages`. Upstream's `CLAUDE_CODE_EFFORT_LEVEL` and `--thinking-display` are not set:
  reasoning is the model server's setting.

## Images

Each task image is built from its `environment/` folder and each verifier image from `tests/` (Harbor's
`hb__<hash>` names), `linux/amd64`, about 25 GB with shared layers. The H100 sandboxes must be able to pull them. Build them once on an amd64 host with buildah,
podman or docker and a registry, then commit both records so later prepares reuse the images (the builder needs
the image configurations and inputs digests in `compose-images.json` as well as `built-images.json`):

```sh
gym eval prepare --benchmark posttrainbench ++prepare_script_args.builds.repository=<registry>/<path>
cp benchmarks/posttrainbench/data/tasks/posttrainbench/{built-images,compose-images}.json benchmarks/posttrainbench/
```

Without the records or a builder, prepare stops and names the tasks.

## Run

Set `OPENSANDBOX_DOMAIN_GPU` and `OPENSANDBOX_API_KEY_GPU` (an OpenSandbox deployment with H100s), `HF_TOKEN`
(read-only: the agent can read it; it needs access to `google/gemma-3-4b-pt` and `Idavidrein/gpqa`) and the judge
key (`INFERENCE_API_KEY` by default). The sandboxes call the policy model, so serve it where they can reach it
(`++use_absolute_ip=true`).

```sh
gym eval prepare --benchmark posttrainbench
gym eval run --benchmark posttrainbench --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/posttrainbench/rollouts.jsonl ++use_absolute_ip=true
```

`++ptb_max_concurrent_episodes=<n>` caps the tasks that run at once (each holds two H100s in turn); a waiting
task gives up after `ptb_queue_timeout_seconds` (default 72 h).
`++prepare_script_args.benchmarks=[gsm8k]` and `++prepare_script_args.models=[qwen3-1.7b]` select tasks, and
`++prepare_script_args.num_hours=1` shortens the agent budget for a smoke run. A selection is not the
benchmark's score.

## Metrics

- `ptb/weighted_avg`: the run score, as upstream's `scripts/aggregate.py` computes it. It is reported only when
  every task has a scored rollout in every repeat; `ptb/tasks_without_score` counts the missing ones.
- `ptb/<benchmark>`: the mean reward over the scored base models.
- `ptb/judge_flagged`, `ptb/ptb_lookup_flagged`, `ptb/eval_failed`, `ptb/no_model`: counts over the scored
  rollouts. A task without a submitted model scores its baseline, as upstream's test.sh records it.
