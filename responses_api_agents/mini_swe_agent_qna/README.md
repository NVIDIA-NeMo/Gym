# mini_swe_agent_qna

Mini-SWE-Agent harness for **SWE-Atlas Codebase QnA**. Drives the
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) bash loop inside a
Gym sandbox (e.g. Apptainer `.sif` images on a cluster), lets the agent explore a
repository to answer an open-ended question, then delegates scoring to the
[`swe_atlas_qna`](../../resources_servers/swe_atlas_qna/README.md) rubric-judge
resources server.

Forked from [`mini_swe_agent_2`](../mini_swe_agent_2/README.md) but specialized
for QnA rather than SWE-Bench:

| | mini_swe_agent_2 (SWE-Bench) | mini_swe_agent_qna |
|---|---|---|
| Config template | built-in `swebench.yaml` | QnA `configs/mswea_qa_config.yaml` |
| Task output | git patch (`submission`) | `/logs/agent/answer.txt` (`<<FINAL_ANSWER>>`) |
| Reward | run unit tests inline | delegate to `swe_atlas_qna` `/verify` (rubric judge) |
| Image | SWE-Bench naming convention | `image_template` filled from `verifier_metadata` |

## How it works

1. `/run` launches the mini-swe-agent loop in a sandbox (Ray task, `SPREAD`
   scheduling). The agent runs read-only bash against the repo at `cwd` (`/app`).
2. The task ends when the agent writes its answer to `answer_path`
   (`/logs/agent/answer.txt`) and submits `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   The harness reads that answer file out of the sandbox.
3. The extracted answer is POSTed to the resources server `/verify`, which grades
   it against the task's rubrics and returns the reward. The full trajectory is
   returned for inspection.

## Key config fields

- `resources_server` — the `swe_atlas_qna` rubric-judge server (owns reward).
- `sandbox_provider` — a Gym sandbox provider; the example config uses the
  Apptainer provider for cluster `.sif` images.
- `image_template` — Python `.format` template for the sandbox image, filled from
  the row's `verifier_metadata` (e.g. `"/lustre/sifs/{sif_basename}"`). When
  `null`, the row's `docker_image` is used directly.
- `mini_swe_config_path` — the QnA mini-swe-agent config template.
- `answer_path` — where the agent writes its answer inside the sandbox.

## Usage

```bash
# Prepare data + configure the judge (see the resources server / benchmark READMEs)
export SWE_ATLAS_QNA_JUDGE_BASE_URL=... SWE_ATLAS_QNA_JUDGE_API_KEY=... SWE_ATLAS_QNA_JUDGE_MODEL=...

# Start servers (Apptainer must be on PATH; point image_template at your .sif dir)
gym env start \
    --config responses_api_agents/mini_swe_agent_qna/configs/mini_swe_agent_qna.yaml \
    --model-type vllm_model \
    '+swe_atlas_qna_mini_swe_agent.responses_api_agents.mini_swe_agent_qna.image_template=/lustre/sifs/{sif_basename}'

# Collect rollouts
gym eval run --no-serve \
    --agent swe_atlas_qna_mini_swe_agent \
    --input resources_servers/swe_atlas_qna/data/example.jsonl \
    --output results/swe_atlas_qna_rollouts.jsonl \
    --num-repeats 1
```

## Licensing Information

- Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- mini-swe-agent: MIT
