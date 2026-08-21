# FrontierScience Research

Open-ended expert science research benchmark from OpenAI's
[FrontierScience](https://huggingface.co/datasets/openai/frontierscience)
release. The public research set contains 60 original subtasks across
chemistry, biology, and physics. Each task has a 10-point rubric and is
counted correct at `>= 7/10`.

## Source

- Dataset: `openai/frontierscience`, file `research/test.jsonl`, split `test`
- Announcement: https://openai.com/index/frontierscience/
- License: Apache 2.0

## Verification

Uses the shared [`frontierscience_judge`](../../resources_servers/frontierscience_judge/)
resource server in `judge_mode: research`. The judge receives the task, the
model answer, and the per-task rubric from the dataset's `answer` field.
It emits `Score: X/10` and `Judgement: YES/NO`; `YES` means the parsed score
is at least 7 points.

The default judge model config is inherited from `frontierscience_judge`:
`openai/gpt-oss-20b` on the public NVIDIA inference API
(`https://integrate.api.nvidia.com/v1`), reading `NVIDIA_API_KEY`. Override
`judge_base_url`, `judge_api_key`, and `judge_model_name` to use another
judge.

## Metrics

- `pass@k/accuracy`, `pass@1[avg-of-k]/accuracy`, `majority@k/accuracy`
  report the OpenAI-style binary correctness threshold (`score >= 7/10`).
- `pass@k/rubric_score` and `pass@1[avg-of-k]/rubric_score` report the
  normalized rubric score as a percentage.
- `chemistry/...`, `biology/...`, and `physics/...` report per-subject
  breakdowns.

## Reproduction note

On 2026-06-09, the 60 public research tasks were reproduced with direct
NVIDIA inference API calls and the research rubric judge. Both policy runs
used `reasoning_effort: high`, `stream: false`, and no temperature override.
The judge model was `openai/openai/gpt-5.5`. For comparison, the
[FrontierScience paper](https://arxiv.org/pdf/2601.21165) reports 17.5% for
Claude Opus 4.5 and 19.4% for OpenAI GPT-5.1 on the Research track.

| Policy model | Endpoint | Max tokens used | Completed | Reproduced accuracy | Paper accuracy | Mean rubric score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `aws/anthropic/claude-opus-4-5` | `/v1/messages` | 64000 | 60/60 | 16.67% | 17.5% | 3.865/10 |
| `openai/openai/gpt-5.1` | `/v1/chat/completions` | 128000 | 59/60 | 21.67% | 19.4% | 4.600/10 on completed rows |

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark frontierscience_research

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark frontierscience_research

# Collect rollouts
gym eval run --no-serve \
    --agent frontierscience_research_frontierscience_judge_simple_agent \
    --input benchmarks/frontierscience_research/data/frontierscience_research_benchmark.jsonl \
    --prompt-config benchmarks/prompts/generic/default.yaml \
    --output results/frontierscience_research_rollouts.jsonl \
    --num-repeats 1
```
