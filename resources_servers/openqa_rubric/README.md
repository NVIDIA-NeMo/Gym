# OpenQA rubric

A training environment for open-ended questions about a video. The policy sees the video and the question and writes
a free-text answer. A judge model grades the final answer against the question's own rubric, and the reward is the
rubric score from 0 to 1. The judge sees the question, the rubric and the final answer; it never sees the video, the
policy's reasoning or a reference answer.

`configs/openqa_rubric.yaml` runs single-turn rows (`simple_agent`, one model call). The same resources server
grades agentic rows (`setting: agentic`) when an agent harness is bound to it.

## Rows

| Field | Content |
| --- | --- |
| `task_id` | The question's ID |
| `setting` | `single_turn` or `agentic` |
| `responses_create_params` | The policy's input, for example the video as an `input_video` part and the question as text |
| `verifier_metadata.question` | The question text the judge sees |
| `verifier_metadata.rubric` | The rubric (below) |

A rubric has `criteria` and optional `contradictions`:

| Field | Content |
| --- | --- |
| `criteria[].id` | Unique ID |
| `criteria[].requirement` | What a correct answer must contain |
| `criteria[].weight_points` | The criterion's points; the weights of all criteria add up to 100 |
| `contradictions[].id` | Unique ID |
| `contradictions[].condition` | A statement that contradicts the correct answer |
| `contradictions[].penalty_points` | Points subtracted when the answer contains it (default 0) |
| `contradictions[].score_cap_points` | The highest score an answer that contains it can get (default: no cap) |

Other keys reach the judge unchanged, for example per-criterion `full_credit`, `partial_credit`, `zero_credit` and
`acceptable_variants`, evidence from the video, or a rubric-wide `answer_policy`. `task_data.py` holds the schema, and
`task.prepare_single_turn(task_id, question, rubric, video_url=...)` builds a single-turn row; the video can be any
URL the policy's model server accepts, such as a local file path or a data URL.

## Grading

1. The final answer is the last assistant message, after any reasoning block (video_mcqa's
   `final_answer_text`).
2. The judge gets a system message and one user message: the grading instructions and a JSON object with the
   question, the rubric and the final answer (`rubric.judge_messages`). Its reply must follow a strict JSON schema
   (`rubric.JUDGMENT_SCHEMA`): the points awarded per criterion, the contradictions it applied, and the score from 0
   to 100.
3. The judgment is accepted only when it scores every criterion once within its weight, copies the penalty and cap
   of each contradiction it applies, and its score equals the sum of the points, minus the penalties, capped by the
   caps and clamped to [0, 100] (`rubric.judgment_points`). The reward is that score / 100.

A judgment that fails this check, a timed-out call and a 408, 429 or 5xx reply are tried again, up to
`judge_max_attempts` attempts in all. Other 4xx replies are not retried. When no attempt succeeds, Gym's judge
failsafe masks the row as `judge_failed` and keeps the answer, so a recovery run judges it again without sampling the
policy.

| `outcome` | When | `reward` | Masked |
| --- | --- | --- | --- |
| `graded` | The judge's checked score | score / 100 | no |
| `empty_answer` | No final answer (reasoning only, an unfinished reasoning block, an empty message) | 0, or `empty_answer_reward` | no |
| `execution_failure` | A tool call in a single-turn row, or an agent run marked as failed | 0 | no |
| `infrastructure_failure` | The model call or the agent's infrastructure failed | 0 | yes |

`empty_answer_reward` is a training option (for example `-0.5`). With it set, the answer must also be the last thing
the model produced: an attempt that ended on a tool call, a tool result or reasoning gets `empty_answer_reward`
instead of a grade for an earlier message. A response with no output at all stays at 0, because Gym's agents verify a
failed run (`execution_failure_reward_zero`) as such a response and require exactly 0.

Metrics: `mean/rubric_score` and `mean/reward` (they differ only when `empty_answer_reward` applies), with
`full_score_rate`, `at_least_0.8_rate` and `outcomes/*` counts. Masked rows are left out of all of them and reported
as coverage. Each verified row carries `rubric_points`, the `judgment`, `judge_usage` and `judge_attempts`.

## Judge

`openqa_rubric_judge` in `configs/openqa_rubric.yaml` is an `openai_model` server: any OpenAI-compatible chat model
with JSON-schema structured output, set with `judge_base_url`, `judge_api_key` and `judge_model_name`. A strong
reasoning model is recommended. Its sampling and reasoning settings go in `judge_chat_completion_create_params`:

```yaml
openqa_rubric:
  resources_servers:
    openqa_rubric:
      judge_chat_completion_create_params:
        messages: []
        reasoning_effort: high
        max_completion_tokens: 32768
```

To use a model served with vLLM instead, point `judge_model_server` at a `vllm_model` server.

## Use in RL training

With NeMo RL, add the config to `env.nemo_gym.config_paths` and set the judge and the training options under
`env.nemo_gym`, as for any Gym environment:

```yaml
env:
  should_use_nemo_gym: true
  should_mask_flagged_samples: true  # keep judge failures and failed model calls out of the loss
  nemo_gym:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/openqa_rubric/configs/openqa_rubric.yaml
    judge_base_url: https://judge.example.com/v1
    judge_api_key: ${oc.env:JUDGE_API_KEY}
    judge_model_name: my-judge-model
    openqa_rubric:
      resources_servers:
        openqa_rubric:
          empty_answer_reward: -0.5
          judge_max_attempts: 12
          judge_endpoint_max_concurrency: 24
          judge_chat_completion_create_params:
            messages: []
            reasoning_effort: high
            max_completion_tokens: 32768
```

Each training row needs `agent_ref: {type: responses_api_agents, name: openqa_rubric_single_turn}`, which
`gym dataset collate` adds for datasets declared in the config. In a mix with other environments, every row's
`agent_ref` routes it to its own environment. Keep the judge separate from the policy: a policy that grades its own
answers can learn to please itself.

## Example data and rollouts

`data/example.jsonl` holds five invented questions with invented rubrics on video_mcqa's five synthetic clips
(`resources_servers/video_mcqa/data/example_videos`, inlined as MP4 data URLs), built with `prepare_single_turn`.
License: Apache 2.0.

```bash
gym env start "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/openqa_rubric/configs/openqa_rubric.yaml]"
gym eval run --no-serve --agent openqa_rubric_single_turn \
    --input resources_servers/openqa_rubric/data/example.jsonl \
    --output resources_servers/openqa_rubric/data/example_rollouts.jsonl --limit 5
```

## Tests

```bash
pytest resources_servers/openqa_rubric/tests
gym env test --resources-server openqa_rubric
```

The tests use invented questions and rubrics and a scripted judge; no API key or network is needed.
