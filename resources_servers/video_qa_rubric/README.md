# Video QA rubric

A training environment for open-ended questions about a video. The policy sees the video and the question and writes
a free-text answer. A judge model grades the final answer against the question's own rubric, and the reward is the
rubric score from 0 to 1. The judge sees the question, the rubric and the final answer; it never sees the video, the
policy's reasoning or a reference answer.

`configs/video_qa_rubric.yaml` runs single-turn rows (`simple_agent`, one model call). The same resources server
grades agentic rows (`setting: agentic`) when an agent harness is bound to it; `configs/video_qa_rubric_agentic.yaml`
runs them with OpenCode in a sandbox that holds the video ([agentic setting](#agentic-setting)).

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
URL the policy's model server accepts: an http(s) or data URL, or a `file://` URL the server is allowed to read (vLLM's
`--allowed-local-media-path`).

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
| `execution_failure` | A tool call in a single-turn row, or an agent run whose response is `failed` or `incomplete` | 0 | no |
| `infrastructure_failure` | The model call of a single-turn row failed | 0 | yes |

Both failures are read from the response's own status with video_mcqa's `response_failure`, so a replayed row scores
the same. In the agentic setting the Environment Server sends verify only the failed runs that its outcome policy
scores, so a failed agent run is not masked.

`empty_answer_reward` is a training option (for example `-0.5`). With it set, the answer must also be the last thing
the model produced: an attempt that ended on a tool call, a tool result or reasoning gets `empty_answer_reward`
instead of a grade for an earlier message. A response with no output at all stays at 0, because Gym's agents verify a
failed run (`execution_failure_reward_zero`) as such a response and require exactly 0.

Metrics: `mean/rubric_score` and `mean/reward` (they differ only when `empty_answer_reward` applies), with
`full_score_rate`, `at_least_0.8_rate` and `outcomes/*` counts. Masked rows are left out of all of them and reported
as coverage. Each verified row carries `rubric_points`, the `judgment`, `judge_usage` and `judge_attempts`.

## Judge

`video_qa_rubric_judge` in `configs/video_qa_rubric.yaml` is an `openai_model` server: any OpenAI-compatible chat model
with JSON-schema structured output, set with `judge_base_url`, `judge_api_key` and `judge_model_name`. A strong
reasoning model is recommended. Its sampling and reasoning settings go in `judge_chat_completion_create_params`:

```yaml
video_qa_rubric:
  resources_servers:
    video_qa_rubric:
      judge_chat_completion_create_params:
        messages: []
        reasoning_effort: high
        max_completion_tokens: 32768
```

A model served with vLLM works through the same `openai_model` server and vLLM's OpenAI-compatible endpoint, as in
the example rollouts below.

## Agentic setting

`configs/video_qa_rubric_agentic.yaml` runs the stock `opencode_agent` (OpenCode 1.17.11) through a
`single_agent_turn_legacy` Environment Server, with the agent, sandbox image, video tools and sandbox settings of the
agentic Video-MME v2, Video-MME and LVBench configs (see video_mcqa's README, "Agentic setting"). Each episode's seed
lends OpenCode a sandbox that holds the row's video, and the episode's close stops it; the judge then grades the final
answer as in the single-turn setting. The sandbox reaches only the policy model server's host, so the judge server
listens on 127.0.0.1.

An agentic row names the video by its path in the sandbox instead of attaching it.
`task.prepare_agentic(task_id, question, rubric, video_path="/testbed/videos/clip.mp4")` builds the user turn (the
question and that path, never the rubric), and the row adds the `video_binding` the seed reads:

```json
"video_binding": {"source": "<host file under video_roots>", "sandbox_path": "videos/clip.mp4", "sha256": "<its SHA-256>"}
```

The seed refuses a row whose video is not a regular file under `video_roots` or does not have that SHA-256, and puts
the video at `/testbed/<sandbox_path>`. Set the sandbox image (`VIDEO_MCQA_SANDBOX_IMAGE`), the OpenSandbox provider
settings (`OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`), the judge, and `video_qa_rubric_agentic_video_roots`.

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
    - resources_servers/video_qa_rubric/configs/video_qa_rubric.yaml
    judge_base_url: https://judge.example.com/v1
    judge_api_key: ${oc.env:JUDGE_API_KEY}
    judge_model_name: my-judge-model
    video_qa_rubric:
      resources_servers:
        video_qa_rubric:
          empty_answer_reward: -0.5
          judge_max_attempts: 12
          judge_endpoint_max_concurrency: 24
          judge_chat_completion_create_params:
            messages: []
            reasoning_effort: high
            max_completion_tokens: 32768
```

Each training row needs `agent_ref: {type: responses_api_agents, name: video_qa_rubric_single_turn}`, which
`gym dataset collate` adds for datasets declared in the config. In a mix with other environments, every row's
`agent_ref` routes it to its own environment. Keep the judge separate from the policy: a policy that grades its own
answers can learn to please itself.

## Example data and rollouts

`data/example.jsonl` holds five invented questions with invented rubrics on video_mcqa's five synthetic clips
(`resources_servers/video_mcqa/data/example_videos`, inlined as MP4 data URLs), built with `prepare_single_turn`.
License: Apache 2.0.

`data/example_rollouts.jsonl` holds one rollout per row from `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
(BF16, one GPU) served by vLLM `0.32.0rc1.dev190+g3f1bcfd82` with the model card's video settings and the model's
default sampling. To fit one GPU, the same served model was also the judge (`max_completion_tokens` 16,384); a training
run should use a separate, stronger judge.

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --max-model-len 131072 \
  --trust-remote-code --video-pruning-rate 0.5 --reasoning-parser nemotron_v3 \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
gym env start "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/video_qa_rubric/configs/video_qa_rubric.yaml]" \
  +policy_base_url=http://127.0.0.1:8000/v1 +policy_api_key=EMPTY \
  +policy_model_name=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  +judge_base_url=http://127.0.0.1:8000/v1 +judge_api_key=EMPTY \
  +judge_model_name=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  ++video_qa_rubric.resources_servers.video_qa_rubric.judge_chat_completion_create_params.max_completion_tokens=16384
gym eval run --no-serve --agent video_qa_rubric_single_turn \
    --input resources_servers/video_qa_rubric/data/example.jsonl \
    --output resources_servers/video_qa_rubric/data/example_rollouts.jsonl --limit 5
```

Three answers earn full points (the blue background and the yellow diamond, one red dot, the bar moving down; the
judge's first judgment of the bar was rejected by the arithmetic check and its second accepted). "An orange
rectangle moves from the left to the right" earns 60 of 100: the direction criterion, but not the square. On the
counter the model runs out of its 16,384 output tokens without an answer (`empty_answer`, 0, no judge call).
`mean/rubric_score` is 0.72.

## Tests

```bash
pytest resources_servers/video_qa_rubric/tests
gym env test --resources-server video_qa_rubric
```

The tests use invented questions and rubrics and a scripted judge; no API key or network is needed.
