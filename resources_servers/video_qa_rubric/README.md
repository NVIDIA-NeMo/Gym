# Video QA rubric

A training environment for open-ended questions about a video. The policy sees the video and the question and writes
a free-text answer. A judge model grades the final answer against the question's own rubric, and the reward is the
rubric score from 0 to 1. The judge sees the question, the rubric and the final answer; it never sees the video or the
policy's reasoning. Each row names its grading protocol: weighted points (the default, without a reference answer) or
reference levels, for question sets graded with a reference answer and a score scale of their own
([Reference levels](#reference-levels)).

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
| `verifier_metadata.grading` | `weighted_points` (the default when absent) or `reference_levels` |
| `verifier_metadata.reference_answer` | `reference_levels` only: the reference answer the judge sees |
| `verifier_metadata.subset` | Optional: the part of the question set the row belongs to, for per-subset metrics |

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
`--allowed-local-media-path`). Its `prompt` (default `Watch the video and answer the question.\n\nQuestion:
{question}`) is the text after the video; a question set that asks its questions as they are passes `"{question}"`.
Its `grading`, `reference_answer` and `subset` arguments fill the fields above.

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

A judgment that fails this check (or the [reference levels](#reference-levels) check), a timed-out call and a 408,
429 or 5xx reply are tried again, up to `judge_max_attempts` attempts in all. Other 4xx replies are not retried. When no attempt succeeds, Gym's judge
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

`empty_answer_reward` is a training option (for example `-0.5`), as in video_mcqa: the reward of an attempt that
produced output but no final answer. The final answer is always the last assistant message, whatever comes after it.
A response with no output at all stays at 0, because Gym's agents verify a failed run
(`execution_failure_reward_zero`) as such a response and require exactly 0.

Metrics: `mean/rubric_score` and `mean/reward` (they differ only when `empty_answer_reward` applies), with
`full_score_rate`, `at_least_0.8_rate` and `outcomes/*` counts. Rows with a `subset` add each subset's mean rubric score
and attempt count (`subsets/<subset>/rubric_score`, `subsets/<subset>/count`) and the mean of those means, each subset
weighing the same (`mean_of_subsets/rubric_score`). Masked rows are left out of all of them and reported as coverage.
Each verified row carries `rubric_points` (or `rubric_level`), the `judgment`, `judge_usage` and `judge_attempts`.

### Reference levels

A row with `grading: reference_levels` carries a `reference_answer`, and its `rubric` may be any JSON object of grading
rules, such as the score rules, accepted phrasings and rejections of a published question set. The judge reads the
question, the reference answer, the rubric and the final answer through the server's `reference_levels` settings, and
returns an integer level; the reward is that level / `max_level`. A server without these settings refuses such rows.

| Setting | Content |
| --- | --- |
| `reference_levels.system_prompt` | The judge's system message |
| `reference_levels.user_template` | The judge's user message, with the row's `{question}`, `{reference_answer}`, `{rubric_json}` (the rubric as `json.dumps(rubric, ensure_ascii=False)`) and `{answer}`; other braces doubled |
| `reference_levels.max_level` | The top of the integer scale (default 5) |

The judge's reply must be exactly the JSON object `{"score": <integer from 0 to max_level>, "reason": <nonempty
text>}`, with one choice that finished normally; no response format is requested, so the prompt asks for it. Anything
else is a rejected judgment, retried as above. The settings are a question set's own, so they live in its config:
`benchmarks/mm_lifelong_v1_1` (MM-Lifelong v1.1, the dataset's grader) is an example. Answers without a final answer
score 0 without a judge call, as in the default protocol.

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
settings (`OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`), the judge, and `video_qa_rubric_agentic_video_roots`. To
train on agentic rows, see [agentic rows with OpenCode](#agentic-rows-with-opencode).

Videos too large to upload for every episode stay in the sandbox image instead: a row with `preloaded_media` names the
image (through the server's `sandbox_preloaded_images`) and pins the image's catalog of them, and `prepare_agentic(...,
text=...)` takes a prompt that describes them its own way (video_mcqa's README, "Media preloaded in the image";
`benchmarks/mm_lifelong_v1_1_agentic` uses it).

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
    judge_api_key: \${oc.env:JUDGE_API_KEY}
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

The judge's key is escaped (`\${...}`). NeMo RL resolves its own config and prints it, so an unescaped
`${oc.env:JUDGE_API_KEY}` would put the key itself into NeMo RL's log and into the global config it hands to Gym.
Escaped, the config keeps the variable's name, and each server process reads the key from its own environment, as
NeMo RL does for the policy's key.

Each training row needs `agent_ref: {type: responses_api_agents, name: video_qa_rubric_single_turn}`, which
`gym dataset collate` adds for datasets declared in the config. In a mix with other environments, every row's
`agent_ref` routes it to its own environment. Keep the judge separate from the policy: a policy that grades its own
answers can learn to please itself.

### Agentic rows with OpenCode

`configs/video_qa_rubric_agentic.yaml` trains on agentic rows with the agent, image and sandbox of the
[agentic setting](#agentic-setting). Each row names the agent,
`agent_ref: {type: responses_api_agents, name: video_qa_rubric_agentic_opencode}`, and binds its video with
`video_binding`. With NeMo RL:

```yaml
env:
  should_use_nemo_gym: true
  should_mask_flagged_samples: true
  nemo_gym:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/video_qa_rubric/configs/video_qa_rubric_agentic.yaml
    video_qa_rubric_agentic_video_roots: [/data/invented_video_qa/references]
    judge_base_url: https://judge.example.com/v1
    judge_api_key: \${oc.env:JUDGE_API_KEY}
    judge_model_name: my-judge-model
    video_qa_rubric_agentic:
      resources_servers:
        video_qa_rubric:
          empty_answer_reward: -0.5
          judge_max_attempts: 12
          judge_chat_completion_create_params:
            messages: []
            reasoning_effort: high
            max_completion_tokens: 32768
    # Raise both to the episodes the trainer keeps in flight; each holds a sandbox.
    video_qa_rubric_agentic_opencode:
      responses_api_agents:
        opencode_agent:
          concurrency: 128
    video_qa_rubric_agentic_environment:
      environment_servers:
        single_agent_turn_legacy:
          max_concurrent_episodes: 128
    # Each video upload is one request that this bounds (opensandbox.yaml: 300 s); see below.
    sandbox:
      opensandbox:
        connection:
          request_timeout_s: 900
```

Set `VIDEO_MCQA_SANDBOX_IMAGE`, or
`video_qa_rubric_agentic.resources_servers.video_qa_rubric.sandbox_config.image`, to an image built from
`resources_servers/video_mcqa/sandbox_image` of the same tree, pinned by digest (video_mcqa's README, "Sandbox
image"), and set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`. The config's limits set the trainer's:

- Each call sends the episode so far and holds, with its output, up to `context_window` (262,144) tokens, and the
  training sequences come from these calls: the policy's sequence length, the model server's `max_model_len` and any
  sequence-packing or micro-batch token limit must hold that many.
- A call asks for up to `max_output_tokens` (32,768) new tokens; the trainer's generation limit must allow them.
- Every call sends the episode's tool images again, up to the video tools' `max_tool_media_images` (8,192): the model
  server's image limit per request.
- Once admitted, `default_episode_timeout_seconds` (6,600 s) is the episode protocol budget (seeding, the agent run and
  grading). Final cleanup runs outside that deadline (`single_agent_turn_legacy` allows up to `cleanup_timeout_seconds`,
  180 s, per cleanup callback), so `/run` can return later than 6,600 s; give the trainer's rollout timeout headroom
  beyond 6,600 s for cleanup (for example, 7,200 s or more).

Compaction is off, so an episode that fills the context ends with a context-limit error and scores 0.

Our training rows for this config are on NVIDIA's DSS: `postOmni_RL_video-openqa-agentic@v0.1` holds 1,877 training
rows and 17 validation rows, and `postOmni_RL_video-openqa-agentic_media@v0.1` holds their 1,103 videos. A row's
`video_binding.source` names its video relative to the media dataset (`references/<sha256>.<ext>`): point the rows at
your copy, and list its `references/` folder in `video_qa_rubric_agentic_video_roots`. We grade them with GPT-6.1-Sol
at `reasoning_effort: max`. Each upload is one request that the provider's `connection.request_timeout_s` (300 s in
`opensandbox.yaml`) bounds (video_mcqa's README, "Seeding"). The largest of these videos, 2.33 GiB, took 239 s to
upload with no other upload running, and the episodes of a group upload the same video at once, so the example
above raises the limit to 900 s.

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
