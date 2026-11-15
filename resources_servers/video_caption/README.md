# Video caption reward

A single-turn training environment for video captioning. The policy sees a video as ordered frames and writes
one caption; the verifier scores that caption with an LLM judge for visual-event recall, precision, or F1,
adds a length reward, and can gate the total on a separate degeneration verdict. The judge is any
OpenAI-compatible chat model reached through a judge `ModelServerRef` and Gym's shared `call_judge` client.

## Configuration and data

`visual_reward_type` accepts `recall` (the default), `precision`, or `f1`. To select it in a training recipe:

```yaml
env:
  nemo_gym:
    video_caption:
      resources_servers:
        video_caption:
          visual_reward_type: f1
          penalize_degeneration: true
```

The equivalent CLI override is
`env.nemo_gym.video_caption.resources_servers.video_caption.visual_reward_type=f1`.
`penalize_degeneration` is independent and defaults to false.

A row carries `responses_create_params.input` (the policy prompt) and `verifier_metadata`:

```json
{"atomic_events": {"visual_events": ["A person walks."]}, "reference_caption": "A person walks across a room."}
```

`atomic_events.visual_events` is the list of reference atomic visual events the judge checks the caption against.
`reference_caption` is the original human visual caption, not a reconstruction from its events; precision and F1
judge the policy's extracted events against it, so those modes require it to be present and non-blank. Recall
rows do not need it. Audio event fields in `atomic_events` are ignored. `task_data.py` declares this schema for
`gym dataset validate` and collation. An optional top-level `source_index` is logged, never graded on.

## Visual score

All judging and token counting use `body.response.output_text`: the final assistant message text after Gym's
response conversion. Separate reasoning items are excluded. A reasoning-only response is an empty final caption
and receives zero without any judge request.

- **Recall R:** check whether each reference visual event is covered by the model caption, using
  `prompt_templates/visual_event_judge.txt`. R is the fraction of reference events covered.
- **Precision P:** first extract atomic visual events from the final model caption in a separate request.
  Then reuse the same coverage prompt, with the human reference caption as the candidate text and the
  predicted events as the list to check. P is the fraction of predicted events covered by the reference
  caption. Reference atomic events are not used as the precision target.
- **F1:** compute both directions in separate requests, then use `2 * R * P / (R + P)`, or zero when both
  scores are zero. It is the F1 of the two event-coverage fractions, not a one-to-one event matching.

Extraction uses `prompt_templates/visual_event_extraction.txt` (configurable through
`event_extraction_prompt_fpath`): the visual-only form of the atomic-event extraction prompt that builds the
reference `atomic_events` offline, keeping its atomicity, independent judgeability, deduplication, chronological
ordering, and timestamp-exclusion instructions; the audio and audio-visual branches are removed. It receives only
`{"visual_caption": "<final predicted caption>"}`, never the reference. Its strict output schema is
`{"visual_events": ["event", "..."]}`. Missing or extra fields, blank events, non-string entries, or invalid JSON
cause retries. An empty event list is valid and gives precision/F1 zero without a coverage request; the length
reward still applies to a nonempty caption.

Recall makes one successful request, precision two (extraction then coverage), and F1 three (recall,
extraction, precision coverage). Optional degeneration adds one separate request after these stages.
None of the judgments is combined into another request. Stages run sequentially within each caption; captions
share the admission semaphore. A failed stage stops subsequent stages. A retry never repeats an earlier
successful stage.

## Overall reward and degeneration

The verifier returns `D * (V + L) / 2`:

- V is the selected recall, precision, or F1 score.
- L is one for `min_caption_tokens`..`max_caption_tokens` final-caption tokens inclusive (100..4000 by default),
  otherwise zero, counted with the policy tokenizer at `caption_tokenizer_path`.
- D is zero when enabled degeneration judging returns `fail`, otherwise one.

Empty captions receive zero total reward, including no length reward. The verifier adds no validity mask or fixed
advantage of its own for a degeneration verdict; it is an ordinary zero reward.

A caption longer than `max_judged_caption_tokens` (16,000 by default; at least `max_caption_tokens`) is scored
without any judge request: V is zero, L is zero because the caption is outside the length window, and the
response sets `caption_over_judge_cap`. The row stays in training as a zero rather than being masked: a runaway
generation is the policy's failure, not a judge outage, and sending it to the judge would exceed the judge's
context (`max_model_len` 40,960 with `max_tokens` 20,480 in `configs/video_caption.yaml`). Count the cap in
policy tokens; a judge with a different tokenizer may need a lower value.

`penalize_degeneration` uses `prompt_templates/degeneration_judge.txt`, configurable through
`degeneration_judge_prompt_fpath`. Its criteria are broad: no degeneration or excessive repetition, a coherent and
relevant video description, and no exposed reasoning or drafting. It receives only the final caption, not reference
text, events, or prior judgments. It returns only `{"verdict":"pass"}` or `{"verdict":"fail"}`; pass requires all
criteria. It excludes factual precision/recall, and length alone is not a failure. A fail gates the entire reward,
including the length reward.

The response exposes `visual_reward_type` and `visual_reward`, plus the individual `visual_recall_reward`,
`visual_precision_reward`, and `visual_f1_reward`. An uncomputed score is null, not a measured zero; a masked
row (see below) carries null for all three. `visual_hits` is recall-only; `precision_visual_hits` and
`predicted_visual_events` carry the precision evidence. `judge_attempts`/`judge_response_text` are recall-only,
with separate `event_extraction_*`, `precision_judge_*`, and `degeneration_judge_*` attempt/response fields.
`length_reward`, `reward_before_degeneration`, `degeneration_reward`, `degeneration_checked`, and
`degeneration_detected` keep their values from before the gate.

## Judge requests and failures

Every stage sends one user message, with no system message, to the judge's `/v1/chat/completions` with a
strict JSON-schema `response_format`, `max_tokens=20480` and `reasoning_effort=low` (all from
`judge_chat_completion_create_params`). Any OpenAI-compatible judge that honours `response_format` works:
`configs/video_caption.yaml` serves one locally through `local_vllm_model`, `configs/video_caption_api_judge.yaml`
uses an API endpoint through `openai_model`. The reference runs used a 27B Qwen3-family reasoning model with
thinking enabled at low reasoning effort; no temperature/top-p override is added.

Gym wraps a judge's separate reasoning in `<think>` tags. The verifier lifts the choice through
`ChatCompletionsConverter` and validates only the final `output_text`, so JSON inside reasoning cannot replace
a missing final answer. Tests exercise the real converter and shared judge client with mocked transport.

Each stage has `judge_max_retries + 1` attempts, each bounded by `judge_attempt_timeout_seconds` after
admission to the `judge_endpoint_max_concurrency` semaphore; queue waiting does not consume the serving
deadline. The server owns this budget: every attempt is one `call_judge(..., max_attempts=1)` HTTP request, so
the shared client's own retries do not nest inside it. A reply that can succeed on another attempt (a 408, 429
or 5xx, a timeout, a lost connection, a malformed or incomplete answer) is retried after a full-jitter backoff
of up to `judge_retry_base_delay_seconds * 2^attempt`, capped at `judge_retry_max_delay_seconds`, so concurrent
verifies do not retry a rate-limited judge in lockstep. Any other 4xx (an unsupported `response_format`, a rejected
credential) fails at once, and a Gym failure that is not a judge failure (a judge provider stop) passes through
untouched. Exhaustion raises `JudgeError`; Gym's `judge_failsafe` wrapper then returns `reward: 0.0` with
`mask_sample: true` and `failure_kind: judge_failed` (also `_ng_failure_class=judge_failed`), so the row drops out
of the training batch instead of teaching a zero. No length reward is added. Cancellation propagates without a
retry.

A row the verifier cannot score is never a 5xx (which would make the environment server replay the episode
and spend another policy rollout). Blank entries in `atomic_events.visual_events`, or a precision/F1 row without a
non-blank `reference_caption`, return `reward: 0.0` with `mask_sample: true`,
`failure_kind: video_caption:unscorable_row` and a `failure_reason`, before any judge call. A missing
`atomic_events`, a missing `visual_events`, or an empty list is rejected with 422 by the request model before
`verify()` runs. `task_data.py` rejects all of these rows at data-validation time (`gym dataset validate`, the
repo's row-drift test).

## Data

Rows are standard Gym rows for `simple_agent`, run through the `single_agent_turn_legacy` environment server
declared in both configs (`video_caption_environment_server`): the policy prompt carries the video as ordered
image frames (or a video input part, depending on the RL framework's media path) followed by the caption
instruction, and `verifier_metadata` carries `atomic_events.visual_events` plus `reference_caption` for
precision/F1. `caption_tokenizer_path` must point at the policy's `tokenizer.json` so the length reward counts
policy tokens.

`data/example.jsonl` holds five synthetic clips (four flat-colour frames each: a sliding rectangle, a growing
circle, a falling triangle, a shrinking bar beside a still square, a brightening background) with three or four
reference events and a reference caption each. They exercise the row format and the verifier end to end; they are
not a benchmark or a baseline. `data/example_rollouts.jsonl` holds one real-model rollout per row through
`configs/video_caption_api_judge.yaml`. Training data comes from your own video caption corpus with per-video
atomic visual events, for example extracted from the human caption with
`prompt_templates/visual_event_extraction.txt`.

## Tests

```sh
gym env test --resources-server video_caption
# or, inside the server's own environment:
cd resources_servers/video_caption && uv sync && uv run pytest tests
```
