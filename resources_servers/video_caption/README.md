# Video caption reward

This single-turn environment scores the final model caption using visual-event recall, precision, or F1,
with the existing length bonus and optional degeneration gate. It uses Gym's `simple_agent`, a judge
`ModelServerRef`, and `call_judge` over the shared client. `local_vllm_model` owns judge processes and
Ray placement groups; this feature does not change core Gym or RL orchestration.

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
The degeneration flag is independent and defaults to false. Existing recipes retain recall behavior.

Rows use the Gym envelope `agent_ref`, `responses_create_params.input`, and `verifier_metadata`:

```json
{"atomic_events": {"visual_events": ["A person walks."]}, "reference_caption": "A person walks across a room."}
```

This example is the verifier metadata, not the policy prompt. `reference_caption` contains the original
human visual caption, not a reconstruction from its events. Precision/F1 requires a nonempty reference
caption and fails before making judge requests when it is missing. Recall-only rows remain compatible
without it. Audio event fields are ignored. Data preparation must populate `reference_caption` from the
human-written visual caption (not a reconstruction from its events) for precision/F1 training.

## Visual score

All model-side judging and token counting use `body.response.output_text`: the final assistant message
text after Gym's response conversion. Separate reasoning items are excluded. A reasoning-only response
is an empty final caption and receives zero without any judge request.

- **Recall R:** check whether each reference visual event is covered by the model caption, using the
  unchanged `prompt_templates/visual_event_judge.txt`. R is the fraction of reference events covered.
- **Precision P:** first extract atomic visual events from the final model caption in a separate request.
  Then reuse the same coverage prompt, with the human reference caption as the candidate text and the
  predicted events as the list to check. P is the fraction of predicted events covered by the reference
  caption. Reference atomic events are not used as the precision target.
- **F1:** compute both directions in separate requests, then use `2 * R * P / (R + P)`, or zero when both
  scores are zero. It is the F1 of the two event-coverage fractions, not a one-to-one event matching.

Extraction uses `prompt_templates/visual_event_extraction.txt` (configurable through
`event_extraction_prompt_fpath`). It is the visual-only form of the atomic-event extraction prompt used to
build the reference `atomic_events` offline. It retains the atomicity, independent judgeability, deduplication,
chronological ordering, and timestamp-exclusion instructions; audio and audio-visual extraction branches
are removed. It receives only `{"visual_caption": "<final predicted caption>"}`, never the reference.
Its strict output schema is `{"visual_events": ["event", "..."]}`. Missing/extra fields, blank events,
non-string entries, or invalid JSON cause retries. An empty event list is valid and gives precision/F1
zero without a coverage request; the existing length bonus still applies to nonempty captions.

Recall makes one successful request, precision two (extraction then coverage), and F1 three (recall,
extraction, precision coverage). Optional degeneration adds one separate request after these stages.
None of the judgments is combined into another request. Stages reuse the existing Qwen judge and execute
sequentially within each caption; captions share the admission semaphore. A failed stage stops subsequent
stages. A retry never repeats an earlier successful stage.

## Overall reward and degeneration

The caption environment returns `D * (V + L) / 2`:

- V is the selected recall, precision, or F1 score.
- L is one for 100–4,000 final-caption tokens inclusive, otherwise zero, using the policy tokenizer.
- D is zero when enabled degeneration judging returns `fail`, otherwise one.

Empty captions and exhausted judge failures receive zero total reward, including no length bonus.
Framework length penalties remain disabled. Existing malformed-thinking penalties, fixed advantages,
and ordinary RL advantage computation are unchanged. No new validity mask or fixed negative advantage
is added for LLM-judged degeneration.

`penalize_degeneration` uses `prompt_templates/degeneration_judge.txt`, configurable through
`degeneration_judge_prompt_fpath`. The user-refined prompt and broad criteria remain unchanged: no
degeneration or excessive repetition, a coherent and relevant video description, and no exposed reasoning
or drafting. It receives only the final caption, not reference text, events, or prior judgments. It returns
only `{"verdict":"pass"}` or `{"verdict":"fail"}`; pass requires all criteria. No defect categories or
quoted evidence are required. It excludes factual precision/recall, and length alone is not a failure.
A fail gates the entire reward, including the length bonus.

The response exposes `visual_reward_type` and `visual_reward`, plus the individual
`visual_recall_reward`, `visual_precision_reward`, and `visual_f1_reward`. An uncomputed score is null,
not a measured zero. `visual_hits` remains recall-only; `precision_visual_hits` and
`predicted_visual_events` retain the precision evidence. `judge_attempts`/`judge_response_text` remain
recall-only, with separate `event_extraction_*`, `precision_judge_*`, and `degeneration_judge_*`
attempt/response fields. `length_reward`, `reward_before_degeneration`, `degeneration_reward`,
`degeneration_checked`, and `degeneration_detected` retain diagnostic values before the gate.

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
the shared client's own retries do not nest inside it. Transport failures (the `judge_failed` event
`call_judge` raises), malformed responses and incomplete output all retry within that budget with exponential
backoff. Exhaustion raises `JudgeError`; Gym's verify wrapper returns reward zero and
`_ng_failure_class=judge_failed`, and no length bonus is added. Cancellation propagates without a retry.

A row the verifier cannot score is never a 5xx (which would make the environment server replay the episode
and spend another policy rollout): blank or missing `atomic_events.visual_events`, or a precision/F1 row
without a nonempty `reference_caption`, returns `reward: 0.0` with `mask_sample: true`,
`failure_kind: video_caption:unscorable_row` and a `failure_reason`, before any judge call. `task_data.py`
rejects the same rows at data-validation time (`gym dataset validate`, the repo's row-drift test).

## Data

Rows are standard Gym training rows for `simple_agent` (`agent_ref` -> `video_caption_simple_agent`): the
policy prompt carries the video as ordered image frames (or a native video part, depending on the RL
framework's media path) followed by the caption instruction, and `verifier_metadata` carries
`atomic_events.visual_events` (reference atomic visual events) plus `reference_caption` for precision/F1.
`caption_tokenizer_path` must point at the policy's `tokenizer.json` so the length bonus counts policy tokens.

`data/example.jsonl` holds one synthetic smoke row (three generated frames of a rectangle sliding right, three
reference events and a reference caption). It exercises the row format and the verifier end to end; it is not a
benchmark or a baseline. Training data comes from your own video caption corpus with per-video atomic visual
events, for example extracted from the human caption with `prompt_templates/visual_event_extraction.txt`.

## Tests

```sh
gym env test --resources-server video_caption
# or, inside the server's own environment:
cd resources_servers/video_caption && uv sync && uv run pytest tests
```
