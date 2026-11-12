# Agentic-MME benchmark

Multimodal agentic reasoning: the policy answers image questions using atomic image tools and, with a
retrieval overlay, web search. This benchmark reports **task-only accuracy** (Track B), not the paper's
process (S/V) scores. Components: resources server [`agentic_mme`](../../resources_servers/agentic_mme/README.md)
(scoring) and agent [`agentic_mme_agent`](../../responses_api_agents/agentic_mme_agent/README.md) (harness, tools,
retrieval).

## Provenance

| | |
|---|---|
| Paper | [arXiv:2604.03016](https://arxiv.org/abs/2604.03016) |
| Data | [Crystal1047/Agentic-MME](https://huggingface.co/datasets/Crystal1047/Agentic-MME) at revision `b9ea9d3f68ff896d83fec666b6143da594c054a8` |
| Reference harness | [ChoS3nE11ven/Agentic-MME](https://github.com/ChoS3nE11ven/Agentic-MME/tree/5e5dfbdc7b40b53dea17d5ce754c14034acb185b) (interface reference only; nothing vendored) |
| Split | The release has one split, named `train`; it is an evaluation set. Do not train on it. |
| Tasks | 418, all tasks in the release; no filtering or synthetic data |
| Licenses | Dataset: Apache-2.0 (dataset card). Gym code: Apache-2.0. The reference harness declares no code license, so the tools are an independent implementation. |

## Task contract

`prepare.py` converts each task into one row:

- **Model-visible input:** a fixed system prompt (tool conventions and the `<answer>...</answer>` format) and a
  user message with the task images, labelled `Image 0`, `Image 1`, ..., followed by the question. Images are embedded
  as data URLs and downscaled to at most 2048×2048 pixels, as in the reference harness.
- **Hidden from the model:** task ID, level, domain, the gold answer as released (value only), the numeric
  tolerance from the task's final answer check (`answer_tolerance`; 29 numeric tasks have one), and process
  checkpoints, all in `verifier_metadata`.
- **Tools:** 14 image tools (crop, rotate, flip, resize, enhance, grayscale, autocontrast, blur, sharpen, denoise,
  edge_detect, invert, equalize, threshold). Each successful operation appends an image the model sees in its next
  turn, capped at 2048×2048 pixels like the inputs and sent as the smaller of PNG and JPEG (quality 95), so long
  episodes stay under hosted endpoints' request-size limits. Retrieval (google_search, google_lens_search,
  fetch_webpage) is off in `config.yaml` and enabled by the `retrieval_*.yaml` overlays, which need provider keys.
  Search results reach the model as the reference harness's compact text (knowledge-graph title, description and
  website; the top 5 results; "people also ask"). Google's answer box is never shown, since it often states the
  answer outright; the provider's full answer stays in the tool trace.
- **Prompting:** prompts are prepared (`prompt_source: prepared`); the harness appends tool results and new images
  to the history and never rewrites earlier turns.

## Scoring and run conditions

- **Reward:** 1 if the final answer is correct, else 0. Higher is better; the aggregate metric is `mean/reward`.
- **Judge:** `config.yaml` binds an LLM correctness judge (default `gpt-4o-mini` through an OpenAI-compatible
  endpoint; override with `agentic_mme_judge_base_url` and `agentic_mme_judge_model_name`). The judge sees the
  question text, the gold answer, the numeric tolerance when the task has one, and the model's final text, not
  the images. The released Track B string match is reported alongside as `string_match_reward`: like
  `check_final_answer_only`, it reads the match type from the gold answer, which the release leaves unset, so every
  task uses `contains`.
- **Answers:** only the terminal assistant message counts. A response cut off at the output-token limit, or a model
  call refused because the context or request size is exceeded, measures the policy: it scores 0 and is recorded in
  `failure_reason` / `policy_error`.
- **Infrastructure failures are not scored.** A failed policy call for any other reason (auth, rate limit, server
  error, timeout, lost connection, unreadable reply) raises out of `/run`, and a failed judge call is masked by
  `judge_failsafe`. Both go to the failures sidecar (`agent_run_error` / `judge_failed`) and are retried or regraded
  on resume. To keep a fixed denominator, count what still fails as 0 in the metrics with
  `count_failure_classes_as_zero`.
- **Budgets:** `config.yaml` sets no round or tool-call limit (`max_rounds: null`, `max_tool_calls: null`): the
  episode ends when the model answers without a tool call or when a model call is refused. Set both to restore the
  component default of 15.
- **Images per request:** every image stays in every later model request. `max_images_per_request` is off by
  default; with a serving engine that limits images per prompt (vLLM `limit_mm_per_prompt.image`), set it to that
  limit, or serve with a limit of at least the input images plus `max_tool_calls`. An image tool call past the limit
  returns a tool error and the model answers with the images it has. A request the engine refuses for its image
  limit is a configuration error and goes to the failures sidecar, not a scored 0.
- **Sampling:** taken from the run (`responses_create_params`); the agent sets no temperature. One repeat by default.

## Prepare and run

```bash
gym eval prepare --benchmark agentic_mme   # or AGENTIC_MME_DATASET_ROOT=<local snapshot>
gym env validate agentic_mme
gym env test agentic_mme
```

Run against an OpenAI-compatible policy endpoint (the judge reads `OPENAI_API_KEY`):

```bash
gym eval run --model-type vllm_model \
    "+config_paths=[benchmarks/agentic_mme/config.yaml]" \
    +split=benchmark +output_jsonl_fpath=results/agentic_mme.jsonl \
    +policy_base_url=<url> "+policy_api_key=\${oc.env:POLICY_API_KEY}" +policy_model_name=<model> \
    +responses_create_params.max_output_tokens=32768
```

To add web retrieval, append one overlay to `config_paths`, for example
`benchmarks/agentic_mme/retrieval_serpapi.yaml`; see the agent README for providers, the shared cache, and the
benchmark-contamination guard.

## Limitations

- Task-only accuracy: process (S/V) scoring is not implemented, so results are not comparable to the paper's Track A.
- Judge-based reward: scores depend on the judge model; `string_match_reward` gives a judge-free reference.
- Retrieval results change over time; use `retrieval.mode: replay` or the shared cache for repeatable runs.
- `experimental: true`: no published result has been reproduced with this integration yet.
