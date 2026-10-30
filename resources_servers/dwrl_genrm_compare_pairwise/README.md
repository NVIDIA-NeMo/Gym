# DWRL Pairwise GenRM Resources Server

A resources server that rewards the rollouts of a prompt by comparing them **in pairs** with a **Generative Reward Model (GenRM)** and fitting **Bradley-Terry (Elo) ratings** to the comparisons. It is used as the reward source for RLHF-style GRPO training (DWRL: the GenRM is a trained judge that is asked which of two responses is better). Its counterpart `dwrl_genrm_compare_pointwise` judges every rollout on its own instead.

## At a glance

- **The rollouts of a prompt are rewarded together.** `/verify` is called once per rollout, but the rollouts of one prompt are grouped into a *cohort*, their responses are compared in pairs, and each rollout's reward is its rating in a model fitted to all of the cohort's comparisons.
- **A reward depends on the whole cohort.** Changing one comparison changes the rating of every response, including ones not in that comparison. This is why `/verify` returns only when the whole cohort has been judged, and why this server (unlike the pointwise one) must run as a single process.
- **The reward is a rating in log-odds units**, not a probability and not a 1-5 score. Without a reference answer the ratings of a cohort sum to zero; with one, a reward is the rating relative to the reference answer.
- **The GenRM is the cost.** A comparison is two GenRM calls (a long evaluation and a 1-token call). A cohort of N rollouts with the `circular` strategy takes N comparisons, plus N more if there is a reference answer.

## How a pair is judged

For a comparison of responses `i` and `j` (and each of the `num_judges_per_comparison` passes):

### 1. The evaluation call

The server fills `dwrl_prompt` and sends it to the GenRM as a single user message through `/v1/chat/completions` (sampling parameters from `genrm_responses_create_params`, e.g. `temperature: 1.0`, `max_tokens: 16384`). The placeholders are:

| Placeholder | Filled with |
|---|---|
| `{context}` | The conversation that the rollouts answer, flattened to text as `System: ...`, `User: ...`, `Assistant: ...`, separated by blank lines. Thinking (`<think>...</think>`, and an unterminated `...</think>` prefix) is removed from assistant turns. |
| `{response_1}`, `{response_2}` | The **final answers** of the two responses (the text of their `output_text` parts). Reasoning items are not shown to the judge. An empty answer becomes the placeholder `None`. |
| `{principle}` | The `principle` of the request if it has one, else `default_principle` from the config. |

The GenRM answers with an analysis of both responses followed by a JSON object, e.g. `{"response_1_analysis": "...", "response_2_analysis": "...", "score_1": 4, "score_2": 3, "ranking": 2}`.

### 2. The score call

The evaluation is appended to the conversation as an assistant turn, followed by the user message `bt_prompt` (*"Based on your evaluation, which response is better? Answer with only a single character: 1 or 2"*). A second call with `max_tokens: 1`, `logprobs: true`, `top_logprobs: 20`, `temperature: 1.0`, `top_p: 1.0` and thinking turned off asks for a single token.

The result of the comparison is the probability `p` that this token is `score_token` (`"1"`, i.e. "response 1 is better"):

1. a top logprob whose token is exactly `score_token`, else
2. one whose token matches after `.lower().strip()` (e.g. `" 1"`), else
3. **0.0, with a warning in the log.** A token outside the top 20 has a probability of at most 1/20, so 0 is the right order of magnitude. This is also what happens for *every* comparison if `score_token` does not match the GenRM's tokens.

### 3. The scores in the JSON are not used

The JSON is parsed for `score_1`, `score_2` and `ranking` (thinking removed first), but **none of them influences the reward**. They only feed the metrics of `/compare`. If the JSON cannot be parsed, a warning is logged and `default_score` is used for the two scores. By default (`genrm_parse_retries: 0`) nothing is retried, since the scores do not influence the reward and a retry would only cost GenRM calls. If `genrm_parse_retries` is greater than 0, an evaluation that cannot be parsed is judged again (the same pair), from scratch (both GenRM calls, with a fresh evaluation), up to that many more times, waiting `genrm_parse_retry_sleep_s` seconds between attempts. The result of the comparison is then the probability from the first attempt that parses, or from the last attempt if none does. An unparseable evaluation is usually one that was cut off at `max_tokens`, so a retry gives it a second chance to complete; each retry costs two more GenRM calls.

### Position of the responses

The probability is that of "response 1 is better", so a GenRM that prefers whatever comes first would bias it. Which responses are shown first depends on the strategy:

- `circular`: every response is response 1 in one comparison and response 2 in another, which balances a position bias.
- `all_pairs`: the response that arrived earlier is always response 1.
- The reference answer is always response 2.

## Rewards

### The Bradley-Terry fit

Every comparison of `i` and `j` with result `p` adds `p` as a fractional win of `i` over `j`, and `1 - p` as a fractional win of `j` over `i`. The reward of a response is its rating `r` in the Bradley-Terry model

```
P(i beats j) = sigmoid(r_i - r_j)
```

fitted to all comparisons of the cohort by unregularized logistic regression (`get_elo_scores` in `utils.py`; scale 1, natural log). Passes of several judges add up, so they act as their mean probability.

- With one comparison of `p = 0.8` between two responses, their ratings differ by `ln(0.8 / 0.2) = 1.39` and are `+0.69` and `-0.69`.
- Without a reference answer the ratings of a cohort sum to zero.
- With a reference answer, it takes part in the fit as an extra response with the highest index (always response 2 in its comparisons), its rating is the zero point, and it is not rewarded. A reward above 0 means that the GenRM prefers the rollout to the reference answer.
- Ratings can be large if the GenRM is very sure: a probability of `1 - 1e-9` is about 20 log-odds. The server asserts that every rating stays below 100 in magnitude.

### Style penalty (optional)

If `group_style_penalty_coeff > 0`, rollouts with a higher *style density* than the others in the cohort get a lower reward. The density is the number of headers, list items, bold spans and emojis (code blocks excluded) per character of the answer. The rollouts' densities are turned into zero-centered weights (lowest density: highest weight) and `weight * group_style_penalty_coeff` is added to each rating. It is off by default and the reference answer is not part of it.

### Length bonuses are disabled

`reasoning_bonus`, `answer_bonus`, `top_percentile` and the two `group_*_length_penalty_coeff` settings are accepted, but the code that applies them is switched off in `aggregate_scores`.

## Cohorts

### Grouping

`verify()` groups rollouts by a key made of the task index (`_ng_task_index`, or else `prompt_id`, when there is one) and a hash of the prompt, the principle and the reference answer. Rollouts are numbered in arrival order. A cohort is full at `num_rollouts_per_prompt` members; the next rollout with the same key starts a new cohort (the same prompt twice in a batch, or in a later epoch). The reference answer is part of the key, so rollouts compared against different reference answers never share a cohort.

If `num_rollouts_per_prompt` is 1 or less, `verify()` returns `default_score` without judging anything.

### Who judges what

A comparison is judged as soon as both of its responses have arrived, by the request that arrived last of the two. A comparison against the reference answer is judged by the request of the rollout, as soon as it arrives. Each comparison is therefore judged exactly once, and the GenRM is busy while the cohort is still filling. For a cohort of 4 with the `circular` strategy:

| Arrival | Judges | Total comparisons started |
|---|---|---|
| rollout 0 | nothing (with a reference: 0 vs reference) | 0 |
| rollout 1 | 0 vs 1 | 1 |
| rollout 2 | 1 vs 2 | 2 |
| rollout 3 | 2 vs 3 and 3 vs 0 | 4 |

### Completion

The rewards are computed by whichever request finishes judging last, not by the one that arrived last. Judging takes a long and highly variable time, so these are generally different requests: the last arrival can finish before an earlier one is done. Until then no request returns. When the last comparison is in, the ratings are fitted and every waiting request is given its rollout's reward.

### Repeated requests

A request can be delivered twice (for example a client that retries after a dropped connection). A request that repeats one that is being handled (same cohort key, same `_ng_rollout_index` and the same response) takes no slot in the cohort and is not judged again: it waits for the reward of the request it repeats. This also covers a repeat that arrives after the cohort has filled up but before it has been finished. If the original request failed, the repeat fails as well. Cancelling a repeat does not cancel the original. A repeat that arrives after the cohort has been finished is not recognized.

### Failures

| What happens | Result |
|---|---|
| A GenRM call fails (connection error, HTTP error, malformed answer) | The comparison gets the neutral probability `default_ranking` (0.5), with an error in the log |
| `score_token` is not in the top 20 logprobs | The comparison gets 0.0, with a warning |
| The evaluation JSON cannot be parsed, and no retries are configured (the default) | No effect on the reward (a warning is logged) |
| The evaluation JSON cannot be parsed, and `genrm_parse_retries > 0` | The evaluation is repeated (the same pair) up to `genrm_parse_retries` times. The comparison uses the probability of the first attempt that parses, or of the last one if none does |
| A request fails or is cancelled (e.g. the client disconnects) | The comparisons that it was to judge are filled in with the neutral result so the rest of the cohort is not left waiting. That request fails, and the others get their rewards |
| The rewards of a cohort cannot be computed (a response with no comparison, or a fit that diverges) | Every request of the cohort fails, instead of waiting forever |

### Known limitations

- **A rollout that never reaches `verify()`** (for example because its generation failed) leaves the other members of its cohort waiting for it. There is no timeout.
- **A late repeat** (after the cohort was finished) starts a cohort of its own and waits for members that never come.
- **Single process.** The cohorts are kept in memory, so with `num_rollouts_per_prompt > 1` a config with `num_workers > 1` is rejected at startup.

## Endpoints

### POST `/verify`

Called by the agent for every rollout. The request is the usual verify request (`responses_create_params`, `response`) plus any fields of the dataset row, since the agent forwards the whole row. The ones that matter here:

- `principle` (optional): the evaluation plan for this prompt. Default: `default_principle`.
- `ref_answer` (optional): the reference answer. Every rollout is also compared with it.
- `_ng_task_index`, `_ng_rollout_index` (added by Gym) and `prompt_id` (optional): identify the cohort and the rollout.

The response is the usual verify response with `reward` set.

### POST `/compare`

A batch API that judges the given responses as a single group, without any grouping or waiting. It uses the configured strategy and has no reference answer.

```json
{
    "conversation_history": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "response_objs": [
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Paris is the capital."}]}]},
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "The capital of France is Paris."}]}]}
    ],
    "principle": "The response should be concise and accurate."
}
```

The response has the rewards and the result of every comparison. `ranking` is the win probability of response `i`, and the metrics summarize the parsed 1-5 scores:

```json
{
    "rewards": [0.42, -0.42],
    "comparison_results": [
        {"response_i": 0, "response_j": 1, "judge_idx": 0, "score_1": 4.0, "score_2": 3.0, "ranking": 0.7},
        {"response_i": 1, "response_j": 0, "judge_idx": 0, "score_1": 3.0, "score_2": 4.0, "ranking": 0.3}
    ],
    "metrics": {"mean_individual_score": 3.5, "std_individual_score": 0.5, "tiebreak_usage_rate": 0.0}
}
```

## Configuration

See `configs/dwrl_genrm_compare_pairwise.yaml`. `num_rollouts_per_prompt` has to be supplied (it is `???`), normally the number of generations per prompt of the training run.

```yaml
dwrl_pairwise_compare_resources_server:
  resources_servers:
    dwrl_genrm_compare_pairwise:
      entrypoint: app.py

      genrm_model_server:        # The model server that runs the GenRM (the dwrl_model block of the same file)
        type: responses_api_models
        name: dwrl_model

      num_rollouts_per_prompt: 8   # = grpo.num_generations_per_prompt
      comparison_strategy: circular
      num_judges_per_comparison: 1

      score_token: "1"
      bt_prompt: "Based on your evaluation, which response is better? Answer with only a single character: 1 or 2"
      dwrl_prompt: |
        ... {context} ... {response_1} ... {response_2} ... {principle} ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genrm_model_server` | ModelServerRef | *required* | The model server that runs the GenRM |
| `genrm_responses_create_params` | object | *required* | Parameters of the evaluation call. The score call copies them and overrides `max_tokens`, `temperature`, `top_p`, `logprobs` and `top_logprobs` |
| `score_token` | string | *required* | The token whose probability is the probability that response 1 is better |
| `bt_prompt` | string | *required* | The question asked after the evaluation to read that probability |
| `dwrl_prompt` | string | *required* | Template of the evaluation prompt with `{context}`, `{response_1}`, `{response_2}`, `{principle}`. Literal braces (e.g. in the JSON example) are written `{{` and `}}` |
| `num_rollouts_per_prompt` | int | `1` | Size of a cohort. At `1` or less `verify()` returns `default_score` without judging |
| `comparison_strategy` | string | `"circular"` | `"circular"` (N comparisons) or `"all_pairs"` (N(N-1)/2) |
| `num_judges_per_comparison` | int | `1` | Number of independent judge passes per pair (at least 1). Each is a full evaluation sampled separately; their probabilities are averaged in the fit |
| `default_principle` | string | *(see config)* | Evaluation plan when the request has none |
| `group_style_penalty_coeff` | float | `0.0` | Style density penalty. `0` disables it |
| `default_ranking` | float | `0.5` | Win probability of a comparison that could not be judged. Must be between 0 and 1: it is added to the win counts that the ratings are fitted to |
| `default_score` | float | `3.0` | The 1-5 scores of an unparseable evaluation, and the reward when `num_rollouts_per_prompt <= 1` |
| `genrm_parse_retries` | int | `0` | How many times an evaluation that cannot be parsed is repeated (the same pair). `0` disables retries. Each retry is two more GenRM calls |
| `genrm_parse_retry_sleep_s` | float | `0.2` | Seconds to wait between a failed attempt and its retry |

Accepted but with no effect on `/verify`: `use_principle`, `aggregator_method` (only `"simple_tiebreaker"` is accepted, whatever it says), `reasoning_bonus`, `answer_bonus`, `top_percentile`, `group_reasoning_length_penalty_coeff`, `group_answer_length_penalty_coeff`, and `debug_logging` (only `/compare` reads it).

### The GenRM server

The packaged yaml also defines `dwrl_model`, a `local_vllm_model` that serves the GenRM in the Gym process group. Set `model` (the GenRM checkpoint), `hf_home`, and `vllm_serve_kwargs` (`tensor_parallel_size`, `data_parallel_size`, `max_model_len`, ...) to fit the nodes it runs on. The prompts (conversation plus two responses plus plan) are long, so the context length has to be comfortably above the longest pair of rollouts. The yaml also defines the `genrm_simple_agent` agents that call this server, with the example dataset.

## Logs and diagnostics

| Output | Meaning |
|---|---|
| `### BT_SCORE:  0.0123` | The probability read from the score call, one line per comparison |
| `Score_1: 4.0   Score_2: 3.0` | The parsed 1-5 scores, one line per comparison |
| `#### ELO_SCORES:` / `#### ELO_SUM:` | The fitted ratings of a cohort, and their sum (about 0 without a reference answer) |
| `Score token '...' not found in the top N logprobs` | The score token was not among the top logprobs. Many of these means a wrong `score_token` |
| `No parseable JSON found in GenRM output` | An evaluation without a parsable JSON, usually cut off at `max_tokens`. Logged by the parser when `genrm_parse_retries` is 0. Harmless for the reward |
| `[GenRM] Parse failed for pair (i, j) (attempt N of M); retrying` | An evaluation could not be parsed and is being repeated (only with `genrm_parse_retries > 0`) |
| `[GenRM] Parse failed for pair (i, j) after M attempts; falling back to defaults` | Every attempt failed to parse. The 1-5 scores take `default_score` and the probability of the last attempt is used |
| `[GenRM] Error in comparison for pair (i, j)` | A GenRM call failed; that comparison got `default_ranking` |
| `[GenRM] Repeated verify request for rollout_index=...` | A repeated request was recognized and is waiting for the original |
| `[GenRM] Failed to aggregate cohort scores` | The ratings of a cohort could not be computed; every request of that cohort failed. The traceback follows |

## Differences from the other GenRM servers

| | `genrm_compare` | `dwrl_genrm_compare_pairwise` (this) | `dwrl_genrm_compare_pointwise` |
|---|---|---|---|
| Judge input | Two responses | Two responses | One response |
| GenRM output used | 1-5 scores and 1-6 ranking from JSON | Probability that response 1 is better (second call) | Probability of "yes" (second call) |
| Reward | Mean of scores with a ranking tiebreak, relative to the group | Bradley-Terry rating fitted over the group's comparisons | The probability itself |
| Depends on other rollouts | Yes | Yes | No |
| Groups rollouts of a prompt | Yes | Yes (cohorts) | No |
| Reference answer | No | Optional, as an anchor of the ratings | No |
| Worker processes | 1 | 1 | Any |

## File structure

```
dwrl_genrm_compare_pairwise/
├── __init__.py
├── app.py                         # Resources server: /verify with cohorts, /compare, the GenRM calls
├── cohort.py                      # Cohort bookkeeping: grouping, comparison slots, repeated requests
├── utils.py                       # Prompt keys, output parsing, Bradley-Terry fit, style penalty
├── configs/
│   └── dwrl_genrm_compare_pairwise.yaml   # Default configuration (including the GenRM model block)
├── data/
│   ├── example.jsonl              # 5 example input rows
│   ├── example_metrics.json       # Statistics of example.jsonl (generated by gym dataset collate)
│   └── example_rollouts.jsonl     # 5 example rollouts (generated from a live run)
├── tests/
│   ├── __init__.py
│   ├── test_app.py                # Server tests: GenRM calls, verify() schedules, failures, repeats
│   ├── test_cohort.py             # Cohort bookkeeping tests
│   └── test_utils.py              # Parsing, prompt keys, Bradley-Terry fit, style penalty tests
└── README.md                      # This file
```

## Example data

`data/` holds the three example files that Gym expects from every resources server (see `CONTRIBUTING.md`, and `_validate_data_single` in `nemo_gym/cli/env.py`, which `gym env test` runs):

| File | What it is | How it is made |
|---|---|---|
| `example.jsonl` | Exactly 5 input rows, the first 5 rows of a training dataset | Hand-maintained. This is the input of everything below |
| `example_metrics.json` | Statistics of `example.jsonl` (number of examples, tools, turns and words, plus a count of every other column) | Generated by `gym dataset collate`. Do not edit it by hand: Gym writes a `*_conflict.json` file if the metrics it computes differ from the checked-in ones |
| `example_rollouts.jsonl` | Exactly 5 rollouts: the example rows run through the agent, so each row also has the policy model's `response` and the server's `reward` | Generated from a live policy model and GenRM (see "Regenerating the files") |

`example_metrics.json` is found through the `datasets` entry of the agents in `configs/dwrl_genrm_compare_pairwise.yaml`, whose `jsonl_fpath` has to point at this folder's own `example.jsonl`.

### The rows

A row is a Gym run request: the conversation to answer and the agent that answers it. Everything else on the row is forwarded to `/verify`.

```json
{
    "responses_create_params": {"input": [{"role": "user", "content": "What is \"SKILL\"?"}], "tools": [], "parallel_tool_calls": false},
    "agent_ref": {"type": "responses_api_agents", "name": "genrm_simple_agent"},
    "dataset": "hs3"
}
```

- `agent_ref.name` is one of the agents that the config defines: `genrm_simple_agent` (policy model with reasoning on) or `genrm_simple_agent_reasoning_off`.
- `principle` (optional): the evaluation plan of this row. Without it `default_principle` is used, which is what the training data does.
- `ref_answer` (optional): a reference answer. Every rollout of the row is also compared with it, and rewards are relative to it. Without it the rollouts are only compared with each other.

### Regenerating the files

Both generated files need Gym's servers, so they are regenerated where the policy model and the GenRM can run, not by hand. Things that are easy to miss: the agents in the config use the model servers `policy_model` and `policy_model_reasoning_off`, and only `policy_model` comes from a Gym config (`responses_api_models/vllm_model/configs`), so define the second one as the training configs do; `dwrl_model` has `model: ???`, which has to be given; and `num_rollouts_per_prompt` is `???` too, which has to be given even to compute the metrics.

```bash
# example_metrics.json (writes data/example_metrics.json next to example.jsonl)
gym dataset collate "+config_paths=[<policy model config>,resources_servers/dwrl_genrm_compare_pairwise/configs/dwrl_genrm_compare_pairwise.yaml]" \
    +dwrl_pairwise_compare_resources_server.resources_servers.dwrl_genrm_compare_pairwise.num_rollouts_per_prompt=5 \
    +output_dirpath=data/dwrl_genrm_compare_pairwise \
    +mode=example_validation
```

**Rollouts need a cohort.** Rewards are fitted over all rollouts of a prompt, and `verify()` waits until `num_rollouts_per_prompt` of them have arrived. The 5 example rows are 5 different prompts, so running them once each would leave every cohort short of rollouts and the run would hang. To get exactly 5 rollouts that form one cohort, run the first row 5 times:

```bash
gym env start "+config_paths=[<policy model config>,resources_servers/dwrl_genrm_compare_pairwise/configs/dwrl_genrm_compare_pairwise.yaml]" \
    +dwrl_pairwise_compare_resources_server.resources_servers.dwrl_genrm_compare_pairwise.num_rollouts_per_prompt=5
gym eval run --no-serve +agent_name=genrm_simple_agent \
    +input_jsonl_fpath=resources_servers/dwrl_genrm_compare_pairwise/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/dwrl_genrm_compare_pairwise/data/example_rollouts.jsonl \
    +limit=1 +num_repeats=5
```

The 5 rows of the result are then 5 rollouts of the first example row, and their `reward` values are the ratings of that one cohort.

The metrics in the repository were produced by Gym's own metrics code (`nemo_gym.train_data_utils`) on the checked-in `example.jsonl`. The `gym` commands above are the generic Gym flow and have not been run for this server.

**How the checked-in `example_rollouts.jsonl` was made:** not with the commands above, but by running the first example row 5 times through the DWRL training setup in NeMo-RL (`env.nemo_gym.is_trajectory_collection=true`, which writes each rollout's `/run` result to `trajectory_collection.jsonl`), with `num_rollouts_per_prompt=5` and the GRPO reward penalties switched off so that `reward` is the server's reward. These are 5 real rollouts of one prompt from a policy model with reasoning on, judged by the GenRM as one cohort. The example row has no `ref_answer`, so the ratings are relative to each other and sum to 0. Two edits were made to the output: the `prompt_str` and `generation_str` fields that vLLM adds to each message were removed (they duplicate the content), and `response.model` was replaced by `policy_model` (it held a checkpoint path).

## Development

```bash
pytest resources_servers/dwrl_genrm_compare_pairwise/tests -v
```

## Related components

- **Pointwise counterpart**: `resources_servers/dwrl_genrm_compare_pointwise/` - judges every rollout on its own, so rewards are returned as soon as the rollout is judged
- **Original server**: `resources_servers/genrm_compare/` - the 1-6 ranking based version these servers were derived from
- **Model server**: `responses_api_models/local_vllm_model/` - serves the GenRM

## License

Apache 2.0 - Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
