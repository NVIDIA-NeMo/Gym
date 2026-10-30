# DWRL Pointwise GenRM Resources Server

A resources server that rewards a rollout by asking a **Generative Reward Model (GenRM)** to judge that one response on its own, and reading the probability that the GenRM thinks it is good. It is used as the reward source for RLHF-style GRPO training (DWRL: the GenRM is a trained pointwise judge). Its counterpart `dwrl_genrm_compare_pairwise` compares the rollouts of a prompt with each other instead.

## At a glance

- **One rollout in, one reward out.** `/verify` is called once per rollout. It judges that rollout with the GenRM and returns its reward as soon as the judgment is done.
- **The reward is a probability** in [0, 1]: the probability that the GenRM gives to `score_token` (`"yes"`) when it is asked "is the response good?" after having written an evaluation of it. With several judge passes (`num_judges_per_comparison`) it is the mean over the passes.
- **Nothing is shared between rollouts.** A rollout is never compared with, grouped with, or waiting for another rollout of the same prompt. A rollout that never arrives, arrives twice, or is cancelled cannot affect any other rollout, and the server can run any number of worker processes.
- **The GenRM is the cost.** Every judge pass is two GenRM calls: a long evaluation (up to `max_tokens`) and a 1-token call. The server itself is asynchronous and I/O bound; throughput is set by the GenRM's vLLM.

## How a rollout is judged

For every `/verify` request, and for each of the `num_judges_per_comparison` passes (run concurrently):

### 1. The evaluation call

The server fills `dwrl_prompt` and sends it to the GenRM as a single user message through `/v1/chat/completions` (sampling parameters from `genrm_responses_create_params`, e.g. `temperature: 1.0`, `max_tokens: 16384`). The placeholders are:

| Placeholder | Filled with |
|---|---|
| `{context}` | The conversation that the rollout answers (`responses_create_params.input`), flattened to text as `System: ...`, `User: ...`, `Assistant: ...`, separated by blank lines. Thinking (`<think>...</think>`, and an unterminated `...</think>` prefix) is removed from assistant turns. |
| `{response}` | The rollout's **final answer** only: the text of its `output_text` parts. Reasoning items are not shown to the judge. An empty answer becomes the placeholder `None`. |
| `{principle}` | The `principle` of the request if it has one, else `default_principle` from the config. |

The GenRM answers with an analysis followed by a JSON object, e.g. `{"response_analysis": "...", "score_1": 4}`.

### 2. The score call

The evaluation is appended to the conversation as an assistant turn, followed by the user message `bt_prompt` (e.g. *"Based on your evaluation, is the original response good? Answer with only a single word: yes or no"*). A second call with `max_tokens: 1`, `logprobs: true`, `top_logprobs: 20`, `temperature: 1.0`, `top_p: 1.0` and thinking turned off (`enable_thinking: false`) asks for a single token.

The reward of the pass is the probability of that token being `score_token`:

1. a top logprob whose token is exactly `score_token`, else
2. one whose token matches after `.lower().strip()` (e.g. `" Yes"`), else
3. **0.0, with a warning in the log.** A token outside the top 20 has a probability of at most 1/20, so 0 is the right order of magnitude. This is also what happens for *every* rollout if `score_token` does not match the GenRM's tokens, so watch for the warning (`Score token 'yes' not found in the top 20 logprobs`).

### 3. The 1-5 score in the JSON is not used

The evaluation's JSON is parsed for `score_1` (thinking removed first), but **it does not influence the reward**. If it cannot be parsed, a warning (`No parseable JSON found in GenRM output`) is logged and `default_score` is used for it. By default (`genrm_parse_retries: 0`) nothing is retried, since the score does not influence the reward and a retry would only cost GenRM calls. If `genrm_parse_retries` is greater than 0, an evaluation that cannot be parsed is judged again, from scratch (both GenRM calls, with a fresh evaluation), up to that many more times, waiting `genrm_parse_retry_sleep_s` seconds between attempts. The reward is then the probability from the first attempt that parses, or from the last attempt if none does. An unparseable evaluation is usually one that was cut off at `max_tokens`, so a retry gives it a second chance to complete; each retry costs two more GenRM calls.

## Reward

```
reward = mean over judge passes of P(score_token | evaluation, bt_prompt)
```

- The scale is a probability, not a 1-5 score. Typical values are low and skewed: a GenRM that is rarely sure that a response is "good" produces many rewards close to 0 and few close to 1. For orientation, the mean reward of one training run was around 0.15-0.19 per step.
- Because GRPO normalizes rewards within a group of rollouts, only the differences between rollouts of a prompt matter to the policy update.
- The mean is a plain running sum divided by the count (`compute_reward` in `utils.py`), so with one judge the reward is exactly the probability from the score call.

### Failures

| What happens | Result for that judge pass |
|---|---|
| The GenRM answers normally | The probability of `score_token` |
| `score_token` is not in the top 20 logprobs | 0.0, with a warning |
| The evaluation JSON cannot be parsed, and no retries are configured (the default) | No effect on the reward (a warning is logged) |
| The evaluation JSON cannot be parsed, and `genrm_parse_retries > 0` | The evaluation is repeated up to `genrm_parse_retries` times. The reward uses the probability of the first attempt that parses, or of the last one if none does |
| An error while talking to the GenRM (connection error, HTTP error, malformed or incomplete answer) | `default_ranking` (0.0 unless configured), with an error in the log |

A failed pass is scored 0.0, the same as a token that is not in the top 20, so a failure is never worth more than a real judgment. The request does not fail: with several judge passes, a pass that failed counts as 0.0 in the mean, and a rollout whose passes all failed gets a reward of 0.0. Nothing but the log distinguishes that from a genuine zero, so look for `[GenRM] Error in comparison` after a GenRM outage.

Do not raise `default_ranking` towards 0.5 or higher: rewards are probabilities that are mostly well below that, so a failure would be worth more than a typical real judgment and a GenRM outage would inflate the rewards. Note that a training config can override the default, so check the `default_ranking` that the run actually uses.

## Endpoint

### POST `/verify`

Called by the agent for every rollout. The request is the usual verify request (`responses_create_params`, `response`) plus any fields of the dataset row, since the agent forwards the whole row. The ones that matter here:

- `principle` (optional): the evaluation plan for this prompt. Default: `default_principle`.
- Anything else (e.g. `ref_answer`, `_ng_task_index`, `_ng_rollout_index`) is accepted and ignored.

The response is the usual verify response with `reward` set.

Properties that follow from every rollout being judged on its own:

- A rollout's reward does not depend on what the other rollouts of its prompt were, or whether they exist.
- A request that fails or is cancelled (e.g. the client disconnects) fails alone.
- A request delivered twice is simply judged twice.
- There is no in-memory state, so `num_workers` can be greater than 1.

The server has no other endpoints. The `/compare` endpoint of the pairwise server that this one was derived from was removed: it could not work for one response.

## Configuration

See `configs/dwrl_genrm_compare_pointwise.yaml`.

```yaml
dwrl_pointwise_compare_resources_server:
  resources_servers:
    dwrl_genrm_compare_pointwise:
      entrypoint: app.py

      genrm_model_server:        # The model server that runs the GenRM (the dwrl_model block of the same file)
        type: responses_api_models
        name: dwrl_model

      genrm_responses_create_params:   # Sampling parameters of the evaluation call
        messages: []
        max_tokens: 16384
        temperature: 1.0
        top_p: 0.95

      num_judges_per_comparison: 1
      score_token: "yes"
      bt_prompt: "Based on your evaluation, is the original response good? Answer with only a single word: yes or no"
      dwrl_prompt: |
        ... {context} ... {response} ... {principle} ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genrm_model_server` | ModelServerRef | *required* | The model server that runs the GenRM |
| `genrm_responses_create_params` | object | *required* | Parameters of the evaluation call. The score call copies them and overrides `max_tokens`, `temperature`, `top_p`, `logprobs` and `top_logprobs` |
| `score_token` | string | *required* | The token whose probability is the reward (`"yes"`) |
| `bt_prompt` | string | *required* | The question asked after the evaluation to read that probability |
| `dwrl_prompt` | string | *required* | Template of the evaluation prompt with `{context}`, `{response}`, `{principle}`. Literal braces (e.g. in the JSON example) are written `{{` and `}}` |
| `num_judges_per_comparison` | int | `1` | Number of independent judge passes (at least 1). Each is a full evaluation sampled separately; the reward is their mean |
| `default_principle` | string | *(see config)* | Evaluation plan when the request has none |
| `default_ranking` | float | `0.0` | Reward of a judge pass whose GenRM call failed (see Failures) |
| `default_score` | float | `3.0` | The 1-5 score of an evaluation that cannot be parsed. Does not influence the reward |
| `genrm_parse_retries` | int | `0` | How many times an evaluation that cannot be parsed is repeated. `0` disables retries. Each retry is two more GenRM calls |
| `genrm_parse_retry_sleep_s` | float | `0.2` | Seconds to wait between a failed attempt and its retry |

Accepted but with no effect: `use_principle`, `aggregator_method`, `reasoning_bonus`, `answer_bonus`, `top_percentile`, `group_reasoning_length_penalty_coeff`, `group_answer_length_penalty_coeff`, `debug_logging`. They are left over from the pairwise server, so that configs written for it still load. A leftover `num_rollouts_per_prompt` key (rollouts are not grouped any more) is ignored.

### The GenRM server

The packaged yaml also defines `dwrl_model`, a `local_vllm_model` that serves the GenRM in the Gym process group. Set `model` (the GenRM checkpoint), `hf_home`, and `vllm_serve_kwargs` (`tensor_parallel_size`, `data_parallel_size`, `max_model_len`, ...) to fit the nodes it runs on. The long prompts (conversation plus response plus plan) need a context length comfortably above the longest rollout. It also defines the `genrm_simple_agent` agents that call this server, with the example dataset.

## Logs and diagnostics

| Output | Meaning |
|---|---|
| `### BT_SCORE:  0.0123` | The probability read from the score call, one line per judge pass |
| `Score_1: 4.0` | The parsed 1-5 score, one line per judge pass |
| `Score token '...' not found in the top N logprobs` | The score token was not among the top logprobs (see above). Many of these means a wrong `score_token` |
| `No parseable JSON found in GenRM output` | An evaluation without a parsable JSON, usually cut off at `max_tokens`. Logged by the parser when `genrm_parse_retries` is 0. Harmless for the reward |
| `[GenRM] Parse failed (attempt N of M); retrying` | An evaluation could not be parsed and is being repeated (only with `genrm_parse_retries > 0`) |
| `[GenRM] Parse failed after M attempts; falling back to defaults` | Every attempt failed to parse. The 1-5 score takes `default_score` and the probability of the last attempt is used |
| `[GenRM] Error in comparison` | A GenRM call failed; that pass got `default_ranking` |

## Differences from the other GenRM servers

| | `genrm_compare` | `dwrl_genrm_compare_pairwise` | `dwrl_genrm_compare_pointwise` (this) |
|---|---|---|---|
| Judge input | Two responses | Two responses | One response |
| GenRM output used | 1-5 scores and 1-6 ranking from JSON | Probability that response 1 is better (second call) | Probability of "yes" (second call) |
| Reward | Mean of scores with a ranking tiebreak, relative to the group | Bradley-Terry rating fitted over the group's comparisons | The probability itself |
| Depends on other rollouts | Yes | Yes | No |
| Groups rollouts of a prompt | Yes | Yes (cohorts) | No |
| Worker processes | 1 | 1 | Any |

## File structure

```
dwrl_genrm_compare_pointwise/
├── __init__.py
├── app.py                         # Resources server: /verify and the two GenRM calls
├── utils.py                       # Output parsing, response text extraction, compute_reward
├── configs/
│   └── dwrl_genrm_compare_pointwise.yaml   # Default configuration (including the GenRM model block)
├── data/
│   ├── example.jsonl              # 5 example input rows
│   ├── example_metrics.json       # Statistics of example.jsonl (generated by gym dataset collate)
│   └── example_rollouts.jsonl     # 5 example rollouts (generated from a live run)
├── tests/
│   ├── __init__.py
│   ├── test_app.py                # Server tests: GenRM calls, token matching, verify() independence, failures
│   └── test_utils.py              # Parsing, text extraction and reward computation tests
└── README.md                      # This file
```

`utils.py` also still holds the helpers of the pairwise server that this one was derived from (comparison pairs, `aggregate_scores`, length bonuses). Nothing in this server uses them any more.

## Example data

`data/` holds the three example files that Gym expects from every resources server (see `CONTRIBUTING.md`, and `_validate_data_single` in `nemo_gym/cli/env.py`, which `gym env test` runs):

| File | What it is | How it is made |
|---|---|---|
| `example.jsonl` | Exactly 5 input rows, the first 5 rows of a training dataset | Hand-maintained. This is the input of everything below |
| `example_metrics.json` | Statistics of `example.jsonl` (number of examples, tools, turns and words, plus a count of every other column) | Generated by `gym dataset collate`. Do not edit it by hand: Gym writes a `*_conflict.json` file if the metrics it computes differ from the checked-in ones |
| `example_rollouts.jsonl` | Exactly 5 rollouts: the example rows run through the agent, so each row also has the policy model's `response` and the server's `reward` | Generated from a live policy model and GenRM (see "Regenerating the files") |

`example_metrics.json` is found through the `datasets` entry of the agents in `configs/dwrl_genrm_compare_pointwise.yaml`, whose `jsonl_fpath` has to point at this folder's own `example.jsonl`.

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
- Any other column (for example `ref_answer`, which the pairwise server uses) is accepted and ignored.

### Regenerating the files

Both generated files need Gym's servers, so they are regenerated where the policy model and the GenRM can run, not by hand. Two things that are easy to miss: the agents in the config use the model servers `policy_model` and `policy_model_reasoning_off`, and only `policy_model` comes from a Gym config (`responses_api_models/vllm_model/configs`), so define the second one as the training configs do; and `dwrl_model` has `model: ???`, which has to be given.

```bash
# example_metrics.json (writes data/example_metrics.json next to example.jsonl)
gym dataset collate "+config_paths=[<policy model config>,resources_servers/dwrl_genrm_compare_pointwise/configs/dwrl_genrm_compare_pointwise.yaml]" \
    +output_dirpath=data/dwrl_genrm_compare_pointwise \
    +mode=example_validation

# example_rollouts.jsonl: 5 rows, each judged on its own, so the 5 example rows are used as they are
gym env start "+config_paths=[<policy model config>,resources_servers/dwrl_genrm_compare_pointwise/configs/dwrl_genrm_compare_pointwise.yaml]"
gym eval run --no-serve +agent_name=genrm_simple_agent \
    +input_jsonl_fpath=resources_servers/dwrl_genrm_compare_pointwise/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/dwrl_genrm_compare_pointwise/data/example_rollouts.jsonl \
    +limit=null
```

The metrics in the repository were produced by Gym's own metrics code (`nemo_gym.train_data_utils`) on the checked-in `example.jsonl`. The `gym` commands above are the generic Gym flow and have not been run for this server.

**How the checked-in `example_rollouts.jsonl` was made:** not with the commands above, but by running the 5 example rows through the DWRL training setup in NeMo-RL (`env.nemo_gym.is_trajectory_collection=true`, which writes each rollout's `/run` result to `trajectory_collection.jsonl`), with the GRPO reward penalties switched off so that `reward` is the server's reward. These are 5 real rollouts of the example rows, in order, from a policy model with reasoning on and the GenRM judge. Two edits were made to the output: the `prompt_str` and `generation_str` fields that vLLM adds to each message were removed (they duplicate the content), and `response.model` was replaced by `policy_model` (it held a checkpoint path). The third rollout (`status: incomplete`) reached `max_output_tokens` before it finished; it is kept as it is.

## Development

```bash
pytest resources_servers/dwrl_genrm_compare_pointwise/tests -v
```

## Related components

- **Pairwise counterpart**: `resources_servers/dwrl_genrm_compare_pairwise/` - compares the rollouts of a prompt with each other and fits ratings
- **Original server**: `resources_servers/genrm_compare/` - the 1-6 ranking based version these servers were derived from
- **Model server**: `responses_api_models/local_vllm_model/` - serves the GenRM

## License

Apache 2.0 - Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
