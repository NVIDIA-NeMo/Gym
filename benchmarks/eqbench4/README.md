# EQ-Bench 4

EQ-Bench 4 measures applied emotional and social intelligence in multi-turn roleplay. The policy plays a helper
(friend, partner, therapist, chatbot, and so on) talking with a simulated person who has hidden traits, sensitivities,
preferences and a situation dynamic. A judge panel compares each conversation with leaderboard models' conversations
of the same scenario, and a Bradley-Terry fit places the policy on the [EQ-Bench 4 leaderboard](https://eqbench.com/)
rating scale.

## Measurement contract

**Provenance.** [EQ-bench/eqbench4](https://github.com/EQ-bench/eqbench4), revision
`93dcb7f5d430433ad49d501e672a98bcff2c9678`, licensed BSD-3-Clause with the Commons Clause condition (Licensor:
Samuel J. Paech). Preparation downloads, and checks against pinned SHA-256 hashes:

- the scenario dataset `src/eqbench/datasets/test.json` (v0.17, 120 scenarios: 60 generated, 60 hand-authored);
- the prompt and rubric files and the prompt-building module `src/eqbench/core/prompts.py` with its schema imports,
  which preparation runs on the downloaded copy so every prompt is rendered exactly as in the leaderboard runs;
- the 28 published result files `results/r*.json` (about 1.7 GB): their transcripts are the opponents, and their
  10,689 pairwise comparisons are the reference rating graph.

Nothing from the upstream repository is committed to Gym; the prepared files and the download cache under `data/`
are gitignored, and the bundled examples and fixtures are synthetic. Gym's integration code is Apache-2.0 and is an
independent implementation; downloaded material keeps its upstream license.

**Task contract.** One row per scenario. The policy opens the conversation and speaks at most eight times, each
time with that turn's system prompt (role, presenting issue, demographics, goals, `Current turn: k of 8`) and the
dialogue so far; its first turn also sees the opening message that states the presenting issue. The persona never
reveals its hidden profile to the policy. After each policy turn the persona simulator replies with a private
`<internal>` monologue, its spoken `<dialogue>`, and a `<status>`; it can end the conversation early. The policy sees
only the dialogue. The agent is `eqbench4_agent` (see its [README](../../responses_api_agents/eqbench4_agent/README.md)).

**Scoring.** For each row the resources server compares the policy's conversation with four published conversations
of the same scenario. The pairwise judge reads both transcripts, the persona's monologues, and the full hidden
persona profile, and names a winner and a margin of one to five `+` on six ability dimensions (bond and rapport,
authenticity, attunement, meeting preferences and needs, emotion sensemaking, emotion management), or `tie`. Each
comparison is judged in both presentation orders by the same judge; the panel (Gemini 3.1 Pro Preview, GPT-5.5,
Claude Opus 4.6) rotates across comparisons. The policy's fraction of a comparison is 0.5 plus or minus half its
average weighted margin as a share of the largest possible margin (30). The row's reward is the mean fraction, in
[0, 1], higher is better.

The headline metric is `eqbench4_elo`: soft Bradley-Terry ratings fitted to the fractional outcomes of the published
comparisons plus the policy's, reported as `1200 + 400 / ln(10) * strength`, the scale of the leaderboard. Its 95%
interval (`eqbench4_elo_ci_low`, `eqbench4_elo_ci_high`) comes from 200 pair-stratified bootstrap resamples, measuring
how the policy's mean gap to the models within three ranks of it moves. `eqbench4_rank` counts the policy among the
29 rated models, `eqbench4_ladder/<model>` lists every published model's rating in the same fit, and
`eqbench4_win_rate` is the mean comparison fraction. Claude Opus 4.8 also scores twelve neutral behavioural traits
per conversation (`eqbench4_trait/<name>`, 0 to 10); they are informational and do not enter the reward or rating.

**Run conditions.** Policy sampling comes from each row: `temperature: 0.7`, `max_output_tokens: 10000`. The persona
uses temperature 0.7 and 10,000 output tokens; the pairwise judges temperature 0.3 and 4,096 tokens; the trait judge
temperature 0.7 and 16,000 tokens. Reasoning is minimal for Gemini, none for GPT-5.5, and Claude runs without
extended thinking, as in EQ-Bench 4. There is no episode deadline: an episode is bounded by eight turns per side,
three persona replies per persona turn, seven policy replies per policy turn, the output-token limits above, each
model server's request timeout, and `judge_timeout_secs` (1,800 s) per judge call. One repeat per scenario.

**Grading path.** Path A: the resources server runs the pairwise judging inside `observe_llm_judge_grading`, calls
every judge through `call_judge` and records each parsed verdict with `record_parsed_judge_verdict`. A comparison
rests on both presentation orders when the judge's replies hold a verdict, and on one when only that order's replies
do (as in EQ-Bench 4); a row records such an order in `unreadable_orders`, and its grading record stays incomplete
because one reply was never parsed. When neither order holds a verdict after `judge_max_attempts`, the judge error
masks the row as `judge_failed`, keeping the conversation for re-judging. The trait judge is informational: its
failure is recorded in `trait_judge_error` and changes nothing else.

**Conversations left out of the score.** As in EQ-Bench 4, a conversation that could not be finished is left out of
the score (the row is masked), whole, whenever it stopped:

| Cause | `failure_kind` | Usual fix |
| --- | --- | --- |
| The policy's prompt overflowed its context window (not asked again) | `model_context_limit` | Serve the policy with a longer context |
| Seven policy replies cut at `max_output_tokens` before any visible text, often all reasoning | `eqbench4:policy_output_budget` | Raise the budget or lower reasoning effort |
| A provider's content filter withheld the reply | `eqbench4:policy_content_filter` | Check the provider's filter |
| Seven finished replies without visible text | `eqbench4:policy_empty_reply` | Inspect the policy's replies |
| The persona gave no usable reply in three attempts | `eqbench4:persona_failed` | Check the persona model |

The agent logs a warning naming the scenario, turn and cause each time. After the run the aggregate metrics report
`eqbench4_conversations` (all conversations), `eqbench4_policy_stopped_rate` and its breakdown
`eqbench4_policy_stopped_rate/<cause>` (with counts in `eqbench4_policy_stopped/<cause>`), and
`eqbench4_persona_failed_rate`; both rates are also key metrics. The resources server logs a run-level warning with
the counts and the usual fix whenever any conversation was left out. In the published leaderboard runs this never
happened to an evaluated model (3,360 conversations); with other models a high rate means the score covers only the
scenarios the model finished, and should be fixed before the score is trusted.

## Prepare and validate

```bash
gym eval prepare --benchmark eqbench4
gym env validate eqbench4
gym env test eqbench4
```

Preparation downloads the pinned files once into `benchmarks/eqbench4/data/upstream/` and writes
`eqbench4_benchmark.jsonl` (120 rows, about 18 MB) and `reference_comparisons.jsonl`. To read a local checkout of
the pinned revision instead, set `+prepare_script_args.source_dir=/path/to/eqbench4` (a Windows checkout with CRLF line
endings is accepted). `opponents_per_scenario` (default 4) sets how many published conversations each row is compared
with.

## Evaluate

The persona and judges share one OpenAI-compatible Chat Completions endpoint. Set them with top-level keys, on the
command line (`++judge_api_key=...`) or in `env.yaml`, which keeps a key out of shell history:

| Key | Default | Sets |
| --- | --- | --- |
| `judge_api_key` | `dummy` | The endpoint's API key; required for a hosted endpoint |
| `judge_base_url` | `https://inference-api.nvidia.com/v1` | The endpoint |
| `eqbench4_persona_model_name` | `gcp/google/gemini-3.1-pro-preview` | Persona simulator |
| `eqbench4_gemini_judge_model_name` | `gcp/google/gemini-3.1-pro-preview` | Pairwise judge |
| `eqbench4_gpt_judge_model_name` | `openai/openai/gpt-5.5` | Pairwise judge |
| `eqbench4_claude_judge_model_name` | `aws/anthropic/bedrock-claude-opus-4-6` | Pairwise judge |
| `eqbench4_trait_judge_model_name` | `aws/anthropic/bedrock-claude-opus-4-8` | Trait judge (not scored) |

`judge_base_url` and `judge_api_key` are Gym's shared judge keys, so an `env.yaml` that sets them for other benchmarks
sets them here too. Change a model ID only where your endpoint names the same model differently: ratings are comparable
to the leaderboard only with these models. Skip the trait judge with
`++eqbench4_benchmark_resources_server.resources_servers.eqbench4.trait_judge_model_server=null`.

A model already served at a host and port (any OpenAI-compatible Chat Completions endpoint):

```bash
gym eval run --benchmark eqbench4 --split benchmark \
  --model-type vllm_model --model <served-model-name> \
  --model-url http://<host>:<port>/v1 --model-api-key <key-or-dummy> \
  ++judge_api_key=<judge-key> \
  --output results/eqbench4.jsonl
```

A shared endpoint may answer 429 when busy. Persona and judge calls are retried up to 36 times with backoff of up to
a minute; give the policy the same with `++policy_model.responses_api_models.vllm_model.max_http_attempts=36`. Add
`--resume` to continue an interrupted run from its saved conversations.

A local Hugging Face checkpoint directory, which Gym serves with vLLM (install Gym with `uv sync --extra ray`):

```bash
gym eval run --benchmark eqbench4 --split benchmark \
  --model-type local_vllm_model --model /path/to/hf_checkpoint \
  ++judge_api_key=<judge-key> \
  --output results/eqbench4.jsonl
```

Add `++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=false` for a served model without a
reasoning parser. A full run makes about 1,920 policy and persona calls, 960 pairwise judge calls with long prompts,
and 120 trait-judge calls.

## Checks against upstream

- Preparation renders byte-identical prompts to the published runs: the persona prompt of all 3,357 conversations;
  the per-turn evaluated-model prompts of all of them except three resumed conversations that recorded a turn twice;
  the trait-judge prompt of 513 sampled conversations; and the pairwise-judge prompt of 651 sampled readings judged
  with the current template.
- Ratings and bootstrap intervals from the published comparisons match upstream's own solver
  (`compute_local_delta_percentile_bootstrap_ratings`) to 1e-12 Elo. The committed `results/elo_results.json` differs
  from re-fitting those comparisons by up to 1.4 Elo; upstream's code gives the same re-fit as Gym.

## Differences from EQ-Bench 4

- **Opponents.** Upstream places a new model with four rounds of adaptive sampling that concentrate comparisons on
  its rank neighbours (up to 718 new comparisons per published model). Gym compares each scenario with four published
  models spread evenly down the ladder, rotating by scenario (480 comparisons; 16 to 19 per published model), so every
  comparison can be judged inside one row. The rating is the same Bradley-Terry fit; the interval reflects this design.
- **Unfinished conversations.** As upstream, a policy turn gets seven replies and an unfinished conversation is left
  out; Gym also records and reports why (see above), and does not ask again after a context-window overflow, which
  would recur.
- **Unparseable judge replies.** A reading with no JSON verdict is asked again, up to three attempts, before the
  comparison falls back to its other order as upstream does. Upstream does not ask again, and drops a comparison
  neither of whose orders parses; Gym masks that row for re-judging instead.
- **Persona corrections.** The correction shown to the persona after an unusable reply is Gym's wording.
- **Presentation.** Respondent codes are seeded by scenario, opponent and order, so a re-judged row shows the judge the
  same prompt; upstream draws them at random. Both orders are always judged, so which model comes first does not
  matter.
- **Reference judgements.** Most published comparisons were judged with earlier variants of the pairwise template;
  Gym judges with the current one, as upstream does for a newly added model.
- The behavioural-model summaries and judge-bias report of the leaderboard site are not produced.

## Limitations

The rating is comparable with the leaderboard only with the default persona model, because the published
conversations were all simulated by Gemini 3.1 Pro Preview; another persona changes what the policy is compared on.
Changing the judges changes the scale in the same way. The simulated persona and the judges are LLMs and have not
been validated against human raters; see the upstream README's limitations.
