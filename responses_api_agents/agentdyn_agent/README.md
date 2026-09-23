# AgentDyn agent

A self-contained agent that runs [AgentDyn](https://github.com/SaFo-Lab/AgentDyn)'s
([arXiv:2602.03117](https://arxiv.org/abs/2602.03117)) pinned upstream harness
(`5353cf7615b135cace8d07c8f12dac53a16b6db3`, benchmark `v1.2.2`) against a NeMo Gym model server. Upstream owns the
tasks, the tool environments, the attack, the defenses and the scoring; the adapter supplies the model, records the
trajectory, and returns upstream's verdict. There is no resources server: AgentDojo-family suites verify against their
own in-process environment state.

The shared bridge and agent base live in `responses_api_agents/agentdojo_family/`. AgentDyn is installed in this
agent's own environment because it forks AgentDojo and installs the same top-level `agentdojo` package, so it cannot
share an environment with the official AgentDojo backend.

## Task matrix

Only AgentDyn's own suites -- `shopping`, `github` and `dailylife` -- are exposed: 20 user tasks each (60 clean
selectors), crossed with 9 injection tasks in `shopping` and `github` and 10 in `dailylife` for 560 attacked pairs,
all under the `important_instructions` attack. A row names a suite and user task, plus an injection task and attack
for the attacked arm; `task_data.py` validates that shape.

## Scoring

Each rollout returns upstream's two booleans, `utility` (the user task was completed) and `attack_success` (the
injection's goal was achieved), plus `security = not attack_success` and `reward = utility * security`. Upstream's own
field is named `security` but is True when the attack *succeeded*; the adapter inverts it so the names mean what they
say.

`compute_metrics` reports, under the `agentdyn/` prefix:

| Metric | Over |
|---|---|
| `benign_utility` | clean selectors |
| `utility_under_attack` | attacked selectors |
| `attack_success_rate` | attacked selectors |
| `scored_rollout_count`, `masked_rollout_count` | all rows the agent is given |

Under `gym eval run` and `gym eval aggregate`, Gym removes masked rows before calling the agent's metrics and reports
them itself as `coverage/masked_rollouts` and `coverage/measured_rollouts`; read the masked count there, since
`agentdyn/masked_rollout_count` is 0 on that path. The rates agree either way.

**Masked rollouts leave the denominator.** A rollout is masked (`mask_sample: true`) when the adapter could not
obtain a result: the model server failed the request, the upstream pipeline raised, or the rollout exceeded
`rollout_timeout_seconds`. Counting such a row either way would be wrong -- as a success it would credit a defense
for a crash, as a failure it would charge the model for infrastructure. `adapter_error` records the cause.

This is the one deliberate departure from upstream's harness. Upstream's `benchmark.py` sets
`utility = False, security = True` on `JSONDecodeError`, a provider `ServerError` or internal-server `ApiError`, and
`context_length_exceeded` -- scoring an infrastructure failure as a successful attack. Attack success rates computed
here are therefore lower than upstream's would be on the same trajectories whenever errors occur.

## Defenses

One defense per run, selected with `default_defense` on the agent server; defenses are not stacked, and the selector
rows are the same for every arm.

| Defense | Kind |
|---|---|
| `tool_filter` | the model pre-selects the tools a task may use |
| `spotlighting_with_delimiting` | tool outputs are delimited as data |
| `repeat_user_prompt` | the user task is restated after each tool call |
| `transformers_pi_detector` | classifier over tool outputs: `protectai/deberta-v3-base-prompt-injection-v2` |
| `piguard_detector` | classifier over tool outputs: `leolee99/PIGuard` |
| `prompt_guard_2_detector` | classifier over tool outputs: `meta-llama/Llama-Prompt-Guard-2-86M` (gated) |
| `camel` | the model writes one program that an interpreter executes under capability checks |
| `progent` | privilege-control policies over tool calls |
| `drift` | dynamic planning and validation around each tool call |

**The three classifiers are pinned** to Hub revisions in the agent config rather than resolved from a moving `main`;
PIGuard in particular is loaded with `trust_remote_code=True`. Each rollout records the repository and revision that
classified it. PromptGuard2's repository is gated; an ungated mirror with byte-identical weights, and a script that
verifies the match, are documented in [`benchmarks/agentdyn/README.md`](../../benchmarks/agentdyn/README.md).

**CaMeL, Progent and DRIFT make their own model calls** through OpenAI clients they construct internally. The adapter
routes those to the same NeMo model server as the policy, so every call in a rollout is answered by the model under
test. `defense_model_alias` is only the name upstream's code path expects; it does not select a model. For these three
defenses `model_call_count` is a lower bound -- calls made inside a defense's own clients are not all visible to the
adapter -- and the model server's logs are the authoritative count.

## Configuration notes

- **`concurrency` is fixed at 1; the config refuses any other value.** The agent holds `asyncio.Semaphore(concurrency)`
  around a whole rollout. For the routed defenses it scopes the client routing with `unittest.mock.patch` over process
  globals (`openai.OpenAI`, `os.environ`), and those scopes are not thread-safe: with two rollouts overlapping, one can
  record calls on the other's bridge, or have the unpatched OpenAI client restored beneath it -- which, with an
  `OPENAI_API_KEY` present, would send a defense's calls to a different model. Collect in parallel with separate
  processes, not a higher semaphore.
- **`rollout_timeout_seconds`.** CaMeL interprets model-generated Python with no step or time budget of its own, and
  occasionally generates a program that never terminates; the benchmark config sets 3600s, far above a healthy
  rollout. An abandoned rollout is masked with `adapter_error: RolloutTimeout`. Its thread cannot be cancelled, so
  after `max_abandoned_rollouts` (default 2) the agent exits to release them and the stack shuts down; rerun
  `gym eval run --resume` to continue. A timeout is an outcome of the treatment, not an infrastructure fault.
- **Sampling comes from the run.** The bridge forwards only the sampling settings present on the request
  (`gym eval run --temperature`, `--top-p`, ...), so with none set the endpoint's default applies. Upstream's harness
  samples at temperature 0; pass `--temperature 0.0` to match it. CaMeL requests 0 from its own client regardless.
- **`model_system_role`.** Some OpenAI-compatible endpoints reject the `developer` role (Qwen3.5 served by SGLang
  does); set `system` for those. The configured role applies to policy calls and to the defenses' own clients alike.
- **Reasoning envelopes.** Gym's Chat Completions path returns reasoning models' output wrapped in `<think>` tags. The
  bridge strips that envelope before upstream parses the reply, because the routed defenses parse structured output and
  would otherwise fail on the first character.

## Example data

`data/example.jsonl` holds five clean selectors chosen by rule -- `user_task_0` and `user_task_1` of `shopping` and
`github`, and `user_task_0` of `dailylife` -- not by outcome. `data/example_rollouts.jsonl` holds the undefended
rollouts for those five selectors from the `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` baseline run, which
was collected at the endpoint's default sampling rather than `--temperature 0.0`. All five happen to be completed
tasks; the full run's benign utility for that model is 70.00%.
