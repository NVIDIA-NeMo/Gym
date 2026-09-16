# Config, Training, And Agent Ref

## Use The Pivot JSONL Directly

The pivot-format JSONL is the dataset. A Gym config should point a train or eval dataset entry
directly at the generated JSONL:

```yaml
example_single_step_tool_use_with_argument_comparison_agent:
  responses_api_agents:
    tool_simulation_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_single_step_tool_use_with_argument_comparison_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: /path/to/pivot.jsonl
        num_repeats: 1
        license: TBD
```

Place the config beside the generated pivot dataset when the user asks for a self-contained pivot
bundle.

## agent_ref Alignment

Each row should carry:

```json
{"type": "responses_api_agents", "name": "example_single_step_tool_use_with_argument_comparison_agent"}
```

The config must define that same agent name, or the launcher must intentionally override it. If the
dataset row and YAML agent name disagree, the row can validate as JSON but still route to the wrong
resource server or model path.

Use the dataset-specific prefix in `agent_ref.name`, not a generic name, when multiple pivot
datasets share the same resource-server family.

## Resource-Server Knobs

For `single_step_tool_use_with_argument_comparison`, the commonly tuned config block is:

```yaml
tool_call_comparator_config:
  word_count_similarity_threshold: 0.1
  floating_point_comparison_threshold: 1.0e-6
  # For datasets with function_call_batch labels:
  parallel_tool_call_rewarding: true
  allow_subset: true
  allow_superset: true
  parallel_tool_call_reward_mode: f1
```

Knobs:

- `word_count_similarity_threshold`: for string arguments with at least two words on both sides,
  split on whitespace, lowercase tokens, count token multiplicities, and compute
  `intersection_count / (expected_word_total + actual_word_total)`. Lower is more permissive;
  higher is stricter. This is a word-overlap threshold in the current code, not a standard IoU.

  **0.5 is the ceiling, not 1.0.** The denominator is a sum, not a union, so two *identical* strings
  score exactly 0.5. Any threshold above 0.5 rejects every multi-word string argument, including a
  perfect reproduction — set 0.7 expecting "strict" and you get "reject everything". Note also that
  the check only applies when both sides have at least two words, so short arguments are unaffected
  by any threshold.

  **Tune from the pairs that change verdict, not the mean.** Re-score recorded rollouts offline at
  several thresholds rather than re-running inference, then read the argument pairs whose verdict
  actually flips and decide which side each belongs on. A mean reward moves smoothly and tells you
  nothing about whether the cut is in the right place.

  **Known limitation.** An argument that is itself a JSON string — a tool whose `arguments` parameter
  nests escaped JSON — is word-matched rather than compared structurally. Shared keys and ids
  dominate the overlap, so a wrong enum value inside it can still pass at any usable threshold.
  Either unwrap nested JSON in the converter, or accept that those arguments are only loosely
  checked. No threshold setting fixes this.
- `floating_point_comparison_threshold`: absolute tolerance for float argument comparison.

### Parallel Tool-Call Knobs

`parallel_tool_call_rewarding` is the master switch and defaults to **false**. While it is off the
number of calls is not part of the verdict: every expected call must still match, but surplus calls
cost nothing. Turn it on for any dataset carrying `function_call_batch` labels — counting calls is
the point of those rows.

Once on, two stages apply. `allow_subset` and `allow_superset` are the cardinality gate, deciding
which response *shapes* are admissible at all; both default to false, so the call count must match
exactly. `parallel_tool_call_reward_mode` then decides how much credit an admissible response earns:
`binary_strict` (all or nothing), `fractional` (matched fraction), or `f1`
(`2 * matched / (expected + actual)`).

Measured against an expected set of two calls:

| response | switch off | on, `f1`, both gates open |
| --- | ---: | ---: |
| both calls, either order | 1.000 | 1.000 |
| only one of the two | 0.000 | 0.667 |
| both calls plus a surplus call | 1.000 | 0.800 |
| right count, one call wrong | 0.000 | 0.500 |

Prefer `f1` for RL. It is the only mode that penalizes missing and surplus calls symmetrically; with
permissive gates the other two reward a policy that emits one easy call and stops, or that spams
every plausible call. `resources_servers/single_step_tool_use_with_argument_comparison/configs/parallel_tool_calls_single_step_tool_use_with_argument_comparison.yaml`
is the maintained worked example.

### Re-Scoring Without Re-Running Inference

`gym eval reverify` recomputes rewards for recorded rollouts against an updated resources server,
which is how to compare verifier settings without paying for inference again. Note that a server
which does not declare `REVERIFY_MODE` defaults to `UNKNOWN`, and reverification then requires
`--force` and writes `unsafe_`-prefixed output;
`single_step_tool_use_with_argument_comparison` currently does not declare it even though its
`verify()` is a pure function of the request.

Use `tool_choice: "auto"` in pivot rows for this workflow. Avoid `tool_choice: "required"` because
some inference engines, including vLLM paths, can treat it as a structured decoding request rather
than ordinary tool-choice behavior.

Each pivot row carries exactly one `expected_action`, and that action covers the whole model call:
`function_call` for a single call, `function_call_batch` for a parallel set, `message` for text.

## Pivot Selection For Training

PivotRL-style training benefits from informative local states. Prefer:

- clean source trajectories with positive reward for the demonstrated pivots.
- tasks whose source trajectory group has mixed rewards when that grouping is available.
- candidate pivots that remain mixed under local on-policy rollout from the downstream or initial policy.

Default profiling rule:

1. Sample at least 8 local rollouts per candidate pivot when feasible.
2. Score with the target verifier.
3. Keep reward groups with both 0 and 1.
4. Discard all-1 groups because they are already easy.
5. Discard all-0 groups because they are usually too hard, underspecified, or mismatched to the verifier.
6. If data is abundant, drop the easiest retained pivots first and keep harder mixed-reward pivots.

Source-task mixed reward is useful but not required. A source model and downstream policy can be far
apart, so downstream local reward variance is the stronger selection signal when it is available.

When every source trajectory carries the same reward — a filtered success-only export, for instance —
this filter has nothing to act on. Say so explicitly in the metrics rather than skipping it silently,
keep the full unfiltered set, and move selection to local on-policy profiling, which is the stronger
signal anyway.

Reference: [Yi et al., "PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost"
(arXiv:2603.21383v1, 2026)](https://arxiv.org/html/2603.21383v1).

## Minimum Acceptance Criteria

Before training:

- sampled rows pass the target resource-server request models.
- `agent_ref.name` matches the config.
- expected tool names exist in `responses_create_params.tools`.
- no `responses_create_params.metadata: null` rows are present.
- no one-element `function_call_batch` rows are present, and `parallel_tool_calls` is `true` wherever
  batch labels are.
- every label self-scores 1.0 under the comparator this config builds.
- the boundary and reasoning-leakage audit is clean when the converter had to segment a flattened
  output list.
- optional provenance is present when needed for debugging, filtering, or later analysis.
- selection metrics show how many candidates were kept, skipped, all-success, all-failure, and mixed.
