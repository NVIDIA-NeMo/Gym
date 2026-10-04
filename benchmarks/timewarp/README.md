# TimeWarp

TimeWarp ([paper](https://arxiv.org/abs/2603.04949), [code](https://github.com/sparklabutah/timewarp)) evaluates
how robust web agents are to temporal changes in web UI. Each of its 231 goals is written independently of any
UI era and runs on Wiki, News and Shop sites rendered in six eras, so a goal becomes six tasks. This benchmark is
the 103-goal test split in all six eras: 618 rows.

- Integration profile: `custom-gym-verifier`, with `simple_agent` driving the browser tools.
- Resources server and task contract: [`resources_servers/timewarp`](../../resources_servers/timewarp/README.md).

## Provenance

| | |
| --- | --- |
| Tasks | [`sparklabutah/timewarp`](https://huggingface.co/datasets/sparklabutah/timewarp) `data.json` at revision `246edb1cc9c4746df68172dad661c97164064cec`, identical to `src/browsergym/timewarp/data/test.raw.json` in `sparklabutah/timewarp` at commit `4978e69` |
| Split | Test: task ids 1-103 (as in BrowserGym's `timewarp.csv` and the dataset's `test.csv`). `prepare(split="train")` writes the 128-goal train split (ids 104-231, 768 rows) |
| Filtering | None. The train split's human-refined plans (`additional_instructions`) are not used: upstream appends them for teacher-trajectory collection, and they state the answer |
| Sites | TimeWarp's Flask apps, hosted by the user (see the resources server README) |
| License | Tasks and site data: MIT. Vendored verifier code: MIT (TimeWarp), with Apache-2.0 modifications |

## Task contract

The prompt ([`prompt.yaml`](prompt.yaml)) describes the three sites and the tools, asks the policy to start with
`observe` and to answer in a final message whose first sentence states the answer, and gives the goal (`intent`)
as the user message. Rows carry the tool schemas, `ui_version`, `start_site`, `sites` and the task's TimeWarp
`eval` block as `verifier_metadata`; the reference answers are never shown to the policy.

## Scoring and run conditions

- Reward: 1.0 when every verifier in the task's `eval_types` accepts the final answer, else 0.0. The headline
  metric is the mean reward; aggregate metrics add success rates per UI version and per site.
- Verifiers: TimeWarp's deterministic `string_match`, `number_match` and `list_match`. One test task (id 32,
  six rows) uses the LLM judge; without `judge_model_server` its rows are masked and excluded from scores.
- Harness: `simple_agent` with `max_steps: 30` (BrowserGym's TimeWarp limit). Observations are Playwright ARIA
  snapshots of at most 12,000 characters per part. Sampling settings, token budgets and repeats are run-time
  choices.
- Scores are not comparable to the paper's AgentLab/BrowserGym numbers; the resources server README lists the
  harness differences.

## Running

Start the six UI versions of the TimeWarp sites (resources server README), then:

```bash
gym eval prepare --benchmark timewarp
gym env validate timewarp
gym eval run --benchmark timewarp --model-type openai_model --split benchmark \
  --limit 6 --num-repeats 1 --output results/timewarp_smoke.jsonl
```

To grade the LLM-judge task, add a judge model server to the run config and set
`++timewarp_benchmark_resources_server.resources_servers.timewarp.judge_model_server.name=<judge>` (with
`type: responses_api_models`).

## Limitations

- The sites are external to Gym and must be running before rollouts start; one process per site and version.
- Observations are text only. The dataset's `test.csv` marks four test goals (ids 72-75, Shop product-photo
  questions; 24 rows) as needing visual information, and an image's alt text does not answer them; 21 more are
  marked as helped by it. Upstream agents may see screenshots.
- No reward profile has been recorded yet.
