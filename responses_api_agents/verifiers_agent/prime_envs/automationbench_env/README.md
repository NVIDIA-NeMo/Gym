# automationbench_env

This is the native V1 port of Zapier's [AutomationBench](https://github.com/zapier/AutomationBench). It contains 600 scored business workflows across sales, marketing, operations, support, finance, and HR. Each task starts with a private simulated workspace containing the apps and records needed for that workflow.

The environment uses `verifiers.v1` and is compatible with Verifiers 0.3.0. The vendored package contains no evaluator runner or Verifiers V0 type.

## Tools

The model receives three bare tool names:

- `api_search(query, top_k=5)` returns a structured JSON object and searches only the services connected to the current task. Results only include operations that have a matching simulator implementation. Each result contains a canonical endpoint ID and real JSON schemas for its path, query parameters, and request body. Google Sheets results also include the available spreadsheet and worksheet IDs and titles.
- `api_fetch(endpoint, path=None, params=None, body=None)` executes a canonical endpoint returned by `api_search`. Its model-visible definition is a `oneOf` containing one concrete schema branch per connected endpoint, so every accepted `path`, `params`, and `body` property is declared in the actual tool definition. The method and internal route are fixed by the endpoint ID.
- `base64_encode(text)` supports API operations that need encoded content.

The model never constructs full URLs. Tenant prefixes stay inside the simulator, so duplicated paths and unresolved tenant placeholders are not part of the public contract. An endpoint from an unconnected service is rejected, and API application failures are returned as MCP tool errors.

The vendored catalog currently exposes 495 implemented operations across 47 simulated services. Unsupported catalog entries are deliberately not advertised.

## Scoring

Each rollout owns one live `WorldState`. Calls are serialized so parallel tool calls cannot overwrite each other's changes. After every `api_fetch`, the AutomationBench rubric scores that live world.

- `partial_credit` is the reward. It is the fraction of scored assertions that pass, using AutomationBench's free-assertion rules.
- `task_completed_correctly` is a metric and equals 1 only when partial credit is exactly 1.

If the model never calls `api_fetch`, the untouched initial world is scored. This matters because a small number of tasks can already satisfy an assertion before any action.

## Taskset

The default taskset contains 600 public scored tasks: 100 tasks in each of the six official domains. The optional `simple` domain adds 200 foundational tasks that are not part of the official public score.

The task builders, world models, rubric, and simulator implementations are vendored from `zapier/AutomationBench` at commit `4a8e1061254004d9dac807054eed33fad7d1ff14`. The upstream license is included in `AUTOMATIONBENCH_LICENSE`.

## Run

From this directory:

```sh
uv sync --all-extras
uv run eval automationbench_env --dry-run
uv run eval automationbench_env --max-turns 50 --no-push
```

The 50-turn limit matches the benchmark prompt's stated tool-use budget. Runs are unlimited by default if `--max-turns` is omitted.
