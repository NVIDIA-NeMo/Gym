# RoleMRC (gym-native)

[RoleMRC](https://huggingface.co/datasets/Junrulu/RoleMRC) is a **role-play
machine-reading-comprehension** benchmark: the model plays a character and
answers questions about supplied passages while respecting the character's
knowledge range, speech style, and instruction priority.

This entry runs RoleMRC through the **gym-native** eval path, in either of the
resources server's two scoring modes — one config each:

| Config | Benchmark name | Mode | Reward | Needs a judge? |
|--------|----------------|------|--------|----------------|
| `config.yaml` | `rolemrc` | `reference` | ROUGE-L vs the gold reply | no |
| `config_judge.yaml` | `rolemrc/config_judge` | `judge` | mean 0/1 over the relevant aspects | **yes** |

Reference mode also reports BLEU / METEOR / BERTScore alongside the reward.
Judge mode scores five aspects — `knowledge_range`, `style_compliance`,
`nested_instruction`, `multi_turn_instruction`, `instruction_priority` — with
one judge call per aspect; which aspects fire depends on the row's `task`. Both
modes break results down by RoleMRC **dimension** (`on_scene_dialogue`,
`multi_turn`, `nested_instruction`, `instruction_priority`).

The judge split is a strict **subset** of the reference split (only rows whose
`task` has an aspect config), so the two benchmarks are not scored over the same
rows and their numbers are not comparable.

## Relationship to the resources server

Scoring, the aspect rubrics, the judge client and all aggregation live in the
`rolemrc` resources server (`resources_servers/rolemrc/`) — see its
[README](../../resources_servers/rolemrc/README.md) for the mode table, the
BERTScore note and the judge wiring. This benchmark only supplies data and
wiring; it chains to that server's `rolemrc.yaml` / `rolemrc_judge.yaml` and
inherits `rolemrc_simple_agent` / `rolemrc_judge_simple_agent`.

## Data shape

RoleMRC rows need **no re-shaping**: `prepare_rolemrc.py` already emits
Responses API shape — the full multi-turn RoleMRC conversation as
`responses_create_params.input`, with `reference` / `task` / `dimension` riding
along — so `prompt_config` is `null` and the pre-built input is used untouched.
`prepare.py` / `prepare_judge.py` only tag each row with the benchmark
`agent_ref` so rows align with the agent selected at eval time.

## Prepare data

```bash
gym eval prepare --benchmark rolemrc                 # reference
gym eval prepare --benchmark rolemrc/config_judge    # judge
```

Either command builds `resources_servers/rolemrc/data/test.jsonl` **and**
`test_judge.jsonl` if missing (one `prepare_rolemrc.py` run writes both,
downloading `Junrulu/RoleMRC` on first use), then writes the tagged
`benchmarks/rolemrc/data/rolemrc{,_judge}_benchmark.jsonl`. All are gitignored.

Set `ROLEMRC_LOCAL_JSONL=/path/to/roleMRC_test.jsonl` to convert a
pre-downloaded file instead of fetching from the Hub.

## Running servers

```bash
# Reference metrics — no judge endpoint needed
gym env start --benchmark rolemrc --model-type vllm_model \
    --model <served-model-name> \
    --model-url http://<vllm-host>:8000/v1 \
    --model-api-key dummy

# LLM-as-judge — a judge endpoint is REQUIRED on top of the policy one
gym env start --benchmark rolemrc/config_judge --model-type vllm_model \
    --model <served-model-name> \
    --model-url http://<vllm-host>:8000/v1 \
    --model-api-key dummy \
    +judge_base_url=https://api.openai.com/v1/ \
    +judge_api_key="$OPEN_AI_KEY" \
    +judge_model_name=gpt-4.1
```

The policy endpoint is not optional: `--model` / `--model-url` /
`--model-api-key` set `policy_model_name` / `policy_base_url` / `policy_api_key`,
and the model server config fails to resolve without them. Use
`--model-type openai_model` instead of `vllm_model` to test a hosted API model.

The judge runs on its own `judge_model` server instance, deliberately separate
from the model under test — pointing it at `policy_model` makes the model grade
its own output.

Reference mode downloads a roberta-large checkpoint on first use (BERTScore is
on by default for parity with upstream); set `include_bertscore: false` on the
resources server for a lightweight ROUGE/BLEU/METEOR-only signal.

## Collecting rollouts and scoring

```bash
gym eval run --no-serve \
    --agent rolemrc_benchmark_simple_agent \
    --input benchmarks/rolemrc/data/rolemrc_benchmark.jsonl \
    --output results/rolemrc_rollouts.jsonl \
    --num-repeats 1
```

Use `rolemrc_judge_benchmark_simple_agent` and
`rolemrc_judge_benchmark.jsonl` for the judge variant.

For a reasoning model, serve it with `--reasoning-parser <name>` — `verify()`
also strips a leading `<think>…</think>` block as a fallback, but an unstripped
reasoning block wrecks the reference metrics.

## License

Dataset: Creative Commons Attribution 4.0 International —
[`Junrulu/RoleMRC`](https://huggingface.co/datasets/Junrulu/RoleMRC).
