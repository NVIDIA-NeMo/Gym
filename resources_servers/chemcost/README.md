# ChemCost

[ChemCost](https://github.com/HeinzVonHank/ChemCost) ("Can Agents Price a Reaction? Evaluating LLMs on Chemical
Cost Reasoning", Wu et al., NeurIPS 2026 Evaluations and Datasets Track, [arXiv:2605.07251](https://arxiv.org/abs/2605.07251))
asks an agent for the cost of buying the materials of a chemical reaction, in US dollars per gram of product.
The agent sees the reaction as it was run in the laboratory: each component's name (a systematic name or
synonym, not the familiar one) and the amount used, the product and its isolated mass, and the scale reference
compound whose 1 mmol defines the costed run. It must work out the structures, the stoichiometry and the
solvents itself, query supplier quotes, pick the cheapest qualifying packs and aggregate. Labels come from a
frozen supplier-quote snapshot, so scoring is deterministic and judge-free.

- Task type: multi-step tool use with the `simple_agent` (two tools, at most 100 model calls)
- Domain: `knowledge`
- Reward: **CTA@25** for one reaction, 0 or 1 (the prediction is within 25% of the label); CTA@10 rides along
- Data: [`YoungerWu/ChemCost`](https://huggingface.co/datasets/YoungerWu/ChemCost) at revision `8af73ec`
  (CC BY-SA 4.0): 1,427 reactions (1,359 single-step, 68 multi-step routes) and the pricing database
  (2,848 chemicals, 121,824 quotes). Both files are checked by SHA-256 (`pricing_data.py`).

This is the authors' open-source release (October 2026). The anonymous review snapshot
(`nips2026-chemcost/chemcost-bench`) has no pricing database and an older task view, and the paper's main
table was run on that protocol; do not compare against it.

## Tools

Both are routes on this server that call the authors' `chemcost` package (MIT, pinned to commit `d7de381` in
`requirements.txt`):

| Tool | What it returns |
|---|---|
| `get_supplier_quotes(smiles_or_name)` | every quote for the chemical in the frozen database, unfiltered: supplier, pack size with unit, price in USD, reported purity (null when not reported) |
| `calculate(expression)` | an arithmetic result (AST whitelist, never executed as code) |

The package resolves a SMILES to a database entry by exact, canonical, stereo-free, tautomer and inorganic-formula
keys, then by name. There is no name-search tool and no molecular-weight tool, as upstream.

## Prompt

`prompts.py` transcribes the upstream view (`prepare_agent_input`), the system prompt and the user prompt of the
tool-using harnesses (`SYSTEM_PROMPT`, `build_task_prompt`) and the tool definitions. `tests/test_prompts.py`
checks every prepared row against the pinned package. Rows are prepared, not templated: the system and user
messages and the tools are in `responses_create_params`.

## Verification

`verify()` reads the terminal assistant message, the same text the upstream Codex harness reads, and parses it with
`answer_parsing.py`, a linear-time port of upstream `ReActAgent._try_parse_final_answer` (the last JSON object with
`predicted_cost_per_gram`, else a labelled per-gram figure; upstream is quadratic in opening braces).
`tests/test_answer_parsing.py` checks the port against the pinned package. The cost goes through upstream `_coerce_cost` and is scored with upstream `cta_at_k`.

| Field | Meaning |
|---|---|
| `cta_at_10`, `cta_at_25` | 1 when the relative error is at most 10% / 25% |
| `abstained` | no numeric prediction: null, a refusal, no final message, or the step or token limit ran out |
| `relative_error` | abs(predicted - label) / label, when there is a prediction |
| `failure_kind` | for an abstention: `agent_limit` when the token or step limit ended it (a message before pending tool calls is not an answer), else `agent_invalid_output`; it stays a scored 0, not a masked sample |

An abstention counts as a failure in the denominator, as upstream. Only a row without a positive label is
masked (`verifier_error`); the release has none. Aggregates: `mean/cta_at_10`, `mean/cta_at_25`,
`mean/abstained`, and `single_step/cta@k`, `multi_step/cta@k`.

## Published numbers

The upstream README reports all 1,427 records with the Codex CLI harness (only the two tools, zero-shot prompt,
no tool-call limit), run on 2026-10-22:

| Model | CTA@10 | CTA@25 | CTA@10 single-step | CTA@10 multi-step |
|---|---|---|---|---|
| gpt-5.6-luna | 64.2% | 72.7% | 65.3% | 41.2% |
| gpt-6-luna | 47.5% | 57.7% | 48.8% | 22.1% |

## What does not line up

- The upstream reference harness is Codex CLI; Gym runs the same prompt and tools through `simple_agent`.
  Codex wraps the prompt in its own instructions, and its default reasoning effort applies.
- Codex sends the system prompt and the task as one user message; Gym sends them as a system and a user
  message, as upstream's Responses API ReAct loop does (`instructions` plus the task).
- Upstream's ReAct loop asks once more for JSON when the step budget runs out; Codex does not, and neither
  does this server.
- Component precision and recall (diagnostics in the paper) and the noise settings are not ported.

## Run

```bash
gym eval prepare --benchmark chemcost
gym env test --resources-server chemcost +should_validate_data=true
```

## Licensing

Integration code: Apache-2.0. `prompts.py` adapts MIT-licensed upstream code (see its header). The dataset
and pricing database are CC BY-SA 4.0, downloaded at preparation time and gitignored.
