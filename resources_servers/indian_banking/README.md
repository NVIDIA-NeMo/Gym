# Description

`indian_banking` is a multi-turn, tool-using customer-support environment for an India-based retail bank. The policy model plays the bank's virtual assistant inside an already-authenticated customer session; a second LLM plays the customer (a role-swapped user simulator driven by a per-task persona and scenario); the assistant serves the customer by calling servicing tools against a per-episode snapshot of a synthetic core-banking database, and by looking up policy in a small knowledge base. Every episode is scored once, at the end, with a dense partial-credit reward built from deterministic checks on the tool calls, the final database state, and the customer-facing messages, plus a bounded LLM-judge score for natural-language assertions.

The environment and all of its data are original and synthetic. No real customer, account, or bank data is included. The task format, the role-swapped user simulator, and the ACTION / DB / COMMUNICATE / NL_ASSERTION evaluation decomposition follow the design of Sierra Research's tau-bench and tau2-bench; the banking domain, tools, database, knowledge base, and tasks were built for this environment.

## Tools

The assistant sees 34 function-calling tools: 33 servicing tools implemented in-process (no network, no external MCP server at runtime) plus `transfer_to_human_agents`, which is a terminal world flag rather than a database operation.

| Family | Tools |
| --- | --- |
| Knowledge base | `search_knowledge_base` |
| Accounts | `get_account_balance`, `get_account_details`, `get_transaction_history` |
| Deposits (FD / RD) | `get_fd_details`, `get_rd_details`, `get_deposit_loan_rates`, `calculate_fd_maturity`, `calculate_rd_maturity`, `create_fd`, `create_rd`, `get_deposit_closure_quote`, `close_deposit`, `update_deposit_renewal` |
| Loans | `get_loan_details`, `get_loan_foreclosure_quote`, `calculate_emi`, `get_gold_rate`, `calculate_gold_loan_ltv` |
| Cards | `get_card_details`, `toggle_card_freeze`, `block_card`, `set_card_controls` |
| Mandates and cheques | `show_mandates`, `cancel_mandate`, `stop_cheque_payment`, `request_cheque_book` |
| Servicing requests | `request_duplicate_statement`, `update_address`, `raise_request`, `get_request_status` |
| Insurance and products | `get_insurance_details`, `get_products_and_offers` |
| Handoff | `transfer_to_human_agents` |

Read tools never change state. Write tools (`create_fd`, `create_rd`, `close_deposit`, `update_deposit_renewal`, `toggle_card_freeze`, `block_card`, `set_card_controls`, `cancel_mandate`, `stop_cheque_payment`, `update_address`, `raise_request`) mutate the episode's database copy and are the only calls that can count as "wrong writes" in the reward; `request_cheque_book` and `request_duplicate_statement` are soft writes, excluded from the wrong-write count. Some writes fail deliberately when the customer's KYC is pending or expired; the assistant is expected to surface the returned message rather than retry.

`transfer_to_human_agents` ends the episode. Tasks whose gold trajectory includes a handoff (out-of-scope requests, disputes that need a human) expect the assistant to call it exactly once after explaining why; tasks that do not expect it treat a spurious handoff as a failed ACTION check.

### Per-episode isolation

`reset()` clones the base database (`data/db.json`) for the session; all tool calls in that episode read and write the clone, which is bound per asyncio task via `contextvars`, so concurrent rollouts in one process never see each other's state. Product catalogues and rate cards are module constants shared by every episode. The clone is discarded when the episode ends.

## User simulator

The customer is an LLM prompted with the task's `user_scenario` (persona, reason for call, what the customer knows and must not volunteer, and how the customer reacts to specific assistant behaviours) plus a fixed set of simulation guidelines: one message per turn, disclose information progressively, never invent facts not in the scenario, and end the conversation with a control token (`###STOP###` when the goal is met, `###TRANSFER###` after a handoff, `###OUT-OF-SCOPE###` when the scenario cannot answer the assistant). The first customer message for each task is precomputed and shipped with the row (`opening_message`), so the policy's first turn is deterministic across rollouts of the same task. Episodes cap at `max_user_turns` customer turns and `max_tool_rounds` consecutive tool rounds (defaults 8 and 12; the agent's `max_steps` should stay at or above 25 because dialogs commonly run 22 to 25 combined steps).

## Knowledge base

`search_knowledge_base` is in-process keyword search over `data/kb.json`, a 59-article corpus (Deposits, Loans, Accounts, Government Schemes, Payments, Cards, Accessibility, Grievance, Insurance, Digital Banking, Lockers, Charges, General, NRI). There is no vector index, embedding model, or network dependency. Results carry a relevance score; the domain policy tells the assistant to refine and re-search on weak hits rather than answer from memory. Tasks that require a policy answer include a `communicate_info` entry the assistant must state to the customer.

## Reward

Each task's `evaluation_criteria` specifies `actions` (gold tool calls, optionally with `compare_args` listing which arguments matter), `communicate_info` (strings the customer must be told), `nl_assertions` (free-text behavioural assertions), and `reward_basis` (which of `ACTION`, `DB`, `COMMUNICATE`, `NL_ASSERTION` are active for that task).

| Component | Question | Evidence | Enters |
| --- | --- | --- | --- |
| ACTION | Were all gold calls made with matching arguments, and were no wrong write calls made? | Assistant tool calls vs. gold actions, using soft argument comparison that ignores ephemeral fields (timestamps, generated ids, free text) | strict and dense |
| DB | Does the final database hash equal the hash produced by replaying the gold actions on a fresh clone? | Normalised DB state | strict and dense |
| COMMUNICATE | Was every `communicate_info` item said to the customer, with no internal identifiers or tool names leaked? | Customer-facing assistant messages | strict and dense |
| NL_ASSERTION | Are the `nl_assertions` satisfied? | Customer-facing transcript only, graded by the judge model | dense only |

An errored tool call never satisfies a gold action unless that gold action carries an explicit `expect_error: true` (tasks that deliberately exercise an error path, e.g. querying a nonexistent service request). A gold action that errors on replay without the marking is a data bug, and `tests/test_app.py::TestShippedDataIntegrity` fails on it.

Three further deterministic floors gate `strict`: an episode with no non-empty customer-facing assistant message never passes (a mute agent cannot outscore a correct refusal-with-explanation); a task may declare `max_tool_calls` (e.g. `0` for a purely conversational task, failing any tool use) or `require_transfer: true` (the episode must end in `transfer_to_human_agents`, read from the engine's transfer flag). All three are code checks, independent of the judge.

`strict` is the product of the binary ACTION, DB, and COMMUNICATE checks that appear in the task's `reward_basis`: 1.0 or 0.0, no partial credit. `dense` is a weighted blend of the same components plus the judge score (weights ACTION 0.40, DB 0.25, COMMUNICATE 0.15, NL_ASSERTION 0.20, renormalised over the active basis). The reported `reward` is:

- `strict == 1`: `1.0 - efficiency_cost`, where the cost (capped at 0.15) grades redundant calls, out-of-order gold calls, and low judge scores on an otherwise passing trajectory, so that a batch of passing rollouts still has reward variance.
- `strict == 0`: `strict + 0.4 * dense`, with conversational credit gated on the assistant having attempted the task (a silent agent does not outscore one that tried and made a small mistake).

The shaping constant is below the smallest increment a true pass earns, so a partial-credit trajectory can never outscore a pass. The judge score never enters `strict`, is capped at a small share of `dense`, sees only the customer-facing transcript (no tool calls, DB state, or gold actions), and fails open (the judge is left out rather than scored as zero) on an endpoint error.

**Call-order strictness.** `core/reward.py` exposes a module constant `SEQ_STRICT`, shipped as `True` (order-strict, the setting used for every number reported for this environment: the 300/300 gold replay, the example rollouts, and the reference GRPO run). Set it to `False` for tau2-comparable set-based scoring. With it off, the ACTION check is set-based, matching the tau2-bench evaluator. With it on, the order of gold tool calls becomes load-bearing for ACTION: `strict` additionally requires that the matched gold calls appear in canonical order (`seq_frac == 1.0`, computed as the longest common subsequence between gold actions and the assistant's calls divided by the number of gold actions), and half of the ACTION argument credit in `dense` is reserved for in-order matches. Pass rates scored with the flag on and off are not comparable, so state which setting was used when reporting results. The `seq_frac` value is always reported in `info`, and out-of-order gold calls always feed the efficiency cost on passing trajectories.

**Multiplicity.** All action signals share one consumption-based semantics: each gold action consumes one matching agent call, so a gold list that repeats an action (a read-after-write verification step - 67 of the 300 tasks do this) requires the agent to repeat it, and one call never satisfies two gold entries. A gold-mandated repeat is never charged as a duplicate by the efficiency cost and never flags the trajectory out of order - a byte-perfect replay of gold scores exactly 1.0. Conversely, a non-errored write call beyond what gold asks for counts as a wrong write, and an unmandated repeated call is charged as a duplicate.

Every terminal step returns the full reward breakdown (`reward`, `strict`, `dense`, per-component values, `action_frac`, `seq_frac`, `bad_writes`, `judge`) in `info`, so aggregation across rollouts works without a separate verify pass.

## Data

Every row is one task, and every task seeds one conversation.

| Split | Rows | Notes |
| --- | --- | --- |
| train | 250 | Training split |
| validation | 50 | Held-out; the 5 `example.jsonl` rows are drawn from it |

300 tasks in total across 50 task families, referencing 177 of the 197 synthetic customers in `data/db.json`. Every task was checked for required fields, an existing customer, real tool names, and clean serialisation. Tasks were generated with an LLM-driven pipeline (not included) and validated programmatically; the tasks, database, knowledge base, agent instruction, and domain policy are all synthetic and ship under Apache 2.0.

All customer, account, transaction, and merchant data is synthetic, and every bank, brand, and merchant name in the data is fictional; any resemblance to real entities is coincidental.

All data ships in-tree under `data/`: `train.jsonl` (250), `validation.jsonl` (50), `example.jsonl` (5 rows drawn from the validation split), `db.json` (customer database), `kb.json` (knowledge base), `agent_instruction.txt`, and `policy.md`. Nothing needs to be downloaded before serving.

A row looks like this (long fields elided):

```json
{
  "responses_create_params": {
    "input": [{"role": "system", "content": "<instructions>..</instructions>\n<policy>..</policy>"}],
    "tools": [{"type": "function", "name": "search_knowledge_base", "..": ".."}, ".. 34 tools .."],
    "parallel_tool_calls": false
  },
  "task_id": "edgablk2x194",
  "customer": "CUST_A0C6A651",
  "user_scenario": {
    "persona": "You are a methodical customer who likes things done step by step ..",
    "instructions": {
      "domain": "banking",
      "reason_for_call": "You spotted fraud on your card; you want it permanently blocked ..",
      "known_info": "You are Divya Krishnamurthy, customer already logged in ..",
      "unknown_info": "..",
      "task_instructions": ".."
    }
  },
  "evaluation_criteria": {
    "actions": [
      {"action_id": "edgablk2x194_0", "name": "get_card_details", "arguments": {"card_ids": ["CARD14974"]}},
      {"action_id": "edgablk2x194_1", "name": "block_card", "arguments": {"card_id": "CARD14974", "reason": "fraud"}, "compare_args": ["card_id", "reason"]},
      {"action_id": "edgablk2x194_2", "name": "get_card_details", "arguments": {"card_ids": ["CARD14974"]}}
    ],
    "communicate_info": [],
    "nl_assertions": ["Agent checks status first, blocks the card exactly once after confirmation, .."],
    "reward_basis": ["ACTION", "DB", "NL_ASSERTION"]
  },
  "initial_state": {},
  "opening_message": "Hi, I think someone has used my card without my permission .."
}
```

The system prompt is `data/agent_instruction.txt` wrapped in `<instructions>` and `data/policy.md` wrapped in `<policy>`; it is part of the task format, so changing either file changes what the policy trains and evaluates on.

**Episode termination.** An episode ends when the simulated customer emits a stop token, when the turn or tool-round caps are reached, or when the policy produces an empty turn (no text, no tool call). A turn that mixes customer-facing text with tool calls runs the tool calls and drops the text by default; with `strict_turn_protocol: true` (recommended for on-policy trainers whose chat template cannot re-render mixed turns byte-identically) such a turn ends the episode instead. Terminated episodes are scored as-is.

## Model servers

Three model roles are wired through the config, and they must not collapse into one:

- `policy_model` (the agent's `model_server`): the model being trained or evaluated. It only ever sees the assistant side of the conversation.
- `user_sim_model_server`: plays the customer. Any instruct model served on an OpenAI-compatible endpoint works; the shipped config declares it as the `user_simulator_model` block (a `responses_api_models.openai_model`) whose base URL, key, and model name are read from `env.yaml` as `user_sim_base_url`, `user_sim_api_key`, and `user_sim_model_name`, alongside the usual `policy_*` keys.
- `judge_model_server`: grades `nl_assertions`. The shipped config points it at the same server as the user simulator, which is fine; what is not fine is pointing it at `policy_model`. The judge must never be the policy: a policy that grades its own assertions is a direct reward-hacking surface, and the judge's share of reward exists precisely because the deterministic checks cannot see the behaviours it grades.

The judge is off when `should_use_judge: false` is set on the resources server; in that case NL_ASSERTION drops out of `dense` and the remaining weights renormalise. The user simulator is required.

## Running

Install the server's dependencies and validate the example data (this is what CI runs):

```bash
gym env test --resources-server indian_banking
```

Add the user-simulator/judge endpoint to `env.yaml` at the repo root (next to the `policy_*` keys), then start the environment with a policy served by vLLM:

```yaml
# env.yaml
user_sim_base_url: http://localhost:8000/v1
user_sim_api_key: dummy
user_sim_model_name: <user-sim-model-name>
```

```bash
gym env start \
    --model-type vllm_model \
    --resources-server indian_banking
```

Produce the collated dataset used for training (all splits ship in-tree; nothing to download):

```bash
gym dataset collate \
    --config resources_servers/indian_banking/configs/indian_banking.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --output-dir data/indian_banking_trajectory_collection \
    --mode train_preparation \
```

Collect rollouts on the example rows (this is also how `data/example_rollouts.jsonl` is produced):

```bash
gym eval run --no-serve \
    --agent indian_banking_agent \
    --input resources_servers/indian_banking/data/example.jsonl \
    --output resources_servers/indian_banking/data/example_rollouts.jsonl \
    --limit 5
```

Regenerate `data/example_metrics.json` after changing the example rows:

```bash
gym dataset collate \
    "+config_paths=[resources_servers/indian_banking/configs/indian_banking.yaml]" \
    +output_dirpath=resources_servers/indian_banking/data \
    +mode=example_validation
```

The deprecated `ng_run` / `ng_test` / `ng_collect_rollouts` entry points still resolve to the same commands (`gym env start`, `gym env test`, `gym eval run`).

For training, the environment exposes `reward` on every terminal step, so any NeMo Gym-compatible RL loop (for example NeMo RL GRPO) can consume it directly. If the policy's chat template strips or rewrites past-turn reasoning, serve it with thinking disabled so multi-turn rollouts stay consistent.

## Synthetic data notice

Every customer, account, card, deposit, loan, mandate, transaction, address,
e-mail address, phone number, employer and scenario in this benchmark is
synthetic, generated for evaluation purposes. The bank itself is fictional. Any
resemblance to real persons, living or dead, or to actual accounts, products or
events is coincidental. Where names of real companies, payees or institutions
appear in the data, they are used only as generic references to make scenarios
read naturally; nothing in this repository describes their actual products,
customers, transactions or policies, and no affiliation or endorsement is
implied. The knowledge-base articles paraphrase publicly available regulatory
and product information for the sole purpose of grounding the simulated
agent; they are not advice and should not be relied upon.

# Licensing information

Code: Apache 2.0<br>
Data: Apache 2.0 (all tasks, customers, accounts, knowledge-base articles, agent instruction, and domain policy are original synthetic data generated for this environment; no real customer or bank data)

Dependencies
- nemo_gym: Apache 2.0
- aiohttp: Apache 2.0 (transitively via nemo_gym; used for judge and user-simulator calls)

Attribution
- The task format, role-swapped user simulator, user-simulation guidelines, and the ACTION / DB / COMMUNICATE / NL_ASSERTION evaluation decomposition build on Sierra Research's tau-bench and tau2-bench (MIT License). `core/action_compare.py` and `core/state_normalize.py` are derived from Sierra Research's tau2-bench evaluator (soft tool-argument comparison and DB-state normalization, respectively) and remain under their original MIT terms (Copyright (c) 2025 Sierra Research; source: https://github.com/sierra-research/tau2-bench/blob/main/LICENSE); the full MIT text is reproduced in the root `ATTRIBUTIONS.md`. `core/prompts/user-sim-guidelines.md` is adapted from tau2-bench's `simulation_guidelines.md` under the same MIT license.
