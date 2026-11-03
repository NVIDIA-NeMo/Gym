# Long conversation simulation

One rollout plays out one long, multi-session user/assistant conversation for synthetic
long-context data. It ports the conversation loop of
[BEAM](https://github.com/mohammadtavakoli78/BEAM) (`answer_generation` and
`ConversationSummaryBuffer` in `src/beam/main.py`, commit `b2da22e`, MIT):

1. The task holds the conversation's sessions (`batches`), each with its pre-generated user
   messages, the session's plan and the plans of earlier sessions.
2. For every user message the **assistant model** answers from a memory of the most recent
   messages plus a running summary written by the **helper model**.
3. The helper model answers yes/no checks: did the assistant ask the user something? If so, the
   **user model** answers it (at most twice). Should the user follow up? If so, the user model
   asks (at most twice).
4. A question whose turn fails is skipped. A session that ends more than
   `batch_shortfall_allowance` messages short of two per question stops the conversation; earlier
   sessions are kept.

The agent calls its three model servers through `/v1/chat/completions` and needs no resources
server. It never verifies anything: the reward is the share of requested sessions the
conversation completed.

## Task

```json
{
  "responses_create_params": {"input": []},
  "topic": "Coding -> Building a personal budget tracker in Flask",
  "theme": "Tracking income and expenses -> Flask project structure, SQLite",
  "previous_plans_summary": "",
  "batches": [
    {"time_anchor": "March-03-2025", "current_plan": "...", "previous_plans": "",
     "messages": ["I'm starting a Flask budget tracker... ->-> 1,2"]}
  ],
  "agent_ref": {"type": "responses_api_agents", "name": "long_conversation_simulation"}
}
```

User messages carry BEAM's `->-> batch,bullet` tag; it is copied into each main question's
`index`. Extra task fields are kept in the rollout record.

## Rollout record

Besides the usual `response` (the last assistant completion, with usage summed over all
assistant calls) and `reward`:

- `chat`: one list of messages per completed session, in BEAM's `chat.pickle` layout:
  `{"role", "id", "content"}`, plus `index`, `question_type` (`main_question`,
  `answer_ai_question`, `followup_question`) on user messages and `time_anchor` on the first.
- `simulation`: sessions requested and completed, where it stopped, skipped questions, and call
  and failure counts per role.

A rollout that completes no session fails, so Gym retries it.

## Configuration

`configs/long_conversation_simulation.yaml` defines the agent and three model servers copied
from `policy_model`: `conversation_assistant_model`, `conversation_user_model` and
`conversation_helper_model`. Override them to use different models or request settings.

| Field | Default | Meaning |
| --- | --- | --- |
| `*_prompt_fpath` | required | BEAM's `ai_assistant_llm_template`, `check_include_question_template`, `check_need_followup_template`, `user_llm_answer_question_template`, `user_llm_ask_followup_question_template` |
| `tiktoken_encoding` | `cl100k_base` | Token estimate for memory budgets; characters // 3 when the encoding cannot load (BEAM's fallback) |
| `call_timeout_seconds` | 1800 | Per model call; vLLM model servers set no timeout of their own |
| `max_call_retries` | 3 | Retries for timeouts, connection errors, HTTP 408/409/425/429/5xx and empty output |
| `assistant_context_tokens` and the other memory fields | BEAM's values | Memory and prompt budgets of `answer_generation` |

Empty model output counts as a failure because a vLLM model server answers a context overflow
with an empty completion unless `propagate_context_overflow_errors` is set.

Rollouts are long: a 128K-setting BEAM conversation takes several hundred sequential calls.
Collect many conversations concurrently rather than relying on per-rollout speed.
