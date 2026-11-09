# Intent matcher (per-trial)

You will see two lists for the same coding task:

- **INTENTS** — pre-extracted atomic intent units from the original human session
- **TRIAL** — messages the simulated user actually sent during one trial

For each INTENT, decide which (if any) TRIAL message best conveys it, and how confident you are.

## What counts as a match

A trial message matches an intent if it conveys substantively the same:
- request / correction / nudge / question / verification / workflow / context

Paraphrasing across languages, different word choice, and different ordering are all fine. The test is whether a reader handed only the trial message would understand the same agent-side action the original intent demanded.

A trial message does **not** match if it raises a separate topic, OR if it covers only a fragment of a compound intent (then assign a lower confidence rather than `null`, e.g. 0.4).

## Use confidence to express partial matches

- `1.0` — verbatim or full paraphrase, no information lost
- `0.7–0.9` — same intent, missing minor specifics
- `0.4–0.6` — same general direction, missing load-bearing details (e.g. intent says "commit the lockfile too" and trial just says "commit")
- `0.1–0.3` — only superficially related; would not trigger the same agent behaviour
- `0.0` or `null match` — not present at all

## One trial message can match multiple intents only if it genuinely covers both

Conservative bias: prefer leaving an intent unmatched than over-assigning a single trial message to many intents.

## Also classify the trial messages

Every trial message that you did NOT pick as a match for any intent goes into `unmatched_trial_msgs`. Tag each:
- `task-relevant-extra` — sensible follow-up the original user happened not to make
- `off-task` — about something unrelated to the task
- `repetitive` — duplicate of an earlier sim message

> Per-message kind/tier/frustration tagging is **not** done here — it lives in
> [`tag_messages.py`](../tag_messages.py) (the multi-label tagger), which writes
> `trial_msg_tags` to the verdict and drives User Correction + User Intent. This
> prompt produces ONLY the match table.

## Output — strict JSON only

Return ONE JSON object. No prose, no markdown fences. Do NOT compute aggregate scores — downstream code does that from the match table.

```json
{
  "schema_version": 2,
  "n_intents": <int>,
  "n_trial_msgs": <int>,
  "per_intent": [
    {
      "intent_id": <int>,
      "matched_trial_idx": <int|null>,
      "match_confidence": <float 0..1>,
      "rationale": "<one short sentence>"
    }
  ],
  "unmatched_trial_msgs": [
    {
      "trial_idx": <int>,
      "category": "task-relevant-extra|off-task|repetitive",
      "rationale": "<one short sentence>"
    }
  ]
}
```

`per_intent` must contain exactly `n_intents` entries, one per intent_id in 0..n_intents-1, in any order.

`unmatched_trial_msgs` contains every trial_idx in 0..n_trial_msgs-1 that does NOT appear as `matched_trial_idx` for any intent.
