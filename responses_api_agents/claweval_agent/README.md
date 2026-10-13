# ClawEval OpenCode agent

This environment turns successful ClawEval SFT records into prompt-only online
RL tasks. The policy never receives recorded assistant turns, reference answers,
judge rubrics, or grader files.

Each rollout stages the selected task's media under `/workspace`, runs stock
OpenCode through the shared Harbor capture proxy, then uploads the task-local
ClawEval grader only after the agent finishes. The native ClawEval score is the
GRPO reward. Capture records retain parent and depth-one child OpenCode sessions
and optional compaction summaries for NeMo-RL multi-trace training.

Generate the 619-row OpenCode-aligned dataset and launch with:

```bash
MODEL_PATH=/path/to/checkpoint \
CLAW_EVAL_BACKEND_DIR=/path/to/live/backend \
bash scripts/claweval/run_sc.sh
```

The launcher uses W&B project `nemorl-v2-subagent-comapction-debug` and the
production GRPO horizon (`max_num_steps=1000000`), not a one-step debug cap.

The native grader follows the benchmark's Inference Hub configuration and
requires `NV_INFERENCE_API_KEY`. Use `scripts/claweval/train_subagents_sync_8n.sh`
for the first on-policy gate; only after that gate has healthy rewards, TMPE, and
a loadable checkpoint should `train_subagents_async_16n.sh` be submitted.
