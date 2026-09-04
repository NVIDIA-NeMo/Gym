# Qwen3.5-122B-A10B on WebVoyager

The reference repository names the model
`Qwen/Qwen3.5-122B-A10B-FP8`. No `120B-A30B` model appears in the inspected
Qwen WebVoyager launcher, so Gym uses the exact referenced model identifier.

The source recipe is recorded in `qwen35_recipe_lock.json`. Its important
settings are:

- four task splits and 16 isolated visual-browser workers per split;
- one TP8 model server per split;
- 262144 maximum model length, 32 maximum sequences, and 0.9 GPU utilization;
- vLLM `qwen3` reasoning parser, `qwen3_coder` tool parser, and automatic tool
  choice enabled;
- temperature 0.1, top-p 0.9, and 32768 maximum output tokens;
- `chat_template_kwargs={"enable_thinking": true}`;
- at most 100 browser steps;
- 20 active screenshot turns, folded in groups of 10, and 100 history turns;
- normalized coordinates in the 0..999 policy space;
- three parse attempts, each with up to five bounded API attempts.

Gym preserves the prompt, XML `computer_use` parser, screenshot resize/folding,
action summaries, terminal semantics, and action executor behavior. The only
intentional worker-safety divergence is a clamp of scroll magnitude to 50,
which prevents a pathological model output from stalling a GUI worker.

The model profile is
`benchmarks/webvoyager/configs/qwen35_122b_a10b.yaml`. Both Qwen and Nano Omni
consume `benchmarks/webvoyager/data/webvoyager.jsonl` and drive the same
`resources_servers/visual_browser` component. Therefore the model protocol is
swappable without changing benchmark tasks or browser behavior.

Use [runbook.md](runbook.md) for the full deployment and denominator checks.
