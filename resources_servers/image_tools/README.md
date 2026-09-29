# Image Tools Pivot Verifier

PivotRL verifier for VLM image-tool trajectories. Scores a single rollout tool call
against the action demonstrated at that turn in an SFT trajectory.

## What a pivot is

One row = one decision point. The trajectory prefix is replayed verbatim
(teacher-forced) and the model emits **one** image-tool call. No tool is executed
and there is no final answer, so this is not outcome-reward RL — the reward is
"did you choose the action the demonstration chose".

Pair it with `tool_simulation_agent` (`max_steps: 1`): one model call, straight to
`/verify`.

## Verification flow

```
 rollout tool call  +  expected (demonstrated) tool call
            |
            v
   Level 0: tool name       same tool?            -> no: reward 0
            |
            v
   Level 1: image target    same img_idx /        -> no: reward 0
            |               img_idx_a+b /
            |               img_indices?
            v
   Level 2: primary arg     per tool family       -> below threshold: reward 0
            |
            v
        reward 1.0   (binary)  or  score  (continuous)
```

`label` and `factor` are **never** compared: a pivot is about which region of which
image, not the caption or the magnification.

### zoom and crop are one decision

`image_zoom_in_tool` and `image_crop_tool` are treated as the **same** tool at Level 0.
The task system prompt defines crop as *"Like image_zoom_in_tool but without
magnification"* — i.e. crop is zoom at factor 1 — and `factor` is already ignored, so
failing a rollout for picking one over the other while calling magnification irrelevant
is self-contradictory, and it punishes a model that localised the right region.

This matters in practice: 88.5% of pivots in the bvstyle dataset expect one of these two
tools, and tool-name mismatch was the single largest failure bucket before the change.

Only this pair is unified. Do **not** collapse whole `tool_family()` groups — those exist
to dispatch argument comparison, not to assert action equivalence.
`find_color_tool` vs `count_objects_tool` (locate vs count) and `image_rotate_tool` vs
`image_flip_tool` (different transforms) are genuinely different decisions and stay
distinct. Set `unify_region_inspect_tools: false` to require exact tool names.

Each verify response records `rollout_tool_name` and `expected_tool_name`, so tool
selection errors can be read as a confusion matrix rather than one aggregate count.

## Tool families

| family   | tools                                            | Level 2 comparison                     |
| -------- | ------------------------------------------------ | -------------------------------------- |
| `bbox`   | `image_zoom_in_tool`, `image_crop_tool`          | IoU of `bbox_2d`                       |
| `point`  | `color_at_tool`                                  | normalized distance on `point_2d`      |
| `color`  | `find_color_tool`, `count_objects_tool`          | channel-wise closeness of `color`      |
| `pair`   | `image_diff_tool`, `image_overlay_tool`          | indices (Level 1) + `alpha`            |
| `multi`  | `image_side_by_side_tool`                        | `img_indices`, order-insensitive       |
| `scalar` | `image_rotate_tool`, `image_flip_tool`           | `degrees` **mod 360**, or `axis`       |

Rotation is compared modulo 360, so `350` and `-10` are the same action.

The parser and IoU are imported from `resources_servers.image_tools`
(`parse_image_tool_calls`, `bbox_iou`) — the same code the image-tools agent uses at
rollout time, so the verifier cannot disagree with the agent about what a tool call is.

## Config

| option                      | default  | meaning                                          |
| --------------------------- | -------- | ------------------------------------------------ |
| `reward_mode`               | `binary` | `binary` (1.0 iff score >= threshold) or `continuous` (score itself) |
| `argument_threshold`        | `0.5`    | IoU / closeness needed for credit                |
| `enable_target_match`       | `true`   | Level 1                                          |
| `numeric_tolerance`         | `1e-6`   | scalar/alpha comparison slack                    |
| `penalize_extra_tool_calls` | `true`   | reward 0 if the rollout emits more than one call |
| `unify_region_inspect_tools`| `true`   | zoom and crop count as the same Level 0 decision |

`reward_mode: continuous` is the softer option: it gives partial credit for a
near-miss box instead of a cliff at IoU 0.5.

## Data

Built from ShareGPT-style SFT trajectories by
`tools/convert_sft_to_image_pivot.py` (in the NeMo-RL repo — conversion scripts do
not belong in Gym). Each row:

```json
{
  "agent_ref": {"type": "responses_api_agents", "name": "image_tools_pivot_agent"},
  "responses_create_params": {
    "input": [ ...trajectory prefix, images as input_image parts... ],
    "metadata": {"extra_body": "{\"stop\": [\"</tool_call>\"], \"include_stop_str_in_output\": true}"}
  },
  "expected_action": {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [...], "img_idx": 0}},
  "metadata": {"source_id": "...", "turn_index": 4, "tool_name": "..."}
}
```

The `stop` on `</tool_call>` is what makes the rollout end *at* the decision.

## Gotchas

- **`license` is required** on `train`/`validation` dataset entries. Without it,
  pydantic falls through to `BenchmarkDatasetConfig` and the server is silently
  demoted to an "almost-server" that never starts — the job dies minutes in with a
  confusing cascade about `prepare_script`. Run
  `python tools/preflight_pivot_config.py <config.yaml>` to catch this offline.
- Source SFT data may elide older tool-produced images, replacing them with
  `[N earlier produced image(s) omitted to bound context]`. Those pivots ask the
  model to decide about an image it can no longer see, which caps achievable reward.

## Testing

```bash
ng_test +entrypoint=resources_servers/image_tools
```
