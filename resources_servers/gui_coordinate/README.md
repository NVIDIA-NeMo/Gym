# Description

GUI coordinate grounding verifier for visual pointing tasks. The model is shown an image and asked to point at a target; the server scores how close the predicted point is to the ground-truth coordinate.

The model must emit its prediction as `<point>(x, y)</point>`, where `x` and `y` are in thousandths of the image width and height (0-1000). The verifier divides them by 1000 to get normalized coordinates and compares against `expected_answer`, which is `"x,y"` in the same normalized 0-1 space.

Reward is a smooth quadratic falloff in Euclidean distance:

- `dist >= max_dist` (default 0.15) → reward 0.0
- otherwise → reward `(1 - dist / max_dist) ** 2`, so an exact hit scores 1.0

If the response contains no parseable `<point>` tag, or `expected_answer` is not a two-number `"x,y"` string, the reward is 0.0.

Data links: ?

# Input format

Each JSONL row carries the prompt (image + instruction) plus the verifier fields:

```json
{
  "responses_create_params": {"input": [{"role": "user", "type": "message", "content": [
    {"type": "input_image", "image_url": "data:image/png;base64,..."},
    {"type": "input_text", "text": "... Respond with the coordinates as <point>(x, y)</point> ..."}
  ]}]},
  "expected_answer": "0.6640,0.8410",
  "max_dist": 0.15,
  "metadata": {"target_color": "yellow"}
}
```

`max_dist` is per-row and optional (defaults to 0.15). `metadata` is passed through and is not used for scoring.

# Example usage

```bash
gym env start \
    --resources-server gui_coordinate \
    --model-type vllm_model &
gym eval run --no-serve \
    --agent gui_coordinate_simple_agent \
    --input resources_servers/gui_coordinate/data/example.jsonl \
    --output resources_servers/gui_coordinate/data/example_rollouts.jsonl
```

The example data needs a vision-capable policy model.

# Licensing information

Code: Apache 2.0
Data: the example images are the ones used by the `circle_click` resources server.

Dependencies
- nemo_gym: Apache 2.0
- fastapi: MIT
