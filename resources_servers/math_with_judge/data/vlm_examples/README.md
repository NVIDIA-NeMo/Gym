# VLM Math Examples

This directory contains JSONL data used by the `math_with_judge` environment for multimodal (image + text) and text-only prompts.


## Record Schema

Each line is one JSON object with the same top-level layout:

```json
{
  "agent_ref": {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"},
  "responses_create_params": {
    "input": [
      {
        "role": "user",
        "type": "message",
        "content": "..." 
      }
    ]
  },
  "question": "...",
  "expected_answer": "..."
}
```

Notes:

- `question` can be `null` in some mixed-source text-only rows.
- `responses_create_params.input[0].content` can be either:
  - an array of multimodal content blocks, or
  - a plain string for text-only samples.

## Multimodal Content Format (used in this folder)

```json
{
  "responses_create_params": {
    "input": [{
      "role": "user",
      "type": "message",
      "content": [
        {
          "type": "input_image",
          "image_url": "/absolute/path/to/image.png",
          "detail": "auto"
        },
        {
          "type": "input_text",
          "text": "Question text (may also include <image> markers)."
        }
      ]
    }]
  }
}
```

This dataset uses `image_url` (typically absolute local paths), not `url`/`data` keys.

## Text-Only Content Format (also present in mixed files)

```json
{
  "responses_create_params": {
    "input": [{
      "role": "user",
      "type": "message",
      "content": "Long text-only prompt..."
    }]
  }
}
```

## Practical Guidance

- Keep one user message in `responses_create_params.input`.
- For image records, keep at least one `input_image` block and one `input_text` block.
- Multiple images are supported by adding multiple `input_image` blocks before `input_text`.
- For text-only rows, keep `content` as a string (as in existing mixed data) unless your downstream pipeline requires block format.
