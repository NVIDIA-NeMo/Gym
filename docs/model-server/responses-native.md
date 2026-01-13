(model-server-responses-native)=
# Responses-Native Models

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #521](https://github.com/NVIDIA-NeMo/Gym/issues/521) for details.
```

Some models provide native support for OpenAI's Responses API, enabling direct tool calling without the Chat Completions to Responses API conversion layer.

---

## What Are Responses-Native Models?

Responses-native models implement the `/v1/responses` endpoint directly, providing:
- Native structured tool calling
- Built-in conversation state management
- Simplified integration without format conversion

## Supported Models

<!-- TODO: Document supported responses-native models (GPT-OSS, etc.) -->

## Benefits Over Chat Completions

| Feature | Chat Completions | Responses API |
|---------|------------------|---------------|
| Tool calling | Via function_call | Native support |
| Multi-turn state | Manual tracking | Built-in |
| Output format | Message-based | Structured items |

## Configuration

<!-- TODO: Add configuration example for responses-native models -->

## Integration with NeMo Gym

When using responses-native models:
1. Model server passes requests directly to `/v1/responses`
2. No conversion between Chat Completions and Responses formats
3. Tool calls and outputs use native Responses API schema

## Implementation Details

<!-- TODO: Document implementation considerations -->

## Migration from Chat Completions

<!-- TODO: Document migration path from chat completions models -->
