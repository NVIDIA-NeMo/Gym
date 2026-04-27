# vllm_audio_model

A drop-in replacement for `vllm_model` that adds audio-content support over a Responses-API-compatible
**sidechannel** (`responses_create_params.metadata.audio_url`).

## Why this exists

OpenAI's Responses API content union is `Union[ResponseInputTextParam, ResponseInputImageParam, ResponseInputFileParam]` — **no audio variant**.
`ResponseInputAudioParam` exists in the openai SDK as an orphan type but is not a member of any input union, so audio content blocks
in `responses_create_params.input.content` get rejected at Gym's simple_agent Pydantic-validation layer
before they ever reach the model server.

vLLM's `/v1/chat/completions` endpoint accepts `audio_url` (data-URI) and `input_audio` (base64+format)
content blocks for any audio-multimodal model. NeMo-Skills' `VLLMMultimodalModel` exploits this by
intercepting at the model-runner layer — it reads audio from per-row metadata, base64-encodes it, and
prepends an audio content block to the user message in the Chat Completions request.

This wrapper is the Gym-side equivalent. It runs on the model side (no agent-layer schema change required),
pulls audio from `metadata["audio_url"]`, and splices an `audio_url` block into the latest user message
in the Chat Completions request that `vllm_model` produces.

## Migration plan

This is **Option A** from the librispeech-pc migration writeup: a no-Gym-core-change unblock for audio benchmarks.
Once OpenAI extends `ResponseInputContentParam` to include audio (or Gym adopts a local extension — Option B),
this wrapper should be deprecated in favor of native audio content blocks.

## Input shape

JSONL rows produced by an audio benchmark's `prepare.py` look like:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Transcribe the audio with proper punctuation and capitalization."}
    ],
    "metadata": {
      "audio_url": "data:audio/wav;base64,<...>"
    }
  },
  "expected_answer": "...",
  "sample_id": "..."
}
```

`metadata` is `Dict[str, str]` per the Responses-API schema, which the simple_agent's Pydantic validator
accepts as opaque. Audio bytes (~50KB base64 per LibriSpeech utterance) ride as a string value.

## What the wrapper does

`_preprocess_chat_completion_create_params` (overridden):

1. Calls the parent implementation (handles the standard Responses → Chat Completions translation +
   reasoning-parser cleanup + extra_body merging).
2. Reads `metadata.audio_url` if present.
3. Strips it from the outgoing metadata (keeps the upstream vLLM request clean).
4. Finds the most recent user message and prepends an `{"type": "audio_url", "audio_url": {"url": ...}}`
   content block to its content list (creating a list if `content` was a plain string).
5. If no user message exists, creates one with just the audio block (defensive).

The `audio BEFORE text` ordering matches NeMo-Skills' `content_text_to_list` placement — required by
some audio models.

## Config

```yaml
policy_model:
  responses_api_models:
    vllm_audio_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: false
      uses_reasoning_parser: false   # most audio LLMs aren't reasoning models
      uses_interleaved_reasoning: false
```

Use this in place of `vllm_model` in any benchmark that needs audio:

```yaml
config_paths:
  - responses_api_models/vllm_audio_model/configs/vllm_audio_model.yaml
  - resources_servers/<your_audio_benchmark>/configs/<your_audio_benchmark>.yaml
```
