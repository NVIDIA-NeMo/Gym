# OpenCode prompt attribution

`prompts/combined.txt` adapts the OpenCode 1.17.11 `anthropic.txt` prompt with
additional visual-task instructions. The system prompt in
`benchmarks/opencode_multimodal_benchmarks/opencode_eval_agent.yaml` reproduces that
version's `default.txt` prompt.

The final-turn directive in `responses_api_models/vllm_model/app_opencode.py`
also comes from this revision. `prompts/combined.txt.license` records the prompt
attribution without adding license text to model input.

Sources:
- https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/packages/core/src/session/runner/max-steps.ts
- https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/packages/opencode/src/session/prompt/anthropic.txt
- https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/packages/opencode/src/session/prompt/default.txt
- https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/LICENSE

MIT License

Copyright (c) 2025 opencode

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
