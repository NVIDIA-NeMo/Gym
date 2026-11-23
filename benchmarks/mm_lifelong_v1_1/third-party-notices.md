# MM-Lifelong provenance

These benchmarks download the MM-Lifelong v1.1 annotations, manifest and checksums from the Hugging Face dataset
`MM-Lifelong/MM-Lifelong` at revision `efa197479b25fbcf4f7881f9218d703c57dfc9bb` when they prepare data; nothing of
the dataset is committed here. Two texts are retained:

- `benchmarks/mm_lifelong_v1_1/grading.yaml` holds the grader's system prompt and user template,
  `v1.1/grading/prompts.json` at that revision, with two template fields renamed.
- `benchmarks/mm_lifelong_v1_1_agentic/prepare.py` holds the `TIME COORDINATES` note of the dataset's agentic runner,
  `src/eval_codex_multivideo.py` at revision `b8a87afff4e24e76f6333937fb7b1c77dd5036fc` (the next revision).

The dataset's videos are third-party recordings whose copyrights remain with their owners; they are not
redistributed, and neither are images or files derived from them. The dataset's license follows:

MIT License

Copyright (c) 2026 MM-Lifelong contributors

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
