# MedXpertQA

[MedXpertQA](https://arxiv.org/abs/2501.18362) (ICML 2025) is an expert-level medical multiple-choice benchmark.
This server runs its official zero-shot CoT protocol from
[TsinghuaC3I/MedXpertQA@5f1db162](https://github.com/TsinghuaC3I/MedXpertQA/tree/5f1db162d0d2b03d62da2d4fe086076a36105e18/eval)
as a two-round Gymnasium episode with `gymnasium_agent`:

1. The dataset prompt (`benchmarks/prompts/eval/medxpertqa/zero_shot_cot.yaml`): system "You are a helpful medical
   assistant.", user "Q: {question}\nA: Let's think step by step."
2. After the reply, the server sends "Therefore, among A through J, the answer is" (the range follows the number of
   options) and grades the second reply.

Grading ports `answer_cleansing` from `eval/utils.py`: the text after the trigger, else after " answer is ", else the
whole reply; then the first standalone option letter, case-sensitive. The tests pin this against outputs of the
upstream function, including its quirks (a reply that starts "I cannot ..." reads as option I). Only the visible reply
of the first round is replayed, as upstream does (`text_only_history: true`). The whole visible reply is graded, so
serve a reasoning model with a reasoning parser: thinking left in the reply is graded too, as upstream would.

`answer_format: boxed` is the upstream DeepSeek-R1 variant for replies the default cleansing misreads: the trigger is
prefixed with "Put your final answer within \boxed{}." and the first `\boxed{<letter>}` is the answer.

Each result's `info` carries `question_id`, `question_type`, `medical_task`, `body_system`, `expected_answer`,
`extracted_answer` and the status of both turns. Metrics report accuracy overall and per `question_type`
(Reasoning, Understanding), `medical_task` and `body_system`. A step without reset state or a row without a valid
label returns `mask_sample` with `failure_kind` `session_lost` or `verifier_error`.

Benchmark: `benchmarks/medxpertqa_text` (Text test split, 2,450 questions).

## Run

```bash
gym eval run --benchmark medxpertqa_text --model-type vllm_model --split benchmark \
    --model <model> --model-url <url> --temperature 0 --output results/medxpertqa_text.jsonl
```

## Licensing

Code: Apache 2.0. Data: MIT ([dataset card](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)).
