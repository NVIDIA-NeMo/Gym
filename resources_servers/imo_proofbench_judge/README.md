# IMO ProofBench Judge

LLM-judge resource server for grading IMO-style proof submissions
against a reference solution and a problem-specific grading rubric.

A strong reasoner (Gemini-2.5-Pro by default, configurable) is asked to
return ``<points>N out of 7</points>``; the rollout is correct iff
``N >= 6``. The judge prompt is byte-identical to NeMo Skills'
``judge/imo_proofbench.yaml``, and the threshold matches Skills'
``is_correct_judgement`` Format-3 rule.

The judge is reached via ``/v1/chat/completions``
(``use_chat_completions_for_judge: true``) so any OpenAI-compatible
endpoint that exposes Gemini-2.5-Pro works — including Google's
``generativelanguage.googleapis.com/v1beta/openai/`` and NVIDIA's
``integrate.api.nvidia.com``.

## Pairing with a benchmark

Use ``benchmarks/imo_proofbench/`` for the IMO-ProofBench dataset.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/imo_proofbench_judge/configs/imo_proofbench_judge.yaml"
ng_run "+config_paths=[$config_paths]" \
    +judge_base_url=https://generativelanguage.googleapis.com/v1beta/openai \
    "+judge_api_key=$GEMINI_API_KEY" \
    +judge_model_name=gemini-2.5-pro
```

## Collecting rollouts (5-example smoke test)

```bash
ng_collect_rollouts \
    +agent_name=imo_proofbench_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/imo_proofbench_judge/data/example.jsonl \
    +output_jsonl_fpath=results/imo_proofbench_judge_rollouts.jsonl \
    +num_repeats=1
```

## Notes

- Start the policy vLLM server with ``--reasoning-parser <name>``
  (e.g. ``deepseek_r1``) so ``<think>…</think>`` is stripped at the
  model edge. Without it, truncated rollouts that never reach the
  closing think tag end up with the full reasoning trace as the
  predicted answer, which the judge then sees in the prompt.
- The judge is asked for ``<points>N out of 7</points>``; only N=0/1/6/7
  are explicitly enumerated by the rubric (matching the original IMO
  ProofBench design). Other integers parse but are below threshold and
  treated as incorrect.
- ``no_judge_score`` is reported when the judge response contains no
  parseable points block (typically empty / truncated judge output).
