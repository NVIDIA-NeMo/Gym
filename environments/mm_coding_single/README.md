# Single-answer multimodal coding

A training environment for four task families (five protocols):
Chart2Code, MMSVG Img2SVG/Text2SVG, Image2Struct LaTeX and Design2Code.
The five synthetic examples exercise input wiring only. They are not a benchmark
score, a training dataset or an accuracy baseline.

The component's [measurement and runtime contract](../../resources_servers/mm_coding_training/README.md)
defines the original execution-gated quality reward, failure masks, source pins,
licensing, limits and required smoke evidence.

```
gym env validate mm_coding_single --json
gym env test mm_coding_single
gym env publish mm_coding_single
```

For a real smoke, configure the trusted judge and remote sandbox provider,
configure a Super 3.5 VL policy endpoint, then start this environment with
`gym env start --environment mm_coding_single --model-type openai_model`.
In another terminal run `gym eval run --no-serve --environment mm_coding_single
--limit 5 --num-repeats 1 --output results/mm_coding_single_smoke.jsonl`.
Inspect all five responses, remote execution receipts, masks and grading
observations. A complete evaluation or long RL run is separate evidence.
