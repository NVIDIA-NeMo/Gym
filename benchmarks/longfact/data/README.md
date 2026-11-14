# LongFact hard-200 evaluation set

`hard200.jsonl` contains 100 `objects` prompts and 100 `concepts` prompts from
all 38 LongFact topics. Prompt IDs and source checksums refer to
`google-deepmind/long-form-factuality` commit
`9d27158d198ced0a9d8271a80147cae580614601`.

Prompts are Copyright 2024 DeepMind Technologies Limited, licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); see the pinned
[upstream attribution and license statement](https://github.com/google-deepmind/long-form-factuality/blob/9d27158d198ced0a9d8271a80147cae580614601/README.md#license-and-disclaimer).
This subset selects and reorders source prompts and adds NeMo Gym fields.
Every row retains its exact upstream training-contamination canary. These
held-out prompts are for evaluation only and must not enter training corpora.

The set contains the 100 prompts per subset with the largest positive
three-way `contradicted_claims` counts in the saved selection source. Ties are
ordered by stable prompt ID. `hard200-manifest.jsonl` records the selection
rank and contradiction count for every prompt.

SHA-256 checksums:

```text
b9ebacd95ea3cce19743da4857842c50c97c52dfe44fcf44defc7e24b91cf66a  hard200.jsonl
b9b6c21d97726e5760bf35085ee3bb7b9912ecb4cf4717f21719d1290ed442c4  hard200-manifest.jsonl
```
