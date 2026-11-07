# LongFact hard-200 evaluation set

`hard200.jsonl` contains 100 `objects` prompts and 100 `concepts` prompts from
all 38 LongFact topics. Prompt IDs and source checksums refer to
`google-deepmind/long-form-factuality` commit
`9d27158d198ced0a9d8271a80147cae580614601`.

The set contains the 100 prompts per subset with the largest positive
three-way `contradicted_claims` counts in the saved selection source. Ties are
ordered by stable prompt ID. `hard200-manifest.jsonl` records the selection
rank and contradiction count for every prompt.

SHA-256 checksums:

```text
3ff4d2e8df1f82b9f1b40435486145f62f1c8852043274997cf7d07ca0374a8e  hard200.jsonl
b9b6c21d97726e5760bf35085ee3bb7b9912ecb4cf4717f21719d1290ed442c4  hard200-manifest.jsonl
```
