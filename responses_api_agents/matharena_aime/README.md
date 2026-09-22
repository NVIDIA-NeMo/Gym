<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MathArena AIME agent

This Responses API agent implements MathArena's AIME format-repair flow. It makes
one policy call, asks the resource server whether strict parsing found an answer,
and makes at most one additional call with the official repair prompt. Incorrect
parseable answers are not retried, and the gold answer is never sent to the
policy model.

The second call receives the full native conversation, including reasoning and
token metadata. Raw turn responses, the retry decision, retry count, and combined
usage are preserved in the verification record.

The benchmark default is 120,000 output tokens per call with thinking enabled.
A repaired rollout can therefore use two policy calls. Gym assigns distinct
repeat seeds through `num_repeats_add_seed`; `seed_offset` shifts those seeds
once and the repair call retains the resulting seed.

The upstream repair-prompt attribution is in [`NOTICE`](NOTICE).

```bash
pytest responses_api_agents/matharena_aime/tests -q
```
