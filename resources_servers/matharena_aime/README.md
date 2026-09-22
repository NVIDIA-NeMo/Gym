<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MathArena AIME verifier

This resource server implements judge-free AIME verification from
[MathArena revision `b89f2f0`](https://github.com/eth-sri/matharena/tree/b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b).
The associated agent asks `/needs_format_retry` whether strict parsing found an
answer, performs at most one formatting-repair turn, and sends the complete
trajectory to `/verify`. Only the final assistant message is scored.

The non-strict verifier supports boxed and fbox answers, symbolic equivalence,
the upstream correction map, and MathArena's last-integer fallback. Gold answers
must be integers from 0 through 999. Parser warnings and response truncation are
reported for review.

The default aggregate metric is `pass@4/accuracy`. A problem passes when any of
its four valid repeats is correct. The server also reports per-language pass@4,
a language-macro pass@4, repeat coverage, missing and duplicate repeats, parser
warnings, retries, truncations, and invalid measurements. Headline metrics are
withheld when the selected task set is incomplete or requires review.

## Parser isolation

The upstream parser requires ANTLR 4.11, while Gym's configuration stack uses
ANTLR 4.9. Startup creates a gitignored parser virtual environment from
`parser-requirements.txt`; those dependencies must not be installed in Gym's
main environment.

Model output is processed by an arithmetic-only AST interpreter before symbolic
comparison. Attributes, subscripts, strings, comprehensions, arbitrary calls,
private names, and oversized expressions are rejected as invalid measurements.
The worker has wall-time, memory, and concurrency limits. Parser failures are
masked instead of counted as wrong answers.

Vendored MathArena sources are hash checked at load time and retain their MIT
license in [`_vendor/LICENSE`](_vendor/LICENSE). Benchmark questions are
downloaded during preparation and are not stored here.

```bash
pytest resources_servers/matharena_aime/tests -q
gym env test --resources-server matharena_aime
```
