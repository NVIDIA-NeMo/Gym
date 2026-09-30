<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conditional-likelihood multiple-choice agent

A native `simple_agent` extension for `/run`: take one raw prompt and ordered
`choices`, request the model server's `/loglikelihood`, select the highest summed
logprob (first choice wins ties), and grade the selected letter with `mcqa`.
It preserves session cookies and records every choice's scores and token IDs.

Rows use the standard MCQA `options` and `expected_answer` fields, plus ordered
`choices` continuations. The output contains the selected letter as `\boxed{A}`;
response metadata explicitly identifies this. Sampling temperature and generation
token caps do not change the choice score. The standard aggregation proxy and
skip-verification behavior are inherited from `simple_agent`. Use the `/run`
endpoint for likelihood evaluation, not the inherited generation-only `/v1/responses`.
