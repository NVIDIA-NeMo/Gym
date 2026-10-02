# Modifications: formatting for NeMo Gym.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Human-likeness scoring based on deviation from human music distributions."""

import json
import math
import statistics as st


SPEC = [
    ("note_len_entropy", "band", "rhythm"),
    ("ioi_mean", "band", "rhythm"),
    ("rhythm_surprisal", "band", "rhythm"),
    ("note_density", "band", "rhythm"),
    ("qualified_note_rate", "band", "rhythm"),
    ("n_chan", "band", "texture"),
    ("polyphony_rate", "low", "texture"),
    ("polyphony_mean", "band", "texture"),
    ("roughness_p90", "band", "acoustic"),
    ("harmonicity_mean", "band", "acoustic"),
    ("key_certainty", "high", "tonal"),
    ("local_key_certainty", "high", "tonal"),
    ("pitch_in_scale", "band", "tonal"),
    ("pitch_min", "band", "register"),
    ("pitch_range", "band", "register"),
    ("voice_leading_cost", "band", "register"),
    ("self_similarity", "band", "structure"),
    ("tension_peaks", "band", "structure"),
]

W = {
    "rhythm": 0.22,
    "texture": 0.20,
    "acoustic": 0.14,
    "tonal": 0.16,
    "register": 0.16,
    "structure": 0.12,
}


def build_ref(dist_json, group="human"):
    """Build reference percentiles from human samples."""
    rows = [r for r in json.load(open(dist_json)) if not r.get("skip") and r["group"] == group]
    ref = {}
    for f, _, _ in SPEC:
        v = sorted(r[f] for r in rows)
        n = len(v)
        ref[f] = dict(
            p05=v[int(0.05 * n)],
            p10=v[int(0.10 * n)],
            p25=v[int(0.25 * n)],
            p50=v[n // 2],
            p75=v[int(0.75 * n)],
            p90=v[int(0.90 * n)],
            p95=v[int(0.95 * n)],
            lo=v[0],
            hi=v[-1],
        )
    ref["_hist"] = {}
    for k in ("_pc_hist", "_iv_hist", "_dur_hist"):
        L = len(rows[0][k])
        ref["_hist"][k] = [sum(r[k][i] for r in rows) / len(rows) for i in range(L)]
    return ref


def _band(x, r):
    if r["p10"] <= x <= r["p90"]:
        return 1.0
    if x < r["p10"]:
        span = max(1e-9, r["p10"] - r["p05"])
        return max(0.0, 1.0 - (r["p10"] - x) / (2 * span))
    span = max(1e-9, r["p95"] - r["p90"])
    return max(0.0, 1.0 - (x - r["p90"]) / (2 * span))


def _low(x, r):
    if x <= r["p25"]:
        return 1.0 if x >= r["p05"] else max(0.0, 1.0 - (r["p05"] - x) / max(1e-9, r["p05"] - r["lo"] + 1e-9))
    span = max(1e-9, r["p95"] - r["p25"])
    return max(0.0, 1.0 - (x - r["p25"]) / span)


def _high(x, r):
    if x >= r["p75"]:
        return 1.0
    span = max(1e-9, r["p75"] - r["p05"])
    return max(0.0, 1.0 - (r["p75"] - x) / span)


def js(p, q):
    p = [x + 1e-12 for x in p]
    q = [x + 1e-12 for x in q]
    sp, sq = sum(p), sum(q)
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    m = [(a + b) / 2 for a, b in zip(p, q, strict=False)]
    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b, strict=False) if x > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def score(feat, ref):
    """Score a feature dict against reference. Returns total/groups/per_feature."""
    if feat.get("skip"):
        return None
    per = {}
    grp = {}
    for f, kind, g in SPEC:
        if f not in feat:
            continue
        r = ref[f]
        x = feat[f]
        s = _band(x, r) if kind == "band" else (_low(x, r) if kind == "low" else _high(x, r))
        per[f] = round(s, 3)
        grp.setdefault(g, []).append(s)
    gs = {g: sum(v) / len(v) for g, v in grp.items()}
    base = sum(W[g] * gs.get(g, 0.5) for g in W) / sum(W.values())
    d = []
    for k in ("_pc_hist", "_iv_hist", "_dur_hist"):
        if k in feat:
            d.append(js(feat[k], ref["_hist"][k]))
    dist_pen = st.mean(d) if d else 0.3
    dist_s = max(0.0, 1.0 - dist_pen / 0.5)
    total = 100 * (0.85 * base + 0.15 * dist_s)
    return dict(
        total=round(total, 1),
        groups={g: round(100 * v, 1) for g, v in gs.items()},
        dist_score=round(100 * dist_s, 1),
        js_mean=round(dist_pen, 4),
        per_feature=per,
    )
