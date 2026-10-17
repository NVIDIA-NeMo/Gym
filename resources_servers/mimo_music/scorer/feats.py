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
"""Assemble core primitives into a complete feature vector for a piece."""

import collections
import math
import statistics as st

from .core import MAJ, MIN, NGram, _ce, _dist, harmonicity, krumhansl_key, parse_midi, roughness


DRUM_CH = 9


def _sonorities(notes, div, grid=0.5):
    if not notes:
        return []
    step = max(1, int(div * grid))
    end = max(n[0] + n[1] for n in notes)
    out = []
    for t in range(0, end + 1, step):
        cur = [(p, v) for (o, d, ch, p, v) in notes if ch != DRUM_CH and o <= t < o + max(d, 1)]
        if cur:
            out.append((t, cur))
    return out


def analyze(path):
    """Extract features from a MIDI file path."""
    m = parse_midi(open(path, "rb").read())
    if not m:
        return {"skip": "bad_midi"}
    div, notes = m["div"], m["notes"]
    if not notes:
        return {"skip": "no_notes"}
    pit = [n for n in notes if n[2] != DRUM_CH]
    drum = [n for n in notes if n[2] == DRUM_CH]
    if not pit:
        return {"skip": "drums_only"}
    R = {}
    R["n_note"] = len(pit)
    R["n_drum"] = len(drum)
    R["n_chan"] = len(set(n[2] for n in pit))
    tempo_bpm = 60_000_000 / m["tempo"]
    total_beats = max(n[0] + n[1] for n in notes) / div
    R["dur_sec"] = round(total_beats / tempo_bpm * 60, 1)
    R["bpm"] = round(tempo_bpm)

    son = _sonorities(pit, div)
    rough = []
    harm = []
    poly = []
    for t, cur in son:
        ps = [p for p, v in cur]
        vs = [v for p, v in cur]
        poly.append(len(ps))
        if len(ps) >= 2:
            rough.append(roughness(ps, vs))
            harm.append(harmonicity(ps))
    R["roughness_mean"] = round(st.mean(rough), 4) if rough else 0.0
    R["roughness_p90"] = round(sorted(rough)[int(0.9 * len(rough))], 4) if rough else 0.0
    R["roughness_std"] = round(st.pstdev(rough), 4) if len(rough) > 1 else 0.0
    R["harmonicity_mean"] = round(st.mean(harm), 4) if harm else 0.0
    R["polyphony_mean"] = round(st.mean(poly), 2) if poly else 0.0
    R["polyphony_rate"] = round(sum(1 for x in poly if x > 1) / len(poly), 3) if poly else 0.0

    hist = [0.0] * 12
    for o, d, ch, p, v in pit:
        hist[p % 12] += d / div
    tonic, minor, r1, r2 = krumhansl_key(hist)
    R["key_tonic"] = tonic
    R["key_minor"] = minor
    R["key_certainty"] = round(r1, 3)
    R["key_margin"] = round(r1 - r2, 3)
    scale = set((tonic + x) % 12 for x in (MIN if minor else MAJ))
    inside = sum(1 for (o, d, ch, p, v) in pit if p % 12 in scale)
    R["pitch_in_scale"] = round(inside / len(pit), 4)
    locs = []
    W = int(div * 8)
    endt = max(n[0] + n[1] for n in pit)
    for t0 in range(0, endt + 1, W):
        h = [0.0] * 12
        for o, d, ch, p, v in pit:
            if t0 <= o < t0 + W:
                h[p % 12] += d / div
        if sum(h) > 0:
            _, _, rr, r2b = krumhansl_key(h)
            locs.append(rr)
    R["local_key_certainty"] = round(st.mean(locs), 3) if locs else 0.0
    R["key_change_rate"] = round(st.pstdev(locs), 3) if len(locs) > 1 else 0.0

    seq = sorted(pit, key=lambda n: (n[0], n[3]))
    top = {}
    for o, d, ch, p, v in seq:
        top[o] = max(top.get(o, 0), p)
    mel = [top[t] for t in sorted(top)]
    out_idx = [i for i, p in enumerate(mel) if p % 12 not in scale]
    resolved = 0
    for i in out_idx:
        if i + 1 < len(mel) and abs(mel[i + 1] - mel[i]) <= 2 and mel[i + 1] % 12 in scale:
            resolved += 1
    R["chromatic_resolution"] = round(resolved / len(out_idx), 3) if out_idx else 1.0

    ces = []
    for t, cur in son:
        ps = [p for p, v in cur]
        vs = [v / 127.0 for p, v in cur]
        c = _ce(ps, vs)
        if c:
            ces.append(c)
    glob = _ce([p for (o, d, ch, p, v) in pit], [d / div for (o, d, ch, p, v) in pit])
    strain = [_dist(c, glob) for c in ces]
    R["tensile_strain_mean"] = round(st.mean(strain), 4) if strain else 0.0
    R["tensile_strain_std"] = round(st.pstdev(strain), 4) if len(strain) > 1 else 0.0
    R["tensile_strain_p90"] = round(sorted(strain)[int(0.9 * len(strain))], 4) if strain else 0.0
    mom = [_dist(ces[i], ces[i + 1]) for i in range(len(ces) - 1)]
    R["cloud_momentum_mean"] = round(st.mean(mom), 4) if mom else 0.0
    diam = []
    for t, cur in son:
        ps = list(set(p % 12 for p, v in cur))
        if len(ps) >= 2:
            pts = [_ce([p]) for p in ps]
            diam.append(max(_dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :]))
    R["cloud_diameter_mean"] = round(st.mean(diam), 4) if diam else 0.0
    if len(strain) > 8:
        mu, sd = st.mean(strain), st.pstdev(strain)
        R["tension_peaks"] = sum(
            1
            for i in range(1, len(strain) - 1)
            if strain[i] > mu + sd and strain[i] >= strain[i - 1] and strain[i] >= strain[i + 1]
        )
        q = len(strain) // 4
        R["phrase_end_drop"] = round(st.mean(strain[:q]) - st.mean(strain[-q:]), 4)
    else:
        R["tension_peaks"] = 0
        R["phrase_end_drop"] = 0.0

    byc = collections.defaultdict(list)
    for o, d, ch, p, v in pit:
        byc[ch].append((o, p))
    vl = []
    for ch, arr in byc.items():
        arr.sort()
        for i in range(len(arr) - 1):
            if arr[i + 1][0] != arr[i][0]:
                vl.append(abs(arr[i + 1][1] - arr[i][1]))
    R["voice_leading_cost"] = round(st.mean(vl), 3) if vl else 0.0
    R["vl_over_octave"] = round(sum(1 for x in vl if x > 12) / len(vl), 4) if vl else 0.0

    iv = [mel[i + 1] - mel[i] for i in range(len(mel) - 1)]
    aiv = [abs(x) for x in iv]
    R["pitch_min"] = min(p for (o, d, ch, p, v) in pit)
    R["pitch_max"] = max(p for (o, d, ch, p, v) in pit)
    R["pitch_range"] = R["pitch_max"] - R["pitch_min"]
    R["mel_interval_mean"] = round(st.mean(aiv), 3) if aiv else 0.0
    R["large_leap_rate"] = round(sum(1 for x in aiv if x > 7) / len(aiv), 4) if aiv else 0.0
    lr = 0
    nl = 0
    for i in range(len(iv) - 1):
        if abs(iv[i]) > 7:
            nl += 1
            if iv[i] * iv[i + 1] < 0 and abs(iv[i + 1]) <= 4:
                lr += 1
    R["leap_resolution"] = round(lr / nl, 3) if nl else 1.0

    def ent(c):
        s = sum(c.values()) or 1
        return round(-sum((v / s) * math.log2(v / s) for v in c.values() if v), 3)

    R["pitch_entropy"] = ent(collections.Counter(p for (o, d, ch, p, v) in pit))
    R["pc_entropy"] = ent(collections.Counter(p % 12 for (o, d, ch, p, v) in pit))

    ons = sorted(set(n[0] for n in notes))
    ioi = [(ons[i + 1] - ons[i]) / div for i in range(len(ons) - 1)]
    R["note_density"] = round(len(pit) / max(1e-9, total_beats), 3)
    R["ioi_mean"] = round(st.mean(ioi), 3) if ioi else 0.0
    R["ioi_entropy"] = ent(collections.Counter(round(x, 2) for x in ioi))
    dur = [d / div for (o, d, ch, p, v) in pit]
    R["note_len_entropy"] = ent(collections.Counter(round(x, 2) for x in dur))
    R["qualified_note_rate"] = round(sum(1 for x in dur if x >= 0.25) / len(dur), 4)
    nb = int(total_beats) + 1
    filled = set(int(o / div) for o in (n[0] for n in notes))
    R["empty_beat_rate"] = round(1 - len(filled) / max(1, nb), 4)
    bars = collections.Counter(int(o / div / 4) for o in (n[0] for n in notes))
    R["empty_bar_rate"] = round(1 - len(bars) / max(1, int(nb / 4) + 1), 4)
    pat = collections.defaultdict(set)
    for o in (n[0] for n in notes):
        pat[int(o / div / 4)].add(round((o % (div * 4)) / div * 4))
    ks = sorted(pat)
    sims = []
    for i in range(len(ks) - 1):
        a, b = pat[ks[i]], pat[ks[i + 1]]
        if a or b:
            sims.append(len(a & b) / len(a | b))
    R["groove_consistency"] = round(st.mean(sims), 3) if sims else 0.0

    if len(mel) > 12:
        pcs = [p % 12 for p in mel]
        ng = NGram(pcs, 12)
        ic, en, ics = ng.ic_entropy(pcs)
        R["surprisal_mean"] = round(ic, 3)
        R["pred_entropy_mean"] = round(en, 3)
        R["surprisal_std"] = round(st.pstdev(ics), 3)
        es = []
        for i, x in enumerate(pcs):
            ctx = pcs[max(0, i - 2) : i]
            ps = [ng.p(ctx, y) for y in range(12)]
            s = sum(ps) or 1
            e = -sum((q / s) * math.log2(q / s) for q in ps if q > 0)
            es.append(e)
        mu_i, mu_e = st.mean(ics), st.mean(es)
        num = sum((ics[i] - mu_i) * (es[i] - mu_e) for i in range(len(ics)))
        den = math.sqrt(sum((x - mu_i) ** 2 for x in ics) * sum((x - mu_e) ** 2 for x in es))
        R["ent_surp_corr"] = round(num / den, 3) if den else 0.0
        if len(ioi) > 12:
            q = [min(15, int(round(x * 4))) for x in ioi]
            ng2 = NGram(q, 16)
            ic2, en2, _ = ng2.ic_entropy(q)
            R["rhythm_surprisal"] = round(ic2, 3)
            R["rhythm_pred_entropy"] = round(en2, 3)
        else:
            R["rhythm_surprisal"] = 0.0
            R["rhythm_pred_entropy"] = 0.0
    else:
        for k in (
            "surprisal_mean",
            "pred_entropy_mean",
            "surprisal_std",
            "ent_surp_corr",
            "rhythm_surprisal",
            "rhythm_pred_entropy",
        ):
            R[k] = 0.0

    if len(mel) > 24:
        ivs = tuple(iv)
        L = 6
        grams = collections.Counter(tuple(ivs[i : i + L]) for i in range(len(ivs) - L))
        rep = sum(c for g, c in grams.items() if c > 1)
        R["motif_recurrence"] = round(rep / max(1, len(ivs) - L), 4)
        R["motif_top_count"] = max(grams.values()) if grams else 0
        R["exact_rep_rate"] = round(sum(c - 1 for c in grams.values() if c > 1) / max(1, len(ivs) - L), 4)
        S = 16
        seglen = max(1, len(mel) // S)
        segs = []
        for s in range(S):
            sl = mel[s * seglen : (s + 1) * seglen]
            if not sl:
                continue
            h = [0] * 12
            for p in sl:
                h[p % 12] += 1
            n2 = sum(h) or 1
            segs.append([x / n2 for x in h])
        sim = []
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                d2 = sum(min(segs[a][k], segs[b][k]) for k in range(12))
                sim.append(d2)
        R["self_similarity"] = round(st.mean(sim), 3) if sim else 0.0
        R["self_sim_std"] = round(st.pstdev(sim), 3) if len(sim) > 1 else 0.0
    else:
        for k in ("motif_recurrence", "motif_top_count", "exact_rep_rate", "self_similarity", "self_sim_std"):
            R[k] = 0.0

    R["_pc_hist"] = [round(x / (sum(hist) or 1), 4) for x in hist]
    ivh = collections.Counter(min(12, abs(x)) for x in iv)
    tot = sum(ivh.values()) or 1
    R["_iv_hist"] = [round(ivh.get(k, 0) / tot, 4) for k in range(13)]
    dh = collections.Counter(min(7, int(round(x * 2))) for x in dur)
    tot2 = sum(dh.values()) or 1
    R["_dur_hist"] = [round(dh.get(k, 0) / tot2, 4) for k in range(8)]
    return R
