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
"""MIDI feature computation: acoustic, tonal, rhythmic, and structural metrics."""

import collections
import math
import struct


def parse_midi(data):
    """Parse MIDI bytes into notes, programs, and tempo."""
    if data[:4] != b"MThd":
        return None
    _fmt, ntrk, div = struct.unpack(">HHH", data[8:14])
    if div == 0 or div & 0x8000:
        return None
    notes = []
    progs = {}
    tempos = []
    i = 14
    for _ in range(ntrk):
        if data[i : i + 4] != b"MTrk":
            break
        ln = struct.unpack(">I", data[i + 4 : i + 8])[0]
        end = i + 8 + ln
        j = i + 8
        t = 0
        last = None
        on = {}
        while j < end:
            dt = 0
            while j < end:
                b = data[j]
                j += 1
                dt = (dt << 7) | (b & 0x7F)
                if not b & 0x80:
                    break
            t += dt
            if j >= end:
                break
            st = data[j]
            if st == 0xFF:
                j += 1
                typ = data[j]
                j += 1
                l2 = 0
                while j < end:
                    b = data[j]
                    j += 1
                    l2 = (l2 << 7) | (b & 0x7F)
                    if not b & 0x80:
                        break
                if typ == 0x51 and l2 == 3:
                    tempos.append((t, (data[j] << 16) | (data[j + 1] << 8) | data[j + 2]))
                j += l2
            elif st in (0xF0, 0xF7):
                j += 1
                l2 = 0
                while j < end:
                    b = data[j]
                    j += 1
                    l2 = (l2 << 7) | (b & 0x7F)
                    if not b & 0x80:
                        break
                j += l2
            else:
                if st & 0x80:
                    last = st
                    j += 1
                s = last
                if s is None:
                    break
                k = s & 0xF0
                ch = s & 0x0F
                if k == 0xC0:
                    progs[ch] = data[j]
                    j += 1
                elif k == 0xD0:
                    j += 1
                elif k == 0x90:
                    p, v = data[j], data[j + 1]
                    j += 2
                    if v > 0:
                        on.setdefault((ch, p), []).append((t, v))
                    else:
                        q = on.get((ch, p))
                        if q:
                            t0, v0 = q.pop(0)
                            notes.append((t0, t - t0, ch, p, v0))
                elif k == 0x80:
                    p = data[j]
                    j += 2
                    q = on.get((ch, p))
                    if q:
                        t0, v0 = q.pop(0)
                        notes.append((t0, t - t0, ch, p, v0))
                else:
                    j += 2
        i = end
    notes.sort()
    return dict(div=div, notes=notes, progs=progs, tempo=(tempos[0][1] if tempos else 500000))


def f0(p):
    return 440.0 * (2 ** ((p - 69) / 12.0))


_S1, _S2, _B1, _B2, _DS = 0.0207, 18.96, 3.51, 5.75, 0.24


def _diss_pair(f1, f2, a1, a2):
    if f1 > f2:
        f1, f2, a1, a2 = f2, f1, a2, a1
    s = _DS / (_S1 * f1 + _S2)
    d = f2 - f1
    return a1 * a2 * (math.exp(-_B1 * s * d) - math.exp(-_B2 * s * d))


def roughness(pitches, vels=None, nharm=6):
    """Sethares roughness for a sonority."""
    if len(pitches) < 2:
        return 0.0
    parts = []
    for idx, p in enumerate(pitches):
        amp = (vels[idx] / 127.0) if vels else 1.0
        base = f0(p)
        for n in range(1, nharm + 1):
            parts.append((base * n, amp / n))
    tot = 0.0
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            tot += _diss_pair(parts[a][0], parts[b][0], parts[a][1], parts[b][1])
    norm = sum(x[1] for x in parts) ** 2
    return tot / norm if norm else 0.0


def harmonicity(pitches, nharm=8):
    """Virtual pitch prominence: how well harmonics align to a common F0."""
    if not pitches:
        return 0.0
    if len(pitches) == 1:
        return 1.0
    fs = [f0(p) for p in pitches]
    best = 0.0
    lo = min(pitches) - 24
    for cand in range(lo, min(pitches) + 1):
        cf = f0(cand)
        score = 0.0
        for f in fs:
            r = f / cf
            n = round(r)
            if n < 1 or n > nharm:
                continue
            dev = abs(r - n) / n
            if dev < 0.03:
                score += (1.0 - dev / 0.03) / math.sqrt(n)
        score /= sum(1.0 / math.sqrt(k + 1) for k in range(len(fs)))
        best = max(best, score)
    return best


_R, _H = 1.0, 0.4


def _spiral(pc_fifths):
    a = pc_fifths * math.pi / 2.0
    return (_R * math.sin(a), _R * math.cos(a), pc_fifths * _H)


_PC2F = {}
for k in range(-12, 13):
    _PC2F.setdefault((k * 7) % 12, k)


def _ce(pitches, weights=None):
    if not pitches:
        return None
    pts = []
    ws = []
    for i, p in enumerate(pitches):
        pts.append(_spiral(_PC2F[p % 12]))
        ws.append(weights[i] if weights else 1.0)
    s = sum(ws) or 1.0
    return tuple(sum(pt[d] * w for pt, w in zip(pts, ws, strict=False)) / s for d in range(3))


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False))) if a and b else 0.0


KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJ = [0, 2, 4, 5, 7, 9, 11]
MIN = [0, 2, 3, 5, 7, 8, 10]


def krumhansl_key(pc_hist):
    """Krumhansl-Schmuckler key detection. Returns (tonic, is_minor, r, runner_up_r)."""
    tot = sum(pc_hist) or 1
    x = [v / tot for v in pc_hist]
    res = []
    for minor, prof in ((0, KS_MAJ), (1, KS_MIN)):
        for t in range(12):
            pr = [prof[(k - t) % 12] for k in range(12)]
            mx, mp = sum(x) / 12, sum(pr) / 12
            num = sum((x[k] - mx) * (pr[k] - mp) for k in range(12))
            den = math.sqrt(sum((x[k] - mx) ** 2 for k in range(12)) * sum((pr[k] - mp) ** 2 for k in range(12)))
            res.append((num / den if den else 0.0, t, minor))
    res.sort(reverse=True)
    return res[0][1], res[0][2], res[0][0], res[1][0]


class NGram:
    """Leave-one-out order-2 variable-order n-gram model for internal predictability."""

    def __init__(self, seq, V):
        self.V = V
        self.uni = collections.Counter(seq)
        self.bi = collections.Counter(zip(seq, seq[1:], strict=False))
        self.tri = collections.Counter(zip(seq, seq[1:], seq[2:], strict=False))
        self.n = len(seq)

    def p(self, ctx, x):
        a = 0.4
        lam = 0.7
        pu = (self.uni[x] + a) / (self.n + a * self.V)
        if not ctx:
            return pu
        c1 = ctx[-1]
        cb = self.bi[(c1, x)]
        tb = sum(v for (u, _), v in self.bi.items() if u == c1)
        pb = (cb + a) / (tb + a * self.V) if tb else pu
        p = lam * pb + (1 - lam) * pu
        if len(ctx) >= 2:
            c2 = ctx[-2]
            ct = self.tri[(c2, c1, x)]
            tt = sum(v for (u, w, _), v in self.tri.items() if u == c2 and w == c1)
            if tt:
                pt = (ct + a) / (tt + a * self.V)
                p = lam * pt + (1 - lam) * p
        return max(p, 1e-9)

    def ic_entropy(self, seq):
        ics = []
        ents = []
        for i, x in enumerate(seq):
            ctx = seq[max(0, i - 2) : i]
            px = self.p(ctx, x)
            ics.append(-math.log2(px))
            ps = [self.p(ctx, y) for y in range(self.V)]
            s = sum(ps) or 1.0
            ents.append(-sum((q / s) * math.log2(q / s) for q in ps if q > 0))
        m = lambda v: sum(v) / len(v) if v else 0.0
        return m(ics), m(ents), ics
