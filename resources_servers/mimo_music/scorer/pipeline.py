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
"""Entry point: ABC extraction + two-gate scoring for verl RL."""

import json
import os
import re
import struct
import subprocess as sp
import tempfile
import traceback

from .feats import analyze
from .score import score as _score_v1


_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baselines", "ref_full4k.json")
_ref = None

_FENCE_RE = re.compile(r"```(?:abc|ABC)?\s*\n(.*?)```", re.DOTALL)
_XA_RE = re.compile(r"X\s*:\s*\d+")

_ABC2MIDI = os.environ.get("ABC2MIDI_BIN") or "abc2midi"


class Abc2MidiMissing(RuntimeError):
    """The `abc2midi` binary is absent from this process's PATH.

    Kept distinct from every other scoring failure because it is the one that
    must not be reported as a reward of 0.0: it applies to every sample rather
    than to one score, so it is indistinguishable from "the policy writes bad
    music" once flattened.
    """


def _get_ref():
    global _ref
    if _ref is None:
        with open(_BASELINE_PATH, encoding="utf-8") as f:
            _ref = json.load(f)
    return _ref


def extract_abc(text):
    if not text:
        return None
    for m in _FENCE_RE.finditer(text):
        body = m.group(1).strip()
        if _XA_RE.search(body):
            return body
    matches = list(_XA_RE.finditer(text))
    if matches:
        return text[matches[-1].start() :].strip()
    return None


def _midi_notes_progs(data):
    if data[:4] != b"MThd":
        return None
    _fmt, ntrk, div = struct.unpack(">HHH", data[8:14])
    if div == 0:
        return None
    notes = []
    progs = []
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
                j += 1
                l2 = 0
                while j < end:
                    b = data[j]
                    j += 1
                    l2 = (l2 << 7) | (b & 0x7F)
                    if not b & 0x80:
                        break
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
                    progs.append((ch, data[j]))
                    j += 1
                elif k == 0xD0:
                    j += 1
                elif k == 0x90:
                    n, v = data[j], data[j + 1]
                    j += 2
                    if v > 0:
                        on.setdefault((ch, n), []).append(t)
                    else:
                        q = on.get((ch, n))
                        if q:
                            t0 = q.pop(0)
                            notes.append((t0, t - t0, ch, n))
                elif k == 0x80:
                    n = data[j]
                    j += 2
                    q = on.get((ch, n))
                    if q:
                        t0 = q.pop(0)
                        notes.append((t0, t - t0, ch, n))
                else:
                    j += 2
        i = end
    return notes, progs


def do(rec):
    key = rec["key"]
    abc = rec.get("abc") or ""
    base = dict(
        key=key,
        id=rec["id"],
        rep=rec["rep"],
        tag=rec.get("tag"),
        lang=rec.get("lang"),
        abc_len=rec.get("abc_len", 0),
        nvoice=rec.get("nvoice", 0),
        latency=rec.get("latency"),
    )
    if not abc.strip():
        return {**base, "skip": "empty_abc"}

    ref = _get_ref()
    with tempfile.TemporaryDirectory() as td:
        ap = os.path.join(td, "a.abc")
        mp = os.path.join(td, "a.mid")
        open(ap, "w", encoding="utf-8").write(abc)
        try:
            p = sp.run([_ABC2MIDI, ap, "-o", mp], capture_output=True, timeout=60)
        except FileNotFoundError as e:
            raise Abc2MidiMissing(
                "abc2midi is not on PATH in this process. The scorer renders MIDI to "
                "extract features, so no score can be computed without it. Install the "
                "`abcmidi` package, or set ABC2MIDI_BIN and make sure PATH is forwarded "
                "to the Ray workers (a driver-side export does not reach them)."
            ) from e
        except Exception as e:
            return {**base, "skip": f"abc2midi:{repr(e)[:60]}"}
        log = (p.stdout + p.stderr).decode("utf-8", "replace")
        err = len(re.findall(r"^Error", log, re.M))
        bar = len(re.findall(r"Bar \d+ has", log))
        if not os.path.exists(mp):
            return {**base, "skip": "no_midi", "err": err, "bar": bar}
        with open(mp, "rb") as f:
            data = f.read()
        try:
            feat = analyze(mp)
        except Exception as e:
            feat = {"skip": f"analyze:{repr(e)[:60]}"}
        try:
            got = _midi_notes_progs(data)
        except Exception:
            got = None

    r = {**base, "err": err, "bar": bar, "blank": 1 if any(not line.strip() for line in abc.split("\n")[:-1]) else 0}

    if feat.get("skip"):
        r["scorer_skip"] = feat["skip"]
    else:
        s = _score_v1(feat, ref)
        r["total"] = s["total"]
        r["groups"] = s["groups"]
        r["n_chan"] = feat.get("n_chan")
        r["is_piano"] = feat.get("is_piano")

    if got and got[0]:
        notes, progs = got
        pit = [n[3] for n in notes]
        m = sum(pit) / len(pit)
        v = sum((x - m) ** 2 for x in pit) / len(pit)
        r["pit_min"] = min(pit)
        r["pit_max"] = max(pit)
        r["pit_range"] = max(pit) - min(pit)
        r["pit_std"] = round(v**0.5, 2)
        r["low_c3"] = round(sum(1 for p in pit if p < 48) / len(pit), 4)
        chp = {}
        for ch, pr in progs:
            chp.setdefault(ch, set()).add(pr)
        r["ch_conflict"] = sum(1 for ch, se in chp.items() if len(se) > 1)

    r["reject"] = (
        1 if (r.get("bar", 0) >= 10 or r.get("err", 0) > 0 or r.get("blank", 0) or r.get("ch_conflict", 0) > 0) else 0
    )
    return r


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    try:
        abc = extract_abc(solution_str)
        if not abc:
            return 0.0

        rec = {
            "key": "rl_rollout",
            "id": 0,
            "rep": 0,
            "abc": abc,
            "tag": None,
            "lang": None,
            "abc_len": len(abc),
            "nvoice": 0,
            "latency": None,
        }
        r = do(rec)

        if r.get("skip") or r.get("reject", 0):
            return 0.0
        if "total" not in r:
            return 0.0

        return max(0.0, min(1.0, float(r["total"]) / 100.0))
    except Abc2MidiMissing:
        raise
    except Exception:
        traceback.print_exc()
        return 0.0
