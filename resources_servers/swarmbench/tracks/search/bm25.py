# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import heapq
import math
import pickle
import re
from array import array

STOP = set("a an and the of in on at to for by was is were be as with from that this it its which who".split())
K1, B = 1.2, 0.75


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def build(texts, path):
    # Postings live in compact arrays so a 40k-page index stays small and loads fast in every worker process.
    post, lens = {}, array("I")
    for d, text in enumerate(texts):
        toks = tokenize(text)
        lens.append(len(toks))
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            p = post.get(t)
            if p is None:
                p = post[t] = (array("I"), array("H"))
            p[0].append(d)
            p[1].append(min(c, 65535))
    with open(path, "wb") as f:
        pickle.dump({"post": post, "lens": lens}, f, protocol=5)


class Index:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.post, lens = data["post"], data["lens"]
        self.n = len(lens)
        avg = sum(lens) / self.n
        self.norm = [K1 * (1 - B + B * L / avg) for L in lens]

    def search(self, query, k):
        terms = sorted(set(tokenize(query)), key=lambda t: len(self.post.get(t, ((),))[0]))
        scores = {}
        for t in terms:
            p = self.post.get(t)
            if p is None:
                continue
            df = len(p[0])
            # Very common terms cost the most and rank the least; skip them once rarer terms have scored.
            if df > self.n // 5 and scores:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            norm = self.norm
            for d, tf in zip(*p):
                scores[d] = scores.get(d, 0.0) + idf * tf * (K1 + 1) / (tf + norm[d])
        return heapq.nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))
