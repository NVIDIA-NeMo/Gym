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
RARE_DF = 25  # words this rare (names, streets) also get trigram postings, so misspelled queries still hit
FUZZY_W = 0.6


def digit_runs(low):
    # "+1 (415) 555-0134", "415.555.0134" and "304-58-1192" all index as one bare digit token.
    out = []
    for m in re.finditer(r"\+?\d[\d ().-]{5,}\d", low):
        d = re.sub(r"\D", "", m.group())
        out.append(d[1:] if len(d) == 11 and d[0] == "1" else d)
    return out


EMAIL_RE = re.compile(r"[a-z0-9._+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")


def tokenize(text):
    # Emails are whole tokens only, so a company's domain does not pull in every employee's signature.
    low = text.lower()
    emails = EMAIL_RE.findall(low)
    words = [t for t in re.findall(r"[a-z0-9]+", EMAIL_RE.sub(" ", low)) if t not in STOP]
    return words + emails + digit_runs(low)


def trigrams(w):
    w = f"^{w}$"
    return {"~" + w[i:i + 3] for i in range(len(w) - 2)}


def fuzzy_words(toks):
    return {t for t in toks if len(t) >= 4 and t.isalpha()}


def build(texts, path):
    docs = [tokenize(t) for t in texts]
    df = {}
    for toks in docs:
        for w in fuzzy_words(toks):
            df[w] = df.get(w, 0) + 1
    post, lens = {}, array("I")
    for d, toks in enumerate(docs):
        lens.append(len(toks))
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for w in fuzzy_words(toks):
            if df[w] <= RARE_DF:
                for g in trigrams(w):
                    tf[g] = tf.get(g, 0) + 1
        for t, c in tf.items():
            p = post.get(t)
            if p is None:
                p = post[t] = (array("I"), array("H"))
            p[0].append(d)
            p[1].append(min(c, 65535))
    with open(path, "wb") as f:
        # sorted keys: the term insertion order follows set iteration, so it changed with PYTHONHASHSEED
        pickle.dump({"post": dict(sorted(post.items())), "lens": lens}, f, protocol=5)


class Index:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.post, lens = data["post"], data["lens"]
        self.n = len(lens)
        avg = sum(lens) / self.n
        self.norm = [K1 * (1 - B + B * L / avg) for L in lens]

    def _add(self, scores, term, weight, allowed):
        p = self.post.get(term)
        if p is None:
            return
        df = len(p[0])
        idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5)) * weight
        norm = self.norm
        for d, tf in zip(*p):
            if allowed is None or d in allowed:
                scores[d] = scores.get(d, 0.0) + idf * tf * (K1 + 1) / (tf + norm[d])

    def search(self, query, k, allowed=None):
        toks = tokenize(query)
        scores = {}
        for t in set(toks):
            self._add(scores, t, 1.0, allowed)
        for w in fuzzy_words(toks):
            grams = trigrams(w)
            for g in grams:
                self._add(scores, g, FUZZY_W / len(grams), allowed)
        return heapq.nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))
