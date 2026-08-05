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

import collections
import math
import re


def toks(s):
    return re.findall(r"[a-z0-9_]+", (s or "").lower())


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [toks(c["title"] + " " + c["content"]) for c in chunks]
        self.dl = [len(d) for d in self.docs]
        self.avgdl = sum(self.dl) / len(self.dl)
        self.df = collections.Counter()
        for d in self.docs:
            for w in set(d):
                self.df[w] += 1
        self.N = len(self.docs)
        self.idf = {w: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for w, df in self.df.items()}
        self.tf = [collections.Counter(d) for d in self.docs]

    def search(self, q, k=5):
        qt = toks(q)
        scores = []
        for i in range(self.N):
            s = 0.0
            tf = self.tf[i]
            dl = self.dl[i]
            for w in qt:
                if w in tf:
                    idf = self.idf.get(w, 0)
                    f = tf[w]
                    s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]
