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
import random

# template -> (hops, subject kind, kind of asserted value)
TEMPLATES = {
    "person_city": (1, "people", "city"),
    "person_year": (1, "people", "year"),
    "company_year": (1, "companies", "year"),
    "company_hq": (1, "companies", "city"),
    "event_city": (1, "events", "city"),
    "founder_city": (2, "companies", "city"),
    "founder_year": (2, "companies", "year"),
    "event_country": (2, "events", "country"),
    "hq_country": (2, "companies", "country"),
    "founder_country": (3, "companies", "country"),
    "founder_host": (3, "companies", "city"),
}
HOP_WEIGHTS = {1: 0.35, 2: 0.4, 3: 0.25}
LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def truth(w, tpl, s):
    cities = w["cities"]
    founder = lambda: w["people"][s["founder"]] if s.get("founder") is not None else {}
    country = lambda c: None if c is None else cities[c]["country"]
    return {
        "person_city": lambda: s["birth_city"],
        "person_year": lambda: s["birth_year"],
        "company_year": lambda: s["founded"],
        "company_hq": lambda: s["hq"],
        "event_city": lambda: s["host"],
        "founder_city": lambda: founder().get("birth_city"),
        "founder_year": lambda: founder().get("birth_year"),
        "event_country": lambda: country(s["host"]),
        "hq_country": lambda: country(s["hq"]),
        "founder_country": lambda: country(founder().get("birth_city")),
        "founder_host": lambda: founder().get("birth_city"),
    }[tpl]()


def distractors(w, tpl, s, rng):
    # Values that sit near the subject: same-name people, sibling companies, other editions, prefixed city variants.
    kind = TEMPLATES[tpl][2]
    cities = w["cities"]
    out = []
    if kind == "year":
        base = truth(w, tpl, s) or rng.randint(1880, 2000)
        out += [base + d for d in (-11, -7, -3, -1, 1, 2, 5, 9)]
        return out
    ids = []
    if s["kind"] == "person":
        ids += s["lived_in"] + [p["birth_city"] for p in w["by_name"][s["name"]] if p is not s]
    if s["kind"] == "company":
        ids += [c["hq"] for c in w["by_root"][s["title"].split()[0]] if c is not s]
        if s["founder"] is not None:
            ids += w["people"][s["founder"]]["lived_in"]
    if s["kind"] == "event":
        ids += [w["events"][e]["host"] for e in s["editions"]]
    ids = [c for c in ids if c is not None]
    t = truth(w, tpl, s)
    if kind == "city" and isinstance(t, int):
        ids += w["city_variants"].get(cities[t]["title"].split()[-1], [])
    if kind == "country":
        return [cities[c]["country"] for c in ids]
    return ids


def display(w, kind, v):
    if kind == "city":
        return w["cities"][v]["title"]
    return str(v)


def accepted(w, kind, v):
    if kind == "city":
        c = w["cities"][v]
        return [c["title"]] + ([c["alias"]] if "alias" in c else [])
    return [str(v)]


def sentence(tpl, subj, val):
    return {
        "person_city": f"{subj} was born in {val}.",
        "person_year": f"{subj} was born in {val}.",
        "company_year": f"{subj} was founded in {val}.",
        "company_hq": f"{subj} is headquartered in {val}.",
        "event_city": f"The {subj} was held in {val}.",
        "founder_city": f"The founder of {subj} was born in {val}.",
        "founder_year": f"The founder of {subj} was born in {val}.",
        "event_country": f"The {subj} was held in a city in {val}.",
        "hq_country": f"{subj} is headquartered in a city in {val}.",
        "founder_country": f"The founder of {subj} was born in a city in {val}.",
        "founder_host": f"The founder of {subj} was born in the city that hosted the {val}.",
    }[tpl]


def index_world(w):
    w["by_name"], w["by_root"], w["city_variants"] = {}, {}, {}
    for p in w["people"]:
        w["by_name"].setdefault(p["name"], []).append(p)
    for c in w["companies"]:
        w["by_root"].setdefault(c["title"].split()[0], []).append(c)
    for i, c in enumerate(w["cities"]):
        w["city_variants"].setdefault(c["title"].split()[-1], []).append(i)


def sample_claims(w, city_events, n, seed):
    rng = random.Random(seed * 7919 + n)
    index_world(w)
    labels = [LABELS[i % 3] for i in range(n)]
    rng.shuffle(labels)
    by_hop = {}
    for tpl, (h, _, _) in TEMPLATES.items():
        by_hop.setdefault(h, []).append(tpl)
    hosted = [i for i, e in enumerate(w["events"]) if e["host"] is not None]
    seen, claims = set(), []
    for i, label in enumerate(labels):
        while True:
            hop = rng.choices(list(HOP_WEIGHTS), weights=list(HOP_WEIGHTS.values()))[0]
            tpl = rng.choice(by_hop[hop])
            _, pool, kind = TEMPLATES[tpl]
            s = rng.choice(w[pool])
            if (tpl, s["title"]) in seen:
                continue
            t = truth(w, tpl, s)
            if (t is None) != (label == "NOT_ENOUGH_INFO"):
                continue
            if tpl == "founder_host":
                if label == "SUPPORTED":
                    if not city_events.get(t):
                        continue
                    ev = rng.choice(city_events[t])
                else:
                    ev = rng.choice(hosted)
                    if w["events"][ev]["host"] == t:
                        continue
                asserted = w["events"][ev]["title"]
            else:
                if label == "SUPPORTED":
                    v = t
                else:
                    cands = [d for d in distractors(w, tpl, s, rng) if d != t]
                    if not cands or rng.random() < 0.25:
                        cands = {"city": range(len(w["cities"])), "country": w["countries"],
                                 "year": range(1860, 2015)}[kind]
                    v = rng.choice(list(cands))
                    if v == t:
                        continue
                asserted = display(w, kind, v)
            break
        seen.add((tpl, s["title"]))
        subj = s["title"]
        if s["kind"] == "company" and "alias" in s and rng.random() < 0.35:
            subj = s["alias"]
        claims.append({"id": f"C{i + 1:04d}", "claim": sentence(tpl, subj, asserted), "label": label,
                       "template": tpl, "hops": TEMPLATES[tpl][0], "subject": s["title"],
                       "value": display(w, kind, t) if label == "REFUTED" else None,
                       "accepted": accepted(w, kind, t) if label == "REFUTED" else []})
    return claims
